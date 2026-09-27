"""Unified End-to-End Entity Disambiguation and Linking Pipeline.

Orchestrates:
1. Named Entity Recognition (NEREngine with domain lexicons and context extraction)
2. Live Web Knowledge Retrieval (WebKnowledgeRetriever via MediaWiki & Wikipedia REST APIs)
3. Context-Aware Semantic Disambiguation (EntityDisambiguator via TF-IDF, token overlap, and type priors)
4. Interactive Knowledge Graph Construction (DisambiguationGraphBuilder via Plotly, NetworkX, and streamlit-agraph)
"""

from typing import List, Dict, Any, Optional
import time
import html
import re

from src.ner_engine import NEREngine
from src.web_retriever import WebKnowledgeRetriever
from src.disambiguator import EntityDisambiguator
from src.graph_builder import DisambiguationGraphBuilder


class EntityDisambiguationPipeline:
    """End-to-End Pipeline for Context-Aware NER, Entity Linking, and Disambiguation."""

    CATEGORY_TO_LABEL = {
        "Technology Company": "ORG",
        "Company": "ORG",
        "Automotive / Brand": "ORG",
        "Fruit / Agriculture": "PRODUCT",
        "Animal / Biology": "ANIMAL",
        "Country / Geography": "GPE",
        "Athlete / Basketball": "PERSON",
        "River / Geography": "LOC",
        "Nature / Geography": "LOC",
        "E-Commerce / Cloud": "ORG",
        "Scientist / Inventor": "PERSON",
        "Drug": "DRUG",
        "Disease": "DISEASE"
    }

    # Keywords in Wikipedia/web extracts that strongly signal a PERSON entity
    PERSON_SIGNAL_WORDS = {
        "professor", "assistant professor", "associate professor", "lecturer",
        "born", "died", "politician", "athlete", "footballer", "cricketer",
        "actor", "actress", "singer", "musician", "director", "author",
        "scientist", "researcher", "engineer", "doctor", "physician",
        "lawyer", "journalist", "businessman", "entrepreneur",
        "linkedin", "biography", "alumnus", "alumni", "graduated",
        "she is", "he is", "she was", "he was", "her career", "his career",
        "instagram", "twitter", "facebook profile"
    }

    # Keywords in extracts that strongly signal an ORG (guard against false PERSON)
    ORG_SIGNAL_WORDS = {
        "founded", "incorporated", "headquarters", "subsidiary", "revenue",
        "employees", "stock exchange", "listed on", "merged with", "acquired"
    }

    def __init__(
        self,
        similarity_threshold: float = 0.15,
        max_candidates: int = 4,
        context_window_size: int = 50,
        enable_cache: bool = True
    ):
        """Initialize pipeline components.

        Args:
            similarity_threshold: Minimum similarity threshold for disambiguation confidence.
            max_candidates: Number of candidate meanings to retrieve per mention.
            context_window_size: Character radius around mentions for local context.
            enable_cache: Whether to cache web knowledge retrieval results.
        """
        self.ner_engine = NEREngine()
        self.web_retriever = WebKnowledgeRetriever()
        self.disambiguator = EntityDisambiguator(similarity_threshold=similarity_threshold)
        self.graph_builder = DisambiguationGraphBuilder()
        self.max_candidates = max_candidates
        self.context_window_size = context_window_size
        self.enable_cache = enable_cache

    def process_text(
        self,
        text: str,
        generate_graph: bool = True
    ) -> Dict[str, Any]:
        """Process an input text through the entire NER and Disambiguation pipeline.

        Args:
            text: Input sentence or document text.
            generate_graph: Whether to generate Plotly and Agraph knowledge graph structures.

        Returns:
            Dictionary containing:
                - input_text: Original text
                - entities: List of disambiguated entity objects with candidate rankings
                - summary_metrics: Counts, average confidence, latency, etc.
                - html_annotated_text: HTML markup with color-coded entity pills
                - plotly_figure: Plotly figure object (if generate_graph=True)
                - agraph_nodes_edges: Tuple of (nodes, edges) for streamlit-agraph
        """
        start_time = time.time()
        if not text or not text.strip():
            return {
                "input_text": text,
                "entities": [],
                "summary_metrics": {
                    "total_entities": 0,
                    "ambiguous_entities": 0,
                    "avg_confidence": 0.0,
                    "processing_time_ms": 0.0,
                    "type_counts": {}
                },
                "html_annotated_text": text,
                "plotly_figure": None,
                "agraph_nodes_edges": ([], [])
            }

        # Step 1: NER Extraction with context windows
        raw_entities = self.ner_engine.extract_entities(
            text,
            context_window_size=self.context_window_size
        )

        # Step 1.5a: Extract text-evidence signals from the full input (runs ONCE)
        # This lets every entity benefit from the full context of the user's input
        input_signals = self.ner_engine.extract_input_signals(text)

        # Step 1.5b: Merge adjacent PERSON spans into full names
        # e.g. ["Kadavala", "Bhavani", "Sirisha"] → ["Kadavala Bhavani Sirisha"]
        raw_entities = self._merge_adjacent_person_spans(text, raw_entities)

        # Step 2 & 3: Retrieval and Disambiguation per entity
        processed_entities = []
        search_logs = []
        ambiguous_count = 0
        total_confidence = 0.0
        type_counts = {}

        for ent in raw_entities:
            mention_text = ent["text"]
            t_search_start = time.time()

            # Fetch external knowledge candidates with live search telemetry
            retrieval_data = self.web_retriever.fetch_candidates_with_telemetry(
                mention_text,
                max_candidates=self.max_candidates
            )
            candidates = retrieval_data["candidates"]
            telemetry = retrieval_data["telemetry"]
            search_latency = telemetry.get("latency_ms", round((time.time() - t_search_start) * 1000, 2))

            # Enrich entity context with its own full name for better TF-IDF scoring
            # When input is short (e.g. just a name list), sentence_context is almost empty
            # → inject the full mention text + full original input for richer context
            ent_for_disambig = dict(ent)
            if len(ent_for_disambig.get("sentence_context", "").split()) < 8:
                ent_for_disambig["sentence_context"] = (
                    mention_text + " " + text
                ).strip()[:300]

            # Disambiguate against context
            disambig_result = self.disambiguator.disambiguate_entity(ent_for_disambig, candidates)

            # ----------------------------------------------------------------
            # ADAPTIVE CONFIDENCE THRESHOLD
            # Determined purely from the user's input text — no external signals.
            # ----------------------------------------------------------------
            best_score = disambig_result.get("selected_match", {}).get("confidence_score", 0.0)
            is_grounded = disambig_result.get("is_grounded", len(candidates) > 0)
            ent_label = disambig_result.get("label", ent.get("label", ""))
            is_single_token = telemetry.get("is_single_token_query", False)
            mention_token_count = len(mention_text.split())

            # Base threshold
            threshold = 0.08

            if ent_label == "PERSON":
                # Rule A: Pure name-list input (no verbs, no sentence structure)
                # "kadavala bhavani sirisha" typed alone → very strict
                if input_signals["is_name_list_only"]:
                    threshold = 0.25

                # Rule B: Input contains student/intern/private-person context words
                elif input_signals["private_person_keywords"]:
                    threshold = 0.22

                # Rule C: Single-token in a RICH SENTENCE with public-figure signals
                # e.g. "Jordan played a key role when Apple launched..."
                # Sentence has verbs AND public signals → famous person reference,
                # NOT a bare private-name query. Use relaxed threshold.
                elif (
                    is_single_token
                    and input_signals["has_sentence_structure"]
                    and input_signals["public_figure_keywords"]
                ):
                    threshold = 0.06  # relaxed: famous person in sentence context

                # Rule C2: Single-token with sentence structure but no public signals
                # Mentioned casually → moderate threshold
                elif is_single_token and input_signals["has_sentence_structure"]:
                    threshold = 0.10

                # Rule C3: Single-token with NO sentence structure (bare name query)
                # User just typed "Jordan" alone → strict (prevent false Wikipedia match)
                elif is_single_token:
                    threshold = 0.20

                # Rule D: Multi-word name (3+ words) → Wikipedia must match most words
                elif mention_token_count >= 3:
                    threshold = 0.18

                # Rule E: Input clearly talks about public figures
                elif input_signals["public_figure_keywords"]:
                    threshold = 0.06  # relaxed — prominent person expected

            # Full-name coverage check for PERSON multi-word mentions:
            # If the best Wikipedia title shares < 50% of the mention's words, demote.
            # Exception: public-figure sentence context → short-forms like "Jordan" for
            # "Michael Jordan" are valid references, so skip the coverage penalty.
            if ent_label == "PERSON" and mention_token_count >= 2:
                best_title = disambig_result.get("selected_match", {}).get("title", "")
                m_words = set(mention_text.lower().split())
                t_words = set(best_title.lower().split())
                word_coverage = len(m_words & t_words) / len(m_words) if m_words else 0.0
                if word_coverage < 0.5 and not input_signals["public_figure_keywords"]:
                    # Fewer than 50% of mention words appear in the Wikipedia title
                    # and no public-figure signals in context → treat as private
                    threshold = max(threshold, 0.35)

            # ----------------------------------------------------------------
            private_individual_signal = telemetry.get("private_individual_signal", False)


            def _make_private_entry(mention: str, score: float, label: str = "PERSON") -> dict:
                """Build a standardised 'unlinked' result dict with correct entity-type messaging."""
                if label == "GPE":
                    kind = "Local / Regional Place"
                    note = (
                        "This place name could not be confidently matched to any verified "
                        "Wikipedia article. It may be a local village, mandal, or geographic "
                        "feature not yet indexed in public knowledge bases."
                    )
                elif label == "ORG":
                    kind = "Local / Unverified Organisation"
                    note = (
                        "This organisation could not be matched to any verified Wikipedia entry. "
                        "It may be a local business, institution, or organisation not yet "
                        "indexed in public knowledge bases."
                    )
                else:
                    kind = "Private / Local Individual"
                    note = (
                        "This entity could not be confidently matched to any verified Wikipedia "
                        "article. It may be a private individual, local entity, or a name not "
                        "yet indexed in public knowledge bases."
                    )
                return {
                    "title": mention,
                    "description": "No reliable public knowledge entry found",
                    "extract": note,
                    "url": f"https://en.wikipedia.org/wiki/Special:Search?search={mention.replace(' ', '_')}",
                    "confidence_score": score,
                    "category": kind,
                    "is_grounded": False
                }

            selected = disambig_result.get("selected_match", {})
            is_web_cand = selected.get("_is_web_grounded", False)
            cand_title = selected.get("title", "")

            # Check name coverage in candidate title
            m_words = set(re.findall(r"\b\w+\b", mention_text.lower()))
            t_words = set(re.findall(r"\b\w+\b", cand_title.lower()))
            coverage = len(m_words & t_words) / len(m_words) if m_words else 0.0

            if is_web_cand and coverage >= 0.5 and best_score >= 0.12:
                # Grounded via verified web / LinkedIn / Academic profile
                is_grounded = True
                disambig_result["is_grounded"] = True
            elif is_grounded and best_score < threshold:
                # Score below adaptive text-evidence threshold → demote to unlinked
                is_grounded = False
                disambig_result["is_grounded"] = False
                disambig_result["selected_match"] = _make_private_entry(mention_text, best_score, ent_label)
            elif not is_grounded:
                disambig_result["selected_match"] = _make_private_entry(mention_text, best_score, ent_label)

            # Check if candidate category can refine the entity label
            selected = disambig_result.get("selected_match", {})
            selected_cat = selected.get("category", "")
            refined_label = disambig_result["label"]

            for cat_key, mapped_label in self.CATEGORY_TO_LABEL.items():
                if cat_key.lower() in selected_cat.lower():
                    refined_label = mapped_label
                    break

            disambig_result["label"] = refined_label

            # --- Web-grounding label override ---------------------------------
            # If NER still shows ORG but the retrieved candidate content contains
            # strong PERSON signals (e.g. "professor", "born", LinkedIn mention),
            # correct the label now before rendering.
            web_corrected_label = self._infer_label_from_candidate(refined_label, selected)
            if web_corrected_label != refined_label:
                disambig_result["label"] = web_corrected_label
                refined_label = web_corrected_label
            # ------------------------------------------------------------------

            disambig_result["start_char"] = ent["start_char"]
            disambig_result["end_char"] = ent["end_char"]
            disambig_result["sentence_context"] = ent["sentence_context"]
            disambig_result["local_context"] = ent["local_context"]
            disambig_result["ner_source"] = ent["source"]
            disambig_result["is_grounded"] = is_grounded
            disambig_result["search_telemetry"] = telemetry

            if disambig_result.get("is_ambiguous", False):
                ambiguous_count += 1

            conf = selected.get("confidence_score", 0.0) if is_grounded else 0.0
            disambig_result["confidence_score"] = conf
            total_confidence += conf

            type_counts[refined_label] = type_counts.get(refined_label, 0) + 1
            processed_entities.append(disambig_result)

            search_logs.append({
                "entity": mention_text,
                "label": refined_label,
                "engine": telemetry.get("engine", "MediaWiki OpenSearch API"),
                "endpoint_url": telemetry.get("endpoint_url", ""),
                "candidates_retrieved": len(candidates),
                "top_title": selected.get("title", "No Match") if is_grounded else "No Real-World Match",
                "top_url": selected.get("url", "") if is_grounded else "",
                "score": conf,
                "latency_ms": search_latency,
                "is_grounded": is_grounded,
                "status": "200 OK (Grounded in Wikipedia)" if is_grounded else "404 (No Real-World Knowledge Entry)"
            })

        # Step 4: Summary Metrics
        total_ents = len(processed_entities)
        avg_confidence = round(total_confidence / total_ents, 4) if total_ents > 0 else 0.0
        latency_ms = round((time.time() - start_time) * 1000, 2)

        summary_metrics = {
            "total_entities": total_ents,
            "ambiguous_entities": ambiguous_count,
            "avg_confidence": avg_confidence,
            "processing_time_ms": latency_ms,
            "type_counts": type_counts,
            "grounding_rate": round(sum(1 for s in search_logs if s.get("is_grounded", False)) / total_ents * 100, 1) if total_ents > 0 else 0.0
        }

        # Step 5: Generate Visual Artifacts
        html_annotated = self._generate_html_annotations(text, processed_entities)

        plotly_fig = None
        agraph_components = ([], [])
        if generate_graph and processed_entities:
            try:
                plotly_fig = self.graph_builder.build_plotly_figure(processed_entities, text)
            except Exception:
                plotly_fig = None

            try:
                agraph_components = self.graph_builder.build_agraph_components(processed_entities, text)
            except Exception:
                agraph_components = ([], [])

        return {
            "input_text": text,
            "entities": processed_entities,
            "search_logs": search_logs,
            "summary_metrics": summary_metrics,
            "html_annotated_text": html_annotated,
            "plotly_figure": plotly_fig,
            "agraph_nodes_edges": agraph_components
        }

    def process_batch(
        self,
        texts: List[str],
        generate_graphs: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple texts sequentially."""
        return [self.process_text(t, generate_graph=generate_graphs) for t in texts]

    # ------------------------------------------------------------------
    # Adjacent PERSON Span Merger
    # ------------------------------------------------------------------

    def _merge_adjacent_person_spans(self, text: str, entities: list) -> list:
        """Merge consecutive spans that together form a single full name.

        Handles the common South-Asian name pattern where a surname is tagged
        as GPE or ORG by spaCy (e.g. "Kadavala [GPE] Bhavani [PERSON] Sirisha [GPE]"
        → "Kadavala Bhavani Sirisha [PERSON]").

        Merging rules:
          - At least one span in the group must be PERSON.
          - Adjacent spans may be PERSON, GPE, or ORG *only if* they are
            single-word, title-case, alpha-only tokens (surname/firstname pattern).
          - The gap between consecutive spans must be whitespace-only.
          - The combined span must be ≤ 6 words.

        Args:
            text: Original input text.
            entities: List of extracted entity dicts (sorted by start_char).

        Returns:
            Updated entity list with merged spans where applicable.
        """
        if len(entities) < 2:
            return entities

        def _is_name_token(ent: dict) -> bool:
            """Return True if this entity span looks like a personal name word."""
            t = ent.get("text", "")
            lbl = ent.get("label", "")
            words = t.strip().split()
            if lbl == "PERSON":
                return True
            # Single-word GPE/ORG that is title-case and alpha-only
            # → very likely a misclassified Indian surname or given name
            if lbl in {"GPE", "ORG"} and len(words) == 1:
                w = words[0]
                if w[0].isupper() and not w.isupper() and w.isalpha() and len(w) >= 3:
                    return True
            return False

        merged = []
        skip = set()

        for i, ent in enumerate(entities):
            if i in skip:
                continue

            # Only start a merge group from a PERSON span
            if ent.get("label") != "PERSON":
                # GPE/ORG that looks like a name-token might be the FIRST word
                # of a name (surname-first pattern). Check if next is PERSON.
                if _is_name_token(ent) and (i + 1) < len(entities):
                    next_ent = entities[i + 1]
                    if next_ent.get("label") == "PERSON":
                        gap = text[ent["end_char"]: next_ent["start_char"]]
                        if gap.strip() == "":
                            # Treat this span as the start of a name group
                            pass  # fall through to merge loop below
                        else:
                            merged.append(ent)
                            continue
                    else:
                        merged.append(ent)
                        continue
                else:
                    merged.append(ent)
                    continue

            # Try to extend this span by absorbing consecutive name-token spans
            combined_text = ent["text"]
            combined_start = ent["start_char"]
            combined_end = ent["end_char"]
            combined_sent = ent.get("sentence_context", "")
            has_person = ent.get("label") == "PERSON"

            j = i + 1
            while j < len(entities) and j not in skip:
                next_ent = entities[j]
                if not _is_name_token(next_ent):
                    break
                if next_ent.get("label") == "PERSON":
                    has_person = True
                # Gap between current end and next start
                gap = text[combined_end: next_ent["start_char"]]
                if gap.strip() != "":
                    gap_words = [w.lower() for w in gap.strip().split()]
                    if not all(w in self.ner_engine.INDIAN_GIVEN_NAMES or w in self.ner_engine.INDIAN_SURNAMES for w in gap_words):
                        break  # non-name words between them → different names
                word_count = len((combined_text + " " + next_ent["text"]).split())
                if word_count > 6:
                    break  # too long to be a single name
                # Merge
                combined_text = combined_text + gap + next_ent["text"]
                combined_end = next_ent["end_char"]
                skip.add(j)
                j += 1

            # Only keep the merge if the group contains at least one PERSON span
            if not has_person:
                merged.append(ent)
                continue

            # Re-compute context window for the merged span
            cw = self.context_window_size
            c_start = max(0, combined_start - cw)
            c_end = min(len(text), combined_end + cw)
            merged_local = text[c_start:c_end].strip()

            merged_ent = dict(ent)
            merged_ent["text"] = combined_text
            merged_ent["label"] = "PERSON"   # merged span is always a person
            merged_ent["start_char"] = combined_start
            merged_ent["end_char"] = combined_end
            merged_ent["local_context"] = merged_local
            merged_ent["sentence_context"] = combined_sent
            merged.append(merged_ent)

        return merged

    # ------------------------------------------------------------------
    # Web-Grounding Label Override
    # ------------------------------------------------------------------

    def _infer_label_from_candidate(self, current_label: str, candidate: dict) -> str:
        """Override a potentially wrong NER label using signals from the web candidate.

        Looks at the candidate's *description* and *extract* text for vocabulary
        that strongly indicates a PERSON or ORG, regardless of what spaCy or the
        category-mapping decided.  Operates only when there is genuine ambiguity
        (i.e. the candidate content provides a clear signal in the opposite direction).

        Args:
            current_label: The label chosen after NER + category-map refinement.
            candidate: The selected disambiguation candidate dict.

        Returns:
            Potentially corrected label string.
        """
        description = (candidate.get("description", "") or "").lower()
        extract = (candidate.get("extract", "") or "").lower()
        combined = description + " " + extract

        person_hits = sum(1 for sig in self.PERSON_SIGNAL_WORDS if sig in combined)
        org_hits = sum(1 for sig in self.ORG_SIGNAL_WORDS if sig in combined)

        # Only override when person signals clearly dominate
        if person_hits >= 1 and org_hits == 0 and current_label == "ORG":
            return "PERSON"

        # Guard: if mislabelled as PERSON but org signals dominate, correct back
        if org_hits >= 2 and person_hits == 0 and current_label == "PERSON":
            return "ORG"

        return current_label


    def _generate_html_annotations(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create styled HTML badges highlighting entities in the text."""
        if not entities:
            return html.escape(text)

        # Sort entities by start_char
        sorted_ents = sorted(entities, key=lambda x: x["start_char"])
        html_parts = []
        last_idx = 0

        # Badge color theme
        LABEL_STYLES = {
            "ORG": ("#2563EB", "#DBEAFE", "#1E40AF"),      # Blue
            "PERSON": ("#7C3AED", "#EDE9FE", "#5B21B6"),   # Purple
            "GPE": ("#0891B2", "#CFFAFE", "#155E75"),      # Cyan
            "LOC": ("#0D9488", "#CCFBF1", "#115E59"),      # Teal
            "PRODUCT": ("#D97706", "#FEF3C7", "#92400E"),  # Amber
            "DRUG": ("#DB2777", "#FCE7F3", "#9D174D"),     # Pink
            "DISEASE": ("#DC2626", "#FEE2E2", "#991B1B"),  # Red
            "ANIMAL": ("#059669", "#D1FAE5", "#065F46"),   # Emerald
            "INDEX": ("#4F46E5", "#EEF2FF", "#3730A3"),    # Indigo
            "CURRENCY": ("#16A34A", "#DCFCE7", "#166534")  # Green
        }
        DEFAULT_STYLE = ("#64748B", "#F1F5F9", "#334155")

        for ent in sorted_ents:
            start = ent["start_char"]
            end = ent["end_char"]

            if start < last_idx:
                continue

            # Add plain text before mention
            html_parts.append(html.escape(text[last_idx:start]))

            label = ent.get("label", "ENTITY")
            selected = ent.get("selected_match", {})
            title = html.escape(selected.get("title", ent["entity_text"]))
            score = selected.get("confidence_score", 0.0)
            url = selected.get("url", "#")

            is_grounded = ent.get("is_grounded", True)
            if is_grounded:
                border_c, bg_c, text_c = LABEL_STYLES.get(label, DEFAULT_STYLE)
                badge_html = (
                    f'<span style="background-color: {bg_c}; color: {text_c}; '
                    f'border: 1px solid {border_c}; border-radius: 6px; padding: 2px 6px; '
                    f'margin: 0 2px; font-weight: 600; display: inline-block;">'
                    f'<a href="{url}" target="_blank" style="color: {text_c}; text-decoration: none;" '
                    f'title="Verified Real-World Grounding: {title} | Confidence: {score:.2f} | Label: {label}">'
                    f'{html.escape(text[start:end])} '
                    f'<span style="font-size: 0.75em; opacity: 0.85; border-left: 1px solid {border_c}; '
                    f'padding-left: 4px; margin-left: 2px;">{label}</span>'
                    f'</a></span>'
                )
            else:
                badge_html = (
                    f'<span style="background-color: #F8FAFC; color: #64748B; '
                    f'border: 1px dashed #94A3B8; border-radius: 6px; padding: 2px 6px; '
                    f'margin: 0 2px; font-weight: 500; display: inline-block;" '
                    f'title="Unlinked Entity: No verified article found in public knowledge base">'
                    f'{html.escape(text[start:end])} '
                    f'<span style="font-size: 0.75em; color: #94A3B8; border-left: 1px dashed #CBD5E1; '
                    f'padding-left: 4px; margin-left: 2px;">{label} (Unlinked)</span>'
                    f'</span>'
                )
            html_parts.append(badge_html)
            last_idx = end

        # Add remaining text
        if last_idx < len(text):
            html_parts.append(html.escape(text[last_idx:]))

        return "".join(html_parts)


if __name__ == "__main__":
    pipeline = EntityDisambiguationPipeline()
    sample = "Steve Jobs and Steve Wozniak founded Apple in Cupertino, while doctors at Mayo Clinic prescribed Metformin."
    res = pipeline.process_text(sample)
    print("\n--- Pipeline Execution Output ---")
    print(f"Entities Found: {len(res['entities'])}")
    for e in res["entities"]:
        match = e["selected_match"]
        print(f"[{e['label']}] '{e['entity_text']}' -> '{match.get('title')}' (Score: {match.get('confidence_score')}, Ambiguous: {e['is_ambiguous']})")
    print(f"Metrics: {res['summary_metrics']}")

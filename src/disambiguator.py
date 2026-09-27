"""Entity Disambiguation and Context-Scoring Engine.

Computes semantic similarity between the input context window and candidate knowledge
extracts using TF-IDF Cosine Similarity, Jaccard Token Overlap, and Entity Type Priors.
Determines whether an entity mention is ambiguous and selects the single context-correct referent.
"""

from typing import List, Dict, Any, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


class EntityDisambiguator:
    """Ranks real-world candidate entities based on surrounding context."""

    LABEL_TYPE_KEYWORDS = {
        "ORG": {"company", "corporation", "organization", "enterprise", "firm", "agency", "inc", "ltd", "headquarters", "office", "ceo", "founded", "employees"},
        "GPE": {"country", "city", "nation", "state", "capital", "territory", "republic", "kingdom"},
        "LOC": {"river", "mountain", "sea", "ocean", "desert", "region", "rainforest", "continent"},
        "PERSON": {
            "player", "athlete", "inventor", "scientist", "actor", "musician", "politician", "founder", "executive",
            "professor", "assistant professor", "associate professor", "engineer", "student", "intern",
            "developer", "scholar", "researcher", "faculty", "academic", "undergraduate", "alumni", "lecturer"
        },
        "DRUG": {"medication", "drug", "pharmaceutical", "treatment", "medicine", "compound", "capsule", "tablet"},
        "DISEASE": {"disease", "infection", "virus", "condition", "illness", "syndrome", "disorder"},
        "ANIMAL": {"species", "mammal", "cat", "animal", "wildlife", "carnivore", "predator", "fauna"},
        "PRODUCT": {"device", "software", "smartphone", "vehicle", "car", "model", "sedan", "hardware", "gadget", "brand"}
    }

    # Context signals that strongly indicate a company/org referent
    # Used to re-score PRODUCT-labelled mentions that are actually companies
    COMPANY_CONTEXT_SIGNALS = {
        "launched", "office", "headquarters", "ceo", "revenue", "employees",
        "announced", "stock", "investors", "acquired", "partnership", "corporate",
        "quarter", "earnings", "billion", "million", "founded", "startup"
    }

    # Minimum score a single retrieved candidate must reach to be considered grounded.
    # Below this, even a lone Wikipedia result is treated as an irrelevant page.
    SINGLE_CANDIDATE_MIN_SCORE = 0.08

    def __init__(self, similarity_threshold: float = 0.15):
        """Initialize disambiguator.

        Args:
            similarity_threshold: Minimum score threshold to consider strong disambiguation confidence
        """
        self.similarity_threshold = similarity_threshold

    def disambiguate_entity(
        self,
        entity_info: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Disambiguate an entity mention against candidate real-world meanings.

        Args:
            entity_info: Dict with mention 'text', 'label', 'sentence_context', 'local_context'
            candidates: List of candidate dicts from WebKnowledgeRetriever

        Returns:
            Disambiguation outcome with:
                - is_ambiguous (bool)
                - selected_match (dict with best candidate and score)
                - all_candidates (list of candidates with individual scores and rankings)
        """
        if not candidates:
            return {
                "entity_text": entity_info["text"],
                "label": entity_info.get("label", "UNKNOWN"),
                "is_ambiguous": False,
                "is_grounded": False,
                "selected_match": {
                    "title": entity_info["text"],
                    "description": "No external knowledge candidate found in public knowledge base",
                    "extract": f"Context: {entity_info.get('sentence_context', '')}",
                    "url": f"https://en.wikipedia.org/wiki/Special:Search?search={entity_info['text']}",
                    "confidence_score": 0.0,
                    "category": "Unlinked / Local Entity",
                    "is_grounded": False
                },
                "all_candidates": []
            }

        # Single candidate case: compute a real score rather than blindly assigning 0.95.
        # A lone Wikipedia result can still be the wrong article (e.g. a disambiguation page
        # or an unrelated topic that happens to share the entity's surface form).
        if len(candidates) == 1:
            input_context = (
                entity_info.get("sentence_context", "") + " " + entity_info.get("local_context", "")
            ).strip()
            label = entity_info.get("label", "")
            cand = dict(candidates[0])
            cand_text = f"{cand.get('title', '')} {cand.get('description', '')} {cand.get('extract', '')}"

            cos_scores = self._compute_tfidf_cosine_similarity(input_context, [cand_text])
            cos_sim = cos_scores[0]
            jaccard = self._compute_token_overlap(input_context.lower(), cand_text.lower())
            type_prior = self._compute_type_prior(label, cand_text.lower(), cand.get("category", ""), input_context)
            title_prior = self._compute_title_exactness_prior(
                entity_info["text"], cand.get("title", ""), input_context
            )

            # Use same weights as multi-candidate path (45/25/15/15)
            computed_score = round(
                min(1.0, max(0.0,
                    (0.45 * cos_sim) +
                    (0.25 * jaccard) +
                    (0.15 * title_prior) +
                    (0.15 * type_prior)
                )),
                4
            )

            cand["confidence_score"] = computed_score
            cand["tfidf_cosine"] = round(cos_sim, 4)
            cand["token_overlap"] = round(jaccard, 4)
            cand["title_prior"] = round(title_prior, 4)
            cand["type_prior"] = round(type_prior, 4)
            cand["rank"] = 1
            cand["is_selected"] = True
            is_grounded_single = computed_score >= self.SINGLE_CANDIDATE_MIN_SCORE
            cand["is_grounded"] = is_grounded_single
            return {
                "entity_text": entity_info["text"],
                "label": label,
                "is_ambiguous": False,
                "is_grounded": is_grounded_single,
                "selected_match": cand,
                "all_candidates": [cand]
            }

        # Multi-candidate case: Ambiguity detection & context scoring
        input_context = (entity_info.get("sentence_context", "") + " " + entity_info.get("local_context", "")).strip()
        label = entity_info.get("label", "")

        scored_candidates = []
        candidate_texts = []
        for cand in candidates:
            # Combine title, description, and extract into searchable text
            combined_text = f"{cand.get('title', '')} {cand.get('description', '')} {cand.get('extract', '')}"
            candidate_texts.append(combined_text)

        # 1. Cosine similarity using TF-IDF
        cosine_scores = self._compute_tfidf_cosine_similarity(input_context, candidate_texts)

        # 2. Token overlap, Title Exactness & Type Prior
        for idx, cand in enumerate(candidates):
            cos_sim = cosine_scores[idx]
            cand_text = candidate_texts[idx].lower()
            cand_title = cand.get("title", "")

            jaccard_score = self._compute_token_overlap(input_context.lower(), cand_text)
            type_prior_score = self._compute_type_prior(label, cand_text, cand.get("category", ""), input_context)

            title_prior_score = self._compute_title_exactness_prior(entity_info["text"], cand_title, input_context)

            # Composite score weights match the documented formula:
            # Score(c,W) = 0.45·TF-IDF + 0.25·Jaccard + 0.15·TitleExact + 0.15·TypePrior
            composite_score = (
                (0.45 * cos_sim) +
                (0.25 * jaccard_score) +
                (0.15 * title_prior_score) +
                (0.15 * type_prior_score)
            )

            # No artificial floor — let low-evidence entities score near 0
            # so the confidence guard in the pipeline can correctly mark them as unlinked
            normalized_score = round(min(1.0, max(0.0, composite_score)), 4)

            cand_entry = dict(cand)
            cand_entry["confidence_score"] = normalized_score
            cand_entry["tfidf_cosine"] = round(cos_sim, 4)
            cand_entry["token_overlap"] = round(jaccard_score, 4)
            cand_entry["title_prior"] = round(title_prior_score, 4)
            cand_entry["type_prior"] = round(type_prior_score, 4)
            scored_candidates.append(cand_entry)

        # Sort candidates by composite confidence score descending
        scored_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)

        for rank, cand in enumerate(scored_candidates, 1):
            cand["rank"] = rank
            cand["is_selected"] = (rank == 1)

        selected_match = scored_candidates[0]

        # Entity is genuinely ambiguous only when:
        #   (a) There are at least 2 candidates that both score above a meaningful floor, AND
        #   (b) The top-2 candidates belong to different semantic categories.
        # This prevents trivially-ranked alternatives (score ~ 0) from setting the flag.
        AMBIG_SCORE_FLOOR = 0.05
        is_ambiguous = False
        if len(scored_candidates) >= 2:
            second = scored_candidates[1]
            top_score = selected_match.get("confidence_score", 0.0)
            second_score = second.get("confidence_score", 0.0)
            top_cat = selected_match.get("category", "").lower()
            second_cat = second.get("category", "").lower()
            both_viable = top_score >= AMBIG_SCORE_FLOOR and second_score >= AMBIG_SCORE_FLOOR
            diff_categories = top_cat != second_cat
            is_ambiguous = both_viable and diff_categories

        return {
            "entity_text": entity_info["text"],
            "label": entity_info.get("label", "UNKNOWN"),
            "is_ambiguous": is_ambiguous,
            "is_grounded": True,
            "selected_match": selected_match,
            "all_candidates": scored_candidates
        }

    def _compute_tfidf_cosine_similarity(self, context: str, candidate_texts: List[str]) -> List[float]:
        """Compute TF-IDF cosine similarity between context and candidates."""
        corpus = [context] + candidate_texts
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            context_vec = tfidf_matrix[0:1]
            cand_vecs = tfidf_matrix[1:]
            sims = cosine_similarity(context_vec, cand_vecs)[0]
            return [float(s) for s in sims]
        except Exception:
            return [0.1] * len(candidate_texts)

    def _compute_token_overlap(self, context: str, candidate_text: str) -> float:
        """Compute Jaccard token overlap between context words and candidate text."""
        ctx_words = {w for w in re.findall(r"\b\w+\b", context) if w not in STOP_WORDS and len(w) > 2}
        cand_words = {w for w in re.findall(r"\b\w+\b", candidate_text) if w not in STOP_WORDS and len(w) > 2}

        if not ctx_words or not cand_words:
            return 0.0

        intersection = ctx_words & cand_words
        union = ctx_words | cand_words
        return len(intersection) / len(union) if union else 0.0

    def _compute_type_prior(self, label: str, candidate_text: str, category: str,
                            context: str = "") -> float:
        """Calculate prior boost if candidate semantics match the predicted NER label.

        Also detects when a PRODUCT-labelled mention is contextually a company/org
        (e.g. 'apple launched its office') and boosts the corporate candidate accordingly.
        """
        combined = (candidate_text + " " + category).lower()
        context_lower = context.lower()

        # Detect company-context signals in the surrounding sentence
        company_context_hit = any(
            sig in context_lower for sig in self.COMPANY_CONTEXT_SIGNALS
        )

        # If PRODUCT-labelled entity is in a company context, treat like ORG for scoring
        effective_label = label
        if label == "PRODUCT" and company_context_hit:
            effective_label = "ORG"

        if not effective_label or effective_label not in self.LABEL_TYPE_KEYWORDS:
            return 0.2

        target_keywords = self.LABEL_TYPE_KEYWORDS[effective_label]
        match_count = sum(1 for kw in target_keywords if kw in combined)
        base_score = min(1.0, 0.25 * match_count)

        # Extra boost: if company context is active AND candidate is explicitly a company,
        # ensure it gets at least 0.5 so it can beat a generic article on pure TF-IDF
        if company_context_hit and any(
            kw in combined for kw in {"company", "corporation", "inc", "ltd", "technology", "software"}
        ):
            base_score = max(base_score, 0.50)

        # In corporate context without musical cues, prioritize commercial/technology enterprises over entertainment companies
        if company_context_hit and any(kw in combined for kw in {"music", "entertainment"}) and not any(
            m in context_lower for m in {"music", "song", "album", "band", "beatles", "record", "concert"}
        ):
            base_score = max(0.2, base_score * 0.7)

        return base_score

    def _compute_title_exactness_prior(self, mention: str, title: str, context: str) -> float:
        """Compute title exactness prior with full multi-word name matching.

        Improvements over the base version:
        - Full word-overlap ratio for multi-word names (handles "Kadavala Bhavani Sirisha")
        - Penalises qualifier articles (disambiguation pages) unless context supports them
        - Partial overlap scoring for long names
        """
        m_clean = mention.strip().lower()
        t_clean = title.strip().lower()

        # Exact match
        if t_clean == m_clean:
            return 1.0

        # Base title before disambiguation parenthesis, dashes, or pipes (web profiles)
        base_title = t_clean.split(" (")[0].split(" - ")[0].split(" | ")[0].strip()
        if base_title == m_clean:
            if "(" in t_clean and ")" in t_clean:
                qualifier = t_clean.split("(")[1].split(")")[0].strip()
                return 0.9 if qualifier in context.lower() else 0.25
            return 1.0

        # Multi-word overlap ratio: what fraction of mention words appear in title?
        m_words = set(m_clean.split())
        t_words = set(base_title.split())
        full_t_words = set(re.findall(r"\b\w+\b", t_clean))
        if len(m_words) > 1:
            overlap = max(
                len(m_words & t_words) / len(m_words) if m_words else 0.0,
                len(m_words & full_t_words) / len(m_words) if m_words else 0.0
            )
            if overlap >= 0.8:
                return 0.90
            if overlap >= 0.5:
                return 0.60
            if overlap >= 0.25:
                return 0.35

        # Single word: substring match
        if m_clean in t_clean:
            return 0.5

        return 0.1


if __name__ == "__main__":
    from src.web_retriever import WebKnowledgeRetriever

    disambiguator = EntityDisambiguator()
    retriever = WebKnowledgeRetriever()

    # Ambiguity test 1: Apple in tech context
    cands = retriever.fetch_candidates("Apple")
    ent1 = {
        "text": "Apple",
        "label": "ORG",
        "sentence_context": "Apple announced quarterly iPhone earnings and Mac revenue in Cupertino.",
        "local_context": "Apple announced quarterly iPhone earnings"
    }
    res1 = disambiguator.disambiguate_entity(ent1, cands)
    print(f"\n[Tech Context] Selected: {res1['selected_match']['title']} (Score: {res1['selected_match']['confidence_score']})")

    # Ambiguity test 2: Apple in agricultural fruit context
    ent2 = {
        "text": "apple",
        "label": "PRODUCT",
        "sentence_context": "The orchard harvest yielded sweet fresh red apple cider and fruit baskets.",
        "local_context": "sweet fresh red apple cider"
    }
    res2 = disambiguator.disambiguate_entity(ent2, cands)
    print(f"[Fruit Context] Selected: {res2['selected_match']['title']} (Score: {res2['selected_match']['confidence_score']})")

"""Named Entity Recognition (NER) Engine.

Extracts context-aware entity mentions using spaCy (en_core_web_sm)
augmented with domain-specific patterns (Healthcare, Finance, Sales, Products).
Captures precise sentence-level and local context windows for downstream disambiguation.
Includes post-processing heuristics to fix common misclassifications (e.g. person names
labelled as ORG by the base model when the name is not in its training vocabulary).
"""

from typing import List, Dict, Any, Optional
import re
import spacy


class NEREngine:
    """Context-Aware Named Entity Recognition engine with domain enrichment."""

    # Common domain terms to augment base model
    # Specialized domain lexicon (unambiguous entities often missed by general NER)
    DOMAIN_LEXICON = {
        # Healthcare / Biotech
        "metformin": ("DRUG", "Healthcare"),
        "aspirin": ("DRUG", "Healthcare"),
        "paracetamol": ("DRUG", "Healthcare"),
        "ibuprofen": ("DRUG", "Healthcare"),
        "covid-19": ("DISEASE", "Healthcare"),
        "diabetes": ("DISEASE", "Healthcare"),
        "biontech": ("ORG", "Healthcare"),
        "pfizer": ("ORG", "Healthcare"),
        "moderna": ("ORG", "Healthcare"),
        "mayo clinic": ("ORG", "Healthcare"),
        "cdc": ("ORG", "Healthcare"),
        "who": ("ORG", "Healthcare"),
        # Finance
        "nasdaq": ("ORG", "Finance"),
        "nyse": ("ORG", "Finance"),
        "blackrock": ("ORG", "Finance"),
        "goldman sachs": ("ORG", "Finance"),
        "federal reserve": ("ORG", "Finance"),
        "wall street": ("LOC", "Finance"),
        "s&p 500": ("INDEX", "Finance"),
        "bitcoin": ("CURRENCY", "Finance"),
        # Sales & Tech
        "salesforce": ("ORG", "Sales"),
        "shopify": ("ORG", "Sales"),
        "iphone": ("PRODUCT", "Technology"),
        "ipad": ("PRODUCT", "Technology"),
        "macbook": ("PRODUCT", "Technology"),
        # Major global brands often typed lowercase
        "google": ("ORG", "Technology"),
        "microsoft": ("ORG", "Technology"),
        "amazon": ("ORG", "Technology"),
        "tesla": ("ORG", "Technology"),
        "meta": ("ORG", "Technology"),
        "netflix": ("ORG", "Technology"),
        "twitter": ("ORG", "Technology"),
        "uber": ("ORG", "Technology"),
        "openai": ("ORG", "Technology"),
        "nvidia": ("ORG", "Technology"),
        "samsung": ("ORG", "Technology"),
        "sony": ("ORG", "Technology"),
        "toyota": ("ORG", "Automotive"),
        "nasa": ("ORG", "Space"),
        "spacex": ("ORG", "Space"),
        "un": ("ORG", "International")
    }

    # Fallback for lowercase polysemous terms — these fire ONLY when spaCy misses the entity.
    # 'apple' as ORG: when context has company signals the disambiguator will pick Apple Inc.
    # 'apple' as PRODUCT: kept for fruit contexts (orchard, cider, harvest etc.)
    # The company context detection in disambiguator.py handles the final call.
    POLYSEMOUS_FALLBACK = {
        "apple": ("ORG", "Technology"),   # default: company (disambiguator adjusts for fruit context)
        "jaguar": ("ORG", "Automotive"),  # Jaguar Cars most common; ANIMAL for wildlife contexts
        "amazon": ("ORG", "Technology"),  # E-commerce giant; LOC (river) resolved by context
        "oracle": ("ORG", "Technology"),
        "adobe": ("ORG", "Technology"),
        "shell": ("ORG", "Energy"),
        "mars": ("GPE", "Space"),          # Planet most commonly; candy bar resolved by context
        "mercury": ("GPE", "Space"),
    }

    # Stopwords and noise spans to filter from named entity extraction
    NOISE_MENTIONS = {"ph.d.", "ph.d", "crm", "ceo", "cfo", "cto", "quarterly", "annual", "b2b"}

    # Labels we typically want to retain for entity linking
    VALID_LABELS = {
        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "FAC", "NORP", "MONEY", "LAW", "LANGUAGE", "DRUG", "DISEASE",
        "INDEX", "CURRENCY", "ANIMAL"
    }

    # Suffixes that strongly indicate a non-person (organisation) span
    ORG_SUFFIXES = {
        "inc", "llc", "ltd", "corp", "co", "plc", "gmbh", "pvt",
        "group", "holdings", "technologies", "solutions", "services",
        "university", "institute", "foundation", "trust", "association",
        "committee", "council", "ministry", "department", "bureau",
        "academy", "school", "college", "hospital", "clinic", "bank",
        "fund", "capital", "ventures", "labs", "systems", "networks",
        "media", "studios", "press", "news", "party", "union"
    }

    # Common Indian/Telugu/South-Asian given names that spaCy frequently
    # misclassifies as GPE or ORG because they are not in its Western corpus.
    # When a single token matches one of these, it is promoted to PERSON.
    INDIAN_GIVEN_NAMES = {
        # Telugu / Andhra Pradesh names
        "sirisha", "bhavani", "deepthi", "deepti", "priyanka", "mounika",
        "sowmya", "sravani", "sravanthi", "swathi", "swetha", "tejaswi",
        "pooja", "lavanya", "keerthi", "harsha", "harshi", "manasa",
        "ramya", "revathi", "rekha", "padma", "pallavi", "sunitha",
        "suneetha", "anitha", "usha", "vasudha", "vaishnavi",
        "nandini", "niharika", "nithya", "kavya", "kavitha", "kaveri",
        "rithika", "ritika", "rohini", "gayatri", "geetha", "geeta",
        "meghana", "megha", "madhuri", "madhavi", "mahima", "malathi",
        "lalitha", "latha", "jyothi", "jyoti", "indira", "indrani",
        "himabindu", "hema", "divya", "devi", "durga", "chaitanya",
        "chaitra", "bindu", "anusha", "anushka", "anjali", "akshitha",
        "aishwarya", "aditi", "akhila", "alekhya", "amulya",
        # Male given names
        "satya", "kadavala", "marada", "venkata", "srinivas", "srikanth",
        "suresh", "ramesh", "mahesh", "ganesh", "naresh", "rajesh",
        "rakesh", "lokesh", "hitesh", "dinesh", "umesh", "nilesh",
        "ravi", "kiran", "arun", "varun", "tarun", "karun",
        "prasad", "prashanth", "prashant", "pavan", "naveen", "praveen",
        "sandeep", "sanjeev", "sanjay", "vijay", "ajay", "nikhil",
        "akhil", "rohit", "mohit", "sohit", "anand", "aakash",
        "arjun", "abhishek", "aditya", "akash", "amith", "anil",
        "anirudh", "anurag", "ashwin", "balaji", "bharath", "bharat",
        "charan", "deepak", "dheeraj", "dileep", "girish", "gopal",
        "hari", "harish", "jagadeesh", "jagadish", "jayaram", "krishna",
        "kumar", "lokesh", "madhu", "manish", "manoj", "murali",
        "nagendra", "narayana", "narendra", "nitin", "omkar", "phanindra",
        "phani", "pradeep", "rajesh", "rama", "ramana", "ramu",
        "reddy", "sagar", "sarath", "satish", "shankar", "shiva",
        "shivam", "siddharth", "siva", "sudhir", "sunil", "suresh",
        "surya", "tirumala", "uday", "vamsi", "venkat", "vikram",
        "vikas", "vinay", "vinayak", "vishal", "vivek", "yashwanth",
    }

    # Known Indian surname patterns — words that appear as FIRST token but are surnames
    # e.g. "Kadavala" in "Kadavala Bhavani Sirisha"
    INDIAN_SURNAMES = {
        "kadavala", "marada", "bandla", "miriyala", "nadella", "reddy", "sharma", "gupta", "singh",
        "kumar", "rao", "naidu", "raju", "babu", "das", "varma",
        "verma", "patel", "joshi", "mishra", "iyer", "iyengar",
        "pillai", "nair", "menon", "krishnan", "subramaniam",
        "venkatesh", "venkataraman", "murthy", "swamy", "choudhary",
        "chowdary", "goud", "yadav", "jain", "agarwal", "agarwala",
        "bhat", "bhatt", "dixit", "srivastava", "shukla", "tiwari",
        "chauhan", "thakur", "pandey", "dubey", "saxena", "khatri",
        "anand", "kapoor", "malhotra", "chopra", "mehta", "seth",
        "bose", "chatterjee", "banerjee", "chakraborty", "mukherjee",
        "ghosh", "roy", "sen", "das", "dutta", "sarkar",
    }

    # Personal honorifics / titles that strongly suggest a PERSON span
    PERSON_TITLES = {
        "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
        "sir", "dame", "lord", "lady", "rev", "reverend",
        "capt", "captain", "col", "colonel", "gen", "general",
        "sgt", "sergeant", "officer", "inspector", "commissioner"
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy NLP model."""
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str, context_window_size: int = 50) -> List[Dict[str, Any]]:
        """Extract named entities with context windows and classifications.

        If the entire input is lowercase (e.g. user typed ``deepthi marada``),
        a title-cased copy is passed to spaCy so that the model recognises
        proper nouns correctly.  All character offsets are reported against the
        *original* (unmodified) text so that UI highlighting remains accurate.
        """
        if not text or not text.strip():
            return []

        # --- Input normalisation: title-case purely-lowercase text ----------
        # spaCy's ``en_core_web_sm`` depends heavily on capitalisation to
        # detect PERSON / ORG spans. When the user types in all lowercase
        # or mixes casing (e.g. 'Deepthi marada'), proper nouns are normalized.
        normalized_text = text
        if text == text.lower() and any(c.isalpha() for c in text):
            # Title-case each word that is currently all-lowercase
            normalized_text = re.sub(
                r"\b([a-z])([a-z']*)\b",
                lambda m: m.group(1).upper() + m.group(2),
                text
            )
        else:
            # Also title-case any lowercase name tokens (e.g. 'marada' in 'Deepthi marada')
            normalized_text = re.sub(
                r"\b([a-z])([a-z']*)\b",
                lambda m: (m.group(1).upper() + m.group(2)) if (m.group(0).lower() in self.INDIAN_GIVEN_NAMES or m.group(0).lower() in self.INDIAN_SURNAMES) else m.group(0),
                text
            )
        # Run spaCy on the (possibly normalised) text; use original for context
        doc = self.nlp(normalized_text)
        entities = []
        covered_indices = set()

        # Step 1: Pre-check specialized domain lexicon (e.g. drugs, finance indexes)
        lower_text = text.lower()
        domain_matches = []
        for term, (domain_label, domain_cat) in self.DOMAIN_LEXICON.items():
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, lower_text):
                start, end = match.span()
                matched_text = text[start:end]
                domain_matches.append({
                    "text": matched_text,
                    "label": domain_label,
                    "start_char": start,
                    "end_char": end,
                    "source": f"domain_{domain_cat.lower()}"
                })

        # Step 1b: Pre-check multi-word Indian / South-Asian personal name sequences
        indian_name_matches = self._detect_indian_names(text, normalized_text)

        # Step 2: SpaCy entities
        spacy_ents = []
        for ent in doc.ents:
            clean_text = ent.text.strip()
            ent_start = ent.start_char
            ent_end = ent.end_char

            # Strip leading English determiners and cardinal numbers ("the", "a", "an", "seven", "two", "7", etc.)
            det_match = re.match(r'^(?:the|a|an|one|two|three|four|five|six|seven|eight|nine|ten|several|many|\d+)\s+', clean_text, re.IGNORECASE)
            if det_match:
                prefix_len = len(det_match.group(0))
                if len(clean_text) > prefix_len + 1:
                    ent_start += prefix_len
                    clean_text = clean_text[prefix_len:].strip()

            # Filter noise words, pure numbers, or invalid labels
            if clean_text.lower() in self.NOISE_MENTIONS:
                continue
            if ent.label_ in {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "DATE", "TIME"}:
                continue
            if ent.label_ not in self.VALID_LABELS:
                continue

            # If spaCy span swallows a known distinct domain entity together with extra words, drop the over-grouped span
            swallowed = [dm for dm in domain_matches if dm["start_char"] >= ent_start and dm["end_char"] <= ent_end]
            if len(swallowed) > 1 or (len(swallowed) == 1 and (len(clean_text) > len(swallowed[0]["text"]) + 6)):
                continue

            # Post-processing: correct label if heuristics suggest misclassification
            corrected_label = self._reclassify_label(clean_text, ent.label_, text)

            spacy_ents.append({
                "text": clean_text,
                "label": corrected_label,
                "start_char": ent_start,
                "end_char": ent_end,
                "source": "spacy_ner"
            })

        # Step 3: Polysemous lowercase fallback (only if not already caught by spaCy)
        poly_matches = []
        for term, (fallback_label, fallback_cat) in self.POLYSEMOUS_FALLBACK.items():
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, text):  # note: case sensitive or lower if not uppercase
                start, end = match.span()
                matched_text = text[start:end]
                # Only if not uppercase (uppercase is handled by spaCy as ORG/PRODUCT)
                if matched_text.islower():
                    poly_matches.append({
                        "text": matched_text,
                        "label": fallback_label,
                        "start_char": start,
                        "end_char": end,
                        "source": f"polysemous_{fallback_cat.lower()}"
                    })

        # Step 4: Combine candidates prioritizing domain lexicon, Indian names, spaCy, then fallback
        all_candidates = domain_matches + indian_name_matches + spacy_ents + poly_matches
        # Sort by start_char, and by length descending to prefer longer/more specific spans
        all_candidates.sort(key=lambda x: (x["start_char"], -(x["end_char"] - x["start_char"])))

        final_raw_ents = []
        for cand in all_candidates:
            cand_indices = set(range(cand["start_char"], cand["end_char"]))
            if not (cand_indices & covered_indices):
                final_raw_ents.append(cand)
                covered_indices.update(cand_indices)

        # Sort by start position
        final_raw_ents.sort(key=lambda x: x["start_char"])

        # Format with sentence context and local context window
        for ent in final_raw_ents:
            start = ent["start_char"]
            end = ent["end_char"]

            # Local context window around mention (from original text)
            c_start = max(0, start - context_window_size)
            c_end = min(len(text), end + context_window_size)
            local_context = text[c_start:c_end].strip()

            # Find containing sentence (doc built from normalized text, same offsets)
            sent_context = text
            for sent in doc.sents:
                if sent.start_char <= start and sent.end_char >= end:
                    sent_context = sent.text.strip()
                    break

            entities.append({
                "text": ent["text"],
                "label": ent["label"],
                "start_char": start,
                "end_char": end,
                "sentence_context": sent_context,
                "local_context": local_context,
                "source": ent["source"]
            })

        return entities

    # ------------------------------------------------------------------
    # Indian / South-Asian Personal Name Detection
    # ------------------------------------------------------------------

    def _detect_indian_names(self, text: str, normalized_text: str) -> List[Dict[str, Any]]:
        """Detect multi-word Indian personal name spans that spaCy often fragments or mislabels."""
        words = list(re.finditer(r"\b[A-Za-z'-]+\b", text))
        matches = []
        if len(words) < 2:
            return matches

        NON_NAME_WORDS = self.NOISE_MENTIONS | {
            "the", "and", "in", "on", "at", "for", "of", "with", "is", "was",
            "are", "were", "by", "from", "to", "a", "an", "as", "into", "reported",
            "said", "announced", "visited", "met", "founded", "launched", "worked",
            "city", "state", "district", "country", "local", "cafe", "hospital",
            "school", "college", "university", "department", "company", "firm",
            "friend", "colleague", "student", "teacher", "professor", "doctor",
            "assistant", "associate", "intern", "engineer", "dear", "young"
        }

        i = 0
        while i < len(words):
            found_span = False
            for span_len in (4, 3, 2):
                if i + span_len <= len(words):
                    span_words = words[i:i + span_len]
                    is_adjacent = True
                    for k in range(len(span_words) - 1):
                        gap = text[span_words[k].end(): span_words[k + 1].start()]
                        if gap.strip() != "":
                            is_adjacent = False
                            break
                    if not is_adjacent:
                        continue

                    lower_tokens = [w.group(0).lower() for w in span_words]
                    if any(t in NON_NAME_WORDS for t in lower_tokens):
                        continue

                    # First token must be a known Indian name
                    if (lower_tokens[0] not in self.INDIAN_GIVEN_NAMES and
                        lower_tokens[0] not in self.INDIAN_SURNAMES):
                        continue

                    lex_hits = sum(
                        1 for t in lower_tokens
                        if t in self.INDIAN_GIVEN_NAMES or t in self.INDIAN_SURNAMES
                    )

                    is_valid_name = False
                    if lex_hits >= 2:
                        is_valid_name = True
                    elif lex_hits >= 1 and span_len == 2:
                        norm_tokens = [normalized_text[w.start():w.end()] for w in span_words]
                        if all(nt[0].isupper() and nt.isalpha() for nt in norm_tokens):
                            is_valid_name = True

                    if is_valid_name:
                        start = span_words[0].start()
                        end = span_words[-1].end()
                        matched_text = normalized_text[start:end]
                        matches.append({
                            "text": matched_text,
                            "label": "PERSON",
                            "start_char": start,
                            "end_char": end,
                            "source": "indian_name_pattern"
                        })
                        i += span_len
                        found_span = True
                        break
            if not found_span:
                i += 1

        return matches

    # ------------------------------------------------------------------
    # Input Signal Extraction — text-evidence based entity profiling
    # ------------------------------------------------------------------

    def extract_input_signals(self, text: str) -> dict:
        """Analyse the raw user input to extract semantic signals that inform grounding decisions.

        This is called ONCE per pipeline run (not per entity) and returns a signals
        dict that all entities in that run can use to self-calibrate their confidence
        thresholds — without relying on any external data source.

        Signals returned:
        - is_name_list_only (bool): Input looks like a bare list of names with no sentence context
        - private_person_keywords (list): Words like "student", "intern", "engineer" found in text
        - public_figure_keywords (list): Words like "CEO", "president", "founded", "elected" found
        - word_count (int): Total word count of input
        - avg_word_length (float): Average word length (short = likely names, long = prose)
        - has_sentence_structure (bool): Input contains a verb/predicate (proper sentence)
        """
        lower = text.lower()
        words = re.findall(r"\b\w+\b", lower)
        word_count = len(words)

        # Signals that suggest entities are private individuals
        PRIVATE_SIGNALS = {
            "student", "undergraduate", "intern", "internship", "fresher",
            "graduate", "postgraduate", "btech", "b.tech", "mtech", "m.tech",
            "mca", "bca", "b.sc", "m.sc", "bachelors", "masters", "phd",
            "employee", "associate", "trainee", "junior", "senior engineer",
            "software engineer", "data analyst", "developer", "programmer",
            "candidate", "applicant", "resume", "cv", "linkedin",
            "college", "university student", "school", "class of",
            "lumora", "systems pvt", "pvt ltd", "private limited"
        }

        # Signals that suggest entities are public/notable figures
        PUBLIC_SIGNALS = {
            "ceo", "cto", "cfo", "president", "prime minister", "minister",
            "senator", "governor", "mayor", "chairman", "founder", "co-founder",
            "elected", "appointed", "awarded", "nobel", "oscar", "grammy",
            "billion", "million", "company", "corporation", "organization",
            "announced", "launched", "published", "authored", "directed",
            "won", "championship", "olympic", "world record"
        }

        # Detect verb presence (simple heuristic: common English verbs)
        COMMON_VERBS = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do",
            "said", "says", "founded", "built", "created", "announced", "works",
            "studied", "earned", "received", "launched", "reported", "joined"
        }

        private_found = [w for w in PRIVATE_SIGNALS if w in lower]
        public_found  = [w for w in PUBLIC_SIGNALS  if w in lower]
        has_verb      = any(v in words for v in COMMON_VERBS)
        avg_len       = sum(len(w) for w in words) / max(1, word_count)

        # A "name list only" input: short, no verbs, no punctuation except commas
        stripped = re.sub(r"[,\s]+", " ", text.strip())
        all_title_case_words = all(
            w[0].isupper() for w in stripped.split() if w.isalpha() and len(w) > 1
        )
        is_name_list_only = (
            word_count <= 12
            and not has_verb
            and all_title_case_words
            and "." not in text
        )

        return {
            "is_name_list_only": is_name_list_only,
            "private_person_keywords": private_found,
            "public_figure_keywords": public_found,
            "word_count": word_count,
            "avg_word_length": round(avg_len, 2),
            "has_sentence_structure": has_verb,
        }

    # ------------------------------------------------------------------
    # Label Reclassification Heuristics
    # ------------------------------------------------------------------

    def _reclassify_label(self, span: str, current_label: str, full_text: str) -> str:
        """Correct common NER misclassifications using rule-based heuristics.

        The spaCy ``en_core_web_sm`` model is trained primarily on Western
        newswire corpora and often mislabels non-English personal names
        (Indian, African, East-Asian, etc.) as ``ORG`` or ``GPE`` because
        those names do not appear in its training vocabulary.

        This method re-inspects the span and its immediate context and
        promotes the label from ``ORG``/``GPE`` → ``PERSON`` when strong
        personal-name signals are found.

        Args:
            span: The extracted entity text (already cleaned / stripped).
            current_label: The label assigned by spaCy.
            full_text: The complete input text (used for title-lookback).

        Returns:
            Corrected entity label string.
        """
        words = span.strip().split()
        if not words:
            return current_label

        lower_span = span.lower()
        lower_words = [w.lower().rstrip(".,;") for w in words]
        first_word_lower = lower_words[0]
        last_word_lower = lower_words[-1]

        # --- Rule 1: Explicit org-suffix → keep/promote to ORG -----------
        if last_word_lower in self.ORG_SUFFIXES or first_word_lower in self.ORG_SUFFIXES:
            return "ORG"
        if any(suf in lower_span for suf in (" inc", " ltd", " llc", " corp", " pvt", " co.")):
            return "ORG"

        # --- Rule 2: Contains digits → likely not a person name ----------
        if re.search(r"\d", span):
            return current_label

        # --- Rule 3: All-caps acronym (e.g. "NASA", "WHO") → keep ORG ---
        if span.isupper() and len(words) == 1 and len(span) <= 6:
            return current_label

        # --- Rule 4: Title/honorific immediately before the span ----------
        # Look back up to 25 chars in the full text
        span_pos = full_text.find(span)
        if span_pos > 0:
            prefix = full_text[max(0, span_pos - 25): span_pos].strip().lower()
            prefix_words = prefix.split()
            if prefix_words and prefix_words[-1].rstrip(".") in self.PERSON_TITLES:
                return "PERSON"

        # --- Rule 5a: Single-word known Indian given name → PERSON --------
        # e.g. spaCy labels "Sirisha" as GPE because it sounds like a place
        if len(words) == 1 and first_word_lower in self.INDIAN_GIVEN_NAMES:
            return "PERSON"

        # --- Rule 5b: Single-word known Indian surname → PERSON -----------
        # e.g. "Kadavala" alone in a name context
        if len(words) == 1 and first_word_lower in self.INDIAN_SURNAMES:
            return "PERSON"

        # --- Rule 6: GPE label but span looks like a personal name --------
        # spaCy tags Indian surnames / first names as GPE (e.g. "Kadavala",
        # "Sirisha") because they superficially resemble place names.
        # Promote to PERSON when:
        #   a) The span contains a known Indian personal name, OR
        #   b) The span is 1–4 alpha-only words without geo indicators and adjacent to name tokens.
        if current_label == "GPE" and 1 <= len(words) <= 4:
            if any(w in self.INDIAN_GIVEN_NAMES or w in self.INDIAN_SURNAMES for w in lower_words):
                return "PERSON"

            all_title_case = all(
                w[0].isupper() and not w.isupper()
                for w in words if w.isalpha()
            )
            has_alpha_only = all(re.match(r"^[A-Za-z'-]+$", w) for w in words)
            geo_indicators = {
                "north", "south", "east", "west", "new", "old", "the",
                "national", "international", "city", "state", "district",
                "town", "village", "mandal", "taluk", "county", "province",
                "republic", "kingdom", "island", "peninsula", "continent"
            }
            has_geo = any(w in geo_indicators for w in lower_words)

            if all_title_case and has_alpha_only and not has_geo:
                # Check neighbours in full_text
                idx = full_text.lower().find(span.lower())
                if idx >= 0:
                    before = full_text[max(0, idx - 30): idx].strip().split()
                    after = full_text[idx + len(span): idx + len(span) + 30].strip().split()
                    neighbour_words = (before[-2:] if before else []) + (after[:2] if after else [])
                    title_case_neighbours = sum(
                        1 for nw in neighbour_words
                        if nw and nw[0].isupper() and nw.isalpha() and len(nw) > 1
                    )
                    if title_case_neighbours >= 1:
                        return "PERSON"
                if first_word_lower in self.INDIAN_GIVEN_NAMES or first_word_lower in self.INDIAN_SURNAMES:
                    return "PERSON"

        # --- Rule 7: 2–4 Title-Case words, ORG-labelled, no digits -------
        # Classic human name pattern: "Deepthi Marada", "Raj Kumar Sharma"
        if current_label == "ORG" and 2 <= len(words) <= 4:
            all_title_case = all(
                w[0].isupper() and not w.isupper()
                for w in words if w.isalpha()
            )
            has_alpha_only = all(re.match(r"^[A-Za-z'-]+$", w) for w in words)
            geo_tokens = {
                "north", "south", "east", "west", "new", "old", "the",
                "national", "international", "global", "world", "city",
                "state", "center", "centre", "club", "team", "agency"
            }
            has_geo = any(w in geo_tokens for w in lower_words)

            if all_title_case and has_alpha_only and not has_geo:
                return "PERSON"

        return current_label


if __name__ == "__main__":
    engine = NEREngine()
    sample = "Apple reported strong profit in Cupertino, while doctors at Mayo Clinic prescribed Metformin."
    results = engine.extract_entities(sample)
    for r in results:
        print(f"[{r['label']}] '{r['text']}' (ctx: {r['local_context']})")

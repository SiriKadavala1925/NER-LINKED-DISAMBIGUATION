"""
=============================================================================
  CONTEXT-AWARE NER & ENTITY DISAMBIGUATION — TERMINAL WALKTHROUGH DEMO
=============================================================================
  This script runs your project step-by-step in the terminal so you can
  see EXACTLY what happens inside the pipeline:

    STEP 1 → Named Entity Recognition   (spaCy + heuristics)
    STEP 2 → Web Knowledge Retrieval     (Wikipedia + DuckDuckGo APIs)
    STEP 3 → Candidate Disambiguation    (TF-IDF, Jaccard, Type Prior)
    STEP 4 → Final Answer                (label, Wikipedia link, confidence)

  Run:  python demo_terminal.py
=============================================================================
"""

import os
import sys
import time
import textwrap

# Add project root dynamically to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ner_engine      import NEREngine
from src.web_retriever   import WebKnowledgeRetriever
from src.disambiguator   import EntityDisambiguator

# Colour helpers
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
PURPLE = "\033[95m"
GREY   = "\033[90m"
BLUE   = "\033[94m"

def banner(title):
    line = "=" * 70
    print(f"\n{CYAN}{BOLD}{line}\n  {title}\n{line}{RESET}")

def section(title):
    print(f"\n{YELLOW}{BOLD}>>  {title}{RESET}")
    print(f"{GREY}{'--' * 33}{RESET}")

def ok(msg):   print(f"  {GREEN}OK  {msg}{RESET}")
def info(msg): print(f"  {BLUE}>>  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}!!  {msg}{RESET}")

DEMO_TEXTS = [
    {"label": "Indian Personal Name (your example)",
     "text":  "deepthi marada"},
    {"label": "Tech + Person (classic ambiguity)",
     "text":  "Elon Musk announced that Tesla will build a new Gigafactory in India."},
    {"label": "Healthcare domain",
     "text":  "Doctors at Mayo Clinic prescribed Metformin for managing diabetes."},
    {"label": "Polysemy: Apple (tech vs. fruit)",
     "text":  "Apple reported record iPhone revenue while fresh apple orchards bloomed."},
]

def run_demo():
    banner("CONTEXT-AWARE NER & ENTITY DISAMBIGUATION -- STEP-BY-STEP DEMO")

    print(f"\n{BOLD}Initialising pipeline components ...{RESET}")
    ner       = NEREngine()
    retriever = WebKnowledgeRetriever()
    disambig  = EntityDisambiguator(similarity_threshold=0.15)
    ok("NEREngine            -- spaCy en_core_web_sm + heuristic reclassifier")
    ok("WebKnowledgeRetriever -- Wikipedia OpenSearch API + DuckDuckGo fallback")
    ok("EntityDisambiguator  -- TF-IDF cosine + Jaccard overlap + type priors")

    for demo in DEMO_TEXTS:
        text = demo["text"]
        banner(f"EXAMPLE: {demo['label']}")
        print(f"\n  {BOLD}Input text:{RESET}  \"{text}\"\n")

        # STEP 1 -- NER
        section("STEP 1 -- Named Entity Recognition (NER)")
        info("Running spaCy en_core_web_sm on your text ...")
        info("Applying heuristic reclassifier to fix label errors ...")

        t0 = time.time()
        entities = ner.extract_entities(text, context_window_size=50)
        ner_ms   = round((time.time() - t0) * 1000, 1)

        if not entities:
            warn("No named entities detected.")
            continue

        print(f"\n  {BOLD}Entities found: {len(entities)}  ({ner_ms} ms){RESET}\n")
        for ent in entities:
            lc = {"PERSON": PURPLE,"ORG": BLUE,"GPE": CYAN,"LOC": CYAN,
                  "PRODUCT": YELLOW,"DRUG": RED,"DISEASE": RED,"ANIMAL": GREEN}.get(ent["label"], GREY)
            print(f"    {lc}{BOLD}[{ent['label']:8}]{RESET}  \"{BOLD}{ent['text']}{RESET}\"")
            print(f"    {GREY}           Source : {ent['source']}{RESET}")
            print(f"    {GREY}           Context: \"{ent['local_context']}\"{RESET}\n")

        # STEP 2 -- Web Retrieval
        section("STEP 2 -- Web Knowledge Retrieval (Wikipedia + DuckDuckGo)")

        all_candidates = {}
        for ent in entities:
            mention = ent["text"]
            info(f"Querying APIs for: \"{mention}\" ...")

            t1 = time.time()
            result    = retriever.fetch_candidates_with_telemetry(mention, max_candidates=4)
            lat_ms    = round((time.time() - t1) * 1000, 1)
            cands     = result["candidates"]
            telemetry = result["telemetry"]

            print(f"\n    {GREY}  Engine  : {telemetry.get('engine', 'Wikipedia OpenSearch API')}{RESET}")
            print(f"    {GREY}  Endpoint: {str(telemetry.get('endpoint_url','N/A'))[:72]}{RESET}")
            print(f"    {GREY}  Status  : HTTP {telemetry.get('http_status','200')}   Latency: {lat_ms} ms{RESET}")
            print(f"    {GREY}  Results : {len(cands)} candidate(s) returned{RESET}\n")

            if cands:
                for i, c in enumerate(cands, 1):
                    snippet = textwrap.shorten(
                        (c.get("extract","") or c.get("description",""))[:120],
                        width=80, placeholder=" ...")
                    print(f"      {GREEN}Candidate {i}:{RESET}  {BOLD}{c.get('title')}{RESET}")
                    print(f"               Category : {c.get('category','General')}")
                    print(f"               Snippet  : {GREY}{snippet}{RESET}")
                    print(f"               URL      : {BLUE}{c.get('url','')}{RESET}\n")
            else:
                warn(f"No public knowledge-base entry for \"{mention}\" -- marked Unlinked.\n")

            all_candidates[mention] = (ent, cands)

        # STEP 3 -- Disambiguation
        section("STEP 3 -- Context-Aware Disambiguation (Scoring & Ranking)")
        info("Scoring each candidate against the input context ...")
        info("Weights: TF-IDF Cosine 45% | Jaccard Overlap 25% | Title Exactness 15% | Type Prior 15%")
        print()

        for mention, (ent, cands) in all_candidates.items():
            result    = disambig.disambiguate_entity(ent, cands)
            all_cands = result["all_candidates"]

            print(f"  {BOLD}Entity: \"{mention}\"{RESET}")
            if len(all_cands) <= 1:
                info("Single candidate -- no scoring needed (confidence = 0.95)")
            else:
                info(f"{len(all_cands)} candidates scored and ranked:")
                for c in all_cands:
                    marker = f"{GREEN}* SELECTED{RESET}" if c.get("is_selected") else f"{GREY}  rank {c['rank']}{RESET}"
                    print(f"      {marker}  {BOLD}{c.get('title')}{RESET}")
                    print(f"               Composite : {c.get('confidence_score',0):.4f}")
                    if "tfidf_cosine" in c:
                        print(f"               TF-IDF    : {c.get('tfidf_cosine',0):.4f} | "
                              f"Jaccard: {c.get('token_overlap',0):.4f} | "
                              f"Type Prior: {c.get('type_prior',0):.4f}")
                    print()
            print()

        # STEP 4 -- Final Output
        section("STEP 4 -- Final Output")
        for mention, (ent, cands) in all_candidates.items():
            result   = disambig.disambiguate_entity(ent, cands)
            selected = result["selected_match"]
            grounded = result.get("is_grounded", False)

            label = ent["label"]
            conf  = selected.get("confidence_score", 0.0)
            title = selected.get("title", mention)
            url   = selected.get("url", "")
            cat   = selected.get("category", "General")

            status_str = f"{GREEN}GROUNDED (linked to Wikipedia){RESET}" if grounded else f"{YELLOW}UNLINKED (no public article){RESET}"
            lc = {"PERSON": PURPLE,"ORG": BLUE,"GPE": CYAN,"LOC": CYAN,
                  "PRODUCT": YELLOW,"DRUG": RED,"DISEASE": RED}.get(label, GREY)

            print(f"  +--------------------------------------------------+")
            print(f"  |  Mention    : \"{mention}\"")
            print(f"  |  NER Label  : {lc}{BOLD}{label}{RESET}")
            print(f"  |  Status     : {status_str}")
            print(f"  |  Linked To  : {BOLD}{title}{RESET}  ({cat})")
            print(f"  |  Confidence : {GREEN if conf >= 0.7 else YELLOW}{conf:.1%}{RESET}")
            print(f"  |  Wikipedia  : {BLUE}{url}{RESET}")
            print(f"  +--------------------------------------------------+\n")

        input(f"  {GREY}[Press ENTER for next example ...]{RESET}\n")

    banner("DEMO COMPLETE")
    print(f"""
  {BOLD}Pipeline summary:{RESET}

  {GREEN}STEP 1{RESET}  spaCy extracts entities. Heuristic reclassifier
          fixes labels for non-Western names (ORG -> PERSON).

  {GREEN}STEP 2{RESET}  For each entity a live HTTP request goes to:
          - Wikipedia OpenSearch REST API  (primary)
          - DuckDuckGo Instant Answer API  (fallback)
          HTTP status codes and latencies are logged.

  {GREEN}STEP 3{RESET}  Multiple Wikipedia candidates are ranked using:
          TF-IDF cosine (45%) + Jaccard (25%) +
          Title exactness (15%) + NER-type prior (15%)

  {GREEN}STEP 4{RESET}  Top-ranked candidate = final answer.
          Grounded = real Wikipedia link.
          Unlinked = no fabricated knowledge, clearly marked.

  {BOLD}Full benchmark:{RESET}  python evaluate.py
""")

if __name__ == "__main__":
    import os
    os.system("color")   # enable ANSI on Windows
    run_demo()

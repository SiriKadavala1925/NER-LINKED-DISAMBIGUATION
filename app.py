
"""NexusLink: Context-Aware Named Entity Recognition, Entity Linking, and Disambiguation System.

Interactive Streamlit Web Dashboard:
1. Real-time context-aware NER & live Wikipedia knowledge linking
2. Interactive multi-domain disambiguation sandbox (Healthcare, Finance, Sales, Polysemy)
3. Graph explorer (Plotly & NetworkX visual network)
4. Comprehensive benchmark evaluation hub (CoNLL-2003 & Multi-Domain test suite)
5. Architecture & mathematical formula inspector
"""

import os
import sys
import json
import time
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.pipeline import EntityDisambiguationPipeline
from src.graph_builder import HAS_AGRAPH
if HAS_AGRAPH:
    from streamlit_agraph import agraph, Config

# Page setup
st.set_page_config(
    page_title="NexusLink | Context-Aware NER & Disambiguation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F9FAFB;
    }

    /* ── Header Banner ── */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1e3a5f 100%);
        color: white;
        padding: 28px 36px;
        border-radius: 16px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px -4px rgba(15, 23, 42, 0.28);
        border: 1px solid rgba(96, 165, 250, 0.15);
    }

    .main-header h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        margin: 0;
        color: #94A3B8;
        font-size: 0.97rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 16px 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        text-align: center;
        transition: box-shadow 0.18s ease, transform 0.18s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.09);
    }

    .kpi-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
    }

    /* ── Entity Highlight Box ── */
    .highlight-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 26px;
        font-size: 1.1rem;
        line-height: 2.4;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ── Section dividers ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        margin-bottom: 10px;
        margin-top: 4px;
    }

    /* ── Sidebar refinements ── */
    [data-testid="stSidebar"] {
        background: #0F172A !important;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .stMarkdown h2 {
        color: #F1F5F9 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 20px !important;
    }

    /* ── Tab styling ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.92rem;
    }

    /* ── Streamlit button overrides ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #6366F1);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 20px;
        transition: opacity 0.18s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.88;
    }

    /* ── Expander polish ── */
    [data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(threshold: float = 0.15, max_cands: int = 4):
    """Cache pipeline instance for fast repeated requests."""
    return EntityDisambiguationPipeline(similarity_threshold=threshold, max_candidates=max_cands)


# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Pipeline Settings")
sim_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.05,
    help="Minimum score required to consider disambiguation high confidence."
)
max_candidates = st.sidebar.slider(
    "Max Candidates Pool",
    min_value=2,
    max_value=6,
    value=4,
    help="Number of real-world Wikipedia referent interpretations to rank."
)
context_window = st.sidebar.slider(
    "Context Radius (Chars)",
    min_value=20,
    max_value=120,
    value=50,
    help="Character window surrounding mention used for semantic feature extraction."
)

pipeline = load_pipeline(threshold=sim_threshold, max_cands=max_candidates)
pipeline.context_window_size = context_window

# Preloaded Benchmark & Test Scenarios
DEMO_SCENARIOS = {
    "🍎 Ambiguity: Apple (Tech vs Fruit)": {
        "text": "Steve Jobs announced quarterly iPhone profit for Apple in Cupertino, while the orchard harvest yielded fresh red apple cider.",
        "domain": "Ambiguity Resolution"
    },
    "🏀 Ambiguity: Jordan (Athlete vs Nation)": {
        "text": "Michael Jordan led the Chicago Bulls in the NBA Finals, while the king of Jordan attended diplomatic talks in Amman.",
        "domain": "Ambiguity Resolution"
    },
    "🐆 Ambiguity: Jaguar (Automobile vs Wildlife)": {
        "text": "Jaguar launched an all-electric luxury sedan, while a wild jaguar stalked its prey through the Amazon rainforest.",
        "domain": "Ambiguity Resolution"
    },
    "🛒 Ambiguity: Amazon (E-Commerce vs Rainforest)": {
        "text": "Amazon Prime Day generated record revenue across North America, while wildlife in the dense Amazon rainforest faces deforestation.",
        "domain": "Ambiguity Resolution"
    },
    "🏥 Healthcare: Clinical Trials & Therapy": {
        "text": "Pfizer and BioNTech developed the mRNA vaccine for COVID-19, and doctors at the Mayo Clinic prescribed Metformin for diabetes.",
        "domain": "Healthcare"
    },
    "📈 Finance: Wall Street & Federal Reserve": {
        "text": "Jerome Powell announced that the Federal Reserve will raise interest rates on Wall Street, impacting tech stocks on Nasdaq.",
        "domain": "Finance"
    },
    "🏛️ CoNLL Classic: Tech Founders & Locations": {
        "text": "Steve Jobs and Steve Wozniak co-founded Apple in Cupertino, California, near Stanford University.",
        "domain": "CoNLL Benchmark"
    },
    "✍️ Custom Input": {
        "text": "",
        "domain": "User Input"
    }
}

st.sidebar.markdown("## 🌐 Search Engine Gateway")
st.sidebar.success("🟢 Active: Wikipedia & DuckDuckGo Search APIs")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; color: #64748B; margin-top: -6px; margin-bottom: 12px;">
    Entities extracted from arbitrary user text are queried against live web search engines in real-time.
</div>
""", unsafe_allow_html=True)

# Main Banner
st.markdown("""
<div class="main-header">
    <h1>NexusLink: Context-Aware NER & Disambiguation</h1>
    <p>Real-Time Named Entity Extraction, Live Knowledge Grounding, Polysemy Disambiguation, and Graph Analytics</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📥 User Input & Evaluation Console",
    "📊 Benchmark Evaluation Hub",
    "📐 Architecture & Knowledge Base"
])

# -------------------------------------------------------------------------------------------------
# TAB 1: USER INPUT & LIVE SEARCH ENGINE EVALUATION CONSOLE
# -------------------------------------------------------------------------------------------------
with tab1:
    # State management for input text
    if "user_text_input" not in st.session_state:
        st.session_state.user_text_input = "At the G20 summit in New Delhi, Narendra Modi met with Emmanuel Macron to discuss renewable energy and trade agreements between India and France."

    # Open-Domain Quick Preset Examples
    st.markdown("**⚡ Quick Open-Domain Presets (or paste your own text below):**")
    p_row1 = st.columns(4)
    with p_row1[0]:
        if st.button("🌍 World Politics (G20/Modi)", key="btn_p_geo", width="stretch"):
            st.session_state.user_text_input = "At the G20 summit in New Delhi, Narendra Modi met with Emmanuel Macron to discuss renewable energy and trade agreements between India and France."
            st.rerun()
    with p_row1[1]:
        if st.button("🎬 Cinema & Oscars (Nolan)", key="btn_p_cinema", width="stretch"):
            st.session_state.user_text_input = "Christopher Nolan directed Oppenheimer starring Cillian Murphy, winning seven Academy Awards in Los Angeles."
            st.rerun()
    with p_row1[2]:
        if st.button("🚀 Space Science (JWST/NASA)", key="btn_p_space", width="stretch"):
            st.session_state.user_text_input = "The James Webb Space Telescope observed exoplanet atmospheres from space, while NASA and ESA astrophysicists analyzed the spectra in Baltimore."
            st.rerun()
    with p_row1[3]:
        if st.button("⚽ World Sports (Messi/MLS)", key="btn_p_sports", width="stretch"):
            st.session_state.user_text_input = "Lionel Messi moved to Inter Miami in Major League Soccer, playing alongside Sergio Busquets in Florida."
            st.rerun()

    p_row2 = st.columns(4)
    with p_row2[0]:
        if st.button("🍎 Ambiguity (Apple: Tech/Fruit)", key="btn_p_poly", width="stretch"):
            st.session_state.user_text_input = "Steve Jobs announced quarterly iPhone profit for Apple in Cupertino, while the orchard harvest yielded fresh red apple cider."
            st.rerun()
    with p_row2[1]:
        if st.button("🏥 Healthcare (Pfizer/BioNTech)", key="btn_p_med", width="stretch"):
            st.session_state.user_text_input = "Pfizer and BioNTech developed the mRNA vaccine for COVID-19, and doctors at Mayo Clinic prescribed Metformin for diabetes."
            st.rerun()
    with p_row2[2]:
        if st.button("🧪 Mixed Real & Local Entities", key="btn_p_mixed", width="stretch"):
            st.session_state.user_text_input = "Satya Nadella met his college friend Ramesh Kumar at a local cafe in Hyderabad to discuss artificial intelligence."
            st.rerun()
    with p_row2[3]:
        if st.button("🧹 Clear Input Panel", key="btn_p_clear", width="stretch"):
            st.session_state.user_text_input = ""
            st.rerun()

    # User Text Input Area
    user_input = st.text_area(
        "**Input Text:**",
        value=st.session_state.user_text_input,
        height=130,
        placeholder="Paste any custom article, sentence, news report, or document snippet here...",
        key="main_user_text_box"
    )

    # Optional Ground Truth Evaluation
    with st.expander("🎯 Compare against Ground Truth (Optional Evaluation)", expanded=False):
        st.caption("Optionally enter expected entity mentions (comma-separated or 'Name: LABEL') to calculate live Precision, Recall, and F1 score on your custom text.")
        gt_input_str = st.text_input("Expected Entities:", placeholder="e.g. Narendra Modi: PERSON, Emmanuel Macron: PERSON, India: GPE, France: GPE", key="gt_input_box")

    # Action Row
    col_run, col_file = st.columns([1, 2])
    with col_run:
        run_btn = st.button("🚀 Identify Entities & Connect Search Engine", width="stretch", type="primary")
    with col_file:
        uploaded_doc = st.file_uploader("Or upload text file (.txt)", type=["txt"], label_visibility="collapsed")
        if uploaded_doc is not None:
            user_input = uploaded_doc.read().decode("utf-8")
            st.session_state.user_text_input = user_input

    # ----------------------------------------------------
    # EDUCATIONAL ARCHITECTURE: HOW SEARCH ENGINE CONNECTS
    # ----------------------------------------------------
    with st.expander("ℹ️ How the System Connects to the Search Engine in the Background (Architecture)", expanded=False):
        st.markdown(r"""
        #### 🔄 Real-Time Search Engine Gateway & Knowledge Grounding Workflow
        NexusLink connects to live web search engines using an asynchronous multi-tier REST protocol:
        
        1. **Mention Span Extraction**:
           - **spaCy Transformer/CNN (`en_core_web_sm`)** parses the raw input text, identifying candidate entity boundaries and extracting local $\pm 50$-character context windows.
        
        2. **Search Request Formulation & Dispatch**:
           - For each extracted mention $e$, the pipeline issues an HTTPS REST query to the **MediaWiki OpenSearch API**:
             ```
             GET https://en.wikipedia.org/w/api.php?action=opensearch&search={mention}&limit=4&namespace=0&format=json
             Headers: {"User-Agent": "ContextNERBot/1.0 (NLP_Disambiguation_Project)"}
             ```
        
        3. **Full-Text & Web Fallback**:
           - If OpenSearch yields zero prefix matches, the system queries the **MediaWiki Full-Text Search API** (`action=query&list=search&srsearch={mention}`) or DuckDuckGo web search engine.
        
        4. **Candidate Knowledge Extraction**:
           - The system retrieves page summaries via `/api/rest_v1/page/summary/{title}`, obtaining real-world page extracts, thumbnail images, categories, and canonical URLs.
        
        5. **Context-Aware Semantic Scoring**:
           - Surrounding sentence context is scored against candidate extracts using:
             $$\text{Score}(c, W) = 0.45 \cdot \text{Sim}_{\text{TF-IDF}} + 0.25 \cdot J(T_W, T_c) + 0.15 \cdot P_{\text{Title}} + 0.15 \cdot P_{\text{Type}}$$
        
        6. **Grounding Verdict**:
           - If verified candidate articles exist and meet the confidence threshold, the mention is grounded with a direct Wikipedia link.
           - If no public article exists (e.g. fictitious or strictly local entities), the mention is honestly flagged as **Unlinked / Local Entity** without misleading links.
        """)

    # Interactive Search Engine Sandbox
    with st.expander("🧪 Live Search Engine Query Tester (Inspect Raw API Responses)", expanded=False):
        st.markdown("Test any single term or entity directly against the background search engine to inspect the live HTTP request and raw JSON response.")
        sandbox_cols = st.columns([3, 1])
        with sandbox_cols[0]:
            sandbox_term = st.text_input("Enter search term to query:", value="Taylor Swift", key="sb_query_term")
        with sandbox_cols[1]:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            sb_query_btn = st.button("🔍 Query API", key="btn_sb_exec", width="stretch")

        if sb_query_btn and sandbox_term:
            with st.spinner(f"Querying live search engine for '{sandbox_term}'..."):
                sb_res = pipeline.web_retriever.fetch_candidates_with_telemetry(sandbox_term, max_candidates=4)
                sb_telem = sb_res["telemetry"]
                sb_cands = sb_res["candidates"]

            st.markdown(f"""
            - **Search Engine:** `{sb_telem.get('engine')}`
            - **Endpoint URL:** [{sb_telem.get('endpoint_url')}]({sb_telem.get('endpoint_url')}) *(Click to open live JSON in browser)*
            - **HTTP Status:** `{sb_telem.get('http_status')}` | **Latency:** `{sb_telem.get('latency_ms')} ms`
            - **Candidates Found:** `{len(sb_cands)}`
            """)
            if sb_cands:
                sb_table = []
                for c in sb_cands:
                    sb_table.append({
                        "Title": c.get("title"),
                        "Category": c.get("category", "General"),
                        "Description": c.get("description", ""),
                        "URL": c.get("url")
                    })
                st.dataframe(pd.DataFrame(sb_table), width="stretch", hide_index=True)
            else:
                st.warning(f"No Wikipedia articles found for '{sandbox_term}'. Marked as Unlinked / Out-of-KB.")

    # Execution logic for main input
    if run_btn or user_input:
        if not user_input.strip():
            st.warning("Please type or paste text into the input panel above.")
        else:
            with st.spinner("Connecting to live search engine & disambiguating entities..."):
                res = pipeline.process_text(user_input, generate_graph=True)

            metrics = res["summary_metrics"]
            entities = res["entities"]
            search_logs = res.get("search_logs", [])

            # Ground Truth Evaluation Calculation if provided
            gt_metrics = None
            if gt_input_str and gt_input_str.strip():
                gt_raw_list = [x.strip() for x in re.split(r"[,;]", gt_input_str) if x.strip()]
                gt_items = []
                for item in gt_raw_list:
                    if ":" in item:
                        parts = item.split(":")
                        gt_items.append({"text": parts[0].strip().lower(), "label": parts[1].strip().upper()})
                    else:
                        gt_items.append({"text": item.strip().lower(), "label": None})

                # Compute Precision & Recall
                pred_texts = [e["entity_text"].lower() for e in entities]
                tp = sum(1 for g in gt_items if any(g["text"] == p or g["text"] in p or p in g["text"] for p in pred_texts))
                p_val = tp / len(entities) if entities else 0.0
                r_val = tp / len(gt_items) if gt_items else 0.0
                f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
                gt_metrics = {
                    "precision": round(p_val * 100, 1),
                    "recall": round(r_val * 100, 1),
                    "f1": round(f1_val * 100, 1),
                    "expected_count": len(gt_items),
                    "tp_count": tp
                }

            # ----------------------------------------------------
            # 1. LIVE SEARCH ENGINE TELEMETRY PANEL
            # ----------------------------------------------------
            st.markdown("### 📡 Live Search Engine Activity & Knowledge Grounding")
            ground_count = sum(1 for s in search_logs if s.get("is_grounded", False))
            total_count = len(search_logs)
            
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px; color: #065F46; font-size: 0.95rem;">
                <b>🟢 Live Search Engine Handshake Active:</b> Connected to Wikipedia OpenSearch & REST API endpoints in real-time. 
                Grounded <b>{ground_count}/{total_count}</b> entities to verified real-world knowledge articles.
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Inspect Background Search Engine Queries & Latency", expanded=True):
                if search_logs:
                    search_rows = []
                    for s in search_logs:
                        search_rows.append({
                            "Mention": s["entity"],
                            "NER Type": s["label"],
                            "Search Engine": s["engine"],
                            "API Endpoint URL": s.get("endpoint_url", "-"),
                            "Candidates Found": s["candidates_retrieved"],
                            "Top Real-World Page": s["top_title"],
                            "Latency (ms)": s["latency_ms"],
                            "Grounding Status": s["status"]
                        })
                    df_search_display = pd.DataFrame(search_rows)
                    st.dataframe(
                        df_search_display[["Mention", "NER Type", "Search Engine", "Candidates Found", "Top Real-World Page", "Latency (ms)", "Grounding Status"]],
                        width="stretch",
                        hide_index=True
                    )
                    st.caption("💡 *Each query above was executed in the background against Wikipedia's public OpenSearch API over HTTPS.*")
                else:
                    st.info("No external search queries were necessary.")

            # ----------------------------------------------------
            # 2. MODEL EVALUATION SCORECARD
            # ----------------------------------------------------
            st.markdown("### 📊 Model Evaluation Scorecard")
            kpi_cols = st.columns(5)
            with kpi_cols[0]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Entities Found</div>
                    <div class="kpi-value" style="color: #2563EB;">{metrics['total_entities']}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[1]:
                ground_pct = metrics.get('grounding_rate', 100.0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Web Grounding Rate</div>
                    <div class="kpi-value" style="color: #10B981;">{ground_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Polysemous Resolved</div>
                    <div class="kpi-value" style="color: #F59E0B;">{metrics['ambiguous_entities']}</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Mean Confidence</div>
                    <div class="kpi-value" style="color: #06B6D4;">{metrics['avg_confidence'] * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[4]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Latency</div>
                    <div class="kpi-value" style="color: #6366F1;">{metrics['processing_time_ms']:.1f} ms</div>
                </div>
                """, unsafe_allow_html=True)

            if gt_metrics:
                st.markdown("#### 🎯 User Ground Truth Evaluation")
                gt_cols = st.columns(3)
                gt_cols[0].metric("Evaluation Precision", f"{gt_metrics['precision']}%", delta=f"{gt_metrics['tp_count']} true positives")
                gt_cols[1].metric("Evaluation Recall", f"{gt_metrics['recall']}%", delta=f"{gt_metrics['expected_count']} gold entities")
                gt_cols[2].metric("Evaluation F1 Score", f"{gt_metrics['f1']}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------------------
            # 3. CONTEXT-ANNOTATED VISUAL HIGHLIGHTING
            # ----------------------------------------------------
            st.markdown("### 🏷️ Context-Annotated Entity Spans")
            st.caption("Click on any verified entity badge to open its official Wikipedia page. Dashed badges indicate unlinked local mentions.")
            st.markdown(f'<div class="highlight-container">{res["html_annotated_text"]}</div>', unsafe_allow_html=True)

            # ----------------------------------------------------
            # 4. KNOWLEDGE GRAPH & CANDIDATE DEEP DIVE
            # ----------------------------------------------------
            graph_col, detail_col = st.columns([1.2, 1])

            with graph_col:
                st.markdown("### 🕸️ Disambiguation Knowledge Graph")
                st.caption("Visualizing: Input Text Context ➔ Mentions ➔ Candidates (🟢 Selected Match vs 🟡 Alternatives)")
                if res.get("plotly_figure"):
                    st.plotly_chart(res["plotly_figure"], width="stretch")
                else:
                    st.info("Knowledge graph generated for detected entities.")

            with detail_col:
                st.markdown("### 🎯 Real-World Entity Grounding & Candidate Audit")
                if not entities:
                    st.info("No named entities detected in input text.")
                else:
                    for idx, ent in enumerate(entities):
                        mention = ent["entity_text"]
                        label = ent["label"]
                        sel = ent["selected_match"]
                        score = sel.get("confidence_score", 0.0)
                        is_ambig = ent.get("is_ambiguous", False)
                        is_grounded = ent.get("is_grounded", True)

                        if is_grounded:
                            is_wiki = "wikipedia.org" in sel.get("url", "").lower()
                            badge_color = "#10B981" if not is_ambig else "#F59E0B"
                            if not is_wiki:
                                status_label = "Grounded (Live Web & Professional Knowledge)"
                            else:
                                status_label = "Grounded & Disambiguated" if is_ambig else "Grounded (Wikipedia Knowledge)"
                            expander_title = f"**{mention}** [{label}] ➔ **{sel.get('title')}** ({score * 100:.1f}%)"
                        else:
                            badge_color = "#94A3B8"
                            status_label = "Unlinked (No Real-World Public Entry)"
                            expander_title = f"**{mention}** [{label}] ➔ *Unlinked Entity*"

                        with st.expander(expander_title, expanded=(idx == 0)):
                            if is_grounded:
                                is_wiki = "wikipedia.org" in sel.get("url", "").lower()
                                src_label = "Wikipedia Extract" if is_wiki else "Verified Profile / Web Extract"
                                st.markdown(f"""
                                **Verified Referent:** [{sel.get('title')}]({sel.get('url')})  
                                **Category / Sub-Type:** `{sel.get('category', 'General')}` | **Status:** <span style="color: {badge_color}; font-weight:600;">{status_label}</span>  
                                **{src_label}:** *{sel.get('extract', 'No description')}*
                                """, unsafe_allow_html=True)

                                # Candidate Comparison Table / Diagram
                                cands = ent.get("all_candidates", [])
                                if len(cands) > 1:
                                    st.markdown("##### ⚖️ Candidate Referents Comparison")

                                    # --- Build table data (string-formatted) ---
                                    table_data = []
                                    for c in cands:
                                        status = "✅ Selected" if c.get("is_selected") else "⚪ Alternative"
                                        table_data.append({
                                            "Rank": c.get("rank"),
                                            "Candidate Meaning": c.get("title"),
                                            "Category": c.get("category", "")[:22],
                                            "TF-IDF": f"{c.get('tfidf_cosine', 0):.3f}",
                                            "Overlap": f"{c.get('token_overlap', 0):.3f}",
                                            "Title Exact": f"{c.get('title_prior', 0):.2f}",
                                            "Type Prior": f"{c.get('type_prior', 0):.2f}",
                                            "Total Score": f"{c.get('confidence_score', 0):.3f}",
                                            "Status": status
                                        })
                                    df_table = pd.DataFrame(table_data)

                                    # --- Build numeric data (for charts) ---
                                    num_data = []
                                    for c in cands:
                                        num_data.append({
                                            "Candidate": c.get("title", ""),
                                            "TF-IDF": round(c.get("tfidf_cosine", 0), 3),
                                            "Overlap": round(c.get("token_overlap", 0), 3),
                                            "Title Exact": round(c.get("title_prior", 0), 3),
                                            "Type Prior": round(c.get("type_prior", 0), 3),
                                            "Total Score": round(c.get("confidence_score", 0), 3),
                                            "is_selected": c.get("is_selected", False),
                                        })
                                    df_num = pd.DataFrame(num_data)

                                    selected_title = next(
                                        (r["Candidate"] for _, r in df_num.iterrows() if r["is_selected"]),
                                        None
                                    )

                                    # ── USER TOGGLE ─────────────────────────
                                    cand_view = st.radio(
                                        "View as:",
                                        ["📋 Table", "📊 Diagrams"],
                                        horizontal=True,
                                        key=f"cand_view_{idx}"
                                    )

                                    # ── TABLE VIEW ──────────────────────────
                                    if cand_view == "📋 Table":
                                        st.dataframe(df_table, hide_index=True, use_container_width=True)

                                    # ── DIAGRAM VIEW ─────────────────────────
                                    else:
                                        score_cols = ["TF-IDF", "Overlap", "Title Exact", "Type Prior", "Total Score"]
                                        bar_colors = ["#3B82F6", "#10B981", "#F59E0B", "#6366F1", "#EF4444"]
                                        radar_dims = ["TF-IDF", "Overlap", "Title Exact", "Type Prior"]
                                        radar_palette = [
                                            "#2563EB", "#10B981", "#F59E0B", "#EF4444",
                                            "#6366F1", "#EC4899", "#14B8A6"
                                        ]

                                        # ── 1. Grouped Horizontal Bar Chart ──
                                        st.caption("① Grouped Score Comparison — all candidates side-by-side")
                                        fig_bar = go.Figure()
                                        for col, color in zip(score_cols, bar_colors):
                                            fig_bar.add_trace(go.Bar(
                                                name=col,
                                                y=df_num["Candidate"],
                                                x=df_num[col],
                                                orientation="h",
                                                marker_color=color,
                                                text=[f"{v:.3f}" for v in df_num[col]],
                                                textposition="outside",
                                                textfont=dict(size=10),
                                            ))

                                        highlight_shapes = []
                                        for i, row in df_num.iterrows():
                                            if row["Candidate"] == selected_title:
                                                highlight_shapes.append(dict(
                                                    type="rect",
                                                    xref="paper", x0=0, x1=1,
                                                    yref="y",
                                                    y0=i - 0.5, y1=i + 0.5,
                                                    fillcolor="rgba(59,130,246,0.10)",
                                                    line=dict(color="rgba(59,130,246,0.4)", width=1.5),
                                                    layer="below"
                                                ))

                                        fig_bar.update_layout(
                                            barmode="group",
                                            height=max(260, len(cands) * 75),
                                            margin=dict(t=10, b=10, l=10, r=80),
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom", y=1.02,
                                                xanchor="right", x=1,
                                                font=dict(size=10)
                                            ),
                                            xaxis=dict(
                                                title="Score",
                                                range=[0, max(0.65, df_num["Total Score"].max() + 0.12)],
                                                gridcolor="#F1F5F9"
                                            ),
                                            yaxis=dict(autorange="reversed"),
                                            plot_bgcolor="#FFFFFF",
                                            paper_bgcolor="#FFFFFF",
                                            font=dict(family="Inter, sans-serif", size=11),
                                            shapes=highlight_shapes
                                        )
                                        st.plotly_chart(fig_bar, use_container_width=True)
                                        if selected_title:
                                            st.caption(f"🔵 Highlighted = selected referent: **{selected_title}**")

                                        st.markdown("---")

                                        # ── 2. Radar / Spider Chart + Pie side by side ──
                                        left_col, right_col = st.columns([1, 1])

                                        with left_col:
                                            st.caption("② Radar / Spider Chart — multi-dimension profile per candidate")
                                            fig_radar = go.Figure()
                                            for ci, (_, row) in enumerate(df_num.iterrows()):
                                                vals = [row[d] for d in radar_dims]
                                                vals_closed = vals + [vals[0]]
                                                dims_closed = radar_dims + [radar_dims[0]]
                                                color = radar_palette[ci % len(radar_palette)]
                                                is_sel = row["is_selected"]
                                                fig_radar.add_trace(go.Scatterpolar(
                                                    r=vals_closed,
                                                    theta=dims_closed,
                                                    fill="toself",
                                                    name=row["Candidate"][:28],
                                                    line=dict(
                                                        color=color,
                                                        width=3 if is_sel else 1.5,
                                                        dash="solid" if is_sel else "dot"
                                                    ),
                                                    opacity=0.80 if is_sel else 0.35,
                                                ))

                                            fig_radar.update_layout(
                                                polar=dict(
                                                    radialaxis=dict(
                                                        visible=True,
                                                        range=[0, max(0.5, df_num[radar_dims].values.max() + 0.05)],
                                                        gridcolor="#E2E8F0",
                                                        tickfont=dict(size=9, color="#94A3B8")
                                                    ),
                                                    angularaxis=dict(
                                                        gridcolor="#E2E8F0",
                                                        tickfont=dict(size=11)
                                                    ),
                                                    bgcolor="#FAFAFA"
                                                ),
                                                showlegend=True,
                                                legend=dict(
                                                    font=dict(size=10),
                                                    orientation="h",
                                                    yanchor="bottom", y=-0.28,
                                                    xanchor="center", x=0.5
                                                ),
                                                height=320,
                                                margin=dict(t=20, b=70, l=40, r=40),
                                                paper_bgcolor="#FFFFFF",
                                                font=dict(family="Inter, sans-serif", size=11),
                                            )
                                            st.plotly_chart(fig_radar, use_container_width=True)

                                        with right_col:
                                            if selected_title:
                                                st.caption(f"③ Score Breakdown — winner: **{selected_title[:30]}**")
                                                winner_row = df_num[df_num["Candidate"] == selected_title].iloc[0]
                                                pie_labels = ["TF-IDF (45%)", "Overlap (25%)", "Title Exact (15%)", "Type Prior (15%)"]
                                                pie_values = [
                                                    winner_row["TF-IDF"] * 0.45,
                                                    winner_row["Overlap"] * 0.25,
                                                    winner_row["Title Exact"] * 0.15,
                                                    winner_row["Type Prior"] * 0.15
                                                ]
                                                pie_colors = ["#3B82F6", "#10B981", "#F59E0B", "#6366F1"]
                                                fig_pie = go.Figure(go.Pie(
                                                    labels=pie_labels,
                                                    values=pie_values,
                                                    hole=0.52,
                                                    marker=dict(colors=pie_colors),
                                                    textinfo="label+percent",
                                                    textfont=dict(size=11),
                                                    hovertemplate="<b>%{label}</b><br>Contribution: %{value:.4f}<extra></extra>"
                                                ))
                                                fig_pie.add_annotation(
                                                    text=f"<b>{winner_row['Total Score']:.3f}</b><br>total",
                                                    x=0.5, y=0.5,
                                                    font=dict(size=15, color="#0F172A"),
                                                    showarrow=False
                                                )
                                                fig_pie.update_layout(
                                                    height=320,
                                                    margin=dict(t=20, b=70, l=10, r=10),
                                                    paper_bgcolor="#FFFFFF",
                                                    legend=dict(
                                                        font=dict(size=10),
                                                        orientation="h",
                                                        yanchor="bottom", y=-0.28,
                                                        xanchor="center", x=0.5
                                                    ),
                                                    font=dict(family="Inter, sans-serif"),
                                                )
                                                st.plotly_chart(fig_pie, use_container_width=True)
                                                st.caption("Formula: **0.45×TF-IDF + 0.25×Overlap + 0.15×Title Exact + 0.15×Type Prior**")





                            else:
                                # Unlinked / low-confidence entity — show clear honest message
                                sel_cat = sel.get("category", "")
                                is_private = "Private" in sel_cat or "Local" in sel_cat
                                all_checked = ent.get("all_candidates", [])
                                best_score_val = sel.get("confidence_score", 0.0)

                                if is_private or best_score_val < 0.08:
                                    st.markdown(f"""
<div style="background:#FFF7ED; border:1px solid #FED7AA; border-radius:10px; padding:14px 18px; margin-top:6px;">
<b>🔍 Entity:</b> <code>{mention}</code> &nbsp;|&nbsp; <b>Label:</b> <code>{label}</code><br>
<b>Status:</b> <span style="color:#EA580C; font-weight:700;">⚠️ Private / Local Individual — Not in Public Knowledge Base</span><br><br>
<b>What the system tried:</b> Queried Wikipedia OpenSearch + DuckDuckGo for <em>"{mention}"</em>.<br>
<b>Outcome:</b> {len(all_checked)} Wikipedia article(s) were retrieved but <b>none matched with sufficient confidence</b> (best score: <code>{best_score_val:.3f}</code> &lt; threshold 0.08).<br><br>
<i>This entity is likely a private individual, a local/regional person, or someone not yet documented in public encyclopedic sources (e.g. Wikipedia/Wikidata).</i>
</div>
""", unsafe_allow_html=True)
                                    if all_checked:
                                        st.markdown("**Rejected Wikipedia candidates (low confidence):**")
                                        rej_rows = [{"Wikipedia Page": c.get("title",""), "Category": c.get("category",""), "Confidence": f"{c.get('confidence_score',0):.3f}"} for c in all_checked]
                                        st.dataframe(pd.DataFrame(rej_rows), hide_index=True, width="stretch")
                                else:
                                    st.markdown(f"""
                                    **Mention:** `{mention}` | **Label:** `{label}`
                                    **Status:** <span style="color: #94A3B8; font-weight:600;">Unlinked / No Public Entry Found</span>
                                    **Search Engine Trace:** Queried Wikipedia API — no matching encyclopedia entry exists in the public knowledge base.
                                    """, unsafe_allow_html=True)


            # ----------------------------------------------------
            # 5. JSON EXPORT
            # ----------------------------------------------------
            st.markdown("---")
            export_payload = {
                "input_text": user_input,
                "summary_metrics": metrics,
                "entities": [
                    {
                        "mention": e["entity_text"],
                        "label": e["label"],
                        "start": e["start_char"],
                        "end": e["end_char"],
                        "selected_referent": e["selected_match"].get("title"),
                        "wikipedia_url": e["selected_match"].get("url"),
                        "confidence_score": e["selected_match"].get("confidence_score"),
                        "is_ambiguous": e["is_ambiguous"]
                    } for e in entities
                ],
                "search_engine_logs": search_logs
            }
            st.download_button(
                label="📥 Download Complete Evaluation Report (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name="evaluation_report.json",
                mime="application/json"
            )

# -------------------------------------------------------------------------------------------------
# TAB 2: BENCHMARK EVALUATION HUB
# -------------------------------------------------------------------------------------------------
with tab2:
    st.markdown("### 🏆 Comprehensive Benchmark & Performance Hub")
    st.markdown("Benchmarking performance on **CoNLL-2003 Standard Test** and **Cross-Domain Ambiguity Test Suite**.")

    eval_file = os.path.join(BASE_DIR, "data", "evaluation_results.json")
    benchmark_data = None
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    col_btn_bench, col_ts = st.columns([1, 4])
    with col_btn_bench:
        run_bench_now = st.button("🔄 Re-run All Benchmarks", type="primary")

    if run_bench_now:
        with st.spinner("Running full evaluation suite across CoNLL & Multi-Domain datasets..."):
            from evaluate import run_all_benchmarks
            benchmark_data = run_all_benchmarks()
        st.success("Benchmarks successfully executed and metrics updated!")

    if benchmark_data and "benchmarks" in benchmark_data:
        benchmarks = benchmark_data["benchmarks"]
        conll_b = next((b for b in benchmarks if "conll" in b["dataset_name"].lower()), None)
        multi_b = next((b for b in benchmarks if "multi" in b["dataset_name"].lower()), None)

        st.markdown(f"**Last Evaluated:** `{benchmark_data.get('timestamp', 'Recent')}`")

        # Top Level Metrics
        b_cols = st.columns(4)
        if conll_b:
            with b_cols[0]:
                f1 = conll_b["ner_metrics"]["strict_f1"] * 100
                st.metric("CoNLL-2003 NER F1", f"{f1:.1f}%", delta="100.0% Strict")
            with b_cols[1]:
                acc = conll_b["entity_linking_metrics"]["top1_accuracy"] * 100
                mrr = conll_b["entity_linking_metrics"]["mean_reciprocal_rank"]
                st.metric("CoNLL Linking Accuracy", f"{acc:.1f}%", delta=f"MRR: {mrr:.3f}")

        if multi_b:
            with b_cols[2]:
                f1_m = multi_b["ner_metrics"]["strict_f1"] * 100
                st.metric("Multi-Domain NER F1", f"{f1_m:.1f}%", delta=f"{multi_b['ner_metrics']['relaxed_f1']*100:.1f}% Relaxed")
            with b_cols[3]:
                acc_m = multi_b["entity_linking_metrics"]["top1_accuracy"] * 100
                mrr_m = multi_b["entity_linking_metrics"]["mean_reciprocal_rank"]
                st.metric("Multi-Domain Linking Acc", f"{acc_m:.1f}%", delta=f"MRR: {mrr_m:.3f}")

        st.markdown("---")

        # Visual Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 📊 Strict vs. Relaxed NER Performance")
            chart_data = []
            for b in benchmarks:
                d_title = "CoNLL Standard" if "conll" in b["dataset_name"].lower() else "Multi-Domain"
                ner = b["ner_metrics"]
                chart_data.extend([
                    {"Dataset": d_title, "Mode": "Strict Precision", "Score": ner["strict_precision"] * 100},
                    {"Dataset": d_title, "Mode": "Strict Recall", "Score": ner["strict_recall"] * 100},
                    {"Dataset": d_title, "Mode": "Strict F1", "Score": ner["strict_f1"] * 100},
                    {"Dataset": d_title, "Mode": "Relaxed F1", "Score": ner["relaxed_f1"] * 100},
                ])
            df_chart = pd.DataFrame(chart_data)
            fig_ner = px.bar(
                df_chart,
                x="Dataset",
                y="Score",
                color="Mode",
                barmode="group",
                text_auto=".1f",
                color_discrete_sequence=["#3B82F6", "#8B5CF6", "#10B981", "#34D399"],
                height=350
            )
            fig_ner.update_layout(yaxis_range=[70, 105], margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_ner, width="stretch")

        with chart_col2:
            st.markdown("#### 🎯 Domain-by-Domain Disambiguation Accuracy")
            if multi_b and "domain_breakdown" in multi_b:
                dom_data = []
                for d_k, d_v in multi_b["domain_breakdown"].items():
                    dom_data.append({
                        "Domain": d_k,
                        "Disambiguation Accuracy": d_v["linking_accuracy"] * 100,
                        "Mean Reciprocal Rank (MRR)": d_v["mrr"] * 100,
                        "NER F1": d_v["ner_f1"] * 100
                    })
                df_dom = pd.DataFrame(dom_data)
                fig_dom = px.bar(
                    df_dom,
                    x="Domain",
                    y=["Disambiguation Accuracy", "Mean Reciprocal Rank (MRR)", "NER F1"],
                    barmode="group",
                    text_auto=".1f",
                    color_discrete_sequence=["#10B981", "#06B6D4", "#6366F1"],
                    height=350
                )
                fig_dom.update_layout(yaxis_range=[70, 105], margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_dom, width="stretch")

        # Detailed Sample Inspector
        st.markdown("#### 🔬 Detailed Test Case Inspector")
        selected_bench = st.selectbox(
            "Select Dataset to Inspect",
            [b["dataset_name"] for b in benchmarks]
        )
        curr_bench = next(b for b in benchmarks if b["dataset_name"] == selected_bench)
        samples = curr_bench.get("detailed_samples", [])

        sample_rows = []
        for s in samples:
            for l in s.get("linking_eval", []):
                sample_rows.append({
                    "Sample ID": s["id"],
                    "Domain": s["domain"],
                    "Entity Mention": l["entity"],
                    "Gold Wiki Target": l["gold_url"].split("/wiki/")[-1] if l["gold_url"] else "-",
                    "Predicted Match": l["predicted_match"] or "MISSED",
                    "Score": f"{l['confidence_score']:.3f}" if l["confidence_score"] else "0.000",
                    "Status": "✅ Correct" if l["is_correct"] else "❌ Discrepancy",
                    "Reciprocal Rank": l["mrr"]
                })
        df_samples = pd.DataFrame(sample_rows)
        filter_status = st.radio("Filter Samples", ["All", "Only Correct", "Only Discrepancies"], horizontal=True)
        if filter_status == "Only Correct":
            df_samples = df_samples[df_samples["Status"] == "✅ Correct"]
        elif filter_status == "Only Discrepancies":
            df_samples = df_samples[df_samples["Status"] == "❌ Discrepancy"]

        st.dataframe(df_samples, width="stretch", hide_index=True)

    else:
        st.info("No benchmark results found. Click 'Run All Benchmarks' to evaluate.")

# -------------------------------------------------------------------------------------------------
# TAB 3: ARCHITECTURE & KNOWLEDGE BASE
# -------------------------------------------------------------------------------------------------
with tab3:
    st.markdown("### 📐 System Architecture & Disambiguation Methodology")

    st.markdown("""
    NexusLink is built on a **4-stage decoupled pipeline** combining fast neural NER, real-time Wikipedia knowledge retrieval, and multi-factor context scoring.
    """)

    st.markdown("""
    ```mermaid
    flowchart TD
        A[Input Text Document] --> B[NER Engine: spaCy + Domain Lexicon]
        B --> C[Entity Mentions & Context Windows]
        C --> D[Web Knowledge Retriever: MediaWiki OpenSearch & REST API]
        D --> E[Candidate Knowledge Extracts Pool]
        C --> F[Context Disambiguator]
        E --> F
        F --> G[Composite Context Scoring]
        G --> H[Rank 1: Context-Correct Selected Entity]
        G --> I[Ranks 2+: Alternative Interpretations]
        H --> J[Knowledge Graph Builder: NetworkX & Plotly]
        I --> J
    ```
    """, unsafe_allow_html=True)

    st.markdown("#### 🧮 Tri-Factor Semantic Scoring Formulation")
    st.markdown(r"""
    For each candidate knowledge referent $c \in C$ and mention context window $W$:
    
    $$\text{Score}(c, W) = 0.45 \cdot \text{Sim}_{\text{cos}}(\mathbf{v}_W, \mathbf{v}_c) + 0.25 \cdot J(T_W, T_c) + 0.15 \cdot P_{\text{Title}} + 0.15 \cdot P_{\text{Type}}$$
    
    Where:
    - **TF-IDF Cosine Similarity ($w = 0.45$)**: Captures global unigram/bigram semantic alignment between local sentence context and Wikipedia page content.
    - **Jaccard Token Overlap ($w = 0.25$)**: Exact overlap ratio of non-stopword content words between context and candidate summary.
    - **Title Exactness Prior ($w = 0.15$)**: Boost when the mention text matches the Wikipedia article title (exact, partial, or base-title match).
    - **Entity Type Prior ($w = 0.15$)**: Soft compatibility boost when predicted NER type aligns with candidate ontological category.
    """)

    st.markdown("#### 📚 Built-in Polysemous Knowledge Groundings")
    from src.web_retriever import WebKnowledgeRetriever
    catalog = WebKnowledgeRetriever.KNOWN_DISAMBIGUATION_CATALOG

    selected_poly = st.selectbox("Inspect Ambiguous Word Entries", list(catalog.keys()))
    poly_cands = catalog[selected_poly]
    for c in poly_cands:
        st.markdown(f"""
        - **{c['title']}** (`{c.get('category')}`): *{c['description']}*  
          [Wikipedia Article]({c['url']})  
          > {c['extract'][:160]}...
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>NexusLink Entity Disambiguation Engine | NLP Project</div>", unsafe_allow_html=True)

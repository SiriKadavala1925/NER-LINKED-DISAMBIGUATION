"""Disambiguation Knowledge Graph Builder.

Constructs interactive graph representations connecting:
[Input Text Context] -> [Entity Mentions] -> [Candidate Real-World Referents]
Distinguishes Selected Context-Correct Matches (Green) from Alternative Meanings (Amber/Gray).
Provides support for:
1. streamlit-agraph (Vis.js physics-driven graph)
2. Plotly Interactive Network Graph
3. NetworkX directed multigraph structure
"""

from typing import List, Dict, Any, Tuple
import networkx as nx
import plotly.graph_objects as go

# Optional agraph imports
try:
    from streamlit_agraph import Node, Edge, Config
    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False


class DisambiguationGraphBuilder:
    """Builds interactive visual graph structures for entity linking and disambiguation."""

    COLOR_PALETTE = {
        "ROOT": "#6366F1",          # Indigo
        "MENTION_DEFAULT": "#3B82F6",# Blue
        "MENTION_ORG": "#2563EB",    # Deep Blue
        "MENTION_PERSON": "#8B5CF6", # Violet
        "MENTION_GPE": "#06B6D4",    # Cyan
        "MENTION_DRUG": "#EC4899",   # Pink
        "SELECTED_MATCH": "#10B981", # Emerald Green
        "ALT_CANDIDATE": "#F59E0B",  # Amber
        "UNLINKED": "#94A3B8"        # Slate Gray
    }

    def __init__(self):
        pass

    def build_networkx_graph(self, pipeline_results: List[Dict[str, Any]], text_preview: str) -> nx.DiGraph:
        """Create NetworkX directed graph from pipeline results."""
        G = nx.DiGraph()

        # Root node
        root_id = "root_context"
        G.add_node(
            root_id,
            label=f"Input Context:\n\"{text_preview[:40]}...\"",
            node_type="ROOT",
            color=self.COLOR_PALETTE["ROOT"],
            size=25,
            title=f"Source Input Text: {text_preview}"
        )

        for ent_idx, ent_res in enumerate(pipeline_results):
            mention_text = ent_res["entity_text"]
            label = ent_res.get("label", "ENTITY")
            mention_id = f"mention_{ent_idx}_{mention_text}"

            mention_color = self.COLOR_PALETTE.get(f"MENTION_{label}", self.COLOR_PALETTE["MENTION_DEFAULT"])
            G.add_node(
                mention_id,
                label=f"{mention_text}\n[{label}]",
                node_type="MENTION",
                entity_type=label,
                color=mention_color,
                size=20,
                title=f"Entity Mention: '{mention_text}' (Type: {label})"
            )
            # Edge from root to mention
            G.add_edge(root_id, mention_id, label="contains", weight=1.0)

            # Add candidate referent nodes
            candidates = ent_res.get("all_candidates", [])
            for cand_idx, cand in enumerate(candidates):
                cand_id = f"cand_{ent_idx}_{cand_idx}_{cand.get('title', 'cand')}"
                is_selected = cand.get("is_selected", False)
                score = cand.get("confidence_score", 0.0)

                node_color = self.COLOR_PALETTE["SELECTED_MATCH"] if is_selected else self.COLOR_PALETTE["ALT_CANDIDATE"]
                prefix = "[MATCH]" if is_selected else "[ALT]"
                node_label = f"{prefix} {cand.get('title', 'Unknown')}\n(Score: {score:.2f})"

                G.add_node(
                    cand_id,
                    label=node_label,
                    node_type="SELECTED" if is_selected else "ALTERNATIVE",
                    color=node_color,
                    size=22 if is_selected else 15,
                    title=f"{cand.get('title')}: {cand.get('description', '')}\nURL: {cand.get('url', '')}\nScore: {score:.4f}",
                    url=cand.get("url", "")
                )

                # Edge from mention to candidate
                edge_label = f"Match ({score:.2f})" if is_selected else f"Alt ({score:.2f})"
                G.add_edge(
                    mention_id,
                    cand_id,
                    label=edge_label,
                    weight=float(score),
                    is_selected=is_selected
                )

        return G

    def build_agraph_components(self, pipeline_results: List[Dict[str, Any]], text_preview: str):
        """Build Node and Edge lists for streamlit-agraph rendering."""
        if not HAS_AGRAPH:
            return [], []

        nodes = []
        edges = []

        root_id = "root_context"
        nodes.append(Node(
            id=root_id,
            label=f"Input Context\n\"{text_preview[:30]}...\"",
            size=26,
            color=self.COLOR_PALETTE["ROOT"],
            font={"color": "#FFFFFF", "size": 13, "bold": True},
            shape="box"
        ))

        for ent_idx, ent_res in enumerate(pipeline_results):
            mention_text = ent_res["entity_text"]
            label = ent_res.get("label", "ENTITY")
            mention_id = f"mention_{ent_idx}"

            mention_color = self.COLOR_PALETTE.get(f"MENTION_{label}", self.COLOR_PALETTE["MENTION_DEFAULT"])
            nodes.append(Node(
                id=mention_id,
                label=f"{mention_text}\n[{label}]",
                size=22,
                color=mention_color,
                font={"color": "#FFFFFF", "size": 12},
                shape="ellipse"
            ))

            edges.append(Edge(
                source=root_id,
                target=mention_id,
                color="#94A3B8",
                width=2
            ))

            candidates = ent_res.get("all_candidates", [])
            for cand_idx, cand in enumerate(candidates):
                cand_id = f"cand_{ent_idx}_{cand_idx}"
                is_selected = cand.get("is_selected", False)
                score = cand.get("confidence_score", 0.0)

                color = self.COLOR_PALETTE["SELECTED_MATCH"] if is_selected else self.COLOR_PALETTE["ALT_CANDIDATE"]
                prefix = "★ SELECTED: " if is_selected else "• ALT: "
                cand_label = f"{prefix}{cand.get('title')}\nScore: {score:.2f}"

                nodes.append(Node(
                    id=cand_id,
                    label=cand_label,
                    size=24 if is_selected else 16,
                    color=color,
                    font={"color": "#FFFFFF", "size": 11, "bold": is_selected},
                    shape="box" if is_selected else "dot"
                ))

                edge_color = "#10B981" if is_selected else "#CBD5E1"
                edge_width = 3 if is_selected else 1
                edges.append(Edge(
                    source=mention_id,
                    target=cand_id,
                    label=f"{score:.2f}",
                    color=edge_color,
                    width=edge_width,
                    dashes=not is_selected
                ))

        return nodes, edges

    def build_plotly_figure(self, pipeline_results: List[Dict[str, Any]], text_preview: str) -> go.Figure:
        """Create a high-performance Plotly interactive network graph visualization."""
        G = self.build_networkx_graph(pipeline_results, text_preview)

        # Spring layout positions
        pos = nx.spring_layout(G, k=1.4, iterations=60, seed=42)

        # Edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#CBD5E1"),
            hoverinfo="none",
            mode="lines"
        )

        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        customdata = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            attr = G.nodes[node]
            node_text.append(attr.get("label", ""))
            node_color.append(attr.get("color", "#3B82F6"))
            node_size.append(attr.get("size", 18) * 1.5)
            customdata.append(attr.get("title", ""))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="bottom center",
            hoverinfo="text",
            hovertext=customdata,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color="#FFFFFF")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="<b>Entity Disambiguation Knowledge Graph</b>",
                    font=dict(size=18, color="#1E293B"),
                    x=0.05
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#F8FAFC",
                paper_bgcolor="#FFFFFF",
                height=520
            )
        )
        return fig


if __name__ == "__main__":
    builder = DisambiguationGraphBuilder()
    mock_results = [
        {
            "entity_text": "Apple",
            "label": "ORG",
            "is_ambiguous": True,
            "all_candidates": [
                {"title": "Apple Inc.", "description": "Tech company", "confidence_score": 0.88, "is_selected": True, "url": "https://en.wikipedia.org/wiki/Apple_Inc."},
                {"title": "Apple", "description": "Fruit", "confidence_score": 0.18, "is_selected": False, "url": "https://en.wikipedia.org/wiki/Apple"}
            ]
        }
    ]
    fig = builder.build_plotly_figure(mock_results, "Apple announced new earnings.")
    print("Graph built successfully with", len(fig.data), "traces.")

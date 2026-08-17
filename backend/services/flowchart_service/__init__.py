"""FlowchartService — reaction pathway graph.

Builds the complete chemistry pathway as a graph where:
  - nodes = compounds (or stages)
  - edges = reactions (carrying all conditions)

Outputs:
  - `to_react_flow()` : JSON consumed by the React Flow frontend.
  - `render_static()` : matplotlib + RDKit PNG for the Word report
    (auto-split into logical sections when too large for one page).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from backend.config import settings
from backend.models import Compound, Job, Reaction, Stage
from backend.services.structure_service import render_structure

logger = logging.getLogger("pca.flowchart")


@dataclass
class FlowNodeSpec:
    id: str
    kind: str
    label: str
    page: int | None = None
    cid: str | None = None
    smiles: str | None = None
    stage_order: int | None = None


@dataclass
class FlowEdgeSpec:
    id: str
    source: str
    target: str
    label: str
    rid: str | None = None


def build_graph(job: Job) -> tuple[list[FlowNodeSpec], list[FlowEdgeSpec]]:
    """Build nodes/edges from the persisted reactions and compounds."""
    compounds: dict[str, Compound] = {c.cid: c for c in job.compounds}
    reactions = sorted(job.reactions, key=lambda r: r.rid)
    stages: dict[str, Stage] = {}
    for s in job.stages:
        stages.setdefault(s.title.lower(), s)

    nodes: list[FlowNodeSpec] = []
    edges: list[FlowEdgeSpec] = []
    used: set[str] = set()

    for c in compounds.values():
        used.add(c.cid)
        nodes.append(FlowNodeSpec(
            id=f"c:{c.cid}",
            kind="compound",
            label=c.name,
            page=c.source_page,
            cid=c.cid,
            smiles=c.smiles,
        ))

    edge_idx = 0
    for rxn in reactions:
        participants = _participants_by_role(rxn)
        reactants = participants.get("reactant", [])
        products = participants.get("product", [])
        intermediates = participants.get("intermediate", [])

        # Fallback: if the participants table is sparse, match from text fields.
        if not reactants and rxn.reactants_text:
            reactants = _fuzzy_match_cids(rxn.reactants_text, compounds)
        if not products and rxn.products_text:
            products = _fuzzy_match_cids(rxn.products_text, compounds)

        # Second-pass: for reactions with only one side, try matching
        # the OTHER text against known compounds too.
        if reactants and not products and rxn.products_text:
            products = _fuzzy_match_cids(rxn.products_text, compounds)
        if products and not reactants and rxn.reactants_text:
            reactants = _fuzzy_match_cids(rxn.reactants_text, compounds)

        if not reactants and not intermediates:
            continue
        sources = intermediates or reactants
        targets = products or intermediates
        if not targets:
            continue
        # Chain: each reactant -> each product; intermediates are waypoints.
        ordered_sources = intermediates + [r for r in reactants if r not in intermediates]
        for src in ordered_sources:
            for tgt in targets:
                if src == tgt:
                    continue
                edge_idx += 1
                edges.append(FlowEdgeSpec(
                    id=f"e:{rxn.rid}:{edge_idx}",
                    source=f"c:{src}",
                    target=f"c:{tgt}",
                    label=_edge_label(rxn),
                    rid=rxn.rid,
                ))

    return nodes, edges


def to_react_flow(job: Job, nodes: list[FlowNodeSpec], edges: list[FlowEdgeSpec]) -> dict:
    """Layout the graph and emit React Flow node/edge JSON."""
    layout = _hierarchical_layout(nodes, edges)

    rxn_by_id = {r.rid: r for r in job.reactions}
    node_json = []
    for n in nodes:
        node_json.append({
            "id": n.id,
            "position": layout.get(n.id, {"x": 0, "y": 0}),
            "data": {
                "kind": n.kind,
                "label": n.label,
                "cid": n.cid,
                "smiles": n.smiles,
                "page": n.page,
                "image_url": f"/api/jobs/{job.id}/structures/{n.cid}.png" if n.cid else None,
            },
        })
    edge_json = []
    for e in edges:
        rxn = rxn_by_id.get(e.rid)
        edge_json.append({
            "id": e.id,
            "source": e.source,
            "target": e.target,
            "label": e.label,
            "data": {"rid": e.rid},
        })
    return {"nodes": node_json, "edges": edge_json}


def _hierarchical_layout(
    nodes: list[FlowNodeSpec], edges: list[FlowEdgeSpec]
) -> dict[str, dict]:
    """Topological multi-layer layout (layer = distance from sources)."""
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n.id)
    for e in edges:
        g.add_edge(e.source, e.target)

    layer: dict[str, int] = {}
    for nid in nx.topological_sort(g):
        preds = list(g.predecessors(nid))
        layer[nid] = max((layer[p] for p in preds), default=0) + 1 if preds else 0

    # Assign columns per layer.
    cols: dict[str, int] = {}
    col_counter: dict[int, int] = {}
    for nid in nx.topological_sort(g):
        l = layer[nid]
        cols[nid] = col_counter.get(l, 0)
        col_counter[l] = col_counter.get(l, 0) + 1

    X, Y, DX, DY = 320, 240, 420, 260  # noqa: N806
    pos = {}
    for nid in g.nodes:
        l = layer.get(nid, 0)
        col = cols.get(nid, 0)
        pos[nid] = {"x": l * DX, "y": col * DY + (DY / 2 if col % 2 == 1 else 0)}
    return pos


def _participants_by_role(rxn: Reaction) -> dict[str, list[str]]:
    from backend.models import reaction_roles
    from backend.database import SessionLocal

    result: dict[str, list[str]] = {}
    with SessionLocal() as db:
        roles = reaction_roles(db, rxn.id)
    for c in rxn.participants:
        role = roles.get(c.id, "reactant")
        result.setdefault(role, [])
        result[role].append(c.cid)
    return result


def _fuzzy_match_cids(text: str, compounds: dict[str, Compound]) -> list[str]:
    """Match compound names from a text string against known compounds."""
    text_lower = text.lower()
    matched: list[str] = []
    for cid, c in compounds.items():
        name_lower = c.name.lower()
        common_lower = (c.common_name or "").lower()
        iupac_lower = (c.iupac_name or "").lower()
        # Exact or substring match on any name variant
        if len(name_lower) > 4 and (
            name_lower in text_lower
            or text_lower in name_lower
            or (common_lower and len(common_lower) > 4 and common_lower in text_lower)
            or (iupac_lower and len(iupac_lower) > 4 and iupac_lower in text_lower)
        ):
            if cid not in matched:
                matched.append(cid)
            continue
        # Word overlap: extract significant words (>5 chars) from compound name
        name_words = {w for w in name_lower.split() if len(w) > 5}
        text_words = set(text_lower.split())
        if name_words and len(name_words & text_words) >= 2:
            if cid not in matched:
                matched.append(cid)
    return matched


def _edge_label(rxn: Reaction) -> str:
    parts = []
    if rxn.type and rxn.type != "Not specified in patent":
        parts.append(rxn.type)
    if rxn.reagents and rxn.reagents != "Not specified in patent":
        parts.append(rxn.reagents)
    if rxn.temperature and rxn.temperature != "Not specified in patent":
        parts.append(rxn.temperature)
    if rxn.time and rxn.time != "Not specified in patent":
        parts.append(rxn.time)
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Static rendering (for the Word report)
# --------------------------------------------------------------------------- #
def render_static(
    job: Job,
    nodes: list[FlowNodeSpec],
    edges: list[FlowEdgeSpec],
    out_dir: Path,
    max_nodes_per_page: int = 9,
) -> list[Path]:
    """Render the flowchart as a set of high-res PNGs, split across pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = _hierarchical_layout(nodes, edges)
    rxn_by_id = {r.rid: r for r in job.reactions}
    compounds = {c.cid: c for c in job.compounds}

    pages: list[Path] = []
    current: list[FlowNodeSpec] = []
    for n in nodes:
        if len(current) >= max_nodes_per_page and current[-1].label != n.label:
            pages.append(_render_png(current, edges, layout, compounds, rxn_by_id, out_dir, len(pages)))
            current = []
        current.append(n)
    if current:
        pages.append(_render_png(current, edges, layout, compounds, rxn_by_id, out_dir, len(pages)))
    return pages


def _render_png(
    nodes: list[FlowNodeSpec],
    edges: list[FlowEdgeSpec],
    layout: dict,
    compounds: dict[str, Compound],
    rxn_by_id: dict[str, Reaction],
    out_dir: Path,
    page_idx: int,
) -> Path:
    import matplotlib
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_ids = {n.id for n in nodes}
    local_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]
    if not local_edges and len(nodes) > 1:
        # Fallback vertical layout when no edges connect this page's nodes.
        order = list(node_ids)
        for i in range(len(order) - 1):
            local_edges.append(FlowEdgeSpec(id=f"fb{i}", source=order[i], target=order[i + 1], label=""))
        local_nodes = {n.id: n for n in nodes}
        for i, nid in enumerate(order):
            layout = dict(layout)
            layout[nid] = {"x": 0, "y": i * 260}

    W = 12.0
    H = max(6.0, (max((layout[n.id]["y"] for n in nodes), default=0) + 260) / 100)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axis("off")

    for n in nodes:
        pos = layout[n.id]
        box_w, box_h = 3.4, 1.9
        x, y = pos["x"] / 100, pos["y"] / 100
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.06", linewidth=1.2,
            edgecolor="#800000", facecolor="#FFF8E7", zorder=3,
        ))
        label_lines = [n.label[:28]]
        c = compounds.get(n.cid or "")
        if c and c.molecular_formula and c.molecular_formula != "Not specified in patent":
            label_lines.append(c.molecular_formula)
        if c and c.molecular_weight:
            label_lines.append(f"MW {c.molecular_weight:.1f}")
        ax.text(x, y, "\n".join(label_lines), ha="center", va="center",
                fontsize=8, zorder=4, color="#3b2a20")

    for e in local_edges:
        s = layout[e.source]
        t = layout[e.target]
        ax.add_patch(FancyArrowPatch(
            (s["x"] / 100 + 1.7, s["y"] / 100), (t["x"] / 100 - 1.7, t["y"] / 100),
            arrowstyle="-|>", mutation_scale=14, color="#8a5a00", lw=1.6, zorder=2,
        ))
        if e.label:
            mx, my = (s["x"] / 100 + t["x"] / 100) / 2, (s["y"] / 100 + t["y"] / 100) / 2
            ax.text(mx, my + 0.15, e.label[:42], ha="center", fontsize=7,
                    color="#5c4632", style="italic")

    out = out_dir / f"flowchart_part_{page_idx + 1}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

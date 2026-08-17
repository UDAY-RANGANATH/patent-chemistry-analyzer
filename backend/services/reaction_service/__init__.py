"""ReactionService — reaction understanding and "What Changed?" engine.

Combines RDKit structural analysis (structure_service) with patent evidence
(RAG) to produce a structured, provenance-stamped reaction explanation.

Everything labelled "AI Chemical Interpretation" is clearly distinguished from
"Patent-stated" and "Chemistry database" facts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from backend.services.structure_service import (
    ChangeSummary,
    analyze,
    classify_reaction,
    what_changed,
)

logger = logging.getLogger("pca.reaction")


@dataclass
class WhatChangedView:
    """UI-ready 'What Changed?' payload for a reaction arrow."""

    before_smiles: str | None = None
    after_smiles: str | None = None
    bond_formed: list[str] = field(default_factory=list)
    bond_broken: list[str] = field(default_factory=list)
    atoms_added: list[str] = field(default_factory=list)
    atoms_removed: list[str] = field(default_factory=list)
    functional_group_added: list[str] = field(default_factory=list)
    functional_group_removed: list[str] = field(default_factory=list)
    formula_before: str | None = None
    formula_after: str | None = None
    similarity: float | None = None
    mcs_smiles: str | None = None
    mcs_coverage: float | None = None
    reaction_types: list[str] = field(default_factory=list)
    confidence: float = 0.0
    basis: str = "RDKit structural analysis + patent evidence"


def build_what_changed(
    reactant_smiles: list[str],
    product_smiles: list[str],
) -> WhatChangedView | None:
    """Compute the structural diff between (combined) reactants and products."""
    if not reactant_smiles or not product_smiles:
        return None

    # Combine multiple reactants/products into one pseudo-molecule each for a
    # coarse formula-level diff; pick the best pair for MCS similarity.
    primary_r = _best_smiles(reactant_smiles)
    primary_p = _best_smiles(product_smiles)
    if not primary_r or not primary_p:
        return None

    cs: ChangeSummary = what_changed(primary_r, primary_p)
    view = WhatChangedView(
        before_smiles=primary_r,
        after_smiles=primary_p,
        bond_formed=cs.bond_formed,
        bond_broken=cs.bond_broken,
        atoms_added=cs.atoms_added,
        atoms_removed=cs.atoms_removed,
        functional_group_added=cs.fg_added,
        functional_group_removed=cs.fg_removed,
        formula_before=cs.formula_before,
        formula_after=cs.formula_after,
        similarity=cs.similarity,
        mcs_smiles=cs.mcs_smiles,
        mcs_coverage=cs.mcs_coverage,
        reaction_types=classify_reaction(primary_r, primary_p),
    )

    # Confidence from MCS coverage: high coverage => high confidence in the diff.
    coverage = cs.mcs_coverage or 0.0
    if cs.similarity is not None and cs.similarity > 0.9:
        view.confidence = 0.95
    elif coverage > 0.7:
        view.confidence = 0.85
    elif coverage > 0.4:
        view.confidence = 0.7
    else:
        view.confidence = 0.5
    return view


def _best_smiles(candidates: list[str]) -> str | None:
    """Choose the most 'structural' candidate (largest molecule) for diffing."""
    best: str | None = None
    best_atoms = -1
    for smi in candidates:
        info = analyze(smi)
        if info.valid:
            # mol wt correlates with size; heavier = more scaffold
            mw = info.molecular_weight or 0
            if mw > best_atoms:
                best_atoms = mw
                best = info.canonical_smiles
    return best


def serialize(view: WhatChangedView | None) -> dict | None:
    if view is None:
        return None
    return {
        "before_smiles": view.before_smiles,
        "after_smiles": view.after_smiles,
        "bond_formed": view.bond_formed,
        "bond_broken": view.bond_broken,
        "atoms_added": view.atoms_added,
        "atoms_removed": view.atoms_removed,
        "functional_group_added": view.functional_group_added,
        "functional_group_removed": view.functional_group_removed,
        "formula_before": view.formula_before,
        "formula_after": view.formula_after,
        "similarity": view.similarity,
        "mcs_smiles": view.mcs_smiles,
        "mcs_coverage": view.mcs_coverage,
        "reaction_types": view.reaction_types,
        "confidence": view.confidence,
        "basis": view.basis,
    }


def what_changed_text(view: WhatChangedView | None) -> str:
    """Human-readable 'What Changed?' paragraph for the report/UI."""
    if view is None:
        return "Not enough structural data to determine what changed."
    lines: list[str] = []
    if view.reaction_types:
        lines.append("Reaction type: " + ", ".join(view.reaction_types) + ".")
    if view.formula_before and view.formula_after:
        if view.formula_before != view.formula_after:
            lines.append(
                f"Molecular formula changed from {view.formula_before} to {view.formula_after}."
            )
        else:
            lines.append(f"Molecular formula ({view.formula_before}) is unchanged.")
    if view.atoms_added:
        lines.append("Atoms added: " + ", ".join(view.atoms_added) + ".")
    if view.atoms_removed:
        lines.append("Atoms removed: " + ", ".join(view.atoms_removed) + ".")
    if view.functional_group_added:
        lines.append("Functional groups introduced: " + ", ".join(view.functional_group_added) + ".")
    if view.functional_group_removed:
        lines.append("Functional groups lost: " + ", ".join(view.functional_group_removed) + ".")
    for b in view.bond_formed:
        lines.append("Bonds formed: " + b + ".")
    for b in view.bond_broken:
        lines.append("Bonds broken: " + b + ".")
    if view.similarity is not None:
        lines.append(f"Tanimoto similarity of starting material and product: {view.similarity:.2f}.")
    if not lines:
        lines.append("The two structures are identical within the resolved data.")
    lines.append(f"(basis: {view.basis})")
    return " ".join(lines)


def merge_participants(*groups: list[dict]) -> dict[str, list[dict]]:
    """Merge LLM-extracted participant lists without duplicate names."""
    merged: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for role, items in [g for g in groups]:
        for item in items or []:
            name = item.get("name") or item.get("compound") or ""
            if not name:
                continue
            key = name.lower().strip()
            seen.setdefault(role, set())
            if key in seen[role]:
                continue
            seen[role].add(key)
            merged.setdefault(role, []).append(item)
    return merged

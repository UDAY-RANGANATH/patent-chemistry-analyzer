"""StructureService — RDKit wrapper.

Responsibilities:
  - SMILES parsing / validation / canonicalization
  - Molecular formula, molecular weight, descriptors
  - Substructure search (functional-group detection)
  - Fingerprints + similarity
  - Stereochemistry checks
  - Structure rendering (PNG / SVG) for the UI and the Word report
  - Maximum Common Substructure (MCS) diff used by "What Changed?"
  - Reaction SMARTS matching (reaction-type classification)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem import rdFMCS

logger = logging.getLogger("pca.structure")


@dataclass
class StructureInfo:
    valid: bool = False
    canonical_smiles: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    exact_mass: float | None = None
    inchi: str | None = None
    inchikey: str | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    aromatic_rings: int | None = None
    stereocenters: int = 0
    functional_groups: list[str] = field(default_factory=list)
    error: str | None = None


FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    "carboxylic acid": "C(=O)[OH]",
    "ester": "C(=O)O[C,N]",
    "amide": "C(=O)[N]",
    "aldehyde": "[CX3H1](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H]",
    "ether": "[CX4][OX2][CX4]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "C#N",
    "nitro": "[NX3](=O)=O",
    "sulfide": "[SX2]",
    "sulfone": "S(=O)(=O)",
    "sulfoxide": "[SX3]=O",
    "thiol": "[SX2H]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "aryl halide": "[cX3][F,Cl,Br,I]",
    "alkyl halide": "[CX4][F,Cl,Br,I]",
    "carbamate": "NC(=O)O",
    "urea": "NC(=O)N",
    "anhydride": "C(=O)OC(=O)",
    "imine": "C=N",
    "oxime": "C=N[OX2H]",
    "hydrazide": "NN(C=O)",
    "sulfonamide": "S(=O)(=O)[N]",
    "carboxylate salt": "C(=O)[O-]",
    "ammonium salt": "[N+;H3,H2,H1,H0]",
    "phosphate": "P(=O)(O)(O)O",
    "thioester": "C(=O)S",
    "boc (tert-butoxycarbonyl)": "OC(=O)OC(C)(C)C",
    "cbz (benzyloxycarbonyl)": "OC(=O)OCc1ccccc1",
    "f-moc (fluorenylmethoxycarbonyl)": "OC(=O)OCC1c2ccccc2-c2ccccc21",
    "benzyl": "[CH2X4]c1ccccc1",
    "methyl": "[CX4H3]",
    "tert-butyl": "C(C)(C)C",
    "phenyl": "c1ccccc1",
}

REACTION_TYPE_SMARTS: dict[str, str] = {
    "esterification": "([#6:1]-[OX2H]) + ([OX2H]-[#6:2]=[OX1]) >> ([#6:1]-[OX2]-[#6:2]=[OX1])",
    "amidation": "([#6]-[OX2H]) + ([NH2:1]-[CX3:2]=[OX1]) >> ([CX3:2](=[OX1])-[N:1])",
    "hydrolysis of ester": "[CX3:1](=O)[OX2]-[#6] >> [CX3:1](=O)[OX2H]",
    "reduction of aldehyde": "[CX3H1:1]=[OX1] >> [CX4H2:1][OX2H]",
    "reduction of ketone": "[#6][CX3:1](=O)[#6] >> [#6][CX3:1][OX2H]",
    "oxidation of alcohol": "[CX4:1][OX2H] >> [CX3:1]=[OX1]",
    "oxidation of aldehyde": "[CX3H1:1]=[OX1] >> [CX3:1](=[OX1])[OX2H]",
    "hydrogenation of alkene": "[CX3:1]=[CX3:2] >> [CX4:1][CX4:2]",
    "halogenation": "[CX4:1]-[H] >> [CX4:1]-[Cl,Br,I]",
    "nitration": "([c:1][H]) >> ([c:1][NX3](=O)=O)",
    "protection of alcohol (acetyl)": "[OX2H:1] >> [OX2:1][C](=O)[CH3]",
    "protection of amine (boc)": "[NX3;H2:1] >> [NX3;H1:1]C(=O)OC(C)(C)C",
    "deprotection (remove boc)": "[NX3;H1:1]C(=O)OC(C)(C)C >> [NX3;H2:1]",
    "salt formation (amine)": "[NX3:1] >> [N+:1][H]",
    "coupling (amide from acid chloride)": "[CX3:1](=O)[Cl] >> [CX3:1](=O)[N]",
}


def parse_smiles(smiles: str | None) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def analyze(smiles: str) -> StructureInfo:
    """Full RDKit analysis of one SMILES string."""
    info = StructureInfo()
    mol = parse_smiles(smiles)
    if mol is None:
        info.error = f"Invalid SMILES: {smiles}"
        return info

    info.valid = True
    info.canonical_smiles = Chem.MolToSmiles(mol)
    info.formula = rdMolDescriptors.CalcMolFormula(mol)
    info.molecular_weight = round(Descriptors.MolWt(mol), 2)
    info.exact_mass = round(Descriptors.ExactMolWt(mol), 4)
    info.inchi = Chem.MolToInchi(mol)
    info.inchikey = Chem.InchiToInchiKey(info.inchi)
    info.logp = round(Crippen.MolLogP(mol), 2)
    info.tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    info.hbd = rdMolDescriptors.CalcNumHBD(mol)
    info.hba = rdMolDescriptors.CalcNumHBA(mol)
    info.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    info.aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    info.stereocenters = len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False))
    info.functional_groups = detect_functional_groups(mol)
    return info


def detect_functional_groups(mol: Chem.Mol) -> list[str]:
    found: list[str] = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                found.append(name)
        except Exception:  # noqa: BLE001
            continue
    return found


def classify_reaction(reactant_smiles: str, product_smiles: str) -> list[str]:
    """Classify a reaction via RDKit reaction SMARTS. Returns matching types."""
    types: list[str] = []
    r_mol = parse_smiles(reactant_smiles)
    p_mol = parse_smiles(product_smiles)
    if r_mol is None or p_mol is None:
        return types
    for name, smarts in REACTION_TYPE_SMARTS.items():
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn.Validate()[0]:
                continue
            matched = False
            for i in range(rxn.GetNumReactantTemplates()):
                if r_mol.HasSubstructMatch(rxn.GetReactantTemplate(i)):
                    matched = True
                    break
            if matched:
                for i in range(rxn.GetNumProductTemplates()):
                    if p_mol.HasSubstructMatch(rxn.GetProductTemplate(i)):
                        types.append(name)
                        break
        except Exception:  # noqa: BLE001
            continue
    return types


def similarity(a: str, b: str, fp_size: int = 2048) -> float | None:
    """Tanimoto similarity between two SMILES (Morgan fingerprints)."""
    ma = parse_smiles(a)
    mb = parse_smiles(b)
    if ma is None or mb is None:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=fp_size)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=fp_size)
    return DataStructs.TanimotoSimilarity(fa, fb)


@dataclass
class ChangeSummary:
    bond_formed: list[str] = field(default_factory=list)
    bond_broken: list[str] = field(default_factory=list)
    atoms_added: list[str] = field(default_factory=list)
    atoms_removed: list[str] = field(default_factory=list)
    formula_before: str | None = None
    formula_after: str | None = None
    fg_added: list[str] = field(default_factory=list)
    fg_removed: list[str] = field(default_factory=list)
    mcs_smiles: str | None = None
    mcs_coverage: float = 0.0  # fraction of product atoms in the MCS
    similarity: float | None = None


def what_changed(before_smiles: str, after_smiles: str) -> ChangeSummary:
    """Structural diff between reactant and product (for 'What Changed?')."""
    cs = ChangeSummary()
    m_before = parse_smiles(before_smiles)
    m_after = parse_smiles(after_smiles)
    if m_before is None or m_after is None:
        return cs

    cs.formula_before = rdMolDescriptors.CalcMolFormula(m_before)
    cs.formula_after = rdMolDescriptors.CalcMolFormula(m_after)
    cs.similarity = similarity(before_smiles, after_smiles)

    rings_before = rdMolDescriptors.CalcNumRings(m_before)
    rings_after = rdMolDescriptors.CalcNumRings(m_after)
    if rings_after > rings_before:
        cs.bond_formed.append(f"new ring closure ({rings_after - rings_before} ring(s) formed)")
    elif rings_after < rings_before:
        cs.bond_broken.append(f"ring opened ({rings_before - rings_after} ring(s) lost)")

    fg_before = set(detect_functional_groups(m_before))
    fg_after = set(detect_functional_groups(m_after))
    cs.fg_added = sorted(fg_after - fg_before)
    cs.fg_removed = sorted(fg_before - fg_after)

    # MCS — the common scaffold; changes are what is NOT in the MCS.
    mcs = rdFMCS.FindMCS(
        [m_before, m_after],
        timeout=30,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
    )
    cs.mcs_smiles = mcs.smartsString if mcs.numAtoms else None
    if mcs.numAtoms and m_after.GetNumAtoms():
        cs.mcs_coverage = round(mcs.numAtoms / m_after.GetNumAtoms(), 3)

    # Formula-level atom add/remove
    from collections import Counter

    def _formula_counts(f: str) -> Counter:
        import re

        c = Counter()
        for m in re.finditer(r"([A-Z][a-z]?)(\d*)", f):
            el, n = m.groups()
            c[el] += int(n or 1)
        return c

    ca, cb = _formula_counts(cs.formula_before or ""), _formula_counts(cs.formula_after or "")
    for el in sorted(set(ca) | set(cb)):
        diff = cb[el] - ca[el]
        if diff > 0:
            cs.atoms_added.append(f"{el}{diff if diff > 1 else ''}")
        elif diff < 0:
            cs.atoms_removed.append(f"{el}{-diff if diff < -1 else ''}")

    # Bond add/remove heuristics via formula H-delta + degree changes are complex;
    # surface a human-readable list from formula differences when available.
    if cs.atoms_added:
        cs.bond_formed.append("new bonds involving " + ", ".join(cs.atoms_added))
    if cs.atoms_removed:
        cs.bond_broken.append("bonds to removed " + ", ".join(cs.atoms_removed))
    if not cs.bond_formed and cs.similarity is not None:
        if cs.similarity < 0.95:
            cs.bond_formed.append("rearrangement within common scaffold (see MCS)")
    return cs


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_structure(
    smiles: str,
    out_path: Path,
    width: int = 360,
    height: int = 260,
    highlight_smarts: str | None = None,
    label: str | None = None,
) -> bool:
    """Render a molecule to PNG. Optionally highlight a substructure match."""
    mol = parse_smiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    highlight_atoms: list[int] = []
    if highlight_smarts:
        pat = Chem.MolFromSmarts(highlight_smarts)
        if pat:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                highlight_atoms = sorted(set(sum((list(m) for m in matches), [])))
    try:
        img = Draw.MolToImage(
            mol,
            size=(width, height),
            highlightAtoms=highlight_atoms or None,
            highlightColor=(0.9, 0.1, 0.1),
            legend=label or None,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("render failed for %s: %s", smiles[:40], exc)
        return False


def render_svg(smiles: str, width: int = 300, height: int = 200) -> str | None:
    """Return an SVG string for a molecule (for web display)."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    try:
        return Draw.MolToSVG(mol, size=(width, height))
    except Exception:  # noqa: BLE001
        return None


def structure_image_bytes(smiles: str, fmt: str = "PNG", width: int = 360, height: int = 260) -> bytes | None:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    try:
        if fmt.upper() == "SVG":
            return Draw.MolToSVG(mol, size=(width, height)).encode("utf-8")
        img = Draw.MolToImage(mol, size=(width, height))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None

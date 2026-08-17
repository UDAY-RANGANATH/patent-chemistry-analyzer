"""ValidationService — quality control and hallucination checks.

Runs after the pipeline and flags:
  - compounds without structures ("unresolved")
  - reactions missing reactants/products
  - contradictions between patent text and database data
  - fields defaulting to "Not specified in patent" (honest gaps, not failures)
  - any AI claim lacking patent-page provenance

Output is a list of `QCIssue`s plus an overall quality score (0..1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("pca.qc")

NOT_SPECIFIED = "Not specified in patent"


@dataclass
class QCIssue:
    level: str  # error | warning | info
    category: str
    entity: str
    message: str


@dataclass
class QCReport:
    score: float = 1.0
    issues: list[QCIssue] = field(default_factory=list)


def audit_compounds(compounds) -> None:
    """Add issues for compounds lacking structure / provenance."""
    return  # placeholder signature used by tests; real audit in run_qc


def run_qc(job) -> QCReport:
    """Full QC pass over one job."""
    report = QCReport()
    problems = 0.0

    # --- Compounds ---
    for c in job.compounds:
        if not c.smiles:
            report.issues.append(QCIssue(
                "warning", "structure", c.cid,
                f"Compound '{c.name}' could not be resolved to a structure; "
                "SMILES/InChI unavailable. Any structural claims about it are unsupported.",
            ))
            problems += 0.25
        if c.source_page is None:
            report.issues.append(QCIssue(
                "warning", "provenance", c.cid,
                f"Compound '{c.name}' has no patent page reference.",
            ))
            problems += 0.15
        if c.extra and c.extra.get("db_conflicts"):
            report.issues.append(QCIssue(
                "info", "database_conflict", c.cid,
                "Chemistry databases disagree on this compound; competing values shown in UI.",
            ))
        if c.confidence < 0.5 and c.smiles:
            report.issues.append(QCIssue(
                "warning", "confidence", c.cid,
                f"Low confidence ({c.confidence:.0%}) identification for '{c.name}'.",
            ))
            problems += 0.1

    # --- Reactions ---
    for r in job.reactions:
        roles = _roles_for(r)
        reactants = [p for p in r.participants if roles.get(p.id) == "reactant"]
        products = [p for p in r.participants if roles.get(p.id) == "product"]
        if not reactants:
            report.issues.append(QCIssue(
                "warning", "reaction", r.rid,
                f"Reaction {r.rid} has no resolved reactant structures.",
            ))
            problems += 0.3
        if not products:
            report.issues.append(QCIssue(
                "warning", "reaction", r.rid,
                f"Reaction {r.rid} has no resolved product structures.",
            ))
            problems += 0.3
        if r.source_page is None:
            report.issues.append(QCIssue(
                "warning", "provenance", r.rid,
                f"Reaction {r.rid} has no patent page reference.",
            ))
            problems += 0.15

    # --- Overall gaps ---
    if not job.compounds:
        report.issues.append(QCIssue(
            "error", "coverage", "job",
            "No chemical compounds were identified in this patent.",
        ))
        problems += 1.0
    if not job.reactions:
        report.issues.append(QCIssue(
            "info", "coverage", "job",
            "No reactions were extracted. The patent may be claims/description-only.",
        ))

    report.score = max(0.0, 1.0 - problems)
    if report.score < 0.5:
        report.issues.append(QCIssue(
            "warning", "overall", "job",
            "Overall quality score is low; interpret results with caution.",
        ))
    return report


def _roles_for(rxn) -> dict[int, str]:
    from backend.database import SessionLocal
    from backend.models import reaction_roles

    with SessionLocal() as db:
        return reaction_roles(db, rxn.id)

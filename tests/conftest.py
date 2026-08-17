"""Pytest fixtures — isolated temp DB + storage, scripted FakeAI, seed helpers.

Environment overrides MUST be set before importing backend modules: the engine
and provider chain are configured once at import time.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Path to the bundled sample patent, used by the API upload tests.
pytest._pca_sample_path = _PROJECT_ROOT / "example_data" / "sample_patent.pdf"

_TMP = Path(tempfile.mkdtemp(prefix="pca_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["PAGE_IMAGE_DIR"] = str(_TMP / "pages")
os.environ["STRUCTURE_DIR"] = str(_TMP / "structures")
os.environ["REPORT_DIR"] = str(_TMP / "reports")
os.environ["MAX_PAGES"] = "50"
# Force a deterministic (non-network) provider so nothing leaks out of tests.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["CHEMSPIDER_API_KEY"] = ""

from backend.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.models import (  # noqa: E402
    Compound,
    Job,
    Page,
    Reaction,
    Section,
    Stage,
)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    init_db()
    yield
    engine.dispose()


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def fake_ai():
    """Scripted AIService: returns canned JSON per prompt kind, offline."""
    from backend.services.ai_service import AIService

    class FakeAI(AIService):
        def chat_json(self, messages, **kwargs):
            content = "\n".join(m.get("content", "") for m in messages).lower()
            if "executive summary" in content or "bibliographic header" in content:
                return {
                    "patent_title": "PROCESS FOR THE PREPARATION OF METHYL 4-HYDROXYBENZOATE",
                    "patent_number": "US 12,345,678 B2", "assignee": "Test Corp",
                    "applicants": "Test Corp", "inventors": "J. Doe",
                    "filing_date": "March 4, 2023", "publication_date": "Not specified in patent",
                    "abstract": "Abstract text.",
                }
            if "chemical substance" in content or '"compounds"' in content or '"compounds":' in content:
                return {
                    "compounds": [
                        {"name": "methyl 4-hydroxybenzoate", "role": "product",
                         "source_page": 1, "source_text": "methyl 4-hydroxybenzoate is prepared.",
                         "context": "final product"},
                        {"name": "4-hydroxybenzoic acid", "role": "reactant",
                         "source_page": 1, "source_text": "from 4-hydroxybenzoic acid.", "context": "SM"},
                        {"name": "methanol", "role": "solvent", "source_page": 1,
                         "source_text": "in methanol.", "context": "solvent"},
                        {"name": "sulfuric acid", "role": "catalyst", "source_page": 1,
                         "source_text": "catalyzed by sulfuric acid.", "context": "catalyst"},
                    ]
                }
            if "manufacturing process" in content or '"stages"' in content:
                return {
                    "stages": [
                        {"order": 1, "title": "Stage 1 - Esterification",
                         "purpose": "form the ester", "starting_material": "4-hydroxybenzoic acid",
                         "reagents": "methanol, sulfuric acid", "conditions": "65 C, 4 h",
                         "reaction": "Fischer esterification", "product": "methyl 4-hydroxybenzoate",
                         "what_changed": "COOH to COOMe", "why_required": "activate acid",
                         "chemistry": "acid-catalyzed esterification", "purification": "crystallization",
                         "yield": "89%", "equipment": "reactor", "patent_ref": "1",
                         "scale": "industrial"},
                    ],
                    "raw_materials": ["4-hydroxybenzoic acid", "methanol"],
                    "process_units": ["Reactor"],
                    "equipment": ["reactor"],
                    "scale_summary": "industrial scale in a 1000 L reactor",
                    "notes": "none",
                }
            if "reaction" in content:
                return {
                    "reactions": [
                        {"name": "Esterification", "type": "esterification",
                         "reactants": ["4-hydroxybenzoic acid", "methanol"],
                         "products": ["methyl 4-hydroxybenzoate"],
                         "reagents": "sulfuric acid", "catalysts": "Not specified in patent",
                         "solvents": "methanol", "temperature": "65 C",
                         "pressure": "Not specified in patent", "time": "4 h",
                         "atmosphere": "Not specified in patent", "yield": "89%",
                         "workup": "recrystallization", "equipment": "round-bottom flask",
                         "source_page": 1, "source_text": "EXAMPLE 1"},
                    ]
                }
            raise ValueError(f"FakeAI: no scripted response for prompt: {content[:80]}")

    return FakeAI()


@pytest.fixture()
def seed_job(db):
    """Factory that inserts a complete job row with compounds/reactions/stages."""

    def _make(job_id: int | None = None, status: str = "complete") -> Job:
        job = Job(
            id=job_id, filename="sample.pdf", original_filename="sample.pdf",
            status=status, progress=1.0, current_stage="Analysis complete",
            patent_title="Test Patent", patent_number="US 1,000,000 B2",
            assignee="Acme", inventors="A. Person", filing_date="Jan 1, 2020",
            page_count=3, file_size_bytes=1234,
            extra={"qc_score": 0.95, "scale_summary": "lab scale",
                   "manufacturing_notes": "notes", "process_units": ["Reactor"]},
        )
        db.add(job)
        db.flush()

        db.add_all([
            Section(job_id=job.id, kind="abstract", page_start=1, page_end=1),
            Section(job_id=job.id, kind="examples", page_start=2, page_end=2),
            Section(job_id=job.id, kind="process", page_start=3, page_end=3),
        ])
        db.add_all([
            Page(job_id=job.id, page_no=p, text=f"patent text page {p}") for p in (1, 2, 3)
        ])

        c1 = Compound(
            job_id=job.id, cid="C1", name="methyl 4-hydroxybenzoate", role="product",
            smiles="COC(=O)c1ccc(O)cc1", iupac_name="methyl 4-hydroxybenzoate",
            molecular_formula="C8H8O3", molecular_weight=152.15, cas_number="99-76-3",
            confidence=0.9, source_basis="database",
            extra={"db_conflicts": [], "providers": ["opsin", "pubchem"]},
        )
        c2 = Compound(
            job_id=job.id, cid="C2", name="4-hydroxybenzoic acid", role="reactant",
            smiles="O=C(O)c1ccc(O)cc1", iupac_name="4-hydroxybenzoic acid",
            molecular_formula="C7H6O3", molecular_weight=138.12, cas_number="99-96-7",
            confidence=0.9, source_basis="database", extra={},
        )
        db.add_all([c1, c2])
        db.flush()

        rxn = Reaction(
            job_id=job.id, rid="R1", name="Esterification", type="esterification",
            reactants_text="4-hydroxybenzoic acid; methanol", products_text="methyl 4-hydroxybenzoate",
            reagents="sulfuric acid", solvents="methanol", temperature="65 C", time="4 h",
            yield_pct="89%", source_page=2, source_text="EXAMPLE 1",
            what_changed='{"reaction_types": ["esterification"], "confidence": 0.9, '
                         '"before_smiles": "O=C(O)c1ccc(O)cc1", "after_smiles": "COC(=O)c1ccc(O)cc1", '
                         '"bond_formed": "O-C", "bond_broken": "O-H", "similarity": 0.95, '
                         '"formula_before": "C7H6O3", "formula_after": "C8H8O3", "basis": "rdkit"}',
            confidence=0.9, extra={},
        )
        db.add(rxn)
        db.flush()
        db.execute(
            __import__("backend.models", fromlist=["compound_reaction"]).compound_reaction
            .insert().values(reaction_id=rxn.id, compound_id=c1.id, role="product")
        )
        db.execute(
            __import__("backend.models", fromlist=["compound_reaction"]).compound_reaction
            .insert().values(reaction_id=rxn.id, compound_id=c2.id, role="reactant")
        )

        db.add_all([
            Stage(job_id=job.id, order_idx=1, title="Esterification",
                  purpose="form the ester", starting_material="4-hydroxybenzoic acid",
                  reagents="methanol, sulfuric acid", conditions="65 C, 4 h",
                  reaction="esterification", product="methyl 4-hydroxybenzoate",
                  what_changed="COOH to COOMe", why_required="activate",
                  chemistry="acid-catalyzed", purification="crystallization",
                  yield_value="89%", equipment="reactor", patent_ref="2", scale="industrial"),
        ])
        db.commit()
        db.refresh(job)
        return job

    return _make

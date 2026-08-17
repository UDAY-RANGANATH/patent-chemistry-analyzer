"""End-to-end pipeline smoke test — FakeAI + fake chemistry, no network.

Exercises the whole orchestration: PDF extraction, sections, RAG indexing,
entity extraction, compound resolution, reaction extraction + RDKit
what-changed, manufacturing reconstruction, QC and summary.
"""

import pytest

from backend.agents import pipeline as pipeline_mod


_LOOKUP = {
    "methyl 4-hydroxybenzoate": "COC(=O)c1ccc(O)cc1",
    "4-hydroxybenzoic acid": "O=C(O)c1ccc(O)cc1",
    "methanol": "CO",
    "sulfuric acid": "OS(=O)(=O)O",
}


@pytest.fixture()
def fake_chemistry(monkeypatch):
    from backend.services.chemistry_service import CompoundRecord, ProviderResult

    def resolve(name):
        key = name.strip().lower()
        smi = _LOOKUP.get(key)
        if not smi:
            return CompoundRecord()
        return CompoundRecord(
            canonical_smiles=smi,
            isomeric_smiles=smi,
            iupac_name=name,
            formula="Cx",
            molecular_weight=100.0,
            inchikey="FAKEKEK-FAKEKEK-1",
            source_provider="opsin",
            confidence=0.9,
            synonyms=[name],
            properties={"providers_consulted": ["opsin"]},
            resolved=True,
        )

    monkeypatch.setattr(pipeline_mod.chemistry_service, "resolve_compound", resolve)
    return resolve


def test_full_pipeline_runs_end_to_end(db, fake_ai, fake_chemistry, monkeypatch):
    from backend.models import Job
    from backend.services.pdf_service import extract_pdf

    pdf = pytest._pca_sample_path
    monkeypatch.setattr(pipeline_mod, "render_structure", lambda smi, path: True)

    job = Job(filename="sample_patent.pdf", original_filename="sample_patent.pdf", status="uploaded")
    db.add(job)
    db.commit()
    db.refresh(job)

    progress = []
    pipeline_mod.run_pipeline(db, job, pdf, progress=lambda s, p, d="": progress.append((s, p)),
                              ai=fake_ai)

    db.refresh(job)
    assert job.status == "complete"
    assert job.progress == 1.0
    assert len(progress) > 5

    assert len(job.compounds) == 4
    by_name = {c.name.lower(): c for c in job.compounds}
    assert by_name["methyl 4-hydroxybenzoate"].smiles == "COC(=O)c1ccc(O)cc1"
    assert by_name["methyl 4-hydroxybenzoate"].structure_valid
    assert by_name["4-hydroxybenzoic acid"].smiles == "O=C(O)c1ccc(O)cc1"
    assert any(c.image_path for c in job.compounds)

    assert len(job.reactions) >= 1
    rxn = job.reactions[0]
    assert rxn.type == "esterification"
    assert rxn.source_page == 1
    assert rxn.what_changed  # RDKit diff serialized
    import json
    view = json.loads(rxn.what_changed)
    assert view["before_smiles"] and view["after_smiles"]
    assert view["formula_before"] != view["formula_after"]

    assert len(job.stages) == 1
    stage = job.stages[0]
    assert stage.title == "Esterification"
    assert stage.starting_material == "4-hydroxybenzoic acid"

    assert (job.extra or {}).get("qc_score") == 1.0
    assert "1000 L reactor" in (job.extra or {}).get("scale_summary", "")
    assert job.patent_title.startswith("PROCESS FOR THE PREPARATION")
    assert job.patent_number == "US 12,345,678 B2"

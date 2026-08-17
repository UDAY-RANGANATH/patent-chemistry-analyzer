"""API smoke tests via FastAPI TestClient against a seeded, in-memory job."""

from fastapi.testclient import TestClient
import pytest

import backend.api as api_module


@pytest.fixture(scope="module")
def client():
    with TestClient(api_module.app) as c:
        yield c


def _upload_pdf(client, monkeypatch, run_called):
    # Never let the real (network) pipeline thread start.
    monkeypatch.setattr(
        api_module,
        "_run_job_in_thread",
        lambda job_id, pdf_path: run_called.append((job_id, pdf_path)),
    )
    sample = pytest._pca_sample_path
    with open(sample, "rb") as fh:
        resp = client.post("/api/upload", files={"file": ("sample_patent.pdf", fh, "application/pdf")})
    return resp


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "provider" in body
    assert "rdkit" in body


def test_upload_accepts_pdf_and_starts_thread(client, monkeypatch):
    run_called = []
    resp = _upload_pdf(client, monkeypatch, run_called)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    assert run_called and run_called[0][0] == job_id


def test_upload_rejects_non_pdf(client):
    resp = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_corrupt_pdf(client):
    resp = client.post("/api/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert resp.status_code == 400
    assert "could not be opened as a PDF" in resp.json()["detail"]


def test_seeded_job_endpoints(client, seed_job):
    job = seed_job()
    jid = job.id

    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert any(j["job_id"] == jid for j in r.json())

    r = client.get(f"/api/jobs/{jid}/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["compound_count"] == 2
    assert body["reaction_count"] == 1
    assert body["stage_count"] == 1
    assert body["confidence"] == 0.95

    r = client.get(f"/api/jobs/{jid}/compounds")
    assert r.status_code == 200
    compounds = r.json()
    assert len(compounds) == 2
    c1 = next(c for c in compounds if c["cid"] == "C1")
    assert c1["cas_number"] == "99-76-3"
    assert c1["image_url"].endswith("structures/C1.png")

    r = client.get(f"/api/jobs/{jid}/reactions")
    assert r.status_code == 200
    rxns = r.json()
    assert len(rxns) == 1
    assert rxns[0]["type"] == "esterification"
    assert rxns[0]["what_changed"]["reaction_types"] == ["esterification"]
    assert rxns[0]["what_changed"]["confidence"] == 0.9
    roles = {p["role"] for p in rxns[0]["participants"]}
    assert {"reactant", "product"} <= roles

    r = client.get(f"/api/jobs/{jid}/stages")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.get(f"/api/jobs/{jid}/flowchart")
    assert r.status_code == 200
    flow = r.json()
    assert flow["nodes"]
    assert flow["edges"]

    r = client.get(f"/api/jobs/{jid}/manufacturing")
    assert r.status_code == 200
    mf = r.json()
    assert mf["stages"][0]["title"] == "Esterification"
    assert mf["scale_summary"] == "lab scale"
    assert mf["process_units"] == ["Reactor"]

    r = client.get(f"/api/jobs/{jid}/sources/2")
    assert r.status_code == 200
    assert r.json()["text"] == "patent text page 2"

    r = client.get(f"/api/jobs/{jid}/sources/99")
    assert r.status_code == 404

    r = client.get(f"/api/jobs/{jid}/structures/C1.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"

    r = client.get(f"/api/jobs/{jid}/structures/C9.png")
    assert r.status_code == 404

    r = client.get(f"/api/jobs/{jid}/report")
    assert r.status_code == 200
    assert r.json()["filename"].endswith(".docx")

    r = client.get(f"/api/jobs/{jid}/report/download")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"  # zip (docx) magic


def test_unknown_job_404(client):
    assert client.get("/api/jobs/9999/summary").status_code == 404
    assert client.get("/api/jobs/9999/compounds").status_code == 404
    assert client.get("/api/jobs/9999/reactions").status_code == 404


def test_report_requires_complete(client, db):
    from backend.models import Job

    job = Job(filename="x.pdf", original_filename="x.pdf", status="processing", progress=0.5)
    db.add(job)
    db.commit()
    resp = client.get(f"/api/jobs/{job.id}/report")
    assert resp.status_code == 400


def test_orphaned_job_recovery(db):
    from backend.models import Job

    for status in ("processing", "uploaded"):
        job = Job(filename=f"{status}.pdf", original_filename=f"{status}.pdf", status=status)
        db.add(job)
    db.commit()

    api_module._recover_orphaned_jobs()

    db.expire_all()
    for job in db.query(Job).filter(Job.status == "failed").all():
        assert "interrupted" in job.error
    assert db.query(Job).filter(Job.status.in_(["processing", "uploaded"])).count() == 0

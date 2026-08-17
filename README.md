# Patent Chemistry Analyzer

Extracts a complete chemical manufacturing picture from a patent PDF:

- **Compounds** — chemical entities with validated names, CAS numbers, SMILES,
  structures (rendered as PNG), and roles (product / reactant / solvent / catalyst).
- **Reactions** — experimental conditions (temperature, pressure, time, yield,
  workup, equipment) with RDKit-based structural-diff reasoning.
- **Manufacturing process** — a reconstructed stage-by-stage industrial route
  with scale summary, process units, and equipment.
- **Quality control** — hallucination checks against the source text with a
  confidence score.
- **Report** — a formatted `.docx` (cover, TOC, overview, compound tables,
  reactions, manufacturing, QC) plus a served flowchart and page-scan view.

The pipeline runs on Groq (default) with automatic fallback to OpenAI, Google,
then a local Ollama model, so nothing is required beyond an API key.

## Architecture

```
frontend/index.html   static single-file web UI (no build step)
backend/api           FastAPI app (upload, jobs, report, recovery)
backend/agents        pipeline, entity/reaction/manufacturing/qc agents
backend/services      pdf (PyMuPDF/pdfplumber/pytesseract), chemistry
                      (OPSIN, PubChem, NIH CIR, ChEBI, RDKit), report, ai
storage/              uploads, SQLite DB (app.db), page scans, structures, reports
```

## Prerequisites

- Python 3.12
- Heavy scientific packages (`rdkit`, `opencv-python-headless`, `numpy`) are
  expected in the **base interpreter**; the venv is created with
  `--system-site-packages` so `requirements.txt` stays minimal.

## Quick start (local)

```powershell
cd patent-chemistry-analyzer

# 1. Create the venv (assumes rdkit/opencv are in the base python)
py -3.12 -m venv --system-site-packages backend\.venv
backend\.venv\Scripts\python -m pip install -r requirements.txt

# 2. Configure
Copy-Item .env.example .env
#    then set GROQ_API_KEY=... in .env (or leave it for local Ollama)

# 3. Run
backend\.venv\Scripts\python -m uvicorn backend.api:app
#    open http://localhost:8000
```

### Tests

```powershell
backend\.venv\Scripts\python -m pytest
```

The suite (39 tests) runs fully offline with a scripted fake AI and a bundled
sample patent — no network or API key required.

## Usage

1. Upload a patent PDF in the web UI (PDFs only, ≤ 200 MB).
2. Watch progress in real time (5 pages ≈ 40 s on Groq).
3. Review Compounds / Reactions / Manufacturing / QC tabs.
4. Download the report, view the flowchart, or page-scan any section.

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a PDF (multipart `file`); starts the pipeline |
| `GET` | `/api/jobs` | List jobs |
| `GET` | `/api/jobs/{id}` | Job status / progress |
| `GET` | `/api/jobs/{id}/summary` | Aggregated analysis summary |
| `GET` | `/api/jobs/{id}/compounds` | Validated compounds |
| `GET` | `/api/jobs/{id}/reactions` | Reactions with conditions |
| `GET` | `/api/jobs/{id}/stages` | Manufacturing stages |
| `GET` | `/api/jobs/{id}/flowchart` | SVG route diagram |
| `GET` | `/api/jobs/{id}/manufacturing` | Scale / units / notes |
| `GET` | `/api/jobs/{id}/sources/{page}` | Raw extracted page text |
| `GET` | `/api/jobs/{id}/structures/{cid}.png` | Structure image |
| `GET` | `/api/jobs/{id}/report` | Report metadata |
| `GET` | `/api/jobs/{id}/report/download` | `.docx` download |
| `GET` | `/api/health` | Health check |

All JSON endpoints return `{"error": ...}` with proper status codes on failure.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Used first if set |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `OPENAI_API_KEY` | *(empty)* | Second in provider chain |
| `GOOGLE_API_KEY` | *(empty)* | Third |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local fallback provider |
| `TESSERACT_PATH` / `TESSERACT_LANG` | `/usr/bin/tesseract` / `eng` | OCR for scanned patents |
| `MAX_PAGES` / `MAX_UPLOAD_MB` | `150` / `200` | Safety limits |

Provider auto-detection order: **GROQ → OPENAI → GOOGLE → OLLAMA**.

## Docker

Dockerfiles are provided but the image has **not been built/verified** on this
machine (Docker is not installed here).

```bash
cp .env.example .env   # set GROQ_API_KEY
docker compose up --build
# app at http://localhost:8000
```

The compose file also starts an optional local Ollama instance (for an
on-prem-only run, leave `GROQ_API_KEY` empty and pull a model once):

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

Storage (uploads, SQLite DB, reports) persists in the `pca_storage` volume.

## Known limitations

- The bundled sample patent is genuinely sparse (no explicit temps/times in
  text), so conditions legitimately report "Not specified in patent".
- Scanned PDFs require Tesseract installed and configured.
- The pipeline runs in a server thread; killing the server leaves jobs marked
  `failed` on next startup (recovered automatically). A task queue is a future
  upgrade for restart-mid-run support.

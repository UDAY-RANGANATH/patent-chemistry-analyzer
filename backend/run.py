"""Dev entrypoint: uvicorn patent_analyzer_backend.api:app --reload"""

from backend.api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

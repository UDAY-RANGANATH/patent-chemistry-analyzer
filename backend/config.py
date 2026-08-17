"""Application configuration loaded from environment / .env file.

All secrets live server-side only. Never import this from the frontend.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- AI providers ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GOOGLE_API_KEY: str = ""
    GOOGLE_MODEL: str = "gemini-2.0-flash"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # --- Chemistry APIs ---
    PUBCHEM_API_URL: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    OPSIN_URL: str = "https://opsin.ch.cam.ac.uk/opsin"
    NIH_CIR_URL: str = "https://cactus.nci.nih.gov/chemical/structure"
    CHEBI_API_URL: str = "https://www.ebi.ac.uk/chebi/api"
    CHEMSPIDER_API_KEY: str = ""
    NIST_WEBBOOK_URL: str = "https://webbook.nist.gov/cgi/cbook.cgi"

    # --- OCR ---
    TESSERACT_PATH: str = ""
    TESSERACT_LANG: str = "eng"

    # --- Storage ---
    DATABASE_URL: str = ""
    UPLOAD_DIR: Path = PROJECT_DIR / "storage" / "uploads"
    PAGE_IMAGE_DIR: Path = PROJECT_DIR / "storage" / "pages"
    STRUCTURE_DIR: Path = PROJECT_DIR / "storage" / "structures"
    REPORT_DIR: Path = PROJECT_DIR / "storage" / "reports"
    CACHE_DIR: Path = PROJECT_DIR / "storage" / "cache"

    # --- Limits ---
    MAX_PAGES: int = 150
    MAX_UPLOAD_MB: int = 200
    CORS_ORIGINS: str = "http://localhost:5173"

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"sqlite:///{(PROJECT_DIR / 'storage' / 'app.db').as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def active_ai_provider(self) -> str:
        if self.GROQ_API_KEY:
            return "groq"
        if self.OPENAI_API_KEY:
            return "openai"
        if self.GOOGLE_API_KEY:
            return "google"
        return "ollama"

    def ensure_dirs(self) -> None:
        for d in (self.UPLOAD_DIR, self.PAGE_IMAGE_DIR, self.STRUCTURE_DIR,
                  self.REPORT_DIR, self.CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

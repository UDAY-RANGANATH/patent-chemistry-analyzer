"""ChemistryService — modular external chemistry API layer.

The AI layer and the rest of the app MUST call this service for any chemical
fact (structure, formula, weight, names). The LLM is never allowed to invent
these; if a fact cannot be confirmed here it is reported as
"Not specified in patent" / "unresolved".

Providers (each is an independent class registered in a registry so new
services — CAS, Reaxys, SciFinder, NIST — can be added later):

  OPSIN    : IUPAC/chemical name -> SMILES  (deterministic, offline-friendly)
  PubChem  : name/SMILES/InChIKey -> full record (PUG REST)
  NIH CIR  : name resolution fallback
  ChEBI    : ontology / biological role enrichment

Conventions:
  - Every provider returns a `ProviderResult` with a confidence score.
  - When providers disagree, a "database conflict" is recorded and BOTH
    values are surfaced (never silently pick one).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.config import settings
from backend.utils import HttpClient, rate_limited, retry

logger = logging.getLogger("pca.chemistry")

_NOT_FOUND = "not found"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ProviderResult:
    provider: str
    smiles: str | None = None
    iupac_name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    inchi: str | None = None
    inchikey: str | None = None
    cas_number: str | None = None
    synonyms: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundRecord:
    """Merged, validated record for one chemical entity."""

    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    iupac_name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    inchi: str | None = None
    inchikey: str | None = None
    cas_number: str | None = None
    synonyms: list[str] = field(default_factory=list)
    source_provider: str = ""
    confidence: float = 0.0
    conflicts: list[dict[str, Any]] = field(default_factory=list)  # database conflict list
    properties: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


# --------------------------------------------------------------------------- #
# Provider base + registry
# --------------------------------------------------------------------------- #
class ChemistryProvider:
    name = "base"

    def resolve_name(self, name: str) -> ProviderResult:  # pragma: no cover - interface
        raise NotImplementedError


class OPSINProvider(ChemistryProvider):
    """IUPAC/chemical name -> SMILES via the OPSIN web service."""

    name = "opsin"

    def __init__(self) -> None:
        self._http = HttpClient(timeout=12.0)

    @rate_limited(2.0)
    @retry(max_attempts=2, base_delay=0.5)
    def resolve_name(self, name: str) -> ProviderResult:
        clean = _clean_name(name)
        if not clean:
            return ProviderResult(provider=self.name, error=_NOT_FOUND)
        try:
            url = f"{settings.OPSIN_URL}/convert/SMILES"
            resp = self._http._client.get(url, params={"q": clean})
            if resp.status_code == 200 and resp.text.strip():
                smi = resp.text.strip()
                return ProviderResult(
                    provider=self.name,
                    smiles=smi,
                    confidence=0.95,
                    extra={"name": clean},
                )
            return ProviderResult(provider=self.name, error=_NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc))

    def close(self) -> None:
        self._http.close()


class PubChemProvider(ChemistryProvider):
    """PubChem PUG REST: name/SMILES/InChIKey -> full record."""

    name = "pubchem"

    def __init__(self) -> None:
        self._http = HttpClient(timeout=12.0)

    @rate_limited(2.0)
    @retry(max_attempts=2, base_delay=0.5)
    def resolve_name(self, name: str) -> ProviderResult:
        clean = _clean_name(name)
        if not clean:
            return ProviderResult(provider=self.name, error=_NOT_FOUND)
        try:
            base = settings.PUBCHEM_API_URL
            url = f"{base}/compound/name/{_urlquote(clean)}/JSON"
            resp = self._http._client.get(url, params={"MaxRecords": "1"})
            if resp.status_code == 404:
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            resp.raise_for_status()
            data = resp.json()
            return self._record_from_pubchem(data)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc))

    def resolve_inchikey(self, inchikey: str) -> ProviderResult:
        try:
            base = settings.PUBCHEM_API_URL
            url = f"{base}/compound/inchikey/{inchikey}/JSON"
            resp = self._http._client.get(url)
            if resp.status_code == 404:
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            resp.raise_for_status()
            return self._record_from_pubchem(resp.json())
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc))

    def _synonyms(self, identifier: str) -> list[str]:
        """Fetch synonym list (contains CAS-like numbers too)."""
        try:
            base = settings.PUBCHEM_API_URL
            url = f"{base}/compound/name/{_urlquote(identifier)}/synonyms/JSON"
            resp = self._http._client.get(url, params={"MaxRecords": "1"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _record_from_pubchem(data: dict) -> ProviderResult:
        pc = data.get("PC_Compounds", [{}])[0]
        props = pc.get("props", [])
        fields: dict[str, Any] = {}
        for prop in props:
            urn = prop.get("urn", {})
            name = urn.get("name") or ""
            label = urn.get("label") or ""
            key = f"{name}|{label}"
            value = prop.get("value", {})
            if "sval" in value:
                fields[key] = value["sval"]
            elif "fval" in value:
                fields[key] = float(value["fval"])
            elif "ival" in value:
                fields[key] = int(value["ival"])

        smiles = fields.get("Absolute|SMILES") or fields.get("Connectivity|SMILES")
        iupac = (fields.get("Preferred|IUPAC Name") or fields.get("Systematic|IUPAC Name")
                 or fields.get("Traditional|IUPAC Name") or fields.get("Markup|IUPAC Name"))
        inchi = fields.get("Standard|InChI")
        inchikey = fields.get("Standard|InChIKey")

        return ProviderResult(
            provider="pubchem",
            smiles=smiles,
            iupac_name=iupac,
            formula=fields.get("|Molecular Formula"),
            molecular_weight=fields.get("|Molecular Weight"),
            inchi=inchi,
            inchikey=inchikey,
            confidence=0.9,
            extra={"cid": pc.get("id", {}).get("id", {}).get("cid")},
        )

    def close(self) -> None:
        self._http.close()


class NIHCIRProvider(ChemistryProvider):
    """NIH CIR (Cactus) — name/SMILES resolution fallback."""

    name = "nih_cir"

    def __init__(self) -> None:
        self._http = HttpClient(timeout=12.0)

    @rate_limited(2.0)
    @retry(max_attempts=1, base_delay=0.5)
    def resolve_name(self, name: str) -> ProviderResult:
        clean = _clean_name(name)
        if not clean:
            return ProviderResult(provider=self.name, error=_NOT_FOUND)
        try:
            url = f"{settings.NIH_CIR_URL}/{_urlquote(clean)}/smiles"
            resp = self._http._client.get(url)
            if resp.status_code == 404:
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            body = resp.text or ""
            # CIR sometimes returns 200 with an HTML error body.
            if not body.strip() or "<" in body[:80] or "not found" in body.lower():
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            smi = body.strip().splitlines()[0]
            if not _looks_like_smiles(smi):
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            return ProviderResult(provider=self.name, smiles=smi, confidence=0.7)
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc))

    def close(self) -> None:
        self._http.close()


class ChEBIProvider(ChemistryProvider):
    """ChEBI ontology enrichment (biological role / classification)."""

    name = "chebi"

    def __init__(self) -> None:
        self._http = HttpClient(timeout=25.0)

    @rate_limited(2.0)
    def resolve_name(self, name: str) -> ProviderResult:
        clean = _clean_name(name)
        if not clean:
            return ProviderResult(provider=self.name, error=_NOT_FOUND)
        try:
            url = f"{settings.CHEBI_API_URL}/search"
            resp = self._http._client.get(
                url,
                params={"query": clean, "maximumRecords": "1", "format": "json"},
            )
            if resp.status_code != 200:
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            data = resp.json()
            entries = data.get("list", {}).get("searchElement", [])
            if not entries:
                return ProviderResult(provider=self.name, error=_NOT_FOUND)
            first = entries[0]
            return ProviderResult(
                provider=self.name,
                iupac_name=first.get("chebiAsciiName"),
                confidence=0.5,
                extra={"chebi_id": first.get("chebiId"), "name": first.get("chebiName")},
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc))

    def close(self) -> None:
        self._http.close()


_PROVIDERS: dict[str, ChemistryProvider] = {}
_REGISTERED: dict[str, Callable[[], ChemistryProvider]] = {
    "opsin": OPSINProvider,
    "pubchem": PubChemProvider,
    "nih_cir": NIHCIRProvider,
    "chebi": ChEBIProvider,
}


def register_provider(name: str, factory: Callable[[], ChemistryProvider]) -> None:
    """Public hook to add new providers (CAS, Reaxys, SciFinder, NIST...)."""
    _REGISTERED[name] = factory
    _PROVIDERS.pop(name, None)


def get_provider(name: str) -> ChemistryProvider:
    if name not in _PROVIDERS:
        _PROVIDERS[name] = _REGISTERED[name]()
    return _PROVIDERS[name]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class ChemistryService:
    """Facade: resolves names via multiple providers and merges records."""

    PROVIDER_ORDER = ["opsin", "pubchem", "nih_cir"]

    def resolve_compound(self, name: str) -> CompoundRecord:
        """Resolve a chemical name to a validated record across providers.

        Returns an unresolved record (resolved=False) when nothing is found.
        Records conflicts when providers disagree.
        """
        import time as _time

        record = CompoundRecord()
        if not _clean_name(name):
            return record
        deadline = _time.monotonic() + 35.0  # hard budget per compound

        results: list[ProviderResult] = []
        for prov_name in self.PROVIDER_ORDER:
            if _time.monotonic() > deadline:
                break
            try:
                res = get_provider(prov_name).resolve_name(name)
            except Exception as exc:  # noqa: BLE001
                res = ProviderResult(provider=prov_name, error=str(exc))
            if res.error:
                continue
            results.append(res)

        if not results:
            return record

        # Best record: highest confidence with a SMILES.
        results.sort(key=lambda r: (r.smiles is not None, r.confidence), reverse=True)
        primary = results[0]

        record.canonical_smiles = primary.smiles
        record.isomeric_smiles = primary.smiles
        record.iupac_name = primary.iupac_name
        record.formula = primary.formula
        record.molecular_weight = primary.molecular_weight
        record.inchi = primary.inchi
        record.inchikey = primary.inchikey
        record.cas_number = primary.cas_number
        record.synonyms = list(primary.synonyms)
        record.source_provider = primary.provider
        record.confidence = primary.confidence
        record.resolved = primary.smiles is not None or primary.inchikey is not None

        # Fetch PubChem synonyms/CAS only when CAS is still missing (1 extra call).
        pubchem_prov = None
        for r_ in results:
            if r_.provider == "pubchem":
                pubchem_prov = r_
                break
        if (pubchem_prov and not record.cas_number and record.resolved
                and _time.monotonic() < deadline):
            try:
                syns = get_provider("pubchem")._synonyms(name)
                if syns:
                    record.synonyms = syns[:15]
                    cas = next((s for s in syns if re.fullmatch(r"\d{2,7}-\d{2}-\d", s)), None)
                    if cas:
                        record.cas_number = cas
            except Exception:  # noqa: BLE001
                pass

        # Complete gaps from other providers (e.g. PubChem fills formula/MW).
        for res in results[1:]:
            if res.smiles and not record.canonical_smiles:
                record.canonical_smiles = res.smiles
                record.isomeric_smiles = res.smiles
            for attr in ("iupac_name", "formula", "molecular_weight", "inchi", "inchikey", "cas_number"):
                val = getattr(res, attr)
                if val and not getattr(record, attr):
                    setattr(record, attr, val)
            if res.smiles and res.smiles != record.canonical_smiles:
                canon_a = _canonical_smiles(record.canonical_smiles or "")
                canon_b = _canonical_smiles(res.smiles)
                if not (canon_a and canon_b and canon_a == canon_b):
                    record.conflicts.append({
                        "field": "SMILES",
                        "provider_a": record.source_provider,
                        "value_a": record.canonical_smiles,
                        "provider_b": res.provider,
                        "value_b": res.smiles,
                    })

        # Formula/MW cross-check between providers.
        formulas = {r.formula for r in results if r.formula}
        if len(formulas) > 1:
            record.conflicts.append({
                "field": "MolecularFormula",
                "values": sorted(formulas),
            })

        record.properties = {"providers_consulted": [r.provider for r in results]}
        return record

    def get_properties(self, name: str) -> dict[str, Any]:
        """Friendly property bag for UI display."""
        rec = self.resolve_compound(name)
        if not rec.resolved:
            return {"resolved": False}
        return {
            "resolved": True,
            "smiles": rec.canonical_smiles,
            "iupac": rec.iupac_name,
            "formula": rec.formula,
            "mw": rec.molecular_weight,
            "inchi": rec.inchi,
            "inchikey": rec.inchikey,
            "cas": rec.cas_number,
            "synonyms": rec.synonyms[:10],
            "provider": rec.source_provider,
            "confidence": rec.confidence,
            "conflicts": rec.conflicts,
        }


chemistry_service = ChemistryService()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean_name(name: str) -> str:
    """Strip patent-noise from a name before querying chemistry APIs."""
    text = name.strip()
    text = re.sub(r"\(compounds?\)\s*$", "", text)
    text = re.sub(r"\s+(or its (salt|solvate)|or a salt thereof)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[;,.:\[\](){}'\"]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


_SMILES_PATTERN = re.compile(r"^[A-Za-z0-9@+\-\[\]()\\/#=.%$,]+$")


def _looks_like_smiles(value: str) -> bool:
    return bool(value) and _SMILES_PATTERN.match(value) and len(value) < 2048


def _canonical_smiles(value: str) -> str | None:
    """RDKit-canonicalized SMILES; returns None when unparseable."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(value)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        return None


def _urlquote(value: str) -> str:
    import urllib.parse

    return urllib.parse.quote(value)

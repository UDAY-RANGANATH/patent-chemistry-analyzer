"""ChemistryService tests — fully offline via registered fake providers."""

import pytest

from backend.services.chemistry_service import (
    ChemistryService,
    CompoundRecord,
    ProviderResult,
    _clean_name,
    register_provider,
)


@pytest.fixture(autouse=True)
def _fake_providers(monkeypatch):
    """Register deterministic fake providers for the whole test module."""

    class FakeOPSIN:
        name = "opsin"

        def resolve_name(self, name):
            if name.lower() == "methanol":
                return ProviderResult(provider="opsin", smiles="CO", confidence=0.95)
            return ProviderResult(provider="opsin", error="not found")

    class FakePubChem:
        name = "pubchem"

        def resolve_name(self, name):
            if name.lower() == "methanol":
                return ProviderResult(
                    provider="pubchem", smiles="CO", iupac_name="methanol",
                    formula="CH4O", molecular_weight=32.04, inchikey="OKKJLVBELUTLKV-UHFFFAOYSA-N",
                    cas_number="67-56-1", confidence=0.9,
                )
            return ProviderResult(provider="pubchem", error="not found")

    class FakeCIR:
        name = "nih_cir"

        def resolve_name(self, name):
            # Deliberately disagreeing provider to exercise conflict detection.
            if name.lower() == "methanol":
                return ProviderResult(provider="nih_cir", smiles="O", confidence=0.7)
            return ProviderResult(provider="nih_cir", error="not found")

    register_provider("opsin", lambda: FakeOPSIN())
    register_provider("pubchem", lambda: FakePubChem())
    register_provider("nih_cir", lambda: FakeCIR())
    yield


def test_resolve_known_compound_merges_fields():
    rec = ChemistryService().resolve_compound("methanol")
    assert rec.resolved
    assert rec.canonical_smiles == "CO"
    assert rec.formula == "CH4O"
    assert rec.molecular_weight == 32.04
    assert rec.cas_number == "67-56-1"
    assert "opsin" in rec.properties["providers_consulted"]


def test_resolve_unknown_returns_unresolved():
    rec = ChemistryService().resolve_compound("fictional-chemical-xyz")
    assert not rec.resolved
    assert rec.canonical_smiles is None


def test_provider_conflict_recorded():
    rec = ChemistryService().resolve_compound("methanol")
    smiles_conflicts = [c for c in rec.conflicts if c["field"] == "SMILES"]
    assert smiles_conflicts


def test_clean_name_strips_noise():
    assert _clean_name("  methyl benzoate (compound) ") == "methyl benzoate"
    assert _clean_name("acid or its salt") == "acid"
    assert _clean_name("Acid.;[]") == "Acid"
    assert _clean_name("  spaced   text  ") == "spaced text"

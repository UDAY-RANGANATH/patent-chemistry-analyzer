"""Agent tests against a scripted FakeAI — including LLM schema-drift robustness."""

import pytest

from backend.agents.entity_agent import ChemicalEntityAgent
from backend.agents.manufacturing_agent import ManufacturingAgent
from backend.agents.reaction_agent import ReactionAnalysisAgent


def test_entity_agent_parses_scripted_output(fake_ai):
    agent = ChemicalEntityAgent(fake_ai)
    mentions = agent.extract_from_chunk("[Patent page 1]\nprepared methyl 4-hydroxybenzoate")
    assert mentions
    ester = next(m for m in mentions if "methyl 4-hydroxybenzoate" in m.name)
    assert ester.role == "product"
    assert ester.source_page == 1


def test_entity_agent_handles_junk_fields(fake_ai):
    from backend.services.ai_service import AIService

    class JunkAI(AIService):
        def chat_json(self, messages, **kwargs):
            return {
                "compounds": [
                    {"name": ["methyl 4-hydroxybenzoate", "paraben"], "role": {"role": "product"},
                     "source_page": "not-a-number", "source_text": {"line": "text"}},
                    {"name": "", "role": "solvent", "source_page": 1, "source_text": "x"},
                    {"name": 12345, "role": "compound", "source_page": 2, "source_text": "y"},
                ]
            }

    agent = ChemicalEntityAgent(JunkAI())
    mentions = agent.extract_from_chunk("[Patent page 1]")
    assert len(mentions) == 2
    assert "methyl 4-hydroxybenzoate" in mentions[0].name  # dict/list coerced to string
    assert mentions[0].source_page is None  # junk int silently dropped
    assert mentions[1].name == "12345"  # numeric name coerced


def test_entity_agent_empty_result():
    from backend.services.ai_service import AIService

    class EmptyAI(AIService):
        def chat_json(self, messages, **kwargs):
            return {"compounds": []}

    agent = ChemicalEntityAgent(EmptyAI())
    assert agent.extract_from_chunk("[Patent page 1]") == []


def test_entity_agent_invalid_json_is_graceful():
    from backend.services.ai_service import AIService

    class BadAI(AIService):
        def chat_json(self, messages, **kwargs):
            raise ValueError("bad json")

    agent = ChemicalEntityAgent(BadAI())
    assert agent.extract_from_chunk("x") == []


def test_reaction_agent_parses_and_type_is_single_value(fake_ai):
    agent = ReactionAnalysisAgent(fake_ai)
    rxns = agent.extract_from_chunk("[Patent page 1]\nEXAMPLE 1")
    assert len(rxns) == 1
    r = rxns[0]
    assert r.type == "esterification"
    assert "4-hydroxybenzoic acid" in r.reactants
    assert r.temperature == "65 C"
    assert r.yield_pct == "89%"
    assert r.source_page == 1


def test_reaction_agent_schema_drift(fake_ai):
    from backend.services.ai_service import AIService

    class DriftAI(AIService):
        def chat_json(self, messages, **kwargs):
            return {
                "reactions": [
                    {"name": ["Esterification"], "type": ["esterification", "condensation"],
                     "reactants": "4-hydroxybenzoic acid; methanol", "products": {"p": "methyl 4-hydroxybenzoate"},
                     "reagents": None, "catalysts": 5, "solvents": ["methanol", "toluene"],
                     "temperature": "65 C", "time": "4 h", "yield": {"v": "89%"},
                     "source_page": "1", "source_text": "text"},
                ]
            }

    agent = ReactionAnalysisAgent(DriftAI())
    r = agent.extract_from_chunk("[Patent page 1]")[0]
    assert "esterification" in r.type
    assert "4-hydroxybenzoic acid" in r.reactants  # semicolon string split? No — joined list tolerated
    assert r.solvents == "methanol; toluene"
    assert r.reagents == "Not specified in patent"
    assert r.yield_pct == "v: 89%"
    assert r.source_page == 1


def test_manufacturing_agent_parses(fake_ai):
    agent = ManufacturingAgent(fake_ai)
    result = agent.extract_from_chunk("[Patent page 3]\nINDUSTRIAL MANUFACTURING PROCESS")
    assert len(result.stages) == 1
    stage = result.stages[0]
    assert stage["title"].startswith("Stage 1")
    assert stage["starting_material"] == "4-hydroxybenzoic acid"
    assert "1000 L reactor" in result.scale_summary
    assert result.process_units == ["Reactor"]


def test_manufacturing_agent_strip_stage_number():
    from backend.services.ai_service import AIService

    class StageAI(AIService):
        def chat_json(self, messages, **kwargs):
            return {"stages": [{"title": "Esterification", "order": 1}],
                    "raw_materials": "acid; alcohol", "process_units": None,
                    "equipment": ["R1"], "scale_summary": "", "notes": ""}

    result = ManufacturingAgent(StageAI()).extract_from_chunk("x")
    assert result.stages[0]["title"] == "Esterification"
    assert result.raw_materials == ["acid", "alcohol"]
    assert result.process_units == []
    assert result.scale_summary == "Not specified in patent"

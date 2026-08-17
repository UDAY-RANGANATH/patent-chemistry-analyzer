"""ManufacturingAgent — reconstructs the manufacturing process from patent text."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.agents.prompts import MANUFACTURING_SYSTEM, MANUFACTURING_USER
from backend.services.ai_service import AIService, system_message, user_message
from backend.utils import coerce_str

logger = logging.getLogger("pca.agent.manufacturing")


@dataclass
class ManufacturingExtract:
    stages: list[dict] = field(default_factory=list)
    raw_materials: list[str] = field(default_factory=list)
    process_units: list[str] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    scale_summary: str = "Not specified in patent"
    notes: str = "Not specified in patent"


class ManufacturingAgent:
    def __init__(self, ai: AIService):
        self.ai = ai

    def extract_from_chunk(self, chunk: str) -> ManufacturingExtract:
        try:
            result = self.ai.chat_json(
                [system_message(MANUFACTURING_SYSTEM),
                 user_message(MANUFACTURING_USER.format(chunk=chunk))],
                max_tokens=2000,
            )
        except ValueError as exc:
            logger.warning("Manufacturing extraction JSON failed: %s", exc)
            return ManufacturingExtract()
        stages = []
        for i, s in enumerate(result.get("stages", []) or []):
            title = coerce_str(s.get("title")).strip() or f"Stage {i + 1}"
            stages.append({
                "order": int(s.get("order") or i + 1),
                "title": title,
                "purpose": _ns(s.get("purpose")),
                "starting_material": _ns(s.get("starting_material")),
                "reagents": _ns(s.get("reagents")),
                "conditions": _ns(s.get("conditions")),
                "reaction": _ns(s.get("reaction")),
                "product": _ns(s.get("product")),
                "what_changed": _ns(s.get("what_changed")),
                "why_required": _ns(s.get("why_required")),
                "chemistry": _ns(s.get("chemistry")),
                "purification": _ns(s.get("purification")),
                "yield": _ns(s.get("yield")),
                "equipment": _ns(s.get("equipment")),
                "patent_ref": _ns(s.get("patent_ref")),
                "scale": _ns(s.get("scale")),
            })
        return ManufacturingExtract(
            stages=stages,
            raw_materials=_as_list(result.get("raw_materials")),
            process_units=_as_list(result.get("process_units")),
            equipment=_as_list(result.get("equipment")),
            scale_summary=_ns(result.get("scale_summary")),
            notes=_ns(result.get("notes")),
        )


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        value = [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    elif not isinstance(value, list):
        value = []
    return [str(x).strip() for x in value if str(x).strip()]


def _ns(value) -> str:
    v = coerce_str(value).strip()
    return v if v else "Not specified in patent"

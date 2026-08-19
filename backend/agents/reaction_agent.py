"""ReactionAnalysisAgent — extracts reactions from experimental examples."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.agents.prompts import REACTION_SYSTEM, REACTION_USER
from backend.services.ai_service import AIService, system_message, user_message
from backend.utils import coerce_str

logger = logging.getLogger("pca.agent.reaction")


@dataclass
class ReactionExtract:
    name: str = ""
    type: str = "Not specified in patent"
    reactants: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    reagents: str = "Not specified in patent"
    catalysts: str = "Not specified in patent"
    solvents: str = "Not specified in patent"
    temperature: str = "Not specified in patent"
    pressure: str = "Not specified in patent"
    time: str = "Not specified in patent"
    atmosphere: str = "Not specified in patent"
    yield_pct: str = "Not specified in patent"
    workup: str = "Not specified in patent"
    equipment: str = "Not specified in patent"
    source_page: int | None = None
    source_text: str = ""


class ReactionAnalysisAgent:
    def __init__(self, ai: AIService):
        self.ai = ai

    def extract_from_chunk(self, chunk: str) -> list[ReactionExtract]:
        try:
            result = self.ai.chat_json(
                [system_message(REACTION_SYSTEM), user_message(REACTION_USER.format(chunk=chunk))],
                max_tokens=2000,
                temperature=0.0,
            )
        except ValueError as exc:
            logger.warning("Reaction extraction JSON failed: %s", exc)
            return []
        extracts: list[ReactionExtract] = []
        for item in result.get("reactions", []):
            if isinstance(item, str):
                extracts.append(ReactionExtract(name=item.strip()[:256]))
                continue
            if not isinstance(item, dict):
                continue
            extracts.append(ReactionExtract(
                name=coerce_str(item.get("name")).strip()[:256],
                type=_ns(item.get("type")),
                reactants=_as_str_list(item.get("reactants")),
                products=_as_str_list(item.get("products")),
                reagents=_ns(item.get("reagents")),
                catalysts=_ns(item.get("catalysts")),
                solvents=_ns(item.get("solvents")),
                temperature=_ns(item.get("temperature")),
                pressure=_ns(item.get("pressure")),
                time=_ns(item.get("time")),
                atmosphere=_ns(item.get("atmosphere")),
                yield_pct=_ns(item.get("yield")),
                workup=_ns(item.get("workup")),
                equipment=_ns(item.get("equipment")),
                source_page=_coerce_int(item.get("source_page")),
                source_text=(item.get("source_text") or "")[:2000],
            ))
        return extracts


def _as_str_list(value) -> list[str]:
    """Coerce an LLM field into a clean list of strings.

    Tolerates the model emitting a bare string, a dict, or a list with junk.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    elif isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        s = coerce_str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _ns(value) -> str:
    v = coerce_str(value).strip()
    return v if v else "Not specified in patent"


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

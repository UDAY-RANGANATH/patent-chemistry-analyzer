"""ChemicalEntityAgent — LLM-driven extraction of chemical entities from patent text.

Runs per chunk (never on the whole patent at once). The AI returns candidate
names + roles + page provenance; the ChemistryService validates each one
afterwards. The agent itself never asserts structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.agents.prompts import ENTITY_SYSTEM, ENTITY_USER
from backend.services.ai_service import AIService, system_message, user_message
from backend.utils import coerce_str

logger = logging.getLogger("pca.agent.entity")


@dataclass
class EntityMention:
    name: str
    role: str = "compound"
    source_page: int | None = None
    source_text: str = ""
    context: str = ""
    mentioned_smiles: str | None = None


class ChemicalEntityAgent:
    def __init__(self, ai: AIService):
        self.ai = ai

    def extract_from_chunk(self, chunk: str) -> list[EntityMention]:
        """Extract entities from one tagged chunk."""
        try:
            result = self.ai.chat_json(
                [system_message(ENTITY_SYSTEM), user_message(ENTITY_USER.format(chunk=chunk))],
                max_tokens=1500,
                temperature=0.0,
            )
        except ValueError as exc:
            logger.warning("Entity extraction JSON failed: %s", exc)
            return []
        mentions: list[EntityMention] = []
        for item in result.get("compounds", []):
            if isinstance(item, str):
                if item.strip():
                    mentions.append(EntityMention(name=item.strip()[:300]))
                continue
            if not isinstance(item, dict):
                continue
            name = coerce_str(item.get("name")).strip()
            if not name:
                continue
            mentions.append(EntityMention(
                name=name[:300],
                role=coerce_str(item.get("role")).strip().lower(),
                source_page=_coerce_int(item.get("source_page")),
                source_text=coerce_str(item.get("source_text")).strip()[:2000],
                context=coerce_str(item.get("context")).strip()[:1000],
                mentioned_smiles=coerce_str(item.get("mentioned_smiles")).strip() or None,
            ))
        return mentions


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

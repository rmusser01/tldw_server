from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.RPG.constants import (
    RPG_ADAPTER_DND5E_SRD,
    RPG_ADAPTER_FATE,
    RPG_ADAPTER_PF2E,
    RPG_ADAPTER_VERSION_V1,
)
from tldw_Server_API.app.core.RPG.errors import RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.models import CheckResult, DiceRollResult, RuleAdapterInfo, RuleLicenseSummary


class RuleAdapter(Protocol):
    adapter_key: str
    adapter_version: str
    display_name: str
    status: str
    source_version: str
    bundled_snippet_policy: str
    license_summary: RuleLicenseSummary
    mechanics_tags: dict[str, str]

    def actor_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def check_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def supported_event_types(self) -> set[str]:
        raise NotImplementedError

    def validate_actor(self, actor_payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_check(self, roller: Any, payload: dict[str, Any]) -> CheckResult:
        raise NotImplementedError

    def content_pack_refs(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def info(self) -> RuleAdapterInfo:
        raise NotImplementedError


_CORE_EVENT_TYPES = frozenset(
    {
        "actor.upserted",
        "clock.updated",
        "faction.upserted",
        "inventory.item.upserted",
        "location.upserted",
        "note.added",
        "npc.upserted",
        "quest.upserted",
        "roll.recorded",
        "rule.reference.added",
        "ruling.added",
        "scene.updated",
    }
)


@dataclass(frozen=True, slots=True)
class StaticRuleAdapter:
    adapter_key: str
    adapter_version: str
    display_name: str
    status: str
    source_version: str
    bundled_snippet_policy: str
    license_summary: RuleLicenseSummary
    mechanics_tags: dict[str, str]
    _actor_schema: dict[str, Any]
    _check_schema: dict[str, Any]
    _check_resolver: Callable[[Any, dict[str, Any]], CheckResult]

    def actor_schema(self) -> dict[str, Any]:
        return deepcopy(self._actor_schema)

    def check_schema(self) -> dict[str, Any]:
        return deepcopy(self._check_schema)

    def supported_event_types(self) -> set[str]:
        return set(_CORE_EVENT_TYPES)

    def validate_actor(self, actor_payload: dict[str, Any]) -> dict[str, Any]:
        return dict(actor_payload)

    def resolve_check(self, roller: Any, payload: dict[str, Any]) -> CheckResult:
        return self._check_resolver(roller, payload)

    def content_pack_refs(self) -> list[dict[str, Any]]:
        return []

    def info(self) -> RuleAdapterInfo:
        return RuleAdapterInfo(
            adapter_key=self.adapter_key,
            adapter_version=self.adapter_version,
            display_name=self.display_name,
            status=self.status,
            source_version=self.source_version,
            bundled_snippet_policy=self.bundled_snippet_policy,
            license_summary=self.license_summary,
            mechanics_tags=dict(self.mechanics_tags),
        )


class RuleAdapterRegistry:
    def __init__(self, adapters: list[RuleAdapter]) -> None:
        self._adapters = {adapter.adapter_key: adapter for adapter in adapters}

    def get(self, adapter_key: str) -> RuleAdapter:
        try:
            return self._adapters[adapter_key]
        except KeyError as exc:
            raise RPGNotFoundError(f"unknown RPG rules adapter: {adapter_key}") from exc

    def list_infos(self) -> list[RuleAdapterInfo]:
        return [self._adapters[key].info() for key in sorted(self._adapters)]


def _base_actor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["actor_id", "name"],
        "properties": {
            "actor_id": {"type": "string", "minLength": 1, "maxLength": 120},
            "name": {"type": "string", "minLength": 1, "maxLength": 200},
            "kind": {"type": "string", "maxLength": 80},
            "traits": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }


def _d20_check_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["check_label"],
        "properties": {
            "check_label": {"type": "string", "minLength": 1, "maxLength": 160},
            "roll_expression": {"type": "string", "default": "1d20", "maxLength": 40},
            "dc": {"type": "integer", "minimum": 0, "maximum": 100},
            "modifier": {"type": "integer", "minimum": -100, "maximum": 100},
        },
        "additionalProperties": True,
    }


def _fate_check_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["check_label", "ladder_target"],
        "properties": {
            "check_label": {"type": "string", "minLength": 1, "maxLength": 160},
            "skill_bonus": {"type": "integer", "minimum": -8, "maximum": 12},
            "ladder_target": {"type": "integer", "minimum": -4, "maximum": 12},
        },
        "additionalProperties": True,
    }


def _required_label(payload: dict[str, Any]) -> str:
    value = payload.get("check_label")
    if not isinstance(value, str) or not value.strip():
        raise RPGValidationError("check_label is required")
    return value.strip()


def _bounded_int(payload: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if type(value) is not int:
        raise RPGValidationError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise RPGValidationError(f"{key} must be between {minimum} and {maximum}")
    return value


def _optional_bounded_int(payload: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _bounded_int(payload, key, default=0, minimum=minimum, maximum=maximum)


def _roll_with_extra_modifier(roller: Any, expression: str, modifier: int) -> DiceRollResult:
    roll = roller.roll(expression)
    if modifier == 0:
        return roll

    combined_modifier = roll.modifier + modifier
    expression_prefix = f"{roll.dice_count}d{roll.sides}"
    combined_expression = f"{expression_prefix}{combined_modifier:+d}" if combined_modifier else expression_prefix

    return DiceRollResult(
        expression=combined_expression,
        values=list(roll.values),
        modifier=combined_modifier,
        total=sum(roll.values) + combined_modifier,
        dice_count=roll.dice_count,
        sides=roll.sides,
        details={"base_expression": roll.expression, "payload_modifier": modifier},
    )


def _resolve_d20_check(roller: Any, payload: dict[str, Any]) -> CheckResult:
    check_label = _required_label(payload)
    roll_expression = payload.get("roll_expression", "1d20")
    if not isinstance(roll_expression, str):
        raise RPGValidationError("roll_expression must be a string")

    modifier = _bounded_int(payload, "modifier", default=0, minimum=-100, maximum=100)
    dc = _optional_bounded_int(payload, "dc", minimum=0, maximum=100)
    roll = _roll_with_extra_modifier(roller, roll_expression, modifier)
    margin = roll.total - dc if dc is not None else None

    return CheckResult(
        check_label=check_label,
        mechanics="d20",
        roll=roll,
        target=dc,
        success=roll.total >= dc if dc is not None else None,
        margin=margin,
        details={"roll_expression": roll_expression, "modifier": modifier},
    )


def _resolve_fate_check(roller: Any, payload: dict[str, Any]) -> CheckResult:
    check_label = _required_label(payload)
    skill_bonus = _bounded_int(payload, "skill_bonus", default=0, minimum=-8, maximum=12)
    target = _optional_bounded_int(payload, "ladder_target", minimum=-4, maximum=12)
    if target is None:
        raise RPGValidationError("ladder_target is required")

    roll = roller.roll_fate(modifier=skill_bonus)
    margin = roll.total - target

    return CheckResult(
        check_label=check_label,
        mechanics="fate",
        roll=roll,
        target=target,
        success=roll.total >= target,
        margin=margin,
        details={"skill_bonus": skill_bonus},
    )


def build_default_adapter_registry() -> RuleAdapterRegistry:
    return RuleAdapterRegistry(
        [
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_DND5E_SRD,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="D&D 5e SRD",
                status="bundled",
                source_version="SRD 5.1",
                bundled_snippet_policy="mechanics_metadata_only",
                license_summary=RuleLicenseSummary(
                    license_name="CC-BY-4.0",
                    source_title="Systems Reference Document 5.1",
                    source_url="https://dnd.wizards.com/resources/systems-reference-document",
                    attribution_required=True,
                    commercial_use_allowed=True,
                    notes="Adapter stores mechanics metadata and citations, not long-form SRD prose.",
                ),
                mechanics_tags={"resolution_family": "d20", "dice": "d20", "genre": "fantasy"},
                _actor_schema=_base_actor_schema(),
                _check_schema=_d20_check_schema(),
                _check_resolver=_resolve_d20_check,
            ),
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_FATE,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="Fate",
                status="bundled",
                source_version="Fate Core",
                bundled_snippet_policy="mechanics_metadata_only",
                license_summary=RuleLicenseSummary(
                    license_name="CC-BY-3.0",
                    source_title="Fate Core System",
                    source_url="https://fate-srd.com/",
                    attribution_required=True,
                    commercial_use_allowed=True,
                    notes="Adapter stores mechanics metadata and citations.",
                ),
                mechanics_tags={"resolution_family": "fate", "dice": "fate", "genre": "generic"},
                _actor_schema=_base_actor_schema(),
                _check_schema=_fate_check_schema(),
                _check_resolver=_resolve_fate_check,
            ),
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_PF2E,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="Pathfinder 2e",
                status="bundled",
                source_version="Remaster",
                bundled_snippet_policy="mechanics_metadata_only",
                license_summary=RuleLicenseSummary(
                    license_name="ORC",
                    source_title="Pathfinder Second Edition Remaster",
                    source_url="https://paizo.com/pathfinder",
                    attribution_required=True,
                    commercial_use_allowed=True,
                    notes="Adapter stores mechanics metadata and citations; user-provided rules packs provide prose.",
                ),
                mechanics_tags={"resolution_family": "d20", "dice": "d20", "genre": "fantasy"},
                _actor_schema=_base_actor_schema(),
                _check_schema=_d20_check_schema(),
                _check_resolver=_resolve_d20_check,
            ),
        ]
    )

"""Versioned, user-authored inputs for a VN generation batch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.prompts import build_prompt_preview

RECIPE_VERSION = 1


class _PromptSnapshot(BaseModel):
    """Validate the prompt fields consumed by generation without coercion."""

    model_config = ConfigDict(strict=True)
    prompt: str
    negative_prompt: str | None


class _AuthoredSlot(BaseModel):
    """Validate stored slot inputs before Retry or worker replay."""

    model_config = ConfigDict(strict=True)
    slot_id: int = Field(gt=0)
    slot_key: str = Field(min_length=1)
    asset_type: str = Field(min_length=1)
    labels: dict[str, Any]
    variant_count: int = Field(ge=0)
    prompt_snapshot: _PromptSnapshot
    requested_backend: str | None
    requested_model: str | None
    width: int | None = Field(gt=0)
    height: int | None = Field(gt=0)
    format: str = Field(min_length=1)
    extra_params: dict[str, Any]
    seeds: list[int | None]

    @model_validator(mode="after")
    def _validate_seed_count(self) -> _AuthoredSlot:
        """Require one recorded seed for each variant the worker will replay."""
        if len(self.seeds) != self.variant_count:
            raise ValueError("variant seed count mismatch")
        return self


class _ExecutionSlot(BaseModel):
    """Validate the pinned public backend/model selection."""

    model_config = ConfigDict(strict=True)
    slot_id: int = Field(gt=0)
    backend: str = Field(min_length=1)
    model: str | None
    local_model_mode: str | None = None
    local_model_path_sha256: str | None = None


def load_recipe(value: Any, *, pack_id: int, owner_user_id: int) -> dict[str, Any]:
    """Parse a stored recipe for the expected pack and owner.

    Args:
        value: Serialized recipe JSON, or a missing legacy value.
        pack_id: Pack that owns the batch.
        owner_user_id: User that requested the batch.

    Returns:
        The validated versioned recipe.

    Raises:
        ValueError: The recipe is absent, malformed, or belongs elsewhere.
    """
    if value is None:
        raise ValueError("vn_asset_recipe_unavailable")
    try:
        recipe = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("vn_asset_recipe_invalid") from exc
    if (
        not isinstance(recipe, dict)
        or recipe.get("version") != RECIPE_VERSION
        or recipe.get("pack_id") != pack_id
        or recipe.get("owner_user_id") != owner_user_id
        or type(recipe.get("primary_character_id")) is not int
        or recipe["primary_character_id"] <= 0
        or not isinstance(recipe.get("slots"), list)
    ):
        raise ValueError("vn_asset_recipe_invalid")
    try:
        for slot in recipe["slots"]:
            _AuthoredSlot.model_validate(slot)
    except ValidationError as exc:
        raise ValueError("vn_asset_recipe_invalid") from exc
    return recipe


def load_execution_recipe(value: Any) -> dict[str, Any]:
    """Validate a stored execution snapshot without rewriting recorded values.

    Args:
        value: Serialized execution JSON, or an absent snapshot.

    Returns:
        The validated snapshot with all recorded metadata preserved.

    Raises:
        ValueError: The snapshot is absent or cannot be faithfully replayed.
    """
    if value is None:
        raise ValueError("vn_asset_execution_recipe_unavailable")
    try:
        recipe = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("vn_asset_execution_recipe_invalid") from exc
    if (
        not isinstance(recipe, dict)
        or recipe.get("version") != RECIPE_VERSION
        or not isinstance(recipe.get("slots"), list)
    ):
        raise ValueError("vn_asset_execution_recipe_invalid")
    try:
        for slot in recipe["slots"]:
            _ExecutionSlot.model_validate(slot)
    except ValidationError as exc:
        raise ValueError("vn_asset_execution_recipe_invalid") from exc
    return recipe


def slot_recipe(recipe: Mapping[str, Any], slot_id: int) -> dict[str, Any]:
    """Return the recorded slot recipe or reject a slot absent from the batch."""
    for slot in recipe["slots"]:
        if isinstance(slot, dict) and slot.get("slot_id") == slot_id:
            return slot
    raise ValueError("vn_asset_recipe_slot_mismatch")


def build_authored_recipe(
    repo: VNAssetPacksRepository,
    pack: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    *,
    owner_user_id: int,
    variant_count: int | None,
) -> dict[str, Any]:
    """Freeze generation inputs and rendered prompts before accepting a batch.

    Args:
        repo: Repository used to read the character and configured world books.
        pack: Current pack record.
        slots: Selected slot records.
        owner_user_id: User requesting the generation.
        variant_count: Optional per-slot variant override.

    Returns:
        A versioned recipe containing one entry per selected slot.

    Raises:
        ValueError: The character or a configured world book is unavailable.
    """
    character = repo.get_character(int(pack["primary_character_id"]))
    if character is None:
        raise ValueError("primary_character_not_found")
    world_book_entries = _world_book_entries(repo, pack)
    dimensions = _json_object(pack.get("default_dimensions_json"))
    recipes: list[dict[str, Any]] = []
    for slot in slots:
        labels = _json_object(slot.get("labels_json"))
        negative_prompt = _join(pack.get("negative_prompt"), slot.get("negative_prompt_template"))
        preview = build_prompt_preview(
            character=character,
            pack_style=pack.get("style_prompt"),
            pack_scenario=pack.get("scenario_notes"),
            negative_prompt=negative_prompt,
            style_lock=_json_object(pack.get("style_lock_json")),
            slot_template=slot.get("prompt_template"),
            labels=labels,
            world_book_entries=world_book_entries,
        )
        count = int(variant_count or slot["variant_count"])
        seed_policy = _json_object(slot.get("seed_policy_json"))
        seed = _positive_int(seed_policy.get("seed")) or _positive_int(seed_policy.get("base_seed"))
        extra_params = dimensions.get("extra_params")
        extra_params = dict(extra_params) if isinstance(extra_params, dict) else {}
        for key in ("steps", "cfg_scale", "sampler"):
            if key in dimensions and key not in extra_params:
                extra_params[key] = dimensions[key]
        recipes.append({
            "slot_id": int(slot["id"]),
            "slot_key": slot["slot_key"],
            "asset_type": slot["asset_type"],
            "labels": labels,
            "variant_count": count,
            "prompt_snapshot": {
                "prompt": preview.prompt,
                "negative_prompt": preview.negative_prompt,
                "token_estimates": preview.token_estimates,
                "omitted_source_counts": preview.omitted_source_counts,
                "warnings": list(preview.warnings),
            },
            "requested_backend": _first_text(slot.get("backend_override"), pack.get("default_backend")),
            "requested_model": _first_text(slot.get("model_override"), pack.get("default_model")),
            "width": _positive_int(slot.get("width")) or _positive_int(dimensions.get("width")),
            "height": _positive_int(slot.get("height")) or _positive_int(dimensions.get("height")),
            "format": (_first_text(dimensions.get("format"), dimensions.get("image_format"), "png") or "png").lower(),
            "extra_params": extra_params,
            "seeds": [seed + index if seed is not None else None for index in range(count)],
        })
    return {
        "version": RECIPE_VERSION,
        "pack_id": int(pack["id"]),
        "owner_user_id": owner_user_id,
        "primary_character_id": int(pack["primary_character_id"]),
        "slots": recipes,
    }


def _world_book_entries(repo: VNAssetPacksRepository, pack: Mapping[str, Any]) -> list[Any]:
    """Read configured world-book entries, rejecting incomplete prompt context."""
    raw_ids = pack.get("source_world_book_ids_json")
    ids = json.loads(raw_ids) if isinstance(raw_ids, str) else (raw_ids or [])
    if not ids:
        return []
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

    books = WorldBookService(repo.db)
    entries: list[Any] = []
    try:
        for raw_id in ids:
            book_id = int(raw_id)
            if books.get_world_book(world_book_id=book_id) is None:
                raise ValueError("vn_asset_world_book_unavailable")
            entries.extend(books.get_entries(world_book_id=book_id, enabled_only=True))
    except Exception as exc:
        raise ValueError("vn_asset_world_book_unavailable") from exc
    return entries


def _json_object(value: Any) -> dict[str, Any]:
    """Decode an optional JSON object without retaining non-object values."""
    if not value:
        return {}
    loaded = json.loads(value) if isinstance(value, str) else value
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    """Return a positive integer or None for absent and invalid values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _first_text(*values: Any) -> str | None:
    """Return the first nonblank textual value in priority order."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _join(*values: Any) -> str | None:
    """Join nonblank prompt fragments with line breaks."""
    parts = [_first_text(value) for value in values]
    return "\n".join(part for part in parts if part) or None

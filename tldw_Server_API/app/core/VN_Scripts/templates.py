"""Built-in VN script starter templates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

ContentRating = Literal["general", "teen", "suggestive", "mature"]


@dataclass(frozen=True)
class VNScriptTemplate:
    """Static template metadata plus deterministic draft construction."""

    id: str
    label: str
    description: str
    category: str
    recommended_content_rating: ContentRating
    required_capabilities: tuple[str, ...]
    preview: dict[str, Any]
    default_title: str
    default_description: str

    def catalog_payload(self) -> dict[str, Any]:
        """Return preview-safe metadata for clients."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "recommended_content_rating": self.recommended_content_rating,
            "required_capabilities": list(self.required_capabilities),
            "preview": deepcopy(self.preview),
            "default_title": self.default_title,
            "default_description": self.default_description,
        }


_TEMPLATES: tuple[VNScriptTemplate, ...] = (
    VNScriptTemplate(
        id="linear_scene",
        label="Linear scene",
        description="A simple one-scene route with narration and an ending.",
        category="starter",
        recommended_content_rating="general",
        required_capabilities=(),
        preview={"flow": ["start"], "operations": ["narrate", "end"]},
        default_title="Linear Scene",
        default_description="A one-scene VN script starter.",
    ),
    VNScriptTemplate(
        id="authored_choices",
        label="Authored choices",
        description="A short route with hand-authored player choices.",
        category="branching",
        recommended_content_rating="general",
        required_capabilities=("authored_choices",),
        preview={"flow": ["start", "accept", "decline"], "operations": ["choice", "narrate", "end"]},
        default_title="Authored Choice Scene",
        default_description="A VN script starter with two authored branches.",
    ),
    VNScriptTemplate(
        id="generated_choice_set",
        label="Generated choice set",
        description="A starter route that asks the generation profile for structured player choices.",
        category="guided_generation",
        recommended_content_rating="general",
        required_capabilities=("generation", "structured_output", "choice_set"),
        preview={"flow": ["start", "generated_choice"], "operations": ["generate", "narrate", "end"]},
        default_title="Generated Choice Scene",
        default_description="A VN script starter that routes generated choices through an authored target.",
    ),
    VNScriptTemplate(
        id="scene_update",
        label="Scene update",
        description="A starter route that requests a structured scene update from the generation profile.",
        category="guided_generation",
        recommended_content_rating="general",
        required_capabilities=("generation", "structured_output", "scene_update"),
        preview={"flow": ["start"], "operations": ["generate", "narrate", "end"]},
        default_title="Generated Scene Update",
        default_description="A VN script starter for generated scene updates.",
    ),
    VNScriptTemplate(
        id="confirm_gated_generation",
        label="Confirm-gated generation",
        description="A starter route that requires user confirmation before applying generated text.",
        category="safety",
        recommended_content_rating="general",
        required_capabilities=("generation", "user_confirmation"),
        preview={"flow": ["start", "cancelled"], "operations": ["generate", "narrate", "end"]},
        default_title="Confirm-Gated Generation",
        default_description="A VN script starter with explicit confirmation and cancel paths.",
    ),
)

_TEMPLATES_BY_ID = {template.id: template for template in _TEMPLATES}


def list_template_catalog() -> list[dict[str, Any]]:
    """Return all built-in templates as sanitized catalog entries."""
    return [template.catalog_payload() for template in _TEMPLATES]


def get_template(template_id: str) -> VNScriptTemplate:
    """Return a built-in template or raise a stable not-found reason."""
    try:
        return _TEMPLATES_BY_ID[template_id]
    except KeyError as exc:
        raise ValueError("template_not_found") from exc


def instantiate_template(
    template_id: str,
    *,
    title: str,
    primary_asset_pack_id: int,
    generation_profile_id: str,
) -> dict[str, Any]:
    """Build a deterministic VN script program draft for a template."""
    get_template(template_id)
    return {
        "schema_version": "vn_script_program.v1",
        "title": title,
        "primary_asset_pack_id": primary_asset_pack_id,
        "entry_label": "start",
        "variables": {},
        "generation_defaults": {"profile_id": generation_profile_id, "persist_model_outputs": True},
        "labels": _template_labels(template_id),
    }


def _template_labels(template_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic label operation blocks for a known starter template."""
    if template_id == "linear_scene":
        return {
            "start": [
                {"op": "narrate", "text": "The scene opens. Replace this line with your setup."},
                {"op": "end"},
            ]
        }
    if template_id == "authored_choices":
        return {
            "start": [
                {"op": "narrate", "text": "A decision point appears."},
                {
                    "op": "choice",
                    "id": "first_choice",
                    "choices": [
                        {"id": "accept", "text": "Take the direct path.", "target": "accept"},
                        {"id": "decline", "text": "Pause and reconsider.", "target": "decline"},
                    ],
                },
            ],
            "accept": [{"op": "narrate", "text": "The direct path continues."}, {"op": "end"}],
            "decline": [{"op": "narrate", "text": "The slower path reveals another detail."}, {"op": "end"}],
        }
    if template_id == "generated_choice_set":
        return {
            "start": [
                {"op": "narrate", "text": "Set the scene before requesting generated choices."},
                {
                    "op": "generate",
                    "scope": "turn",
                    "max_choices": 2,
                    "output_schema": "choice_set",
                    "on_generated_choice": "generated_choice",
                },
                {"op": "end"},
            ],
            "generated_choice": [{"op": "narrate", "text": "Handle the selected generated choice here."}, {"op": "end"}],
        }
    if template_id == "scene_update":
        return {
            "start": [
                {"op": "narrate", "text": "Describe the current scene state."},
                {"op": "generate", "scope": "scene", "max_choices": 1, "output_schema": "scene_update"},
                {"op": "end"},
            ]
        }
    if template_id == "confirm_gated_generation":
        return {
            "start": [
                {"op": "narrate", "text": "Prepare the player for generated continuation."},
                {
                    "op": "generate",
                    "scope": "turn",
                    "max_choices": 1,
                    "requires_user_confirm": True,
                    "on_cancel": "cancelled",
                },
                {"op": "end"},
            ],
            "cancelled": [{"op": "narrate", "text": "Generation was cancelled; continue with authored text."}, {"op": "end"}],
        }
    raise ValueError("template_not_found")

"""Shared taxonomy for Persona Visual starter production recipes.

The starter catalog exposes source asset groups and timed runtime animation
outputs as separate concepts. Keeping the taxonomy in this small module avoids
drift between immutable fixture validation and public API schema validation.
"""

BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS = frozenset(
    {
        "identity_brief",
        "neutral_anchor",
        "preview_image",
        "model_sheet",
        "static_talking_sheet",
        "static_reaction_sheet",
        "required_state_loops",
        "animation_strips",
        "animation_atlas",
        "custom_state_variants",
    }
)
BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS = frozenset(
    {
        "identity_brief",
        "neutral_anchor",
        "preview_image",
        "model_sheet",
        "static_talking_sheet",
        "static_reaction_sheet",
    }
)
BUDDY_VISUAL_ANIMATION_OUTPUT_IDS = frozenset(
    {
        "required_state_loops",
        "animation_strips",
        "animation_atlas",
        "custom_state_variants",
    }
)

__all__ = [
    "BUDDY_VISUAL_ANIMATION_OUTPUT_IDS",
    "BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS",
    "BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS",
]

from tldw_Server_API.app.core.VN_Assets.prompts import (
    PromptBudgets,
    build_prompt_preview,
    estimate_prompt_tokens,
)


long_character = {
    "name": "Mira",
    "description": "curious archivist with silver hair " * 20,
    "personality": "patient, exacting, warm, observant " * 20,
    "scenario": "researching an abandoned orbital archive " * 20,
    "first_message": "Welcome back to the stacks. " * 20,
    "creator_notes": "Keep the same uniform silhouette. " * 20,
    "style_notes": "soft ink lines, brass accessories " * 20,
}

very_long_world_book = {
    "priority": 5,
    "summary": "The orbital archive contains layered districts, ceremonial doors, and old constellations. " * 40,
}


def test_prompt_preview_preserves_slot_and_style_before_world_book() -> None:
    preview = build_prompt_preview(
        character=long_character,
        world_book_entries=[very_long_world_book],
        pack_style="watercolor, consistent outfit",
        slot_template="full body sprite, happy expression",
        labels={"expression": "happy"},
        budgets=PromptBudgets(character=20, world_book=20, pack=20, slot=20, total=60),
    )

    assert "full body sprite" in preview.prompt
    assert "watercolor" in preview.prompt
    assert preview.omitted_source_counts["world_book"] > 0
    assert preview.warnings


def test_prompt_preview_is_deterministic_with_stable_label_ordering() -> None:
    labels_a = {"pose": "standing", "expression": "happy", "variant": 2}
    labels_b = {"variant": 2, "expression": "happy", "pose": "standing"}

    first = build_prompt_preview(
        character={"name": "Mira", "description": "Archivist"},
        world_book_entries=["Archive hallway"],
        pack_style="watercolor",
        slot_template="full body sprite",
        labels=labels_a,
    )
    second = build_prompt_preview(
        character={"description": "Archivist", "name": "Mira"},
        world_book_entries=["Archive hallway"],
        pack_style="watercolor",
        slot_template="full body sprite",
        labels=labels_b,
    )

    assert first.prompt == second.prompt
    assert "expression=happy" in first.prompt
    assert first.prompt.index("expression=happy") < first.prompt.index("pose=standing")
    assert first.prompt.index("pose=standing") < first.prompt.index("variant=2")


def test_negative_prompt_and_pack_context_are_preserved_before_lower_priority_sources() -> None:
    preview = build_prompt_preview(
        character=long_character,
        world_book_entries=[very_long_world_book],
        pack_scenario="moonlit archive reading room",
        pack_style="clean watercolor sprite sheet",
        negative_prompt="blurry, extra fingers, mismatched outfit",
        slot_template="waist-up sprite, thoughtful expression",
        labels={"expression": "thoughtful"},
        budgets=PromptBudgets(character=8, world_book=8, pack=40, slot=18, total=64),
    )

    assert "waist-up sprite" in preview.prompt
    assert "moonlit archive" in preview.prompt
    assert "clean watercolor" in preview.prompt
    assert "Negative prompt:" in preview.prompt
    assert "blurry" in preview.negative_prompt
    assert preview.prompt.index("clean watercolor") < preview.prompt.index("Character name")
    assert preview.omitted_source_counts["character"] > 0
    assert preview.omitted_source_counts["world_book"] > 0


def test_long_pack_scenario_does_not_crowd_out_style_or_negative_prompt() -> None:
    preview = build_prompt_preview(
        character={"name": "Mira"},
        world_book_entries=["Archive hallway"],
        pack_scenario="scenario detail " * 100,
        pack_style="watercolor, consistent outfit",
        negative_prompt="blurry, extra fingers",
        slot_template="sprite",
        labels={"expression": "happy"},
        budgets=PromptBudgets(character=4, world_book=4, pack=18, slot=12, total=40),
    )

    assert "watercolor" in preview.prompt
    assert "Negative prompt:" in preview.prompt
    assert "blurry" in preview.prompt
    assert "scenario detail" not in preview.prompt
    assert preview.omitted_source_counts["pack"] > 0


def test_prompt_preview_respects_total_token_budget() -> None:
    preview = build_prompt_preview(
        character=long_character,
        world_book_entries=[very_long_world_book],
        pack_scenario="archive ceremony " * 10,
        pack_style="watercolor, consistent outfit " * 10,
        slot_template="full body sprite, happy expression",
        labels={"expression": "happy", "pose": "front"},
        budgets=PromptBudgets(character=30, world_book=30, pack=30, slot=20, total=55),
    )

    assert estimate_prompt_tokens(preview.prompt) <= 55
    assert preview.token_estimates["total"] <= 55


def test_prompt_preview_reserves_separator_tokens_inside_total_budget() -> None:
    preview = build_prompt_preview(
        character={"name": "Mira"},
        world_book_entries=["Archive hallway"],
        pack_style="watercolor",
        slot_template="sprite",
        labels={"expression": "happy"},
        budgets=PromptBudgets(character=4, world_book=4, pack=4, slot=4, total=4),
    )

    assert estimate_prompt_tokens(preview.prompt) <= 4
    assert preview.token_estimates["total"] <= 4


def test_prompt_preview_handles_character_ceiling_separator_edge_case() -> None:
    preview = build_prompt_preview(
        character={},
        world_book_entries=[],
        pack_style="b",
        slot_template="a",
        labels={},
        budgets=PromptBudgets(character=0, world_book=0, pack=100, slot=100, total=2),
    )

    assert estimate_prompt_tokens(preview.prompt) <= 2
    assert preview.token_estimates["total"] <= 2


def test_prompt_preview_reports_omissions_and_warnings_when_truncated() -> None:
    preview = build_prompt_preview(
        character=long_character,
        world_book_entries=[very_long_world_book],
        pack_style="watercolor",
        slot_template="sprite",
        labels={"expression": "neutral"},
        budgets=PromptBudgets(character=5, world_book=5, pack=5, slot=5, total=20),
    )

    assert preview.omitted_source_counts["character"] > 0
    assert preview.omitted_source_counts["world_book"] > 0
    assert preview.warnings
    assert any("truncated" in warning for warning in preview.warnings)

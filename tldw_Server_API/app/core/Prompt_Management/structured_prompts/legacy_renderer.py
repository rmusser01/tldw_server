from typing import Literal

from .models import PromptDefinition, PromptLegacySnapshot


def render_single_text_legacy_snapshot(
    text: str,
    target_role: Literal["system", "user"],
) -> PromptLegacySnapshot:
    """Place one compiled string in exactly its target field without trimming."""
    return PromptLegacySnapshot(
        system_prompt=text if target_role == "system" else "",
        user_prompt=text if target_role == "user" else "",
    )


def render_legacy_snapshot(
    messages: list[dict[str, str]],
    definition: PromptDefinition,
) -> PromptLegacySnapshot:
    separator = definition.assembly_config.block_separator
    system_roles = set(definition.assembly_config.legacy_system_roles)
    user_roles = set(definition.assembly_config.legacy_user_roles)

    system_prompt = separator.join(message["content"] for message in messages if message["role"] in system_roles)
    user_prompt = separator.join(message["content"] for message in messages if message["role"] in user_roles)
    return PromptLegacySnapshot(system_prompt=system_prompt, user_prompt=user_prompt)

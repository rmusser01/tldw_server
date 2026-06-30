"""
Character Card V3 parser and validator (minimal, spec-aligned).

Supports cards with:
- spec: "chara_card_v3"
- spec_version: e.g., "3.0"
- data: { name, description, personality, scenario, first_mes, mes_example, ... }

Maps to DB schema fields consistent with V1/V2 parsers.
"""

from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Character_Chat.modules.character_validation import (
    MAX_V2_ALTERNATE_GREETINGS,
    MAX_V2_TAG_LENGTH,
    MAX_V2_TAGS,
    MAX_V2_TEXT_FIELD_LENGTHS,
    validate_character_book,
)


def validate_v3_card(card_data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        # Top-level spec markers are helpful but not mandatory for leniency
        data = card_data.get("data", card_data)
        if not isinstance(data, dict):
            errors.append("'data' node must be a dictionary for v3")
            return False, errors
        required_fields = {
            "name": str,
            "description": str,
            "first_mes": str,
        }
        for field, expected_type in required_fields.items():
            if field not in data or data[field] is None:
                errors.append(f"Missing required field '{field}' in v3 data")
            elif not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' must be of type '{expected_type.__name__}'.")

        if isinstance(data.get("name"), str) and not data["name"].strip():
            errors.append("Field 'name' cannot be blank.")

        optional_fields = {
            "personality": str,
            "scenario": str,
            "mes_example": str,
            "creator_notes": str,
            "system_prompt": str,
            "post_history_instructions": str,
            "alternate_greetings": list,
            "tags": list,
            "creator": str,
            "character_version": str,
            "extensions": dict,
        }
        for field, expected_type in optional_fields.items():
            if field in data and not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' must be of type '{expected_type.__name__}'.")

        for field_name, max_len in MAX_V2_TEXT_FIELD_LENGTHS.items():
            field_value = data.get(field_name)
            if isinstance(field_value, str) and len(field_value) > max_len:
                errors.append(f"Field '{field_name}' exceeds max length of {max_len} characters.")

        alternate_greetings = data.get("alternate_greetings")
        if isinstance(alternate_greetings, list):
            if len(alternate_greetings) > MAX_V2_ALTERNATE_GREETINGS:
                errors.append(
                    f"Field 'alternate_greetings' exceeds max entries of {MAX_V2_ALTERNATE_GREETINGS}."
                )
            for idx, greeting in enumerate(alternate_greetings):
                if not isinstance(greeting, str):
                    errors.append(f"Field 'alternate_greetings[{idx}]' must be of type 'str'.")
                    continue
                if len(greeting) > MAX_V2_TEXT_FIELD_LENGTHS["first_mes"]:
                    errors.append(
                        f"Field 'alternate_greetings[{idx}]' exceeds max length of "
                        f"{MAX_V2_TEXT_FIELD_LENGTHS['first_mes']} characters."
                    )

        tags = data.get("tags")
        if isinstance(tags, list):
            if len(tags) > MAX_V2_TAGS:
                errors.append(f"Field 'tags' exceeds max entries of {MAX_V2_TAGS}.")
            for idx, tag in enumerate(tags):
                if not isinstance(tag, str):
                    errors.append(f"Field 'tags[{idx}]' must be of type 'str'.")
                    continue
                if len(tag) > MAX_V2_TAG_LENGTH:
                    errors.append(f"Field 'tags[{idx}]' exceeds max length of {MAX_V2_TAG_LENGTH} characters.")

        if "character_book" in data:
            if not isinstance(data["character_book"], dict):
                errors.append("Field 'character_book' must be a dictionary.")
            else:
                is_valid_book, book_errors = validate_character_book(data["character_book"])
                if not is_valid_book:
                    errors.extend(book_errors)
        return (len(errors) == 0), errors
    except Exception as e:
        logger.error(f"Unexpected error validating v3 card: {e}")
        return False, [str(e)]


def parse_v3_card(card_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        is_valid, errors = validate_v3_card(card_data)
        if not is_valid:
            logger.warning("Invalid v3 character card: {}", "; ".join(errors))
            return None

        data = card_data.get("data", card_data)
        if not isinstance(data, dict):
            return None
        parsed = {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "personality": data.get("personality", ""),
            "scenario": data.get("scenario", ""),
            "first_message": data.get("first_mes", ""),
            "message_example": data.get("mes_example", ""),
            "creator_notes": data.get("creator_notes", ""),
            "system_prompt": data.get("system_prompt", ""),
            "post_history_instructions": data.get("post_history_instructions", ""),
            "alternate_greetings": data.get("alternate_greetings", []),
            "tags": data.get("tags", []),
            "creator": data.get("creator", ""),
            "character_version": data.get("character_version", ""),
            "extensions": data.get("extensions", {}) or {},
        }
        image_value = data.get("char_image") or data.get("image")
        if image_value is not None:
            parsed["image_base64"] = image_value
        if not parsed["name"]:
            return None
        return parsed
    except Exception as e:
        logger.error(f"Error parsing v3 card: {e}")
        return None


__all__ = ["validate_v3_card", "parse_v3_card"]

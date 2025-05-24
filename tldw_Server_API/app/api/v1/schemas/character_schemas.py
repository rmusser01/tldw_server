# character_schemas.py
# Description:
#
# Imports
import json
from typing import Optional, Any, List, Union, Dict
#
# Third-party imports
from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import FieldValidationInfo
#
######################################################################################################################
#
# --- Pydantic Schemas ---

class CharacterBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = Field(None, examples=["A brave knight"])
    personality: Optional[str] = Field(None, examples=["Stoic and honorable"])
    scenario: Optional[str] = Field(None, examples=["Guarding the ancient ruins"])
    system_prompt: Optional[str] = Field(None, examples=["You are a helpful character."])
    post_history_instructions: Optional[str] = None
    first_message: Optional[str] = Field(None, examples=["Greetings, traveler!"])
    message_example: Optional[str] = Field(None, examples=["<START>\nUSER: Hello\nASSISTANT: Hi there!\n<END>"])
    creator_notes: Optional[str] = None
    alternate_greetings: Optional[Union[List[str], str]] = Field(None,
                                                                 description="List of strings or a JSON string representation of a list.",
                                                                 examples=[["Hello!", "Good day!"]])
    tags: Optional[Union[List[str], str]] = Field(None,
                                                  description="List of strings or a JSON string representation of a list.",
                                                  examples=[["fantasy", "knight"]])
    creator: Optional[str] = None
    character_version: Optional[str] = None
    extensions: Optional[Union[Dict[str, Any], str]] = Field(None,
                                                             description="Dictionary or a JSON string representation of a dictionary.")
    image_base64: Optional[str] = Field(None,
                                        description="Base64 encoded image string (without 'data:image/...;base64,' prefix).")

    @field_validator("alternate_greetings", "tags", "extensions", mode="before")
    @classmethod
    def parse_json_string(cls, value: Any, info: FieldValidationInfo) -> Any:  # Corrected type hint for info
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON string for field '{info.field_name}': {value[:100]}...")
                if info.field_name in ["alternate_greetings", "tags"]: return []
                if info.field_name == "extensions": return {}
                return value
        return value


class CharacterCreate(CharacterBase):
    name: str = Field(..., examples=["Sir Gideon"])


class CharacterUpdate(CharacterBase):
    pass  # All fields optional


class CharacterResponse(CharacterBase):
    id: int
    version: int
    image_present: bool = False
    model_config = {"from_attributes": True}


class CharacterImportResponse(BaseModel):
    message: str
    character: Optional[CharacterResponse] = None


class DeletionResponse(BaseModel):
    message: str
    character_id: int

#
# End of character_schemas.py
######################################################################################################################

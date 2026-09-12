"""Versioned structured prompt models and shared single-text recipe bounds."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from pydantic_core import PydanticCustomError

SINGLE_TEXT_RECIPE_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "max_blocks": 100,
        "max_variables": 100,
        "max_label_length": 200,
        "max_key_length": 128,
        "max_content_length": 20000,
        "max_separator_length": 20,
        # V2 ordering must round-trip exactly through JavaScript numbers.
        "max_abs_order": 9007199254740991,
        # Enforced after substitution by the single-text renderer, not schema validation.
        "max_rendered_output_length": 100000,
    }
)


class PromptLegacySnapshot(BaseModel):
    system_prompt: str
    user_prompt: str


class PromptAssemblyResult(BaseModel):
    messages: list[dict[str, str]]
    legacy: PromptLegacySnapshot


class ValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class PromptVariableDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    label: str | None = None
    description: str | None = None
    required: bool = False
    default_value: Any = None
    input_type: str = "text"
    options: list[str] | None = None
    max_length: int | None = Field(default=None, ge=1)


class PromptBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    role: Literal["system", "developer", "user", "assistant"]
    kind: str | None = None
    content: str
    enabled: bool = True
    order: int
    is_template: bool = False


class PromptAssemblyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legacy_system_roles: list[str] = Field(default_factory=lambda: ["system", "developer"])
    legacy_user_roles: list[str] = Field(default_factory=lambda: ["user"])
    block_separator: str = "\n\n"


class MultiMessagePromptDefinitionV1(BaseModel):
    """The original multi-message definition; never accepts recipe versions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    format: Literal["structured"] = "structured"
    variables: list[PromptVariableDefinition] = Field(default_factory=list)
    blocks: list[PromptBlock] = Field(default_factory=list)
    assembly_config: PromptAssemblyConfig = Field(default_factory=PromptAssemblyConfig)


# Existing assembler, conversion, API, and interop imports require this v1-only name.
PromptDefinition = MultiMessagePromptDefinitionV1


class SingleTextRecipeVariableDefinition(PromptVariableDefinition):
    """Recipe declarations use the existing template grammar and bounded labels."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=SINGLE_TEXT_RECIPE_LIMITS["max_key_length"],
        pattern=r"^[A-Za-z0-9_]+$",
    )
    label: str | None = Field(default=None, max_length=SINGLE_TEXT_RECIPE_LIMITS["max_label_length"])


class SingleTextRecipeBlock(PromptBlock):
    """An ordered recipe section, retaining the existing block storage fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(..., min_length=1, max_length=SINGLE_TEXT_RECIPE_LIMITS["max_key_length"])
    name: str = Field(..., min_length=1, max_length=SINGLE_TEXT_RECIPE_LIMITS["max_label_length"])
    content: str = Field(..., max_length=SINGLE_TEXT_RECIPE_LIMITS["max_content_length"])
    order: int = Field(
        ...,
        ge=-SINGLE_TEXT_RECIPE_LIMITS["max_abs_order"],
        le=SINGLE_TEXT_RECIPE_LIMITS["max_abs_order"],
    )
    section_key: str | None = Field(default=None, max_length=SINGLE_TEXT_RECIPE_LIMITS["max_key_length"])

    @field_validator("order", mode="before")
    @classmethod
    def normalize_json_integer_order(cls, value: Any) -> Any:
        """JSON 1 and 1.0 are equivalent; never coerce strings or booleans."""
        if type(value) is float and value.is_integer():
            limit = SINGLE_TEXT_RECIPE_LIMITS["max_abs_order"]
            if value > limit:
                raise PydanticCustomError("less_than_equal", "Order must be at most {le}.", {"le": limit})
            if value < -limit:
                raise PydanticCustomError("greater_than_equal", "Order must be at least {ge}.", {"ge": -limit})
            return int(value)
        return value

    @field_validator("id")
    @classmethod
    def require_nonblank_id(cls, value: str) -> str:
        """Reject IDs skipped by duplicate detection without rewriting stored identity."""
        if not value.strip():
            raise PydanticCustomError("invalid_block_id", "Recipe block IDs must contain a non-whitespace character.")
        return value


class SingleTextRecipeAssemblyConfig(BaseModel):
    """A recipe targets exactly one field with one deterministic render format."""

    model_config = ConfigDict(extra="forbid", strict=True)

    assembly_mode: Literal["single_text"]
    target_role: Literal["system", "user"]
    render_format: Literal["xml", "markdown", "freeform"]
    block_separator: str = Field(default="\n\n", max_length=SINGLE_TEXT_RECIPE_LIMITS["max_separator_length"])


class SingleTextRecipeDefinitionV2(BaseModel):
    """Explicitly identified single-text recipe with no persisted runtime values."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2]
    format: Literal["structured"]
    definition_kind: Literal["single_text_recipe"]
    variables: list[SingleTextRecipeVariableDefinition] = Field(
        default_factory=list,
        max_length=SINGLE_TEXT_RECIPE_LIMITS["max_variables"],
    )
    blocks: list[SingleTextRecipeBlock] = Field(
        default_factory=list,
        max_length=SINGLE_TEXT_RECIPE_LIMITS["max_blocks"],
    )
    assembly_config: SingleTextRecipeAssemblyConfig

    @model_validator(mode="before")
    @classmethod
    def require_integer_version(cls, value: Any) -> Any:
        """Preserve JSON numeric identity even when callers bypass the union."""
        return _require_integer_schema_version(value)


def normalize_recipe_schema_version(value: Any) -> Any:
    """Copy the v2 JSON numeric equivalent 2.0 to 2 without mutating callers."""
    if isinstance(value, Mapping):
        version = value.get("schema_version")
        if type(version) is float and version == 2.0:
            return {**value, "schema_version": 2}
    return value


def _require_integer_schema_version(value: Any) -> Any:
    """Allow the v2 JSON numeric equivalent, never bool/string coercion."""
    value = normalize_recipe_schema_version(value)
    version = value.get("schema_version") if isinstance(value, Mapping) else getattr(value, "schema_version", None)
    if type(version) is not int:
        raise ValueError("schema_version must be an integer")
    return value


StructuredPromptDefinition = Annotated[
    MultiMessagePromptDefinitionV1 | SingleTextRecipeDefinitionV2,
    Field(discriminator="schema_version"),
    BeforeValidator(_require_integer_schema_version),
]
_DEFINITION_ADAPTER = TypeAdapter(StructuredPromptDefinition)


def parse_prompt_definition(
    definition: Mapping[str, Any] | MultiMessagePromptDefinitionV1 | SingleTextRecipeDefinitionV2,
) -> MultiMessagePromptDefinitionV1 | SingleTextRecipeDefinitionV2:
    """Parse the explicit version without migration; semantic checks live in validator."""
    return _DEFINITION_ADAPTER.validate_python(definition)

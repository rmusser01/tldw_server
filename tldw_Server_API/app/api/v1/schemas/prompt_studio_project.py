# prompt_studio_project.py
# Project and prompt schemas for Prompt Studio

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .prompt_studio_base import ConstraintDefinition, FieldDefinition, ProjectStatus, TimestampMixin, UUIDMixin

########################################################################################################################
# Project Schemas

class ProjectBase(BaseModel):
    """Base project model"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=2000, description="Project description")
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT, description="Project status")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

class ProjectCreate(ProjectBase):
    """Project creation request"""
    pass

class ProjectUpdate(BaseModel):
    """Project update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[ProjectStatus] = None
    metadata: Optional[dict[str, Any]] = None

class ProjectResponse(ProjectBase, TimestampMixin, UUIDMixin):
    """Project response model"""
    id: int
    user_id: str
    version: int = 1

    model_config = ConfigDict(from_attributes=True)

class ProjectListItem(BaseModel):
    """Simplified project for list views"""
    id: int
    uuid: UUID
    name: str
    description: Optional[str]
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    prompt_count: Optional[int] = 0
    test_case_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)

########################################################################################################################
# Signature Schemas

class SignatureBase(BaseModel):
    """Base signature model"""
    name: str = Field(..., min_length=1, max_length=255, description="Signature name")
    input_schema: list[FieldDefinition] = Field(..., description="Input field definitions")
    output_schema: list[FieldDefinition] = Field(..., description="Output field definitions")
    constraints: Optional[list[ConstraintDefinition]] = Field(None, description="Validation constraints")
    validation_rules: Optional[dict[str, Any]] = Field(None, description="Additional validation rules")

class SignatureCreate(SignatureBase):
    """Signature creation request"""
    project_id: int = Field(..., description="Parent project ID")

class SignatureUpdate(BaseModel):
    """Signature update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    input_schema: Optional[list[FieldDefinition]] = None
    output_schema: Optional[list[FieldDefinition]] = None
    constraints: Optional[list[ConstraintDefinition]] = None
    validation_rules: Optional[dict[str, Any]] = None

class SignatureResponse(SignatureBase, TimestampMixin, UUIDMixin):
    """Signature response model"""
    id: int
    project_id: int

    model_config = ConfigDict(from_attributes=True)

class SignatureValidateRequest(BaseModel):
    """Request to validate data against a signature"""
    signature_id: int
    data: dict[str, Any]
    validate_inputs: bool = True
    validate_outputs: bool = False

class SignatureValidateResponse(BaseModel):
    """Signature validation response"""
    valid: bool
    errors: Optional[list[dict[str, str]]] = None

########################################################################################################################
# Prompt Schemas

class PromptModule(BaseModel):
    """Configuration for a prompt module (CoT, ReAct, etc.)"""
    type: str = Field(..., description="Module type")
    enabled: bool = Field(default=True)
    config: Optional[dict[str, Any]] = Field(None, description="Module-specific configuration")

class FewShotExample(BaseModel):
    """Few-shot example for prompts"""
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    explanation: Optional[str] = None

class PromptBase(BaseModel):
    """Base prompt model"""
    name: str = Field(..., min_length=1, max_length=255, description="Prompt name")
    system_prompt: Optional[str] = Field(None, max_length=50000, description="System prompt")
    user_prompt: Optional[str] = Field(None, max_length=50000, description="User prompt template")
    prompt_format: Literal["legacy", "structured"] = Field(
        "legacy",
        description="Whether the prompt stores raw legacy fields or a structured definition.",
    )
    prompt_schema_version: Optional[int] = Field(
        None,
        ge=1,
        description="Structured prompt schema version when prompt_format is 'structured'.",
    )
    prompt_definition: Optional[dict[str, Any]] = Field(
        None,
        description="Structured prompt definition when prompt_format is 'structured'.",
    )
    few_shot_examples: Optional[list[FewShotExample]] = Field(None, description="Few-shot examples")
    modules_config: Optional[list[PromptModule]] = Field(None, description="Module configurations")
    change_description: Optional[str] = Field(None, max_length=500, description="Description of changes")

class PromptCreate(PromptBase):
    """Prompt creation request"""
    project_id: int = Field(..., description="Parent project ID")
    signature_id: Optional[int] = Field(None, description="Associated signature ID")
    parent_version_id: Optional[int] = Field(None, description="Parent version for versioning")

class PromptUpdate(BaseModel):
    """Prompt update request (creates new version)"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    system_prompt: Optional[str] = Field(None, max_length=50000)
    user_prompt: Optional[str] = Field(None, max_length=50000)
    prompt_format: Optional[Literal["legacy", "structured"]] = None
    prompt_schema_version: Optional[int] = Field(None, ge=1)
    prompt_definition: Optional[dict[str, Any]] = None
    few_shot_examples: Optional[list[FewShotExample]] = None
    modules_config: Optional[list[PromptModule]] = None
    change_description: str = Field(..., min_length=1, max_length=500, description="Required change description")

class PromptResponse(PromptBase, TimestampMixin, UUIDMixin):
    """Prompt response model"""
    id: int
    project_id: int
    signature_id: Optional[int]
    version_number: int
    parent_version_id: Optional[int]

    model_config = ConfigDict(from_attributes=True)

class PromptVersion(BaseModel):
    """Prompt version in history"""
    id: int
    uuid: UUID
    version_number: int
    name: str
    change_description: Optional[str]
    created_at: datetime
    parent_version_id: Optional[int]
    prompt_format: Literal["legacy", "structured"] = "legacy"
    prompt_schema_version: Optional[int] = None
    prompt_definition: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class PromptCompareRequest(BaseModel):
    """Request to compare two prompt versions"""
    prompt_id_1: int
    prompt_id_2: int
    include_metrics: bool = Field(default=False, description="Include performance metrics")

class PromptCompareResponse(BaseModel):
    """Prompt comparison response"""
    prompt_1: PromptResponse
    prompt_2: PromptResponse
    differences: dict[str, Any]
    metrics_comparison: Optional[dict[str, Any]] = None


class StructuredPromptPreviewRequest(BaseModel):
    """Request to preview assembled prompt messages."""
    project_id: int
    signature_id: Optional[int] = None
    prompt_format: Literal["legacy", "structured"] = "legacy"
    system_prompt: Optional[str] = Field(None, max_length=50000)
    user_prompt: Optional[str] = Field(None, max_length=50000)
    prompt_schema_version: Optional[int] = Field(None, ge=1)
    prompt_definition: Optional[dict[str, Any]] = None
    few_shot_examples: Optional[list[FewShotExample]] = None
    modules_config: Optional[list[PromptModule]] = None
    variables: dict[str, Any] = Field(default_factory=dict)


class StructuredPromptPreviewResponse(BaseModel):
    """Preview response with assembled messages and legacy compatibility text."""
    prompt_format: Literal["legacy", "structured"]
    prompt_schema_version: Optional[int] = None
    rendered_text: Optional[str] = None
    assembled_messages: list[dict[str, str]] = Field(default_factory=list)
    legacy_system_prompt: str = ""
    legacy_user_prompt: str = ""


class StructuredPromptConvertRequest(BaseModel):
    """Convert raw prompt fields into a structured prompt definition."""
    project_id: int
    system_prompt: Optional[str] = Field(None, max_length=50000)
    user_prompt: Optional[str] = Field(None, max_length=50000)


class StructuredPromptConvertResponse(BaseModel):
    """Structured prompt conversion result."""
    prompt_format: Literal["structured"] = "structured"
    prompt_schema_version: int = 1
    prompt_definition: dict[str, Any]
    extracted_variables: list[str] = Field(default_factory=list)
    legacy_system_prompt: str = ""
    legacy_user_prompt: str = ""

########################################################################################################################
# Generation Requests

class PromptGenerateRequest(BaseModel):
    """Request to generate a prompt from description"""
    project_id: int
    task_description: str = Field(..., min_length=10, max_length=5000, description="Task description")
    task_type: Optional[str] = Field(None, description="Type of task (classification, extraction, etc.)")
    include_cot: bool = Field(default=True, description="Include chain-of-thought reasoning")
    include_examples: bool = Field(default=True, description="Generate few-shot examples")
    target_model: Optional[str] = Field(None, description="Target LLM model")

class PromptImproveRequest(BaseModel):
    """Request to improve an existing prompt"""
    prompt_id: int
    improvement_goals: Optional[list[str]] = Field(None, description="Specific improvement goals")
    test_case_ids: Optional[list[int]] = Field(None, description="Test cases to optimize against")
    preserve_structure: bool = Field(default=True, description="Preserve prompt structure")

class ExampleGenerateRequest(BaseModel):
    """Request to generate few-shot examples"""
    prompt_id: int
    num_examples: int = Field(default=3, ge=1, le=10, description="Number of examples to generate")
    use_test_cases: bool = Field(default=True, description="Generate from test cases if available")
    diversity_factor: float = Field(default=0.7, ge=0.0, le=1.0, description="Example diversity")

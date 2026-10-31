# prompt_studio_optimization_requests.py
# Request models for optimization endpoints

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    MODEL_CONFIG_ALIASES,
    NATIVE_DURABLE_OPTIMIZATION_STRATEGIES,
    normalize_durable_optimization_config,
    normalize_optimization_strategy,
    reconcile_optimization_model_config_aliases,
    reconcile_optimization_strategy,
)

from .prompt_studio_base import DEFAULT_MAX_TEST_CASES


class CompareStrategiesRequest(BaseModel):
    """Request model for comparing optimization strategies."""
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    prompt_id: int = Field(..., description="Prompt to optimize")
    # Accept optional project_id for back-compat with older clients/tests
    project_id: Optional[int] = Field(
        None,
        description="Project ID (optional; inferred from prompt if omitted)",
    )
    test_case_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_MAX_TEST_CASES,
        description="Test cases for evaluation",
    )
    strategies: list[str] = Field(
        ...,
        min_length=2,
        max_length=len(NATIVE_DURABLE_OPTIMIZATION_STRATEGIES),
        description="Unique supported strategies to compare",
    )
    # Back-compat: accept config as alias for model_configuration
    model_configuration: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "model_configuration",
            "model_config",
            "llm_model_config",
            "config",
        ),
        serialization_alias="model_config",
        description="Model configuration",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_configuration(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        aliases = (
            "model_configuration",
            "model_config",
            "llm_model_config",
            "config",
        )
        normalized["model_configuration"] = (
            reconcile_optimization_model_config_aliases(
                normalized,
                aliases=aliases,
                allow_default_when_missing=True,
                reject_sensitive=True,
            )
        )
        for alias in aliases[1:]:
            normalized.pop(alias, None)
        return normalized

    @field_validator("test_case_ids")
    @classmethod
    def _validate_test_case_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("test_case_ids must be unique")
        return value

    @field_validator("strategies")
    @classmethod
    def _validate_strategies(cls, value: list[str]) -> list[str]:
        normalized = [normalize_optimization_strategy(strategy) for strategy in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("strategies must be unique")
        return normalized


class OptimizationSimpleCreateRequest(BaseModel):
    """Minimal optimization job creation payload (compat endpoint)."""
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    prompt_id: Optional[int] = None
    initial_prompt_id: Optional[int] = None
    config: dict[str, Any] = Field(default_factory=dict)
    # The prompt remains authoritative for project ownership; project_id is
    # accepted only for compatibility with older clients and the WebUI.
    project_id: Optional[int] = None
    strategy: Optional[str] = None
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    llm_model_config: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "model_config",
            "model_configuration",
            "llm_model_config",
        ),
        serialization_alias="model_config",
    )
    test_case_ids: Optional[list[int]] = Field(
        None,
        min_length=1,
        max_length=DEFAULT_MAX_TEST_CASES,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        raw_config = normalized.get("config")
        if raw_config is None:
            raw_config = {}
        inner_model_supplied = isinstance(raw_config, Mapping) and any(
            alias in raw_config for alias in MODEL_CONFIG_ALIASES
        )
        normalized_config = normalize_durable_optimization_config(
            raw_config,
            reject_sensitive=True,
        )
        outer_model_supplied = any(
            alias in normalized for alias in MODEL_CONFIG_ALIASES
        )
        if outer_model_supplied:
            outer_model_config = reconcile_optimization_model_config_aliases(
                normalized,
                allow_default_when_missing=False,
                reject_sensitive=True,
            )
            if (
                inner_model_supplied
                and outer_model_config != normalized_config["model_config"]
            ):
                raise ValueError("Model configuration aliases conflict")
            normalized_config["model_config"] = outer_model_config

        raw_strategy = normalized.get("strategy")
        selected_strategy = reconcile_optimization_strategy(
            raw_strategy,
            normalized_config.get("optimizer_type"),
            normalized_config.get("strategy"),
            default="iterative",
        )
        normalized_config["optimizer_type"] = selected_strategy
        normalized_config.pop("strategy", None)
        normalized_config = normalize_durable_optimization_config(
            normalized_config,
            reject_sensitive=True,
        )
        if raw_strategy is not None:
            normalized["strategy"] = selected_strategy
        for alias in MODEL_CONFIG_ALIASES:
            normalized.pop(alias, None)
        normalized["model_config"] = normalized_config["model_config"]
        normalized["config"] = normalized_config
        return normalized

    @field_validator("test_case_ids")
    @classmethod
    def _validate_test_case_ids(
        cls,
        value: Optional[list[int]],
    ) -> Optional[list[int]]:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("test_case_ids must be unique")
        return value

    @model_validator(mode="after")
    def _require_one_id(self):
        if not self.prompt_id and not self.initial_prompt_id:
            raise ValueError("One of prompt_id or initial_prompt_id must be provided")
        return self

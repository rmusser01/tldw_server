"""
Benchmark Registry for managing and running evaluation benchmarks.

Provides a centralized registry for benchmark configurations and mappings
to appropriate evaluation types.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from tldw_Server_API.app.core.Evaluations.benchmark_utils import (
    BaseEvaluation,
    BullshitDetectionEvaluation,
    CodeGenerationEvaluation,
    HonestyEvaluation,
    InstructionFollowingEvaluation,
    MultipleChoiceEvaluation,
)

logger = logger


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    description: str
    evaluation_type: str
    dataset_source: str  # URL, file path, or HuggingFace dataset ID
    dataset_format: str  # json, jsonl, csv, huggingface
    field_mappings: dict[str, str] = field(default_factory=dict)
    evaluation_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_type": self.evaluation_type,
            "dataset_source": self.dataset_source,
            "dataset_format": self.dataset_format,
            "field_mappings": self.field_mappings,
            "evaluation_params": self.evaluation_params,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BenchmarkConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkConfig':
        """Load from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class BenchmarkRegistry:
    """Registry for managing benchmarks."""

    # Mapping of evaluation types to classes
    EVALUATION_TYPES = {
        "multiple_choice": MultipleChoiceEvaluation,
        "code_generation": CodeGenerationEvaluation,
        "instruction_following": InstructionFollowingEvaluation,
        "honesty": HonestyEvaluation,
        "bullshit_detection": BullshitDetectionEvaluation,
        "simpleqa": None,  # Loaded dynamically
        "function_calling": None,  # To be implemented
        "multi_turn": None,  # To be implemented
    }

    def __init__(self):
        self.benchmarks: dict[str, BenchmarkConfig] = {}
        self._load_default_benchmarks()

    def _load_default_benchmarks(self):
        """Load default benchmark configurations. Every registered benchmark builds an evaluator and loads
        rows in its evaluator's fields (TASK-15487); a benchmark with no evaluator is not registered."""
        # SimpleQA: OpenAI's test set (simple-evals), a CSV with "problem" and "answer" columns
        self.register(BenchmarkConfig(
            name="simpleqa",
            description="OpenAI SimpleQA - Measuring short-form factuality",
            dataset_source="https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
            evaluation_type="simpleqa",
            dataset_format="csv",
            field_mappings={
                "question": "problem",
                "answer": "answer",
            },
            evaluation_params={
                "grading_model": "openai",
                "strict_grading": True,
            },
            metadata={
                "paper": "https://openai.com/index/introducing-simpleqa/",
                "size": 4326,
                "topics": ["history", "science", "technology", "art", "geography"]
            }
        ))

        # SimpleQA Verified: the curated set ships in the repo as a zip holding one CSV ("problem", "answer", "topic")
        repo_root = Path(__file__).resolve().parents[4]
        verified_archive = repo_root / "Helper_Scripts" / "Evals" / "simpleqa-verified.zip"
        if verified_archive.is_file():
            dataset_source = str(verified_archive)
            self.register(BenchmarkConfig(
                name="simpleqa_verified",
                description="SimpleQA Verified - curated factuality set (local)",
                evaluation_type="simpleqa",
                dataset_source=dataset_source,
                dataset_format="csv",
                field_mappings={
                    "question": "problem",
                    "answer": "answer",
                    "topic": "topic"
                },
                evaluation_params={
                    "grading_model": "openai",
                    "strict_grading": True,
                },
                metadata={
                    "source": "local_repo",
                    "path": dataset_source
                }
            ))

        # Bullshit Benchmark (packaged local snapshot)
        self.register(BenchmarkConfig(
            name="bullshit_benchmark",
            description="Bullshit Benchmark v2 - nonsensical premise detection",
            evaluation_type="bullshit_detection",
            dataset_source="builtin://bullshit_benchmark_v2",
            dataset_format="custom",
            field_mappings={},
            evaluation_params={},
            metadata={
                "source": "bullshit-benchmark",
                "snapshot_version": "2.0",
                "resource_package": "tldw_Server_API.app.core.Evaluations.data.bullshit_benchmark",
                "resource_file": "questions_v2.json",
            },
        ))

        # MMLU Pro (needs the datasets package: the data-tools extra). Its rows: question, options, answer (a letter),
        # answer_index, category.
        self.register(BenchmarkConfig(
            name="mmlu_pro",
            description="MMLU Pro - Enhanced multiple choice with 10 options and reasoning",
            evaluation_type="multiple_choice",
            dataset_source="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            dataset_format="huggingface",
            field_mappings={
                "question": "question",
                "choices": "options",
                "correct_answer": "answer",
                "category": "category"
            },
            evaluation_params={
                "num_choices": 10,
                "require_reasoning": True
            },
            metadata={
                "paper": "https://arxiv.org/abs/2406.01574",
                "domains": 14,
                "size": 12000
            }
        ))

        # GPQA Diamond (needs the datasets package; the dataset is gated: accept its terms on HuggingFace and set
        # HF_TOKEN). The loader shapes its columns into question, choices, correct_answer and category.
        self.register(BenchmarkConfig(
            name="gpqa_diamond",
            description="Graduate-level science questions (physics, chemistry, biology)",
            evaluation_type="multiple_choice",
            dataset_source="https://huggingface.co/datasets/Idavidrein/gpqa",
            dataset_format="huggingface",
            field_mappings={
                "question": "question",
                "choices": "choices",
                "correct_answer": "correct_answer",
                "category": "category"
            },
            evaluation_params={
                "num_choices": 4,
                "require_reasoning": True
            },
            metadata={
                "difficulty": "graduate",
                "domains": ["physics", "chemistry", "biology"]
            }
        ))

        # Simple Bench
        self.register(BenchmarkConfig(
            name="simple_bench",
            description="Reasoning problems with detailed scenarios",
            evaluation_type="multiple_choice",
            dataset_source="https://github.com/simple-bench/SimpleBench/blob/main/simple_bench_public.json",
            dataset_format="json",
            field_mappings={
                "question": "prompt",
                "correct_answer": "answer"
            },
            evaluation_params={
                "num_choices": 6,
                "require_reasoning": False
            }
        ))

    def register(self, config: BenchmarkConfig) -> None:
        """Register a benchmark configuration."""
        self.benchmarks[config.name] = config
        logger.info(f"Registered benchmark: {config.name}")

    def unregister(self, name: str) -> None:
        """Unregister a benchmark."""
        if name in self.benchmarks:
            del self.benchmarks[name]
            logger.info(f"Unregistered benchmark: {name}")

    def get(self, name: str) -> Optional[BenchmarkConfig]:
        """Get a benchmark configuration."""
        return self.benchmarks.get(name)

    def load_dataset(self, benchmark_name: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Load a registered benchmark using the common dataset parser."""
        from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset

        if self.get(benchmark_name) is None:
            return []
        return load_benchmark_dataset(benchmark_name, limit=limit, registry=self)

    def list_benchmarks(self) -> list[str]:
        """List all registered benchmark names."""
        return list(self.benchmarks.keys())

    def get_benchmark_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a benchmark."""
        config = self.get(name)
        if not config:
            return {}

        return {
            "name": config.name,
            "description": config.description,
            "evaluation_type": config.evaluation_type,
            "dataset_source": config.dataset_source,
            "metadata": config.metadata
        }

    def create_evaluator(self, benchmark_name: str) -> Optional[BaseEvaluation]:
        """Create an evaluator instance for a benchmark."""
        config = self.get(benchmark_name)
        if not config:
            logger.error(f"Benchmark not found: {benchmark_name}")
            return None

        # Special handling for SimpleQA
        if config.evaluation_type == "simpleqa":
            try:
                from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQAEvaluation
                evaluator = SimpleQAEvaluation(
                    name=config.name,
                    **config.evaluation_params
                )
                return evaluator
            except Exception as e:
                logger.exception(f"Failed to create SimpleQA evaluator: {e}")
                return None

        eval_class = self.EVALUATION_TYPES.get(config.evaluation_type)
        if not eval_class:
            logger.error(f"Evaluation type not implemented: {config.evaluation_type}")
            return None

        # Create evaluator with benchmark-specific parameters
        try:
            evaluator = eval_class(
                name=config.name,
                **config.evaluation_params
            )
            return evaluator
        except Exception as e:
            logger.exception(f"Failed to create evaluator for {benchmark_name}: {e}")
            return None

    def save_config(self, config: BenchmarkConfig, path: str) -> None:
        """Save benchmark configuration to file."""
        path = Path(path)

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved benchmark config to {path}")

    def load_config(self, path: str) -> Optional[BenchmarkConfig]:
        """Load benchmark configuration from file."""
        path = Path(path)

        if not path.exists():
            logger.error(f"Config file not found: {path}")
            return None

        try:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                config = BenchmarkConfig.from_yaml(str(path))
            else:
                with open(path) as f:
                    data = json.load(f)
                config = BenchmarkConfig.from_dict(data)

            return config

        except Exception as e:
            logger.exception(f"Failed to load config from {path}: {e}")
            return None


# Global registry instance
_registry = BenchmarkRegistry()

def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry."""
    return _registry

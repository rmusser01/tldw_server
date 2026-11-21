"""
Dataset loaders for various benchmark formats.

Supports loading from:
- Local files (JSON, JSONL, CSV)
- URLs
- HuggingFace datasets
- Custom formats
"""

import contextlib
import csv
import io
import json
import os
import random
import zipfile
from collections.abc import Generator
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from tldw_Server_API.app.core.http_client import fetch, fetch_json

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Evaluations.benchmark_registry import BenchmarkRegistry

logger = logger


class DatasetLoader:
    """Base class for dataset loading."""

    @staticmethod
    def load_json(source: str) -> list[dict[str, Any]]:
        """Load JSON dataset from file or URL."""
        if source.startswith(('http://', 'https://')):
            data = fetch_json(method="GET", url=source, timeout=15)
        else:
            with open(source, encoding='utf-8') as f:
                data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ['data', 'items', 'questions', 'examples', 'samples', 'dataset']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no common key, wrap in list
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

    @staticmethod
    def load_jsonl(source: str) -> list[dict[str, Any]]:
        """Load JSONL dataset from file or URL."""
        data = []

        if source.startswith(('http://', 'https://')):
            # Simpler: fetch full text then split lines (avoids bespoke streaming logic)
            r = fetch(method="GET", url=source, timeout=30)
            try:
                for line in r.text.splitlines():
                    if line:
                        data.append(json.loads(line))
            finally:
                with contextlib.suppress(Exception):
                    r.close()
        else:
            with open(source, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

        return data

    @staticmethod
    def load_csv(source: str, delimiter: str = ',') -> list[dict[str, Any]]:
        """Load CSV dataset from file or URL."""
        if source.startswith(('http://', 'https://')):
            r = fetch(method="GET", url=source, timeout=15)
            try:
                data = r.text
            finally:
                with contextlib.suppress(Exception):
                    r.close()
            return [dict(row) for row in csv.DictReader(io.StringIO(data, newline=""), delimiter=delimiter)]

        with open(source, encoding='utf-8', newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]

    @staticmethod
    def load_huggingface(dataset_id: str, split: str = 'test',
                        limit: Optional[int] = None,
                        revision: Optional[str] = None,
                        name: Optional[str] = None) -> list[dict[str, Any]]:
        """Load dataset from HuggingFace; ``name`` is the dataset's config (GPQA has one per subset).

        Note: This requires the 'datasets' library to be installed.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.exception("HuggingFace datasets library not installed. "
                        "Install with: pip install datasets")
            return []

        try:
            # Load dataset
            resolved_revision = revision or os.getenv("HF_DATASETS_DEFAULT_REVISION")
            dataset = load_dataset(dataset_id, name=name, split=split, revision=resolved_revision)  # nosec B615

            # Convert to list of dicts
            data = []
            for i, item in enumerate(dataset):
                if limit and i >= limit:
                    break
                data.append(dict(item))

            return data

        except Exception as e:
            logger.exception(f"Failed to load HuggingFace dataset {dataset_id}: {e}")
            return []

    @staticmethod
    def stream_large_file(source: str, format: str = 'jsonl',
                         chunk_size: int = 1000) -> Generator[list[dict[str, Any]], None, None]:
        """Stream large datasets in chunks."""
        if format == 'jsonl':
            chunk = []

            if source.startswith(('http://', 'https://')):
                r = fetch(method="GET", url=source, timeout=60)
                try:
                    for line in r.text.splitlines():
                        if line:
                            chunk.append(json.loads(line))
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                finally:
                    with contextlib.suppress(Exception):
                        r.close()
            else:
                with open(source, encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            chunk.append(json.loads(line))
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []

            # Yield remaining items
            if chunk:
                yield chunk

        else:
            # For non-streaming formats, load all and chunk
            if format == 'json':
                data = DatasetLoader.load_json(source)
            elif format == 'csv':
                data = DatasetLoader.load_csv(source)
            else:
                raise ValueError(f"Unsupported format for streaming: {format}")

            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]


class BenchmarkDatasetLoader:
    """Specialized loaders for specific benchmarks."""

    @staticmethod
    def load_mmlu_pro(source: Optional[str] = None) -> list[dict[str, Any]]:
        """Load MMLU Pro dataset."""
        if not source:
            # Default to HuggingFace
            return DatasetLoader.load_huggingface("TIGER-Lab/MMLU-Pro", split="test")

        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id, split="test")
        else:
            # Load from file/URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)

    @staticmethod
    def load_simple_bench(source: Optional[str] = None) -> list[dict[str, Any]]:
        """Load Simple Bench dataset."""
        if not source:
            source = "https://raw.githubusercontent.com/simple-bench/SimpleBench/main/simple_bench_public.json"

        data = DatasetLoader.load_json(source)

        # Simple Bench has questions embedded in prompt with choices
        processed = []
        for item in data:
            # Extract choices from prompt (they're embedded as A., B., etc.)
            prompt = item.get("prompt", "")

            # Basic parsing - would need more sophisticated parsing for production
            import re
            choices_pattern = r'([A-F])\.\s*([^\n]+)'
            matches = re.findall(choices_pattern, prompt)

            choices = [match[1].strip() for match in matches]

            processed.append({
                "question": prompt.split('\n\n')[0] if '\n\n' in prompt else prompt,
                "choices": choices,
                "correct_answer": item.get("answer", ""),
                "question_id": item.get("question_id", ""),
                "_original": item
            })

        return processed

    @staticmethod
    def load_gpqa(source: Optional[str] = None,
                 subset: str = "diamond") -> list[dict[str, Any]]:
        """Load a GPQA subset: one config of the gated HuggingFace dataset (split ``train``), or a file in its columns."""
        if not source:
            data = DatasetLoader.load_huggingface("Idavidrein/gpqa", split="train", name=f"gpqa_{subset}")
        elif source.startswith("huggingface://"):
            data = DatasetLoader.load_huggingface(source.replace("huggingface://", ""), split="train", name=f"gpqa_{subset}")
        elif source.endswith('.csv'):
            data = DatasetLoader.load_csv(source)
        elif source.endswith('.jsonl'):
            data = DatasetLoader.load_jsonl(source)
        else:
            data = DatasetLoader.load_json(source)
        return [BenchmarkDatasetLoader._shape_gpqa_row(row, i) for i, row in enumerate(data)]

    @staticmethod
    def _shape_gpqa_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        """A GPQA row holds the correct answer and three incorrect ones in named columns. The choices are
        shuffled once per item, seeded by its Record ID, so a run is reproducible and A is not always right."""
        if "Correct Answer" not in row:
            return row  # already in the benchmark's fields (a local file)
        correct = str(row["Correct Answer"]).strip()
        choices = [correct] + [str(row[f"Incorrect Answer {n}"]).strip() for n in (1, 2, 3)]
        random.Random(str(row.get("Record ID") or index)).shuffle(choices)  # nosec B311 - reproducible order, not security
        return {
            "question": row.get("Question", ""),
            "choices": choices,
            "correct_answer": chr(65 + choices.index(correct)),
            "category": row.get("Subdomain") or row.get("High-level domain") or "general",
            "_original": row,
        }

    @staticmethod
    def load_simpleqa(source: Optional[str] = None) -> list[dict[str, Any]]:
        """Load a SimpleQA dataset: OpenAI's CSV (the default), a URL, a local file, a zip holding a CSV, or a directory."""
        if not source:
            source = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id)
        elif source.endswith('.csv'):
            return DatasetLoader.load_csv(source)
        elif source.startswith("http"):
            # Load from URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
        else:
            # Load from local path (zip, file or directory)
            from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQADataset
            p = Path(source)
            if p.suffix == ".zip":  # the shipped simpleqa-verified.zip: one CSV inside
                with zipfile.ZipFile(p) as archive:
                    member = next(n for n in archive.namelist() if n.endswith(".csv"))
                    with archive.open(member) as raw:
                        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))
            if p.is_dir():
                # Prefer a *_verified.jsonl or first jsonl/json
                candidates = list(p.glob("*.jsonl")) + list(p.glob("*.json"))
                if not candidates:
                    logger.error(f"No dataset files found in directory: {source}")
                    return []
                # Heuristic: pick file with 'verified' in name first
                candidates.sort(key=lambda x: ("verified" not in x.name.lower(), x.name))
                source_file = str(candidates[0])
                return SimpleQADataset.load_from_file(source_file)
            else:
                return SimpleQADataset.load_from_file(source)

    @staticmethod
    def load_bullshit_benchmark(source: Optional[str] = None) -> list[dict[str, Any]]:
        """Load and normalize the bullshit benchmark dataset."""
        if source and source.startswith("builtin://"):
            builtin_id = source.replace("builtin://", "", 1)
            if builtin_id not in {"bullshit_benchmark_v2", "bullshit_benchmark"}:
                raise ValueError(f"Unsupported builtin bullshit benchmark source: {source}")
            source = None

        if source:
            if source.startswith(("http://", "https://")):
                payload = fetch_json(method="GET", url=source, timeout=15)
            else:
                with open(source, encoding="utf-8") as f:
                    payload = json.load(f)
        else:
            pkg = "tldw_Server_API.app.core.Evaluations.data.bullshit_benchmark"
            with resources.files(pkg).joinpath("questions_v2.json").open(
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Bullshit benchmark payload must be a JSON object")
        techniques = payload.get("techniques")
        if not isinstance(techniques, list):
            raise ValueError(
                "Bullshit benchmark payload must contain a top-level 'techniques' list"
            )

        rows: list[dict[str, Any]] = []
        for technique in techniques:
            if not isinstance(technique, dict):
                continue
            technique_id = str(technique.get("technique", "")).strip()
            technique_description = str(technique.get("description", "")).strip()
            questions = technique.get("questions", [])
            if not isinstance(questions, list):
                continue
            for question in questions:
                if not isinstance(question, dict):
                    continue
                required_fields = ("id", "question", "nonsensical_element", "domain")
                missing_fields = [field for field in required_fields if field not in question]
                if missing_fields:
                    raise ValueError(
                        f"Missing required field(s) {missing_fields} for technique={technique_id}"
                    )
                rows.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "nonsensical_element": question["nonsensical_element"],
                        "domain": question["domain"],
                        "technique": technique_id,
                        "technique_description": technique_description,
                        "is_control": bool(question.get("is_control", False)),
                    }
                )

        rows.sort(key=lambda row: str(row.get("id", "")))
        return [row for row in rows if not row.get("is_control", False)]


def load_benchmark_dataset(benchmark_name: str,
                          source: Optional[str] = None,
                          limit: Optional[int] = None,
                          *,
                          registry: Optional["BenchmarkRegistry"] = None,
                          **kwargs) -> list[dict[str, Any]]:
    """Load dataset for a specific benchmark.

    Args:
        benchmark_name: Name of the benchmark
        source: Optional custom source (file, URL, or HF dataset)
        limit: Maximum number of samples to load
        registry: Registry for resolving a benchmark's configured defaults
        **kwargs: Additional benchmark-specific parameters

    Returns:
        List of dataset items, in the benchmark's own fields: the registry's field_mappings are applied here,
        so every caller (both run routes and the CLIs) gets the shape its evaluator formats (TASK-15487).
    """
    from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry

    config = (registry if registry is not None else get_registry()).get(benchmark_name)
    registered_source = config.dataset_source if config else None
    if benchmark_name == "simpleqa_verified" and not (source or registered_source):
        return []

    # Map benchmark names to loaders
    loaders = {
        "mmlu_pro": BenchmarkDatasetLoader.load_mmlu_pro,
        "simple_bench": BenchmarkDatasetLoader.load_simple_bench,
        "simpleqa": lambda s: BenchmarkDatasetLoader.load_simpleqa(s or registered_source),
        "simpleqa_verified": lambda s: BenchmarkDatasetLoader.load_simpleqa(s or registered_source),
        "bullshit_benchmark": BenchmarkDatasetLoader.load_bullshit_benchmark,
        "gpqa": BenchmarkDatasetLoader.load_gpqa,
        "gpqa_diamond": lambda s: BenchmarkDatasetLoader.load_gpqa(s, subset="diamond"),
    }

    if benchmark_name in loaders:
        data = loaders[benchmark_name](source)
    else:
        # Configured sources need not have file extensions. An explicit source keeps its own format.
        format_hint = None
        if not source and config:
            source = config.dataset_source
            format_hint = config.dataset_format
        if not source:
            logger.error(f"No source specified for unknown benchmark: {benchmark_name}")
            return []

        if source.startswith("huggingface://") or format_hint == "huggingface":
            dataset_id = source.removeprefix("huggingface://")
            data = DatasetLoader.load_huggingface(dataset_id)
        elif format_hint == "jsonl" or (format_hint is None and source.endswith('.jsonl')):
            data = DatasetLoader.load_jsonl(source)
        elif format_hint == "json" or (format_hint is None and source.endswith('.json')):
            data = DatasetLoader.load_json(source)
        elif format_hint == "csv" or (format_hint is None and source.endswith('.csv')):
            data = DatasetLoader.load_csv(source)
        else:
            logger.error(f"Cannot determine format for source: {source}")
            return []

    if config and config.field_mappings:
        # Merge, not replace: a loader that already shaped its rows (simple_bench) keeps them
        data = [
            {**item, **{target: item[field] for target, field in config.field_mappings.items() if field in item}}
            for item in data
        ]

    # Apply limit if specified
    if limit and limit > 0:
        data = data[:limit]

    return data


def validate_dataset_format(data: list[dict[str, Any]],
                           required_fields: list[str]) -> tuple[bool, list[str]]:
    """Validate that dataset has required fields.

    Args:
        data: Dataset to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    if not data:
        return False, ["Dataset is empty"]

    # Check first item for fields
    first_item = data[0]
    missing_fields = []

    for field in required_fields:
        if field not in first_item:
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields

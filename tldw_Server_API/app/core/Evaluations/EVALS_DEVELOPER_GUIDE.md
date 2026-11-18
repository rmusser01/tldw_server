# Evaluations Module - Developer Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Adding New Evaluation Types](#adding-new-evaluation-types)
4. [Creating Custom Benchmarks](#creating-custom-benchmarks)
5. [Extending the System](#extending-the-system)
6. [API Reference](#api-reference)
7. [Testing Guidelines](#testing-guidelines)
8. [Performance Optimization](#performance-optimization)
9. [Contributing](#contributing)

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                     CLI Interface                        │
│                  (cli/evals_cli.py)                     │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  Benchmark Registry                      │
│              (benchmark_registry.py)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • Benchmark Registration                        │   │
│  │ • Configuration Management                      │   │
│  │ • Evaluator Factory                            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────┬───────────────────┘
          │                           │
┌─────────▼──────────┐      ┌────────▼──────────────────┐
│  Dataset Loaders   │      │   Evaluation Types        │
│(benchmark_loaders) │      │  (benchmark_utils.py)     │
│ • JSON/JSONL       │      │  • BaseEvaluation         │
│ • CSV              │      │  • MultipleChoice         │
│ • HuggingFace      │      │  • CodeGeneration         │
│ • Custom           │      │  • InstructionFollowing   │
└────────────────────┘      │  • NextTokenCapture       │
                            └──────────┬──────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Evaluation Manager             │
                    │    (evaluation_manager.py)          │
                    │  • Result Storage                   │
                    │  • History Tracking                 │
                    │  • Metric Calculation               │
                    └──────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Extensibility**: Easy to add new evaluation types and benchmarks
3. **Configuration-Driven**: YAML configs define behavior
4. **Type Safety**: Pydantic models for validation
5. **Async Support**: Efficient parallel processing

## Core Components

### 1. BaseEvaluation Class

The foundation for all evaluation types:

```python
from benchmark_utils import BaseEvaluation
from typing import Dict, Any

class CustomEvaluation(BaseEvaluation):
    """Custom evaluation implementation."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, "Custom evaluation description")
        # Initialize custom parameters
        self.custom_param = kwargs.get('custom_param', 'default')

    def format_for_custom_metric(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for evaluation."""
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_prompt": self._create_prompt(input_data),
            "input_data": self._prepare_input(input_data),
            "scoring_criteria": self._define_criteria(),
            "metadata": {"type": "custom"}
        }

    def parse_response(self, response: str) -> Any:
        """Parse model response."""
        # Custom parsing logic
        return parsed_data

    def score(self, predicted: Any, expected: Any) -> float:
        """Calculate score."""
        # Custom scoring logic
        return score
```

### 2. BenchmarkConfig

Configuration structure for benchmarks:

```python
from benchmark_registry import BenchmarkConfig

config = BenchmarkConfig(
    name="my_benchmark",
    description="Description",
    evaluation_type="multiple_choice",  # Must match registered type
    dataset_source="path/to/data.json",
    dataset_format="json",
    field_mappings={
        "question": "q",  # Map your field names
        "answer": "a"
    },
    evaluation_params={
        "custom_param": "value"
    },
    metadata={
        "version": "1.0",
        "author": "Name"
    }
)
```

### 3. Dataset Loaders

Implementing custom dataset loaders:

```python
from benchmark_loaders import DatasetLoader

class CustomFormatLoader(DatasetLoader):
    """Loader for custom data format."""

    @staticmethod
    def load(source: str) -> List[Dict[str, Any]]:
        """Load data from custom format."""
        # Custom loading logic
        data = []
        with open(source) as f:
            # Parse custom format
            pass
        return data
```

### 4. Evaluation Manager

Managing evaluation lifecycle:

```python
from evaluation_manager import EvaluationManager

manager = EvaluationManager()

# Store evaluation results
eval_id = await manager.store_evaluation(
    evaluation_type="benchmark_name",
    input_data={"questions": [...]},
    results={"scores": [...], "metrics": {...}},
    metadata={"model": "gpt-4", "timestamp": "..."}
)

# Retrieve history
history = await manager.get_history(
    evaluation_type="benchmark_name",
    limit=100
)
```

## Adding New Evaluation Types

### Step 1: Create Evaluation Class

```python
# In benchmark_utils.py or separate module

class DialogueEvaluation(BaseEvaluation):
    """Evaluation for dialogue quality."""

    def __init__(self, name: str = "dialogue",
                 max_turns: int = 10):
        super().__init__(name, "Dialogue evaluation")
        self.max_turns = max_turns

    def format_for_custom_metric(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        conversation = input_data.get("conversation", [])

        evaluation_prompt = f"""
        Evaluate this dialogue for:
        1. Coherence across turns
        2. Appropriate responses
        3. Context maintenance

        Dialogue:
        {self._format_conversation(conversation)}

        Model Response:
        {{response}}

        Score 0.0-1.0 with explanation.
        """

        return {
            "name": f"{self.name}_quality",
            "description": self.description,
            "evaluation_prompt": evaluation_prompt,
            "input_data": {
                "conversation": conversation,
                "response": "{model_response}"
            },
            "scoring_criteria": {
                "coherence": "Maintains context",
                "appropriateness": "Suitable responses",
                "engagement": "Engaging dialogue"
            }
        }
```

### Step 2: Register with Registry

```python
# In benchmark_registry.py

class BenchmarkRegistry:
    EVALUATION_TYPES = {
        # ... existing types ...
        "dialogue": DialogueEvaluation,  # Add new type
    }
```

### Step 3: Create Configuration

```yaml
# configs/dialogue_benchmark.yaml
name: dialogue_benchmark
description: Dialogue quality evaluation
evaluation_type: dialogue
dataset_source: ./data/dialogues.json
dataset_format: json
field_mappings:
  conversation: dialogue_turns
  expected_response: gold_response
evaluation_params:
  max_turns: 10
```

## Creating Custom Benchmarks

### Complete Example: Math Reasoning Benchmark

```python
# math_reasoning_benchmark.py

from typing import List, Dict, Any
import json
from benchmark_utils import BaseEvaluation

class MathReasoningEvaluation(BaseEvaluation):
    """Evaluation for mathematical reasoning."""

    def __init__(self, name: str = "math_reasoning"):
        super().__init__(name, "Mathematical reasoning evaluation")

    def format_for_custom_metric(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        problem = input_data.get("problem", "")
        solution_steps = input_data.get("solution_steps", [])
        final_answer = input_data.get("final_answer", "")

        evaluation_prompt = f"""
        Evaluate this mathematical solution:

        Problem: {problem}

        Expected Steps:
        {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))}

        Expected Answer: {final_answer}

        Model Solution:
        {{response}}

        Evaluate based on:
        1. Correct reasoning steps (40%)
        2. Mathematical accuracy (40%)
        3. Final answer (20%)

        SCORE: X.X EXPLANATION: [detailed evaluation]
        """

        return {
            "name": self.name,
            "description": self.description,
            "evaluation_prompt": evaluation_prompt,
            "input_data": input_data,
            "scoring_criteria": {
                "reasoning": "Logical step progression",
                "accuracy": "Mathematical correctness",
                "answer": "Correct final answer"
            }
        }

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse mathematical solution."""
        import re

        # Extract final answer
        answer_pattern = r'(?:answer|result).*?(\d+\.?\d*)'
        match = re.search(answer_pattern, response, re.IGNORECASE)

        return {
            "full_solution": response,
            "final_answer": match.group(1) if match else None,
            "has_steps": '\n' in response or 'step' in response.lower()
        }

    def score(self, parsed: Dict[str, Any], expected: str) -> float:
        """Score mathematical solution."""
        score = 0.0

        # Check final answer
        if parsed.get("final_answer") == expected:
            score += 0.5

        # Check for reasoning steps
        if parsed.get("has_steps"):
            score += 0.3

        # Basic solution presence
        if len(parsed.get("full_solution", "")) > 50:
            score += 0.2

        return min(score, 1.0)


# Dataset generator
def create_math_dataset():
    """Generate math reasoning dataset."""
    problems = [
        {
            "problem": "If a train travels 120 miles in 2 hours, what is its average speed?",
            "solution_steps": [
                "Identify given values: distance = 120 miles, time = 2 hours",
                "Apply formula: speed = distance / time",
                "Calculate: 120 / 2 = 60"
            ],
            "final_answer": "60",
            "category": "rate_problems",
            "difficulty": "easy"
        },
        # Add more problems...
    ]

    return {
        "benchmark_name": "Math Reasoning Benchmark",
        "version": "1.0",
        "problems": problems
    }


# Runner script
async def run_math_benchmark(api_name: str = "openai"):
    """Run the math reasoning benchmark."""
    from benchmark_registry import get_registry
    from evaluation_manager import EvaluationManager

    # Register benchmark
    registry = get_registry()
    config = BenchmarkConfig(
        name="math_reasoning",
        description="Mathematical reasoning problems",
        evaluation_type="math_reasoning",
        dataset_source="./data/math_problems.json",
        dataset_format="json",
        field_mappings={
            "question": "problem",
            "answer": "final_answer"
        }
    )
    registry.register(config)

    # Load dataset
    dataset = create_math_dataset()

    # Run evaluation
    evaluator = MathReasoningEvaluation()
    manager = EvaluationManager()

    results = []
    for problem in dataset["problems"]:
        # Get model response (implement your API call)
        model_response = await get_model_response(problem["problem"], api_name)

        # Parse and score
        parsed = evaluator.parse_response(model_response)
        score = evaluator.score(parsed, problem["final_answer"])

        results.append({
            "problem": problem["problem"],
            "score": score,
            "model_response": model_response,
            "expected": problem["final_answer"]
        })

    # Store results
    await manager.store_evaluation(
        evaluation_type="math_reasoning",
        input_data={"problems": dataset["problems"]},
        results={"scores": results},
        metadata={"api": api_name}
    )

    return results
```

## Extending the System

### Adding New Scoring Methods

```python
# custom_scoring.py

from typing import Any

class CustomScorer:
    """Custom scoring implementation."""

    @staticmethod
    def score_semantic_similarity(text1: str, text2: str) -> float:
        """Score using semantic similarity."""
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding1 = model.encode(text1)
        embedding2 = model.encode(text2)

        similarity = util.cos_sim(embedding1, embedding2).item()
        return similarity

    @staticmethod
    def score_code_execution(code: str, test_cases: List[Dict]) -> float:
        """Score by executing code against tests."""
        import subprocess
        import tempfile

        passed = 0
        for test in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
                f.write(code)
                f.write(f"\n\n# Test\nprint({test['input']})")
                f.flush()

                try:
                    result = subprocess.run(
                        ['python', f.name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.stdout.strip() == str(test['output']):
                        passed += 1
                except:
                    pass

        return passed / len(test_cases) if test_cases else 0
```

### Adding New Data Sources

```python
# custom_data_source.py

from typing import List, Dict, Any
import json
import urllib.parse
import urllib.request

class APIDataSource:
    """Load datasets from API endpoints."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def load_from_api(self, endpoint: str, params: Dict = None) -> List[Dict[str, Any]]:
        """Load dataset from API."""
        url = endpoint
        if params:
            url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Transform to standard format
        return self._transform_api_data(data)

    def _transform_api_data(self, data: Any) -> List[Dict[str, Any]]:
        """Transform API data to standard format."""
        # Implementation depends on API structure
        pass


class DatabaseDataSource:
    """Load datasets from database."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def load_from_query(self, query: str) -> List[Dict[str, Any]]:
        """Load dataset from SQL query."""
        import sqlite3

        conn = sqlite3.connect(self.connection_string)
        cursor = conn.execute(query)

        columns = [desc[0] for desc in cursor.description]
        data = []

        for row in cursor:
            data.append(dict(zip(columns, row)))

        conn.close()
        return data
```

### Custom Result Analyzers

```python
# custom_analyzers.py

import numpy as np
from typing import List, Dict, Any

class AdvancedAnalyzer:
    """Advanced analysis tools."""

    @staticmethod
    def calculate_confidence_interval(scores: List[float],
                                     confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for scores."""
        import scipy.stats as stats

        mean = np.mean(scores)
        std_err = stats.sem(scores)
        interval = stats.t.interval(
            confidence,
            len(scores) - 1,
            loc=mean,
            scale=std_err
        )
        return interval

    @staticmethod
    def perform_regression_analysis(results: List[Dict[str, Any]]) -> Dict:
        """Analyze score trends."""
        from sklearn.linear_model import LinearRegression

        # Extract features and scores
        X = []  # Features
        y = []  # Scores

        for r in results:
            # Extract relevant features
            features = [
                len(r.get('question', '')),
                r.get('difficulty_numeric', 1),
                # Add more features
            ]
            X.append(features)
            y.append(r['score'])

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        return {
            'coefficients': model.coef_.tolist(),
            'intercept': model.intercept_,
            'r_squared': model.score(X, y)
        }

    @staticmethod
    def cluster_errors(failed_results: List[Dict[str, Any]],
                       n_clusters: int = 5) -> Dict:
        """Cluster similar errors."""
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer

        # Encode error texts
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [r['model_answer'] for r in failed_results]
        embeddings = model.encode(texts)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)

        # Group by cluster
        clustered = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered:
                clustered[cluster_id] = []
            clustered[cluster_id].append(failed_results[i])

        return clustered
```

## API Reference

### Core Classes

#### BaseEvaluation
```python
class BaseEvaluation:
    def __init__(self, name: str, description: str = "")
    def format_for_custom_metric(self, input_data: Dict[str, Any]) -> Dict[str, Any]
    def parse_response(self, response: str) -> Any
    def score(self, predicted: Any, expected: Any) -> float
```

#### BenchmarkRegistry
```python
class BenchmarkRegistry:
    def register(self, config: BenchmarkConfig) -> None
    def unregister(self, name: str) -> None
    def get(self, name: str) -> Optional[BenchmarkConfig]
    def list_benchmarks(self) -> List[str]
    def create_evaluator(self, benchmark_name: str) -> Optional[BaseEvaluation]
    def load_dataset(self, benchmark_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]
```

#### EvaluationManager
```python
class EvaluationManager:
    async def store_evaluation(self, evaluation_type: str, input_data: Dict[str, Any],
                              results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str
    async def get_history(self, evaluation_type: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: int = 50, offset: int = 0) -> Dict[str, Any]
    async def compare_evaluations(self, evaluation_ids: List[str],
                                 metrics_to_compare: Optional[List[str]] = None) -> Dict[str, Any]
    async def evaluate_custom_metric(self, metric_name: str, description: str,
                                    evaluation_prompt: str, input_data: Dict[str, Any],
                                    scoring_criteria: Dict[str, Any],
                                    api_name: str = "openai") -> Dict[str, Any]
```

### Utility Functions

```python
# benchmark_utils.py
def parse_multiple_choice_answer(response: str, choices: List[str] = None) -> str
def score_multiple_choice(predicted: str, correct: Union[str, int], choices: List[str] = None) -> float
def parse_function_call(response: str) -> Dict[str, Any]
def validate_function_call(predicted: Dict[str, Any], expected: Dict[str, Any],
                          strict_params: bool = False) -> float
def calculate_accuracy_by_category(results: List[Dict[str, Any]],
                                 category_field: str = "category") -> Dict[str, Dict[str, float]]
def format_benchmark_summary(results: List[Dict[str, Any]], benchmark_name: str) -> str

# benchmark_loaders.py
def load_dataset_from_json(file_path: str) -> List[Dict[str, Any]]
def load_dataset_from_jsonl(file_path: str) -> List[Dict[str, Any]]
def load_dataset_from_url(url: str, format: str = "auto") -> List[Dict[str, Any]]
def load_benchmark_dataset(benchmark_name: str, source: Optional[str] = None,
                          limit: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]
def validate_dataset_format(data: List[Dict[str, Any]],
                           required_fields: List[str]) -> Tuple[bool, List[str]]
```

## Testing Guidelines

### Unit Tests

```python
# test_evaluation_types.py

import pytest
from benchmark_utils import MultipleChoiceEvaluation

class TestMultipleChoiceEvaluation:
    def test_parse_answer(self):
        evaluator = MultipleChoiceEvaluation()

        # Test various formats
        assert evaluator.parse_response("A") == "A"
        assert evaluator.parse_response("The answer is B") == "B"
        assert evaluator.parse_response("(C)") == "C"
        assert evaluator.parse_response("D.") == "D"

    def test_scoring(self):
        evaluator = MultipleChoiceEvaluation()

        assert evaluator.score("A", "A") == 1.0
        assert evaluator.score("B", "A") == 0.0
        assert evaluator.score("a", "A") == 1.0  # Case insensitive

    def test_format_for_metric(self):
        evaluator = MultipleChoiceEvaluation()

        data = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "correct_answer": 1
        }

        formatted = evaluator.format_for_custom_metric(data)

        assert "question" in formatted["input_data"]
        assert "correct_answer" in formatted["input_data"]
        assert formatted["input_data"]["correct_answer"] == "B"
```

### Integration Tests

```python
# test_benchmark_integration.py

import pytest
import asyncio
from benchmark_registry import get_registry, BenchmarkConfig
from evaluation_manager import EvaluationManager

@pytest.mark.asyncio
async def test_full_benchmark_flow():
    """Test complete benchmark execution flow."""

    # Register benchmark
    registry = get_registry()
    config = BenchmarkConfig(
        name="test_benchmark",
        description="Test",
        evaluation_type="multiple_choice",
        dataset_source="test_data.json",
        dataset_format="json",
        field_mappings={"question": "q", "answer": "a"},
        evaluation_params={}
    )
    registry.register(config)

    # Create evaluator
    evaluator = registry.create_evaluator("test_benchmark")
    assert evaluator is not None

    # Run evaluation
    manager = EvaluationManager()
    test_data = {"question": "Test?", "choices": ["A", "B"], "correct_answer": 0}

    formatted = evaluator.format_for_custom_metric(test_data)

    # Store results
    eval_id = await manager.store_evaluation(
        evaluation_type="test_benchmark",
        input_data=test_data,
        results={"score": 1.0}
    )

    assert eval_id is not None

    # Retrieve history
    history = await manager.get_history(evaluation_type="test_benchmark")
    assert history["total_count"] > 0
```

### Performance Tests

```python
# test_performance.py

import time
import pytest
from benchmark_loaders import load_dataset_from_json

def test_large_dataset_loading(benchmark):
    """Test loading performance with large datasets."""

    def load_large_dataset():
        # Create test dataset
        data = [{"q": f"Question {i}", "a": f"Answer {i}"}
                for i in range(10000)]

        # Time loading
        start = time.time()
        loaded = load_dataset_from_json("large_test.json")
        return time.time() - start

    # Use pytest-benchmark
    result = benchmark(load_large_dataset)
    assert result < 1.0  # Should load in under 1 second
```

## Performance Optimization

### Batch Processing

```python
# Efficient batch processing
async def process_batch(items: List[Dict], evaluator, api_client, batch_size: int = 10):
    """Process items in batches."""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]

        # Create async tasks
        tasks = [
            evaluate_single(item, evaluator, api_client)
            for item in batch
        ]

        # Run in parallel
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    return results
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedEvaluator:
    """Evaluator with caching."""

    def __init__(self):
        self.cache = {}

    def _cache_key(self, input_data: Dict) -> str:
        """Generate cache key."""
        data_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def evaluate(self, input_data: Dict) -> float:
        """Evaluate with caching."""
        key = self._cache_key(input_data)

        if key in self.cache:
            return self.cache[key]

        # Perform evaluation
        result = self._evaluate_impl(input_data)
        self.cache[key] = result

        return result
```

### Memory Management

```python
# Stream processing for large datasets
def process_large_dataset_streaming(file_path: str, evaluator):
    """Process large dataset with streaming."""

    with open(file_path) as f:
        for line in f:
            item = json.loads(line)

            # Process single item
            result = evaluator.evaluate(item)

            # Yield result instead of accumulating
            yield result

            # Explicit cleanup if needed
            del item
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/tldw_server
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev,evals]"

# Run tests
pytest tests/Evaluations/ -v

# Run linting
black app/core/Evaluations/
flake8 app/core/Evaluations/
mypy app/core/Evaluations/
```

### Code Style Guidelines

1. **Type Hints**: Always use type hints
2. **Docstrings**: Google-style docstrings for all public methods
3. **Error Handling**: Explicit error messages with context
4. **Logging**: Use loguru for consistent logging
5. **Testing**: Minimum 80% code coverage

### Submitting Benchmarks

1. Create benchmark configuration in `configs/`
2. Add dataset in `data/` or provide loader
3. Write tests in `tests/`
4. Update documentation
5. Submit pull request with:
   - Benchmark description
   - Performance metrics
   - Example results

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New benchmark
- [ ] New evaluation type
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Documentation updated

## Benchmark Details (if applicable)
- Dataset size:
- Evaluation type:
- Expected runtime:
- Memory requirements:

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation added/updated
```

## Advanced Topics

### Multi-Model Evaluation

```python
# Evaluate multiple models simultaneously
async def compare_models(benchmark_name: str, models: List[str]):
    """Run benchmark on multiple models."""
    dataset = load_benchmark_dataset(benchmark_name)

    results = {}
    for model in models:
        model_results = await run_benchmark(
            dataset,
            model_name=model
        )
        results[model] = model_results

    # Generate comparison
    comparison = analyze_model_differences(results)
    return comparison
```

### Adaptive Evaluation

```python
# Adjust difficulty based on performance
class AdaptiveEvaluator:
    """Evaluator that adapts to model performance."""

    def __init__(self):
        self.performance_history = []
        self.current_difficulty = "medium"

    def select_next_question(self, question_pool: Dict[str, List]):
        """Select question based on performance."""
        if len(self.performance_history) < 5:
            return random.choice(question_pool[self.current_difficulty])

        recent_performance = np.mean(self.performance_history[-5:])

        if recent_performance > 0.8:
            self.current_difficulty = "hard"
        elif recent_performance < 0.5:
            self.current_difficulty = "easy"
        else:
            self.current_difficulty = "medium"

        return random.choice(question_pool[self.current_difficulty])
```

### Real-time Monitoring

```python
# Monitor evaluation progress in real-time
class EvaluationMonitor:
    """Real-time evaluation monitoring."""

    def __init__(self):
        self.start_time = None
        self.processed = 0
        self.total = 0
        self.current_score = 0

    def start(self, total_items: int):
        """Start monitoring."""
        self.start_time = time.time()
        self.total = total_items
        self.processed = 0

    def update(self, score: float):
        """Update progress."""
        self.processed += 1
        self.current_score = (
            (self.current_score * (self.processed - 1) + score)
            / self.processed
        )

        # Calculate metrics
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed
        eta = (self.total - self.processed) / rate

        # Log progress
        logger.info(
            f"Progress: {self.processed}/{self.total} "
            f"({self.processed/self.total:.1%}) | "
            f"Score: {self.current_score:.3f} | "
            f"Rate: {rate:.1f}/s | "
            f"ETA: {eta:.0f}s"
        )
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors | Missing dependencies | `pip install -e ".[evals]"` |
| Dataset not loading | Wrong path or format | Check path and validate format |
| Low scores | Prompt mismatch | Adjust prompt template |
| Memory errors | Large dataset | Use streaming or batching |
| Slow evaluation | Serial processing | Enable parallel processing |

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
export EVALS_DEBUG=1
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_evaluation():
    """Profile evaluation performance."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Run evaluation
    run_benchmark("test_benchmark")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

## Resources

- [API Documentation](api_docs.md)
- [Benchmark Catalog](benchmarks.md)
- [Example Notebooks](notebooks/)
- [Community Forum](https://forum.example.com)
- [Issue Tracker](https://github.com/your-repo/issues)

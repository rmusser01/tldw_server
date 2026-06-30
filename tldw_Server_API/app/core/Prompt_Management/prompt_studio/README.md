# Prompt Studio Documentation

## Overview

Prompt Studio is a comprehensive prompt engineering platform integrated into tldw_server. It combines DSPy's programmatic optimization capabilities with Anthropic Console's testing features to provide a powerful environment for developing, testing, and optimizing LLM prompts.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Features

### Core Capabilities
- **Multi-Provider Support**: Integration with 15+ LLM providers (OpenAI, Anthropic, Groq, Mistral, etc.)
- **Test Management**: Create, import/export, and generate test cases
- **Evaluation System**: Comprehensive metrics for prompt performance assessment
- **Optimization Engine**: 5 different optimization strategies (MIPRO, Bootstrap, Genetic, etc.)
- **Real-time Updates**: WebSocket support for live progress tracking
- **Job Queue**: Async processing via core Jobs with retry logic and prioritization
- **Version Control**: Full prompt versioning with history tracking

### Key Components
1. **Projects & Signatures**: Organize prompts with DSPy-style signatures
2. **Test Cases**: Comprehensive test case management with golden sets
3. **Evaluations**: Run prompts against test cases with multiple metrics
4. **Optimizations**: Automated prompt improvement strategies
5. **Monitoring**: Built-in metrics and health checks

## Architecture

```
prompt_studio/
├── Database Layer
│   ├── PromptStudioDatabase.py     # Extended database operations
│   └── Schemas (8 tables)           # Projects, prompts, tests, etc.
├── Core Components
│   ├── prompt_executor.py          # LLM execution engine
│   ├── evaluation_metrics.py       # Metric calculations
│   ├── test_runner.py              # Test execution
│   └── optimization_engine.py      # Optimization strategies
├── Management Systems
│   ├── test_case_manager.py        # Test CRUD operations
│   ├── jobs_adapter.py             # Core Jobs adapter (job lifecycle/status)
│   ├── job_types.py                # Shared job type/status enums
│   └── event_broadcaster.py        # Real-time events
├── API Layer
│   ├── endpoints/                  # FastAPI endpoints
│   └── schemas/                     # Pydantic models
└── Supporting Systems
    ├── auth_permissions.py         # Authentication & authorization (deprecated; use core AuthNZ/JWT + RBAC)
    ├── monitoring.py                # Metrics & health checks
    └── evaluation_reports.py        # Report generation
```

## Installation

### Prerequisites
- Install the project requirements:
```bash
pip install -e .[dev]
```

- Optional extras:
  - Redis: enables distributed/shared rate limiting and background coordination via the core AuthNZ limiter
  - Playwright and other tools depending on your workflows (see repository requirements)

### Database Setup
The database tables are automatically created on first use. To manually initialize:

```python
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

db = PromptStudioDatabase("path/to/prompts.db", "client_id")
# Tables are created automatically
```
For PostgreSQL deployments, provide a configured backend via `DB_Manager`. Idempotency mappings are supported and used by endpoints that accept an `Idempotency-Key` header.

## Quick Start

### 1. Create a Project
```python
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

db = PromptStudioDatabase("prompts.db", "my_client")

project_id = db.create_project(
    name="My First Project",
    description="Testing prompt optimization"
)
```

### 2. Define a Signature
```python
signature = {
    "name": "Summarization",
    "instruction": "Summarize the given text",
    "input_schema": [
        {"name": "text", "type": "string", "required": True},
        {"name": "max_words", "type": "integer", "required": False}
    ],
    "output_schema": [
        {"name": "summary", "type": "string"},
        {"name": "key_points", "type": "array"}
    ]
}

signature_id = db.create_signature(project_id, signature)
```

### 3. Create a Prompt
```python
prompt = {
    "name": "Basic Summarizer",
    "content": "Summarize this text in {max_words} words:\n\n{text}",
    "system_prompt": "You are a helpful assistant that creates concise summaries."
}

prompt_id = db.create_prompt(project_id, signature_id, prompt)
```

### 4. Create Test Cases
```python
from prompt_studio.test_case_manager import TestCaseManager

manager = TestCaseManager(db)

test_case = manager.create_test_case(
    project_id=project_id,
    name="News Article Test",
    inputs={
        "text": "Long news article text...",
        "max_words": 100
    },
    expected_outputs={
        "summary": "Expected summary...",
        "key_points": ["point1", "point2"]
    },
    is_golden=True
)
```

### 5. Run Evaluation
```python
from prompt_studio.test_runner import TestRunner

runner = TestRunner(db)

result = await runner.run_single_test(
    prompt_id=prompt_id,
    test_case_id=test_case["id"],
    model_config={
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "parameters": {"temperature": 0.7}
    }
)

print(f"Score: {result['scores']['aggregate_score']}")
```

### 6. Optimize Prompt
```python
from prompt_studio.optimization_engine import OptimizationEngine

engine = OptimizationEngine(db)

optimization_result = await engine.optimize(
    optimization_id=optimization_id  # Created via API
)

print(f"Improvement: {optimization_result['improvement']*100:.1f}%")
```

## MCTS Strategy (Canary)

The MCTS-based optimizer (optimizer_type="mcts") explores multi-step prompt sequences with UCT-guided tree search.
It is disabled by default. Enable in development or explicitly with flags:

- Set `PROMPT_STUDIO_ENABLE_MCTS=true` to enable everywhere, or
- Set `APP_ENV=dev` (or `ENVIRONMENT=dev`) and keep canary default `PROMPT_STUDIO_ENABLE_MCTS_CANARY=true`.

Knobs (strategy_params):
- `mcts_simulations`: number of simulations
- `mcts_max_depth`: maximum depth of the search tree
- `mcts_exploration_c`: UCT exploration constant
- `prompt_candidates_per_node`: candidate expansions per node
- `score_dedup_bin`: dedup bin size for scorer
- `ws_throttle_every`: throttle iteration WS events and persisted iterations (default ≈ n_sims/50)
- `trace_top_k`: how many top candidates to store in final trace

Observability:
- Iteration WS events are throttled; see `/api/v1/prompt-studio/ws`.
- Metrics: `prompt_studio.mcts.*` including best_reward and errors_total (see monitoring.py).
- Debug decisions: set `PROMPT_STUDIO_MCTS_DEBUG_DECISIONS=true` to include per-depth top scored candidates in the final trace and log UCT choices.

Final metrics include `tree_nodes`, `avg_branching`, `tokens_spent`, `duration_ms`, `best_reward`, error counters, `applied_params`, and a compact `trace`.

## Core Components

### Projects and Signatures

**Projects** organize related prompts and tests:
```python
project = {
    "name": "Customer Support Bot",
    "description": "Prompts for customer service",
    "tags": ["production", "support"],
    "config": {
        "default_model": "gpt-4",
        "max_test_cases": 100
    }
}
```

**Signatures** define input/output schemas (DSPy-style):
```python
signature = {
    "name": "QuestionAnswer",
    "instruction": "Answer the question based on context",
    "input_schema": [
        {"name": "question", "type": "string"},
        {"name": "context", "type": "string"}
    ],
    "output_schema": [
        {"name": "answer", "type": "string"},
        {"name": "confidence", "type": "number"}
    ]
}
```

### Test Case Management

```python
# Import from CSV
from prompt_studio.test_case_io import TestCaseIO

io = TestCaseIO(manager)
imported, errors = io.import_from_csv(
    project_id=project_id,
    csv_data=csv_content,
    auto_generate_names=True
)

# Generate test cases
from prompt_studio.test_case_generator import TestCaseGenerator

generator = TestCaseGenerator(manager)
cases = generator.generate_diverse_cases(
    project_id=project_id,
    signature_id=signature_id,
    num_cases=10
)

# Export to JSON
json_data = io.export_to_json(
    project_id=project_id,
    include_golden_only=True
)
```

### Evaluation Metrics

Available metrics:
- **Text Matching**: exact_match, fuzzy_match, levenshtein, token_overlap
- **Structured Data**: json_match, json_schema_valid
- **Classification**: accuracy, precision, recall, f1_score
- **Numerical**: mae, mse, rmse
- **Custom**: regex_match, contains, length_match

```python
from prompt_studio.evaluation_metrics import EvaluationMetrics, MetricType

metrics = EvaluationMetrics()
scores = metrics.evaluate(
    output="Generated text",
    expected="Expected text",
    metrics=[MetricType.FUZZY_MATCH, MetricType.TOKEN_OVERLAP]
)
```

### Optimization Strategies

#### MIPRO (Multi-Instruction Prompt Optimization)
```python
config = {
    "strategy": "mipro",
    "target_metric": "accuracy",
    "min_improvement": 0.01,
    "max_iterations": 20
}
```

#### Bootstrap Few-Shot
```python
config = {
    "strategy": "bootstrap",
    "num_examples": 3,
    "selection_strategy": "diverse"  # or "best", "random"
}
```

#### Hyperparameter Tuning
```python
config = {
    "strategy": "hyperparameter",
    "params_to_optimize": ["temperature", "max_tokens", "top_p"],
    "search_method": "bayesian"
}
```

#### Iterative Refinement
```python
config = {
    "strategy": "iterative",
    "max_iterations": 10
}
```

#### Genetic Algorithm
```python
config = {
    "strategy": "genetic",
    "population_size": 10,
    "generations": 20,
    "mutation_rate": 0.1
}
```

#### MCTS (Sequence Optimization)
Use Monte Carlo Tree Search inspired exploration to refine prompt sequences. The MCTS core performs full tree search with UCT selection, sibling deduplication via score bins, and contextual generation over decomposed segments. This MVP includes early stopping and optional pruning via a cheap heuristic quality scorer.

```python
config = {
    "optimizer_type": "mcts",  # or "strategy": "mcts" (legacy key)
    "max_iterations": 20,
    "target_metric": "accuracy",
    "strategy_params": {
        "mcts_simulations": 20,
        "mcts_max_depth": 4,
        "mcts_exploration_c": 1.4,
        "prompt_candidates_per_node": 3,
        "score_dedup_bin": 0.1,
        "early_stop_no_improve": 5,
        "token_budget": 50000,
        "min_quality": 0.0  # 0..10 (heuristic pruning)
    }
}
```

Notes:
- Progress is streamed via Prompt Studio WebSocket (iteration, current/best scores).
- Heuristic pruning is controlled by `min_quality` (0-10). Lower = fewer prunes.
- OpenAPI example for MCTS is available under `/api/v1/prompt-studio/optimizations/create`.

## API Reference

### Base URL
```
http://localhost:8000/api/v1/prompt_studio
```

### Endpoints

#### Projects
- `POST /projects/create` - Create new project
- `GET /projects/list` - List projects
- `GET /projects/{id}` - Get project details
- `PUT /projects/{id}` - Update project
- `DELETE /projects/{id}` - Delete project

#### Prompts
- `POST /prompts/create` - Create prompt
- `GET /prompts/{id}` - Get prompt
- `GET /prompts/{id}/versions` - Get version history
- `POST /prompts/{id}/revert` - Revert to version

#### Test Cases
- `POST /test_cases/create` - Create test case
- `POST /test_cases/bulk` - Bulk create
- `POST /test_cases/import` - Import from CSV/JSON
- `POST /test_cases/export/{project_id}` - Export test cases
- `POST /test_cases/generate` - Auto-generate tests

##### Program Evaluator Runner Flag (MVP)
To mark a test case as a code/program evaluation (non-executing MVP), include a `runner` hint in either `expected_outputs` or `inputs`:

```json
{
  "name": "Generate Python function",
  "inputs": { "prompt": "Write a function add(a,b).", "runner": "python" },
  "expected_outputs": { "runner": "python" }
}
```

When the environment flag `PROMPT_STUDIO_ENABLE_CODE_EVAL=true` is set, the TestRunner will enable the Program Evaluator for code-like outputs. Additionally, you can enable per project by setting `{"enable_code_eval": true}` in the project `metadata` field. The evaluator extracts Python code blocks from LLM output, runs them in a sandboxed subprocess (no network/files; import whitelist), and maps objective/constraints to a reward in [-1..10]. If disabled, a heuristic text-based reward is used instead.

#### Evaluations
- `POST /evaluations/create` - Create evaluation
- `GET /evaluations/{id}` - Get results
- `GET /evaluations/{id}/report` - Get report
- `POST /evaluations/compare` - Compare prompts

#### Optimizations
- `POST /optimizations/create` - Start optimization
- `GET /optimizations/{id}` - Get status
- `POST /optimizations/cancel/{id}` - Cancel
- `GET /optimizations/strategies` - List strategies

### Authentication

Include API key in headers:
```http
X-API-Key: pstudio_your_api_key_here
```

Or use JWT token:
```http
Authorization: Bearer your_jwt_token_here
```

## Usage Examples

### Complete Workflow Example

```python
import asyncio
from prompt_studio import PromptStudio

async def optimize_customer_support_prompt():
    # Initialize
    studio = PromptStudio(api_key="your_key")

    # Create project
    project = await studio.create_project(
        name="Customer Support Optimization",
        description="Improve response quality"
    )

    # Define signature
    signature = await studio.create_signature(
        project_id=project.id,
        name="SupportResponse",
        inputs=["customer_query", "product_info"],
        outputs=["response", "satisfaction_prediction"]
    )

    # Create initial prompt
    prompt = await studio.create_prompt(
        project_id=project.id,
        signature_id=signature.id,
        content="""
        Customer Query: {customer_query}
        Product Info: {product_info}

        Please provide a helpful response.
        """,
        system_prompt="You are a friendly customer support agent."
    )

    # Import test cases from CSV
    test_cases = await studio.import_test_cases(
        project_id=project.id,
        file_path="support_test_cases.csv"
    )

    # Run baseline evaluation
    baseline = await studio.evaluate(
        prompt_id=prompt.id,
        test_case_ids=[tc.id for tc in test_cases],
        model="gpt-3.5-turbo"
    )

    print(f"Baseline score: {baseline.overall_score:.2f}")

    # Optimize using MIPRO
    optimization = await studio.optimize(
        prompt_id=prompt.id,
        strategy="mipro",
        test_cases=test_cases,
        max_iterations=20
    )

    # Get optimized prompt
    optimized = await studio.get_prompt(optimization.optimized_prompt_id)

    # Run final evaluation
    final = await studio.evaluate(
        prompt_id=optimized.id,
        test_case_ids=[tc.id for tc in test_cases],
        model="gpt-3.5-turbo"
    )

    print(f"Final score: {final.overall_score:.2f}")
    print(f"Improvement: {(final.overall_score - baseline.overall_score)*100:.1f}%")

    # Generate report
    report = await studio.generate_report(
        evaluation_id=final.id,
        format="pdf"
    )

    print(f"Report saved to: {report.path}")

# Run
asyncio.run(optimize_customer_support_prompt())
```

### WebSocket Real-time Updates

```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8000/ws/prompt_studio?client_id=my_client');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'evaluation_progress':
            console.log(`Progress: ${data.data.progress}%`);
            break;
        case 'optimization_iteration':
            console.log(`Iteration ${data.data.iteration}: current=${data.data.current_metric}, best=${data.data.best_metric}`);
            break;
        case 'job_completed':
            console.log('Job completed!', data.data);
            break;
    }
};

// Subscribe to specific optimization
ws.send(JSON.stringify({
    type: 'subscribe',
    entity_type: 'optimization',
    entity_id: 123
}));
```

## Best Practices

### 1. Test Case Design
- **Golden Sets**: Mark high-quality test cases as golden for regression testing
- **Diversity**: Include edge cases, typical cases, and challenging examples
- **Balanced**: Ensure coverage of all input variations
- **Realistic**: Use real-world data when possible

### 2. Prompt Optimization
- **Start Simple**: Begin with basic prompts and iterate
- **Measure Everything**: Track metrics throughout optimization
- **Version Control**: Keep all prompt versions for rollback
- **A/B Testing**: Compare multiple variants simultaneously

### 3. Performance
- **Batch Processing**: Use bulk operations for efficiency
- **Caching**: Results are cached automatically
- **Rate Limiting**: Respect provider rate limits
- **Async Operations**: Use async/await for concurrent execution

### 4. Security
- **API Keys**: Store securely, never in code
- **Permissions**: Use role-based access control
- **Audit Logging**: All actions are logged
- **Data Privacy**: Sensitive data should be anonymized

## Troubleshooting

### Common Issues

#### Database Lock Errors
```python
# Use connection pooling
db = PromptStudioDatabase(
    db_path="prompts.db",
    client_id="client",
    pool_size=5
)
```

#### Rate Limiting
```python
# Configure retry logic
config = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "backoff_factor": 2.0
}
```

#### Memory Issues with Large Evaluations
```python
# Use streaming and pagination
results = runner.run_evaluation_streaming(
    evaluation_id=eval_id,
    batch_size=10,
    max_concurrent=3
)
```

### Debugging

Enable debug logging:
```python
from loguru import logger
logger.add("prompt_studio_debug.log", level="DEBUG")
```

Check metrics:
```python
from tldw_Server_API.app.core.Jobs.manager import JobManager
from prompt_studio.monitoring import PromptStudioHealthCheck

jm = JobManager()  # uses JOBS_DB_URL if configured
health = PromptStudioHealthCheck(db, job_manager=jm)
status = health.check_health()
print(json.dumps(status, indent=2))
```

## Configuration

### Environment Variables
```bash
# Database
PROMPT_STUDIO_DB_PATH=/path/to/prompts.db

# API Keys (for LLM providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Performance
PROMPT_STUDIO_MAX_WORKERS=4
PROMPT_STUDIO_CACHE_SIZE=1000

# Monitoring
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Prompt Studio job processing (core Jobs)
JOBS_DB_URL=postgresql://...      # Optional; enables Postgres-backed core Jobs
PROMPT_STUDIO_JOBS_QUEUE=default  # Queue for Prompt Studio jobs
PROMPT_STUDIO_JOBS_LEASE_SECONDS=60  # WorkerSDK lease duration
PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS=10  # Renew when lease is within threshold
PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS=5  # Random jitter added to renew interval

# Prompt Studio job quotas (per-user, via core Jobs)
PROMPT_STUDIO_MAX_CONCURRENT_JOBS=10  # Maps to JOBS_QUOTA_MAX_INFLIGHT_PROMPT_STUDIO
PROMPT_STUDIO_JOBS_MAX_QUEUED=100     # Maps to JOBS_QUOTA_MAX_QUEUED_PROMPT_STUDIO
PROMPT_STUDIO_JOBS_SUBMITS_PER_MIN=30 # Maps to JOBS_QUOTA_SUBMITS_PER_MIN_PROMPT_STUDIO
# Per-user overrides (optional):
# JOBS_QUOTA_MAX_INFLIGHT_PROMPT_STUDIO_USER_<user_id>=...
# Per-user AuthNZ policy overrides (preferred):
# limits.prompt_studio_max_concurrent_jobs
# limits.prompt_studio_max_queued_jobs
# limits.prompt_studio_submits_per_min

# Legacy prompt_studio_job_queue (kept for compatibility)
TLDW_PS_JOB_LEASE_SECONDS=60      # Legacy lease duration
TLDW_PS_HEARTBEAT_SECONDS=0       # Legacy heartbeat; if 0, derived from lease

# Shared rate limiting (via core AuthNZ limiter)
# Set REDIS_URL to enable Redis-backed distributed limits
REDIS_URL=redis://localhost:6379/0

# Test mode (bypasses certain checks; do not enable in production)
TEST_MODE=false
```

### Config File (prompt_studio_config.yaml)
```yaml
database:
  path: ./prompts.db
  pool_size: 5
  timeout: 30

execution:
  max_concurrent_tests: 10
  default_timeout: 60
  retry_attempts: 3

optimization:
  max_iterations: 50
  early_stopping_patience: 5
  min_improvement: 0.001

monitoring:
  metrics_enabled: true
  export_interval: 60
  health_check_interval: 30
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Part of tldw_server project - see main LICENSE file.
## Program Evaluator (Feature Flag)

The Program Evaluator executes code (runner="python") for test cases and maps objective/constraint success to a reward.
It is disabled by default and guarded by a feature flag and per-project controls.

Enable:
- `PROMPT_STUDIO_ENABLE_CODE_EVAL=true` to request global code evaluation.
- `PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL=true` to acknowledge that subprocess execution is unsafe without OS-level isolation.
- `PROMPT_STUDIO_ALLOW_PROJECT_CODE_EVAL=true` to let project metadata `{ "enable_code_eval": true }` request code evaluation.
- `PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL_IN_PRODUCTION=true` is additionally required in production-like environments (`tldw_production=true`, `ENVIRONMENT=production`, `APP_ENV=prod`, etc.).

Runtime knobs:
- `PROMPT_STUDIO_CODE_EVAL_TIMEOUT_MS` (wall timeout)
- `PROMPT_STUDIO_CODE_EVAL_MEM_MB` (memory ceiling)
- `PROMPT_STUDIO_CODE_EVAL_IMPORT_WHITELIST` (comma-separated module allowlist)

Runner convention:
- Test cases that should be executed must declare `runner="python"` and provide inputs/expected_outputs compatible with the evaluator.
- The evaluator extracts fenced code blocks or heuristically detects Python in the output.
- Execution uses a restricted subprocess with resource limits (best-effort, POSIX RLIMIT); no network/files; import whitelist only. This is not an OS-level sandbox.
- On non-POSIX (e.g., Windows), the sandbox is best-effort only; use extra caution.

Status and safety:
- If disabled, a heuristic text-based reward is used instead.
- Do not enable in untrusted multi-tenant contexts unless the deployment provides its own container, gVisor, microVM, or equivalent isolation boundary.
- Long-running loops are mitigated with CPU/memory/time limits; still use with care.

See also:
- Docs/Guides/Prompt_Studio_Program_Evaluator.md
- OpenAPI example `mcts_with_program_evaluator` under `/api/v1/prompt-studio/optimizations/create`.

## Strategy Overview (Pros/Cons & Cost)

| Strategy | Pros | Cons | Cost Expectation |
| --- | --- | --- | --- |
| iterative | Simple, predictable | May get stuck in local optima | Low-Medium |
| bootstrap | Improves few-shot grounding | Needs quality traces | Medium |
| genetic | Explores diverse space | Parameter-sensitive | Medium-High |
| mcts (canary) | Strong on hard tasks; sequence-aware | Higher token usage; latency | High |

For MCTS, token usage scales with simulations × depth × candidates. Use `token_budget`, `ws_throttle_every`, and caching to control costs.

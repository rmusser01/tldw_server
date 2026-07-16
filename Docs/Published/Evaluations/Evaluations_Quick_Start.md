# Evaluations Module - Quick Start Guide

Get started with evaluations in 5 minutes! This guide covers the essentials for using the tldw_server evaluation system.

## Prerequisites

### 1. Start the tldw_server
```bash
cd tldw_server
python -m uvicorn tldw_Server_API.app.main:app --reload
```

### 2. Verify API is accessible
```bash
curl http://localhost:8000/health
```

### 3. Check API documentation
Visit http://localhost:8000/docs for interactive API documentation.

## Authentication

The tldw_server supports two authentication modes:

### Single-User Mode (Default for Development)
```bash
# Generate a strong key once and set SINGLE_USER_API_KEY (new format)
python -m tldw_Server_API.app.core.AuthNZ.initialize
# Copy SINGLE_USER_API_KEY into your environment
export API_KEY="${SINGLE_USER_API_KEY}"
```

### Multi-User Mode (Production)
```bash
# Use your personal API key
export API_KEY="your-personal-api-key"
```

**Note**: Authentication mode is configured in `tldw_Server_API/Config_Files/config.txt` or via environment variables.

## Your First Evaluation (No LLM Required)

### 1. Exact Match Evaluation

Test simple string matching without needing any LLM API keys:

```python
import requests
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "http://localhost:8000"

# Create an exact match evaluation
eval_data = {
    "name": "capital_cities_test",
    "eval_type": "exact_match",
    "eval_spec": {
        "threshold": 1.0  # Require 100% match
    },
    "dataset": [  # You MUST provide either 'dataset' or 'dataset_id'
        {
            "input": {"output": "Paris"},
            "expected": {"output": "Paris"}  # Will pass
        },
        {
            "input": {"output": "London"},
            "expected": {"output": "London"}  # Will pass
        },
        {
            "input": {"output": "Berlin"},
            "expected": {"output": "Berlin"}  # Will pass
        },
        {
            "input": {"output": "Rome"},
            "expected": {"output": "Madrid"}  # Will fail
        }
    ]
}

# Create the evaluation
response = requests.post(
    f"{BASE_URL}/api/v1/evaluations",
    json=eval_data,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

if response.status_code == 201:
    eval_id = response.json()["id"]
    print(f"✅ Created evaluation: {eval_id}")
else:
    print(f"❌ Error: {response.status_code} - {response.json()}")
    exit(1)

# Run the evaluation
run_data = {
    "target_model": "none",  # Not evaluating a model, just data
    "config": {
        "temperature": 0,
        "max_workers": 4,
        "timeout_seconds": 30
    }
}

run_response = requests.post(
    f"{BASE_URL}/api/v1/evaluations/{eval_id}/runs",
    json=run_data,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

if run_response.status_code == 202:
    run_id = run_response.json()["id"]
    print(f"🏃 Started run: {run_id}")
else:
    print(f"❌ Error starting run: {run_response.json()}")
    exit(1)

# Check run status
import time
time.sleep(2)  # Wait for evaluation to complete

status_response = requests.get(
    f"{BASE_URL}/api/v1/evaluations/runs/{run_id}",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

status_data = status_response.json()
print(f"📊 Status: {status_data['status']}")

# Get results if completed (included in run payload)
if status_data["status"] == "completed":
    aggregate = status_data["results"]["aggregate"]
    print(f"\n📊 Results:")
    print(f"  Pass Rate: {aggregate['pass_rate']:.0%}")
    print(f"  Mean Score: {aggregate['mean_score']:.2f}")
    print(f"  Total Samples: {aggregate['total_samples']}")
    print(f"  Failed Samples: {aggregate['failed_samples']}")
else:
    print(f"⏳ Run status: {status_data['status']}")
    print(f"   Progress: {status_data.get('progress', {})}")
```

**Expected Output:**
```
✅ Created evaluation: eval_xxxxxxxxxxxx
🏃 Started run: run_xxxxxxxxxxxx
📊 Status: completed

📊 Results:
  Pass Rate: 75%
  Mean Score: 0.75
  Total Samples: 4
  Failed Samples: 1
```

## LLM-Based Evaluation (Requires API Keys)

### 2. Configure LLM API Keys

Add your API keys to `tldw_Server_API/Config_Files/config.txt`:

```ini
[API]
# Add the API keys for the providers you want to use
openai_api_key = sk-your-openai-key-here
anthropic_api_key = sk-ant-your-anthropic-key-here
groq_api_key = gsk_your_groq_key_here

# Optional: Configure default models
openai_model = gpt-4
anthropic_model = claude-sonnet-4-5
```

### 3. Summarization Quality Evaluation (G-Eval)

Evaluate summary quality using the G-Eval framework:

```python
# Evaluate a summary using G-Eval
summary_eval = {
    "name": "news_summary_quality",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "summarization",  # Required for model_graded
        "evaluator_model": "gpt-4",   # Model to use for evaluation
        "metrics": ["fluency", "relevance", "consistency", "coherence"],
        "threshold": 0.7  # Scores >= 0.7 pass
    },
    "dataset": [
        {
            "input": {
                "source_text": """
                The James Webb Space Telescope has captured unprecedented images
                of distant galaxies, revealing details about the early universe
                that were previously impossible to observe. The telescope's infrared
                capabilities allow it to see through cosmic dust and detect light
                from the first stars and galaxies that formed after the Big Bang.
                Scientists are particularly excited about the discovery of several
                galaxies that appear to be much larger and more mature than expected
                for their age, challenging current models of galaxy formation.
                """,
                "summary": "Webb telescope reveals surprising early universe details, showing unexpectedly large and mature galaxies."
            }
        }
    ]
}

response = requests.post(
    f"{BASE_URL}/api/v1/evaluations",
    json=summary_eval,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

if response.status_code == 201:
    eval_id = response.json()["id"]
    print(f"Created G-Eval evaluation: {eval_id}")

    # Run the evaluation
    run_response = requests.post(
        f"{BASE_URL}/api/v1/evaluations/{eval_id}/runs",
        json={"config": {"temperature": 0}},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    print(f"Started evaluation run: {run_response.json()['id']}")
else:
    print(f"Error: {response.json()}")
```

## Common Use Cases

### Check if Text Contains Keywords

```python
keyword_eval = {
    "name": "keyword_check",
    "eval_type": "includes",
    "eval_spec": {
        "threshold": 0.8  # Must include 80% of keywords
    },
    "dataset": [
        {
            "input": {"output": "The quick brown fox jumps over the lazy dog"},
            "expected": {"includes": ["quick", "fox", "dog"]}  # Will pass (3/3 = 100%)
        },
        {
            "input": {"output": "The cat sat on the mat"},
            "expected": {"includes": ["cat", "dog", "bird"]}  # Will fail (1/3 = 33%)
        }
    ]
}
```

### Fuzzy String Matching

```python
fuzzy_eval = {
    "name": "fuzzy_match_test",
    "eval_type": "fuzzy_match",
    "eval_spec": {
        "threshold": 0.85  # 85% similarity required (0-1 scale)
    },
    "dataset": [
        {
            "input": {"output": "The capital of France is Paris"},
            "expected": {"output": "Paris is the capital city of France"}  # Similar enough
        }
    ]
}
```

### RAG System Evaluation

```python
rag_eval = {
    "name": "rag_quality",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "rag",
        "evaluator_model": "gpt-4",  # Or "anthropic", "groq", etc.
        "metrics": ["relevance", "faithfulness"],
        "threshold": 0.75
    },
    "dataset": [
        {
            "input": {
                "query": "What is Python?",
                "contexts": [
                    "Python is a high-level, interpreted programming language.",
                    "It was created by Guido van Rossum and first released in 1991.",
                    "Python emphasizes code readability and simplicity."
                ],
                "response": "Python is a high-level programming language created by Guido van Rossum in 1991, known for its readability and simplicity."
            },
            "expected": {
                "answer": "Python is a high-level programming language"  # Optional ground truth
            }
        }
    ]
}
```

## Working with Datasets

### Create a Reusable Dataset

```python
# Create a dataset
dataset_request = {
    "name": "qa_test_dataset",
    "description": "Question-answer pairs for testing",
    "samples": [
        {
            "input": {"question": "What is 2+2?"},
            "expected": {"answer": "4"}
        },
        {
            "input": {"question": "What is the capital of Japan?"},
            "expected": {"answer": "Tokyo"}
        }
    ]
}

dataset_response = requests.post(
    f"{BASE_URL}/api/v1/evaluations/datasets",
    json=dataset_request,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

dataset_id = dataset_response.json()["id"]
print(f"Created dataset: {dataset_id}")

# Use the dataset in an evaluation
eval_with_dataset = {
    "name": "qa_evaluation",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset_id": dataset_id  # Reference the dataset instead of inline data
}

eval_response = requests.post(
    f"{BASE_URL}/api/v1/evaluations",
    json=eval_with_dataset,
    headers={"Authorization": f"Bearer {API_KEY}"}
)
```

## Understanding Results

All evaluation results include:

- **pass_rate**: Percentage of samples that met the threshold (0-1)
- **mean_score**: Average score across all samples (0-1)
- **std_dev**: Standard deviation showing consistency
- **total_samples**: Number of samples evaluated
- **failed_samples**: Number of samples below threshold

### Score Interpretation
- Scores range from 0 (worst) to 1 (best)
- The `threshold` in `eval_spec` determines pass/fail
- For multiple metrics, the average score is compared to threshold

```python
def print_results(run_id):
    """Pretty print evaluation results"""
    response = requests.get(
        f"{BASE_URL}/api/v1/evaluations/runs/{run_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )

    if response.status_code == 200:
        data = response.json()
        results = data["results"]["aggregate"]

        print(f"""
📊 Evaluation Results:
  ✅ Pass Rate: {results['pass_rate']:.1%}
  📈 Mean Score: {results['mean_score']:.3f} / 1.000
  📊 Std Dev: {results['std_dev']:.3f}
  📋 Total Samples: {results['total_samples']}
  ❌ Failed: {results['failed_samples']}
        """)

        # Show metric breakdown if available
        if "by_metric" in data["results"]:
            print("\n📊 Metrics Breakdown:")
            for metric, stats in data["results"]["by_metric"].items():
                print(f"  {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")
    else:
        print(f"Error getting results: {response.status_code}")
```

## Monitoring Progress

### Option 1: Polling

```python
def wait_for_completion(run_id, max_wait=60):
    """Poll for run completion"""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{BASE_URL}/api/v1/evaluations/runs/{run_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )

        data = response.json()
        status = data["status"]

        if status == "completed":
            return True
        elif status == "failed":
            print(f"Run failed: {data.get('error_message', 'Unknown error')}")
            return False

        # Show progress
        progress = data.get("progress", {})
        if progress.get("total_samples", 0) > 0:
            percent = (progress.get("completed_samples", 0) / progress["total_samples"]) * 100
            print(f"Progress: {percent:.0f}% ({progress['completed_samples']}/{progress['total_samples']})")

        time.sleep(2)

    print("Timeout waiting for completion")
    return False
```

### Option 2: Poll More Frequently for Near-Real-Time

```python
def stream_like_progress(run_id):
    """Poll every second to approximate real-time updates."""
    import time

    while True:
        response = requests.get(
            f"{BASE_URL}/api/v1/evaluations/runs/{run_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        data = response.json()
        print(f"status={data['status']} progress={data.get('progress', {})}")
        if data["status"] in ["completed", "failed", "cancelled"]:
            break
        time.sleep(1)
```

## Error Handling

### Common Error Responses

```python
def handle_api_error(response):
    """Handle API errors consistently"""
    if response.status_code == 200 or response.status_code == 201:
        return response.json()

    error_data = response.json()

    # Handle different error response formats
    if "error" in error_data:
        error = error_data["error"]
    elif "detail" in error_data:
        if isinstance(error_data["detail"], dict) and "error" in error_data["detail"]:
            error = error_data["detail"]["error"]
        else:
            error = {"message": str(error_data["detail"])}
    else:
        error = {"message": f"Unknown error: {error_data}"}

    print(f"❌ Error {response.status_code}: {error.get('message', 'Unknown error')}")
    if "type" in error:
        print(f"   Type: {error['type']}")
    if "code" in error:
        print(f"   Code: {error['code']}")

    return None
```

## Troubleshooting

### Issue: "API key required for openai"
**Solution**: Add your OpenAI API key to the config file:
```bash
# Edit the config file
nano tldw_Server_API/Config_Files/config.txt

# Add under [API] section:
openai_api_key = sk-your-api-key-here
```

### Issue: 401 Unauthorized
**Solution**: Check your API key:
```python
# For development/single-user mode:
headers = {"Authorization": "Bearer YOUR_API_KEY"}

# For production/multi-user mode:
headers = {"Authorization": "Bearer your-actual-api-key"}
```

### Issue: 404 Not Found on legacy eval paths
**Solution**: Use the correct API path with `/api/` prefix:
```python
# Wrong:
url = "http://localhost:8000/v1/evals"

# Correct:
url = "http://localhost:8000/api/v1/evaluations"
```

### Issue: "Either dataset_id or dataset must be provided"
**Solution**: Always include either inline data or reference a dataset:
```python
# Option 1: Inline dataset
eval_data = {
    "name": "test",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [...]  # Required!
}

# Option 2: Reference existing dataset
eval_data = {
    "name": "test",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset_id": "dataset_xxx"  # Required!
}
```

### Issue: Rate limiting (429 Too Many Requests)
**Solution**: The default rate limits are:
- Create operations: 100/minute
- Read operations: 100/minute
- Run operations: 50/minute

Add delays between requests or batch operations.

### Issue: Results show "pending" or "running" indefinitely
**Solution**: Check the run status for errors:
```python
response = requests.get(f"{BASE_URL}/api/v1/evaluations/runs/{run_id}", headers=headers)
data = response.json()
if "error_message" in data:
    print(f"Run error: {data['error_message']}")
```

## Best Practices

1. **Start Simple**: Test with `exact_match` or `includes` before using LLM evaluations
2. **Small Batches**: Start with 3-5 samples before scaling up
3. **Use Temperature 0**: For consistent, reproducible results
4. **Save Datasets**: Reuse datasets across multiple evaluation runs
5. **Monitor Costs**: LLM-based evaluations consume API tokens
6. **Handle Errors**: Always check response status codes
7. **Use Appropriate Metrics**: Choose metrics that align with your use case

## Next Steps

- **[User Guide](../User_Guides/Server/Evaluations_User_Guide.md)** - Detailed usage patterns and examples
- **[API Reference](../API-related/Evaluations_API_Reference.md)** - Complete endpoint documentation
- **[Developer Guide](../Code_Documentation/Evaluations_Developer_Guide.md)** - Extend the evaluation system
- **Interactive API Docs** - http://localhost:8000/docs when server is running

## Quick Reference Card

```python
# Essential imports
import requests
import json
import time

# Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "http://localhost:8000"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Evaluation types
types = ["model_graded", "exact_match", "fuzzy_match", "includes"]

# Model-graded subtypes
subtypes = ["summarization", "rag", "response_quality"]

# Common metrics
g_eval_metrics = ["fluency", "consistency", "relevance", "coherence"]
rag_metrics = ["relevance", "faithfulness", "answer_similarity"]

# API endpoints
endpoints = {
    "create_eval": "POST /api/v1/evaluations",
    "get_eval": "GET /api/v1/evaluations/{eval_id}",
    "list_evals": "GET /api/v1/evaluations",
    "create_run": "POST /api/v1/evaluations/{eval_id}/runs",
    "get_run": "GET /api/v1/evaluations/runs/{run_id}",
    "create_dataset": "POST /api/v1/evaluations/datasets",
    "get_dataset": "GET /api/v1/evaluations/datasets/{dataset_id}"
}
```

Happy evaluating! 🎉

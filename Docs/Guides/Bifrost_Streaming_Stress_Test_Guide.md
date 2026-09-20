# Bifrost Streaming Stress-Testing Guide

This guide shows how to run the streaming load tests and Scenario A sweeps described in `Stress-Test-Plan.md` using the Python harnesses and Make targets in this repo.

It is focused on `/api/v1/chat/completions` streaming behavior (TTFT and per-chunk latency) and comparing HTTP/1.1 vs HTTP/2 under increasing concurrency.

---

## 1. Prerequisites

### 1.1 Environment

- Python 3.11+.
- Dependencies installed (from repo root):

  ```bash
  pip install -e .[dev]
  ```

- A running `tldw_server` instance:
  - For quick local tests, you can use mock mode via the Makefile:

    ```bash
    make server-up-dev HOST=127.0.0.1 PORT=8000 API_KEY=REPLACE-ME
    ```

    This starts FastAPI with:
    - `AUTH_MODE=single_user`
    - `SINGLE_USER_API_KEY=$API_KEY`
    - `CHAT_FORCE_MOCK=1` (no real provider cost)
    - `STREAMS_UNIFIED=1` (unified SSE streaming enabled)

  - Or run your own Bifrost + tldw_server stack and point the load harness at the gateway URL.

### 1.2 Auth and Base URL

Export the base URL and API key for convenience:

```bash
export BASE_URL=http://127.0.0.1:8000
export SINGLE_USER_API_KEY=your-key-here
```

Most commands below assume:

- `BASE_URL` points to either:
  - The FastAPI server directly, or
  - The Bifrost gateway in front of it.
- `SINGLE_USER_API_KEY` is valid for `X-API-KEY` auth.

---

## 2. One-Off Streaming Load Run (Sanity Check)

Before running full sweeps, do a single streaming load run to ensure everything works.

From repo root:

```bash
make load-chat-stream \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o \
  LOAD_CONCURRENCY=50 \
  LOAD_STREAMS_PER_CLIENT=1 \
  LOAD_PROMPT_BYTES=256
```

What this does:

- Calls `Helper_Scripts/load_tests/chat_streaming_load.py` with:
  - `/api/v1/chat/completions` and `stream=true`.
  - `LOAD_CONCURRENCY` concurrent clients.
  - `LOAD_STREAMS_PER_CLIENT` sequential streams per client.
  - Approximate prompt size `LOAD_PROMPT_BYTES`.
- Prints:
  - Success/failure counts and error samples.
  - TTFT p50/p95/p99.
  - Inter-chunk latency p50/p95/p99 (across all streams).

If this passes without errors, you are ready for Scenario A sweeps.

---

## 3. Scenario A: Baseline Streaming Sweeps

Scenario A focuses on:

- Short/medium prompts (256–1k bytes).
- Short/medium outputs.
- Increasing concurrency (50 → 800) to observe tail latency and protocol behavior.

The repo exposes canonical Scenario A sweeps via Make targets that wrap:

- `Helper_Scripts/load_tests/chat_streaming_sweep.py`
- `Helper_Scripts/load_tests/chat_streaming_load.py`

### 3.1 Short Prompt, HTTP/1.1

From repo root:

```bash
make scenario-a-short-http1 \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o
```

Defaults:

- Concurrency steps: `50 100 200 400 800` (configurable via `SCENARIO_A_CONC_STEPS`).
- Prompt size: `256` bytes (configurable via `SCENARIO_A_SHORT_PROMPT_BYTES`).
- Streams per client: `1` (configurable via `LOAD_STREAMS_PER_CLIENT`).

The script prints a table like:

```text
conc  total  ok    err   err%   ttft_p50  ttft_p95  ttft_p99  chunk_p50  chunk_p95  chunk_p99
  50   1000   998     2   0.2     80.1     95.3    120.5      25.4      40.7      75.2
 100   2000  1990    10   0.5     90.4    110.1    150.2      28.3      46.9      82.1
 ...
```

Interpretation:

- `conc`: number of concurrent streaming clients.
- `total`: total streams run for that step.
- `ok` / `err`: successes and failures.
- `err%`: error rate per step.
- `ttft_*`: time-to-first-token (ms).
- `chunk_*`: inter-chunk latency across all streams (ms).

### 3.2 Short Prompt, HTTP/2

Run the same sweep with HTTP/2:

```bash
make scenario-a-short-http2 \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o
```

This allows a direct comparison between HTTP/1.1 and HTTP/2 tail latency at the same concurrency and prompt sizes.

### 3.3 Longer Prompt, HTTP/1.1 and HTTP/2

To simulate “longer” prompts within Scenario A:

- Uses `SCENARIO_A_LONG_PROMPT_BYTES` (default `1024` bytes).

HTTP/1.1 sweep:

```bash
make scenario-a-long-http1 \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o
```

HTTP/2 sweep:

```bash
make scenario-a-long-http2 \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o
```

You can adjust prompt size:

```bash
make scenario-a-long-http2 \
  BASE_URL=$BASE_URL \
  API_KEY=$SINGLE_USER_API_KEY \
  CHAT_MODEL=gpt-4o \
  SCENARIO_A_LONG_PROMPT_BYTES=2048
```

---

## 4. Custom Sweeps and JSON Output

If you want full control (e.g., different concurrencies or JSON summaries for plotting), call the sweep script directly.

### 4.1 Custom Concurrency Steps

Example: HTTP/2 with custom steps and larger prompts:

```bash
python Helper_Scripts/load_tests/chat_streaming_sweep.py \
  --base-url "$BASE_URL" \
  --api-key "$SINGLE_USER_API_KEY" \
  --model gpt-4o \
  --concurrency-steps 50 150 300 600 \
  --streams-per-client 2 \
  --prompt-bytes 1024 \
  --http2
```

### 4.2 JSON Summary for Plots

Add `--json-out` to write a summary file:

```bash
mkdir -p .benchmarks

python Helper_Scripts/load_tests/chat_streaming_sweep.py \
  --base-url "$BASE_URL" \
  --api-key "$SINGLE_USER_API_KEY" \
  --model gpt-4o \
  --concurrency-steps 50 100 200 400 800 \
  --streams-per-client 2 \
  --prompt-bytes 256 \
  --http2 \
  --json-out .benchmarks/chat_stream_scenario_a_short_http2.json
```

This JSON includes:

- `base_url`, `model`, `concurrency_steps`, `streams_per_client`, `prompt_bytes`, `http2`.
- `steps`: per-concurrency metrics:
  - `total_streams`, `successes`, `failures`, `error_rate`.
  - `ttft_p50_ms`, `ttft_p95_ms`, `ttft_p99_ms`.
  - `chunk_p50_ms`, `chunk_p95_ms`, `chunk_p99_ms`.

You can use this in notebooks or plotting tools to compare runs (e.g., Bifrost vs non-Bifrost, different models, different OS/kernel tuning).

---

## 5. Comparing Bifrost vs Direct Server Path

To compare gateway behavior:

1. **Run sweeps against the direct FastAPI server**:

   ```bash
   export BASE_URL=http://127.0.0.1:8000  # or appropriate host
   # Run Scenario A short/long, HTTP/1.1 and HTTP/2
   make scenario-a-short-http1 BASE_URL=$BASE_URL API_KEY=$SINGLE_USER_API_KEY
   make scenario-a-short-http2 BASE_URL=$BASE_URL API_KEY=$SINGLE_USER_API_KEY
   ```

2. **Run sweeps against Bifrost**:

   ```bash
   export BASE_URL=http://127.0.0.1:9000  # example Bifrost URL
   make scenario-a-short-http1 BASE_URL=$BASE_URL API_KEY=$SINGLE_USER_API_KEY
   make scenario-a-short-http2 BASE_URL=$BASE_URL API_KEY=$SINGLE_USER_API_KEY
   ```

3. **Compare**:
   - TTFT p99 across concurrencies.
   - Chunk latency p99 across concurrencies.
   - Error rates and failure profiles.

This gives a concrete view of whether Bifrost keeps tail latency low for streaming at your target concurrency.

---

## 6. Extending Beyond Scenario A

The current harnesses are focused on Scenario A (baseline streaming). To move toward the full `Stress-Test-Plan.md`:

- **Scenario B (Long Context & Large Outputs)**:
  - Use the same scripts but:
    - Increase `--prompt-bytes` significantly (for long prompts).
    - Reduce `--concurrency-steps` (e.g. 10, 20, 50, 100).
  - Ensure the backend model supports the target context length (e.g. 32k–128k tokens).

- **Scenario C (High Concurrency & Churn)**:
  - Use higher concurrencies (1k–2k) and multiple `streams_per_client`.
  - For real churn (disconnects, early termination), you can:
    - Extend `chat_streaming_load.py` workers to randomly abort streams.
    - Record TTFT/chunk latency under churn and observe resource usage.

- **Scenarios D/E (Failure Storms & Semantic Cache)**:
  - Combine this harness with:
    - Fault injection in the gateway or provider client.
    - Semantic cache-enabled configs (see `Stress-Test-Plan.md`).
  - Use the same TTFT/chunk metrics plus error rates and vector DB metrics.

Refer back to `Stress-Test-Plan.md` for detailed scenario definitions, metrics, and infrastructure/OS tuning suggestions.

---

## 7. Summary

Quick checklist for running baseline streaming stress tests:

1. Install dependencies: `pip install -e .[dev]`.
2. Start the server (or Bifrost + server), ideally with `CHAT_FORCE_MOCK=1` for cost-free tests.
3. Sanity-check with:

   ```bash
   make load-chat-stream BASE_URL=$BASE_URL API_KEY=$SINGLE_USER_API_KEY
   ```

4. Run canonical Scenario A sweeps:
   - Short prompt, HTTP/1.1: `make scenario-a-short-http1 ...`
   - Short prompt, HTTP/2: `make scenario-a-short-http2 ...`
   - Longer prompt, HTTP/1.1: `make scenario-a-long-http1 ...`
   - Longer prompt, HTTP/2: `make scenario-a-long-http2 ...`

5. Optionally, run custom sweeps with `chat_streaming_sweep.py` and save JSON summaries in `.benchmarks/`.
6. Use the results to:
   - Compare HTTP/1.1 vs HTTP/2.
   - Compare Bifrost vs direct FastAPI.
   - Feed into the broader stress plan (long-context, churn, failures, semantic caching).

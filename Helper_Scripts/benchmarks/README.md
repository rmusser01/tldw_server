# Benchmark Scripts

## Native Batch STT

`stt_bench.py` benchmarks batch transcription through tldw_server's native
`SttProviderRegistry` and `SttProviderAdapter` contract. It uses deterministic
strict and normalized WER/CER—no Pipecat, LLM judge, automatic model download,
or server process.

Read the exact
[STT benchmark protocol](../../Docs/Development/STT_Benchmark_Protocol.md)
before collecting or publishing results. Start from
[`stt_benchmark_manifest.example.jsonl`](stt_benchmark_manifest.example.jsonl),
but treat it only as a schema example. Replace or independently verify every
illustrative value, including IDs, references, language/profile/suite choices,
tags, paths, source metadata, durations, and checksums. Corpus audio is not
included.

Quick start:

```bash
source .venv/bin/activate

python Helper_Scripts/benchmarks/stt_bench.py validate \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt

python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt \
  --profile regression \
  --mode neutral-v1 \
  --text-retention errors-only \
  --target faster-whisper=large-v3 \
  --run local-regression-v1

python Helper_Scripts/benchmarks/stt_bench.py report \
  --run .benchmarks/stt/local-regression-v1

python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline .benchmarks/stt/baseline/summary.json \
  --candidate .benchmarks/stt/candidate/summary.json
```

`--target` is repeatable and uses `provider=model`. All model files and optional
dependencies must already be installed. Any loopback or remote audio target
requires the separate `--allow-network-targets` flag. V1 has no dry-run plan
preview: inspect the configured provider endpoint and its privacy/retention
terms before giving that flag, because successful preflight proceeds directly
to execution. Local run artifacts are written under `.benchmarks/stt/`, ignored
by Git, and may still contain sensitive transcripts.

## LLM Gateway

Overview
- `llm_gateway_bench.py` is a minimal async load generator for the Chat API (`/api/v1/chat/completions`).
- It sweeps concurrency levels and reports latency percentiles, error rates, and streaming TTFT.
- `email_search_bench.py` benchmarks Stage-1 email operator search (`search_email_messages`) with optional synthetic fixture generation.

Email Search Benchmark (M1-009)
- Script: `Helper_Scripts/benchmarks/email_search_bench.py`
- Optional query mix fixture: `Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc`
- Optional workload trace fixture: `Helper_Scripts/benchmarks/email_search_workload_trace.sample.json`
- Output: JSON report containing:
  - dataset profile (message count, label cardinality, date span)
  - cold pass summary (reopen DB for each query)
  - warm pass summary (warmups + measured runs)
  - optional SQLite query-plan captures (when `--capture-query-plans` is enabled)
  - p50/p95 checks against PRD NFR targets

Examples
- Build deterministic fixture + run benchmark:

  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --ensure-fixture \
    --fixture-messages 20000 \
    --runs 30 \
    --warmup-runs 5 \
    --out .benchmarks/email_search_report.json

- Benchmark existing tenant without fixture writes:

  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path /path/to/media.db \
    --tenant-id user:1 \
    --runs 20 \
    --warmup-runs 3

- Use custom query mix JSON:

  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --query-mix-file Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc

- Use workload trace-derived query mix and capture SQLite query plans:

  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --workload-trace-file Helper_Scripts/benchmarks/email_search_workload_trace.sample.json \
    --workload-top-n 15 \
    --capture-query-plans \
    --out .benchmarks/email_search_report.json

Recommended Server Settings (for safe local benchmarking)
- Quick start (recommended):

  make server-up-dev HOST=127.0.0.1 PORT=8000 API_KEY=dev-key-123

  This starts uvicorn with:
  - `AUTH_MODE=single_user`
  - `SINGLE_USER_API_KEY=$API_KEY`
  - `DEFAULT_LLM_PROVIDER=openai`
  - `CHAT_FORCE_MOCK=1` (no upstream calls)
  - `STREAMS_UNIFIED=1` (enables SSE metrics)

- Manual alternative:

  AUTH_MODE=single_user \
  SINGLE_USER_API_KEY=dev-key-123 \
  CHAT_FORCE_MOCK=1 \
  DEFAULT_LLM_PROVIDER=openai \
  STREAMS_UNIFIED=1 \
  python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000 --reload

  Notes:
  - `CHAT_FORCE_MOCK=1` avoids hitting real upstream providers; responses are mocked and fast.
  - In multi-user mode, supply a Bearer token instead of `X-API-KEY`.

Examples
- Non-streaming, 1/2/4/8 concurrency for 20s each:

  python Helper_Scripts/benchmarks/llm_gateway_bench.py \
    --base-url http://127.0.0.1:8000 \
    --path /api/v1/chat/completions \
    --api-key "$SINGLE_USER_API_KEY" \
    --concurrency 1 2 4 8 \
    --duration 20

- Streaming with concurrency=16 for 30s:

  python Helper_Scripts/benchmarks/llm_gateway_bench.py \
    --stream \
    --concurrency 16 \
    --duration 30 \
    --api-key "$SINGLE_USER_API_KEY"

- Stop on error-rate > 5% or p99 > 5s:

  python Helper_Scripts/benchmarks/llm_gateway_bench.py \
    --concurrency 1 2 4 8 16 32 \
    --duration 20 \
    --max-error-rate 0.05 \
    --latency-p99-sla-ms 5000

What It Measures
- Per step: total, successes/failures, RPS, p50/p90/p95/p99 (ms)
- If `--stream`: TTFT (p50/p95) in ms
- Optional server-side metrics deltas from `/metrics` (Prometheus):
  - `http_requests_total{endpoint="/api/v1/chat/completions",status="..."}` by status
  - Use `--metrics-url` to point to a different metrics endpoint

Tips
- Increase `--prompt-bytes` to simulate larger inputs.
- Use `--provider openai --model gpt-4o-mini` with `CHAT_FORCE_MOCK=1` for consistent, fast results.
- Optionally export to JSON with `--out results.json`.

Locust (Open-Loop RPS)
- File: `Helper_Scripts/benchmarks/locustfile.py`
- Closed-loop (headless):

  locust -f Helper_Scripts/benchmarks/locustfile.py --host http://127.0.0.1:8000 \
    --headless -u 50 -r 10 -t 2m

- Approximate open-loop RPS plan via env:

  TLDW_RPS_PLAN="10:30,20:30,40:60,20:30,10:30" \
  TLDW_TASKS_PER_USER_PER_SEC=1 \
  locust -f Helper_Scripts/benchmarks/locustfile.py --host http://127.0.0.1:8000 --headless -t 3m

- Optional env vars:
  - `TLDW_BENCH_PATH` (default `/api/v1/chat/completions`)
  - `TLDW_BENCH_PROVIDER` (default `openai`)
  - `TLDW_BENCH_MODEL` (default `gpt-4o-mini`)
  - `TLDW_BENCH_STREAM` (`1|true|yes|on` to enable streaming)
  - `TLDW_BENCH_PROMPT_BYTES` (default 256)
  - `SINGLE_USER_API_KEY` or `TLDW_BENCH_BEARER_TOKEN`
  - `TLDW_TASKS_PER_USER_PER_SEC` (default 1; used with RPS plan)

Notes
- Streaming in Locust: total request time includes consuming the stream; a synthetic TTFT metric is emitted as `request_type=TTFT`, `name=chat:stream_ttft`.

Monitoring Stack (Prometheus + Grafana)
- Compose files: `Dockerfiles/Monitoring/`
- Start stack:

  docker compose -f Dockerfiles/Monitoring/docker-compose.monitoring.yml up -d

- Prometheus scrapes `host.docker.internal:8000/metrics` by default (adjust `Dockerfiles/Monitoring/prometheus.yml`).
- Grafana at http://localhost:3000 (admin/admin). The `LLM Gateway` dashboard is auto-provisioned from `Docs/Monitoring/Grafana_Dashboards/`.
- To enable SSE panels (enqueue→yield), set on the server: `STREAMS_UNIFIED=1`.
 - Linux note: if `host-gateway` is unsupported, change the Prometheus target to your host IP (e.g., `172.17.0.1:8000`).

One‑Command Full Run
- Start monitoring + run both sweeps (non-stream and stream) and print links:

  make bench-full BASE_URL=http://127.0.0.1:8000 API_KEY=$SINGLE_USER_API_KEY \
    FULL_CONCURRENCY="1 2 4 8" FULL_STREAM_CONCURRENCY="4 8 16" FULL_DURATION=20

  Results are saved to `.benchmarks/bench_nonstream.json` and `.benchmarks/bench_stream.json`. Open Grafana at:
  - http://localhost:3000/d/tldw-llm-gateway
  - Login: admin / admin
  - Tip: ensure the server runs with `STREAMS_UNIFIED=1` for SSE metrics.

Make Targets (summary)
- `server-up-dev`        — run uvicorn in mock mode with SSE metrics enabled
- `monitoring-up`        — start Prometheus (9090) + Grafana (3000)
- `monitoring-down`      — stop monitoring stack
- `monitoring-logs`      — tail monitoring logs
- `bench-sweep`          — non-stream concurrency sweep (writes `.benchmarks/bench_nonstream.json`)
- `bench-stream`         — streaming sweep (writes `.benchmarks/bench_stream.json`)
- `bench-rps`            — Locust RPS plan (open-loop approx)
- `bench-full`           — monitoring-up + both sweeps + helpful links

# Research Workspace Authenticated Power-User UAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify GitHub issue #2607 and RW-UAT-028 through a clean CDP-controlled authenticated Research Workspace journey against real backend, ingestion, retrieval, and model-provider services.

**Certification rule:** RW-UAT-028 remains `Partial` and issue #2607 remains open until every gate in this plan passes. Environment limitations and delegated follow-ups may explain a blocker, but they do not satisfy a gate. TASK-12020.24 must also reach an evidence-backed terminal state before certification.

**Architecture:** Use task-owned disposable configuration, databases, logs, browser profiles, and evidence under `/private/tmp/task-12020-37`; do not inherit repository `config.txt`, `.env`, developer databases, browser credentials, or public frontend fallback credentials. Exercise the research loop in an isolated single-user stack and sharing permissions in a separate isolated multi-user stack with real owner/member/non-member identities and team or organization membership. Drive Chrome only through Playwright `connectOverCDP`.

**Scope boundary:** RW-UAT-030 owns the broad destructive-recovery certification. This task rolls forward its current evidence and performs only the live remove/underlying-media/Undo checks needed by RW-UAT-028 and TASK-12020.32/.33. It must not silently claim or duplicate the rest of RW-UAT-030.

**Tech Stack:** FastAPI, SQLite for single-user state, the repository-supported PostgreSQL multi-user fixture, Next.js, React, Playwright `connectOverCDP`, Chrome DevTools Protocol, llama.cpp OpenAI-compatible API, Vitest, pytest.

**Authoritative commands:**

```bash
#!/usr/bin/env bash
# Save this block as the task-owned orchestration script and run it from the
# repository root. Any command, assertion, readiness timeout, or cleanup failure
# makes the run non-authoritative.
set -Eeuo pipefail
REPO_ROOT="$(pwd -P)"
RUN_ROOT=/private/tmp/task-12020-37
RUN_STARTED_AT="$(date -u +%s)"
SINGLE_API_PID=""
SINGLE_WEB_PID=""
SINGLE_CHROME_PID=""
MULTI_API_PID=""
MULTI_WEB_PID=""
MULTI_CHROME_PID=""
CLONE_WORKER_PID=""
PROVIDER_OBSERVER_PID=""

cleanup() {
  local original_status=$?
  local cleanup_status=0
  local attempt
  local port
  local pid
  local residual
  local running
  trap - EXIT INT TERM
  for pid in "$CLONE_WORKER_PID" "$MULTI_CHROME_PID" "$MULTI_WEB_PID" \
    "$MULTI_API_PID" "$SINGLE_CHROME_PID" "$SINGLE_WEB_PID" \
    "$SINGLE_API_PID" "$PROVIDER_OBSERVER_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  pkill -TERM -f "$RUN_ROOT/single/chrome-profile" 2>/dev/null || true
  pkill -TERM -f "$RUN_ROOT/multi/chrome-profile" 2>/dev/null || true
  docker rm --force task1202037-postgres >/dev/null 2>&1 || true
  for attempt in $(seq 1 20); do
    running=0
    for pid in "$CLONE_WORKER_PID" "$MULTI_CHROME_PID" "$MULTI_WEB_PID" \
      "$MULTI_API_PID" "$SINGLE_CHROME_PID" "$SINGLE_WEB_PID" \
      "$SINGLE_API_PID" "$PROVIDER_OBSERVER_PID"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then running=1; fi
    done
    if (( running == 0 )); then break; fi
    sleep 0.25
  done
  for pid in "$CLONE_WORKER_PID" "$MULTI_CHROME_PID" "$MULTI_WEB_PID" \
    "$MULTI_API_PID" "$SINGLE_CHROME_PID" "$SINGLE_WEB_PID" \
    "$SINGLE_API_PID" "$PROVIDER_OBSERVER_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      cleanup_status=1
      kill -KILL "$pid" 2>/dev/null || true
    fi
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  for port in 18170 18171 18172 18173 18174 18175 19099 55437; do
    residual="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$residual" ]]; then
      cleanup_status=1
      kill -TERM $residual 2>/dev/null || true
    fi
  done
  sleep 1
  for port in 18170 18171 18172 18173 18174 18175 19099 55437; do
    residual="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$residual" ]]; then kill -KILL $residual 2>/dev/null || true; fi
  done
  sleep 0.25
  for port in 18170 18171 18172 18173 18174 18175 19099 55437; do
    if [[ -n "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]]; then
      cleanup_status=1
    fi
  done
  if [[ -n "$(docker ps -aq --filter name=^/task1202037-postgres$)" ]]; then
    cleanup_status=1
  fi
  if (( original_status != 0 )); then exit "$original_status"; fi
  exit "$cleanup_status"
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null; then return 0; fi
    sleep 1
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

# Disposable roots and single-user credential. Config/env files intentionally
# start empty because all active settings are passed through the allowlist. A
# pre-existing root or container fails closed instead of reusing stale state.
umask 077
test ! -e "$RUN_ROOT"
test -z "$(docker ps -aq --filter name=^/task1202037-postgres$)"
test -z "$(find "$REPO_ROOT" -maxdepth 3 -type f \( -name '.env' -o -name '.ENV' \) -print -quit)"
for port in 18170 18171 18172 18173 18174 18175 19099 55437; do
  test -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)"
done
install -d -m 700 \
  "$RUN_ROOT/single/home" "$RUN_ROOT/single/tmp" "$RUN_ROOT/single/logs" \
  "$RUN_ROOT/multi/home" "$RUN_ROOT/multi/tmp" "$RUN_ROOT/multi/logs" \
  "$RUN_ROOT/multi/postgres" "$RUN_ROOT/runner"
node Docs/Reviews/assets/2026-07-15-research-workspace-power-user-uat/runtime-path-audit.mjs \
  before --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/runtime-paths-before.json"
printf '%s\n' '[API-Routes]' 'disable = llm, llamacpp' \
  >"$RUN_ROOT/single/config.txt"
chmod 600 "$RUN_ROOT/single/config.txt"
install -m 600 /dev/null "$RUN_ROOT/single/.env"
printf '%s\n' '[API-Routes]' 'disable = llm, llamacpp' \
  >"$RUN_ROOT/multi/config.txt"
chmod 600 "$RUN_ROOT/multi/config.txt"
install -m 600 /dev/null "$RUN_ROOT/multi/.env"
printf '%s\n' 'modules: []' >"$RUN_ROOT/single/mcp-modules.yaml"
printf '%s\n' 'modules: []' >"$RUN_ROOT/multi/mcp-modules.yaml"
chmod 600 "$RUN_ROOT/single/mcp-modules.yaml" "$RUN_ROOT/multi/mcp-modules.yaml"
export TASK_12020_37_API_KEY="$(openssl rand -hex 24)"

# Provider proof
curl -fsS http://127.0.0.1:9099/health
curl -fsS http://127.0.0.1:9099/v1/models
curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf","messages":[{"role":"user","content":"Reply with exactly: task-12020-37-provider-ready"}],"max_tokens":256}' \
  --output "$RUN_ROOT/single/direct-provider-completion.json" \
  http://127.0.0.1:9099/v1/chat/completions
.venv/bin/python -c 'import json; p="/private/tmp/task-12020-37/single/direct-provider-completion.json"; d=json.load(open(p, encoding="utf-8")); assert d["choices"][0]["message"]["content"].strip()'

# Transparent provider observer: forwards to llama.cpp without substituting a
# model and records only redacted method/path/model/status/body-digest/nonce data.
node Docs/Reviews/assets/2026-07-15-research-workspace-power-user-uat/provider-observer.mjs \
  --listen 127.0.0.1:19099 --upstream http://127.0.0.1:9099 \
  --log "$RUN_ROOT/provider-observer.jsonl" \
  >"$RUN_ROOT/single/logs/provider-observer.log" 2>&1 &
PROVIDER_OBSERVER_PID=$!
wait_for_url http://127.0.0.1:19099/healthz

# Offline embedding fixture proof. This is the only model directory read outside
# the task root; all mutable caches remain task-owned.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/single/home \
  XDG_CACHE_HOME=/private/tmp/task-12020-37/single/cache \
  HF_HOME=/private/tmp/task-12020-37/single/huggingface \
  SENTENCE_TRANSFORMERS_HOME=/private/tmp/task-12020-37/single/sentence-transformers \
  TORCH_HOME=/private/tmp/task-12020-37/single/torch \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -c 'from sentence_transformers import SentenceTransformer; p="/Users/macbook-dev/Documents/GitHub/tldw_server2/models/embeddings/sentence-transformers__all-MiniLM-L6-v2"; v=SentenceTransformer(p).encode(["task-12020-37 embedding smoke"]); assert tuple(v.shape) == (1, 384)'

# Single-user API; the runner supplies only the documented allowlisted variables.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/single/home \
  TMPDIR=/private/tmp/task-12020-37/single/tmp \
  PYTHONPYCACHEPREFIX=/private/tmp/task-12020-37/single/cache/pycache \
  TLDW_CONFIG_FILE=/private/tmp/task-12020-37/single/config.txt \
  TLDW_ENV_FILE=/private/tmp/task-12020-37/single/.env \
  AUTH_MODE=single_user SINGLE_USER_API_KEY="$TASK_12020_37_API_KEY" \
  DATABASE_URL=sqlite:////private/tmp/task-12020-37/single/users.db \
  USER_DB_BASE_DIR=/private/tmp/task-12020-37/single/user_databases \
  JOBS_DB_PATH=/private/tmp/task-12020-37/single/jobs.db \
  SCHEDULER_DATABASE_URL=sqlite:////private/tmp/task-12020-37/single/scheduler.db \
  SCHEDULER_BASE_PATH=/private/tmp/task-12020-37/single/scheduler \
  WORKFLOWS_SCHEDULER_ENABLED=false \
  WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-37/single/watchlist-templates \
  SYSTEM_LOG_FILE_PATH=/private/tmp/task-12020-37/single/logs/system.jsonl \
  MCP_MODULES_CONFIG=/private/tmp/task-12020-37/single/mcp-modules.yaml \
  MCP_AUDIT_LOG_FILE=/private/tmp/task-12020-37/single/logs/mcp-audit.log \
  AUDIT_STORAGE_MODE=per_user \
  AUDIT_SHARED_DB_PATH=/private/tmp/task-12020-37/single/audit-shared.db \
  CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-37/single/circuit-breakers.db \
  REDIS_ENABLED=false WORKFLOWS_DB_MAINTENANCE_ENABLED=false \
  XDG_CACHE_HOME=/private/tmp/task-12020-37/single/cache \
  EMBEDDINGS_STORAGE_ALLOWLIST_ROOT=/private/tmp/task-12020-37/single \
  EMBEDDINGS_MODEL_STORAGE_DIR=/private/tmp/task-12020-37/single/cache/embedding-models \
  HF_HOME=/private/tmp/task-12020-37/single/huggingface \
  SENTENCE_TRANSFORMERS_HOME=/private/tmp/task-12020-37/single/sentence-transformers \
  TORCH_HOME=/private/tmp/task-12020-37/single/torch \
  RAG_SEMANTIC_CACHE_DIR=/private/tmp/task-12020-37/single/cache/rag-semantic \
  RAG_FLASHRANK_CACHE_DIR=/private/tmp/task-12020-37/single/cache/flashrank \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  EMBEDDINGS_DEFAULT_PROVIDER=huggingface \
  EMBEDDINGS_DEFAULT_MODEL=/Users/macbook-dev/Documents/GitHub/tldw_server2/models/embeddings/sentence-transformers__all-MiniLM-L6-v2 \
  CUSTOM_OPENAI_API_IP=http://127.0.0.1:19099/v1 \
  CUSTOM_OPENAI_API_MODEL=Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  CUSTOM_OPENAI_API_KEY=task-local \
  DEFAULT_LLM_PROVIDER=custom-openai-api RAG_DEFAULT_LLM_PROVIDER=custom-openai-api \
  WORKFLOWS_EGRESS_BLOCK_PRIVATE=false WORKFLOWS_EGRESS_ALLOWED_PORTS=19099 \
  .venv/bin/python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 18170 \
  >"$RUN_ROOT/single/logs/api.log" 2>&1 &
SINGLE_API_PID=$!
wait_for_url http://127.0.0.1:18170/api/v1/health

# Reject active frontend environment files, then start the WebUI without inherited
# credentials or public auth fallbacks.
test -z "$(find apps/tldw-frontend -maxdepth 1 -type f -name '.env*' ! -name '*.example' -print -quit)"
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/single/home \
  TMPDIR=/private/tmp/task-12020-37/single/tmp \
  NEXT_PUBLIC_API_URL=http://127.0.0.1:18170 \
  bun --cwd apps/tldw-frontend run dev:webpack -- -H 127.0.0.1 -p 18171 \
  >"$RUN_ROOT/single/logs/webui.log" 2>&1 &
SINGLE_WEB_PID=$!
wait_for_url http://127.0.0.1:18171/research-workspace

# Launch the exact browser under test. The runner uses chromium.connectOverCDP
# against this endpoint; no computer-control or Playwright-launched browser is used.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/single/home \
  TMPDIR=/private/tmp/task-12020-37/single/tmp \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=18172 \
  --user-data-dir=/private/tmp/task-12020-37/single/chrome-profile \
  --disable-extensions \
  --disable-component-extensions-with-background-pages \
  --no-first-run --no-default-browser-check about:blank \
  >"$RUN_ROOT/single/logs/chrome.log" 2>&1 &
SINGLE_CHROME_PID=$!
wait_for_url http://127.0.0.1:18172/json/version
node /private/tmp/task-12020-37/runner/research-workspace-power-user-uat.mjs \
  --web-url http://127.0.0.1:18171 --cdp-url http://127.0.0.1:18172 \
  --run-started-at "$RUN_STARTED_AT"

# Multi-user PostgreSQL fixture. Secrets are generated into the task-owned shell
# environment and are redacted from evidence; literal values are never committed.
export TASK_12020_37_PG_PASSWORD="$(openssl rand -hex 24)"
export TASK_12020_37_JWT_SECRET="$(openssl rand -hex 32)"
export TASK_12020_37_MCP_JWT_SECRET="$(openssl rand -hex 32)"
export TASK_12020_37_MCP_API_KEY_SALT="$(openssl rand -hex 32)"
export TASK_12020_37_SESSION_KEY="$(.venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export TASK_12020_37_SUPPORT_KEY="$(openssl rand -hex 24)"
export TASK_12020_37_OWNER_PASSWORD="Uat!$(openssl rand -hex 16)Aa1"
export TASK_12020_37_MEMBER_PASSWORD="Uat!$(openssl rand -hex 16)Aa1"
export TASK_12020_37_REQUESTER_PASSWORD="Uat!$(openssl rand -hex 16)Aa1"
docker run --detach --name task1202037-postgres \
  --env POSTGRES_USER=tldw_uat \
  --env POSTGRES_PASSWORD="$TASK_12020_37_PG_PASSWORD" \
  --env POSTGRES_DB=task1202037 \
  --mount type=bind,source="$RUN_ROOT/multi/postgres",target=/var/lib/postgresql/data \
  --publish 127.0.0.1:55437:5432 postgres:18
until docker exec task1202037-postgres pg_isready -U tldw_uat -d task1202037; do sleep 1; done

# Initialize the same PostgreSQL schema the server will use. TASK-12020.38 must
# make the sharing tables part of this canonical initialization path.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/multi/home \
  TMPDIR=/private/tmp/task-12020-37/multi/tmp \
  PYTHONPYCACHEPREFIX=/private/tmp/task-12020-37/multi/cache/pycache \
  TLDW_CONFIG_FILE=/private/tmp/task-12020-37/multi/config.txt \
  TLDW_ENV_FILE=/private/tmp/task-12020-37/multi/.env \
  AUTH_MODE=multi_user PROFILE=multi-user-postgres \
  DATABASE_URL="postgresql://tldw_uat:$TASK_12020_37_PG_PASSWORD@127.0.0.1:55437/task1202037" \
  JWT_SECRET_KEY="$TASK_12020_37_JWT_SECRET" \
  SESSION_ENCRYPTION_KEY="$TASK_12020_37_SESSION_KEY" \
  MCP_JWT_SECRET="$TASK_12020_37_MCP_JWT_SECRET" \
  MCP_API_KEY_SALT="$TASK_12020_37_MCP_API_KEY_SALT" \
  USER_DB_BASE_DIR=/private/tmp/task-12020-37/multi/user_databases \
  JOBS_DB_PATH=/private/tmp/task-12020-37/multi/jobs.db \
  SCHEDULER_DATABASE_URL=sqlite:////private/tmp/task-12020-37/multi/scheduler.db \
  SCHEDULER_BASE_PATH=/private/tmp/task-12020-37/multi/scheduler \
  WORKFLOWS_SCHEDULER_ENABLED=false \
  WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-37/multi/watchlist-templates \
  SYSTEM_LOG_FILE_PATH=/private/tmp/task-12020-37/multi/logs/system.jsonl \
  MCP_MODULES_CONFIG=/private/tmp/task-12020-37/multi/mcp-modules.yaml \
  MCP_AUDIT_LOG_FILE=/private/tmp/task-12020-37/multi/logs/mcp-audit.log \
  AUDIT_STORAGE_MODE=per_user \
  AUDIT_SHARED_DB_PATH=/private/tmp/task-12020-37/multi/audit-shared.db \
  CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-37/multi/circuit-breakers.db \
  REDIS_ENABLED=false WORKFLOWS_DB_MAINTENANCE_ENABLED=false \
  XDG_CACHE_HOME=/private/tmp/task-12020-37/multi/cache \
  EMBEDDINGS_STORAGE_ALLOWLIST_ROOT=/private/tmp/task-12020-37/multi \
  EMBEDDINGS_MODEL_STORAGE_DIR=/private/tmp/task-12020-37/multi/cache/embedding-models \
  HF_HOME=/private/tmp/task-12020-37/multi/huggingface \
  SENTENCE_TRANSFORMERS_HOME=/private/tmp/task-12020-37/multi/sentence-transformers \
  TORCH_HOME=/private/tmp/task-12020-37/multi/torch \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  RAG_SEMANTIC_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/rag-semantic \
  RAG_FLASHRANK_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/flashrank \
  .venv/bin/python -m tldw_Server_API.app.core.AuthNZ.initialize --non-interactive

# Multi-user API uses the same explicit task-owned runtime boundary.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/multi/home \
  TMPDIR=/private/tmp/task-12020-37/multi/tmp \
  PYTHONPYCACHEPREFIX=/private/tmp/task-12020-37/multi/cache/pycache \
  TLDW_CONFIG_FILE=/private/tmp/task-12020-37/multi/config.txt \
  TLDW_ENV_FILE=/private/tmp/task-12020-37/multi/.env \
  AUTH_MODE=multi_user PROFILE=multi-user-postgres \
  DATABASE_URL="postgresql://tldw_uat:$TASK_12020_37_PG_PASSWORD@127.0.0.1:55437/task1202037" \
  JWT_SECRET_KEY="$TASK_12020_37_JWT_SECRET" \
  SESSION_ENCRYPTION_KEY="$TASK_12020_37_SESSION_KEY" \
  MCP_JWT_SECRET="$TASK_12020_37_MCP_JWT_SECRET" \
  MCP_API_KEY_SALT="$TASK_12020_37_MCP_API_KEY_SALT" \
  ENABLE_REGISTRATION=true REQUIRE_REGISTRATION_CODE=false \
  ENABLE_ADMIN_E2E_TEST_MODE=true \
  TLDW_ADMIN_E2E_SUPPORT_KEY="$TASK_12020_37_SUPPORT_KEY" \
  TLDW_ADMIN_E2E_ADMIN_PASSWORD="$TASK_12020_37_OWNER_PASSWORD" \
  TLDW_ADMIN_E2E_OWNER_PASSWORD="$TASK_12020_37_OWNER_PASSWORD" \
  TLDW_ADMIN_E2E_SUPER_ADMIN_PASSWORD="$TASK_12020_37_OWNER_PASSWORD" \
  TLDW_ADMIN_E2E_MEMBER_PASSWORD="$TASK_12020_37_MEMBER_PASSWORD" \
  TLDW_ADMIN_E2E_REQUESTER_PASSWORD="$TASK_12020_37_REQUESTER_PASSWORD" \
  USER_DB_BASE_DIR=/private/tmp/task-12020-37/multi/user_databases \
  JOBS_DB_PATH=/private/tmp/task-12020-37/multi/jobs.db \
  SCHEDULER_DATABASE_URL=sqlite:////private/tmp/task-12020-37/multi/scheduler.db \
  SCHEDULER_BASE_PATH=/private/tmp/task-12020-37/multi/scheduler \
  WORKFLOWS_SCHEDULER_ENABLED=false \
  WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-37/multi/watchlist-templates \
  SYSTEM_LOG_FILE_PATH=/private/tmp/task-12020-37/multi/logs/system.jsonl \
  MCP_MODULES_CONFIG=/private/tmp/task-12020-37/multi/mcp-modules.yaml \
  MCP_AUDIT_LOG_FILE=/private/tmp/task-12020-37/multi/logs/mcp-audit.log \
  AUDIT_STORAGE_MODE=per_user \
  AUDIT_SHARED_DB_PATH=/private/tmp/task-12020-37/multi/audit-shared.db \
  CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-37/multi/circuit-breakers.db \
  REDIS_ENABLED=false WORKFLOWS_DB_MAINTENANCE_ENABLED=false \
  XDG_CACHE_HOME=/private/tmp/task-12020-37/multi/cache \
  EMBEDDINGS_STORAGE_ALLOWLIST_ROOT=/private/tmp/task-12020-37/multi \
  EMBEDDINGS_MODEL_STORAGE_DIR=/private/tmp/task-12020-37/multi/cache/embedding-models \
  HF_HOME=/private/tmp/task-12020-37/multi/huggingface \
  SENTENCE_TRANSFORMERS_HOME=/private/tmp/task-12020-37/multi/sentence-transformers \
  TORCH_HOME=/private/tmp/task-12020-37/multi/torch \
  RAG_SEMANTIC_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/rag-semantic \
  RAG_FLASHRANK_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/flashrank \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  EMBEDDINGS_DEFAULT_PROVIDER=huggingface \
  EMBEDDINGS_DEFAULT_MODEL=/Users/macbook-dev/Documents/GitHub/tldw_server2/models/embeddings/sentence-transformers__all-MiniLM-L6-v2 \
  CUSTOM_OPENAI_API_IP=http://127.0.0.1:19099/v1 \
  CUSTOM_OPENAI_API_MODEL=Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  CUSTOM_OPENAI_API_KEY=task-local \
  DEFAULT_LLM_PROVIDER=custom-openai-api RAG_DEFAULT_LLM_PROVIDER=custom-openai-api \
  WORKFLOWS_EGRESS_BLOCK_PRIVATE=false WORKFLOWS_EGRESS_ALLOWED_PORTS=19099 \
  .venv/bin/python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 18173 \
  >"$RUN_ROOT/multi/logs/api.log" 2>&1 &
MULTI_API_PID=$!
wait_for_url http://127.0.0.1:18173/api/v1/health

curl -fsS -H "X-TLDW-Admin-E2E-Key: $TASK_12020_37_SUPPORT_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scenario":"jwt_admin"}' \
  --output /private/tmp/task-12020-37/multi/seed.json \
  http://127.0.0.1:18173/api/v1/test-support/admin-e2e/seed
curl -fsS -H "X-TLDW-Admin-E2E-Key: $TASK_12020_37_SUPPORT_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"principal_key":"jwt_owner"}' \
  --output /private/tmp/task-12020-37/multi/owner-session.json \
  http://127.0.0.1:18173/api/v1/test-support/admin-e2e/bootstrap-jwt-session
curl -fsS -H "X-TLDW-Admin-E2E-Key: $TASK_12020_37_SUPPORT_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"principal_key":"jwt_non_admin"}' \
  --output /private/tmp/task-12020-37/multi/member-session.json \
  http://127.0.0.1:18173/api/v1/test-support/admin-e2e/bootstrap-jwt-session

# Start the clean multi-user WebUI/browser and let the runner create the owner
# source, register a true outsider, add/remove organization membership, and inject
# returned HttpOnly cookies into isolated owner/member/outsider contexts.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/multi/home \
  TMPDIR=/private/tmp/task-12020-37/multi/tmp \
  NEXT_PUBLIC_API_URL=http://127.0.0.1:18173 \
  bun --cwd apps/tldw-frontend run dev:webpack -- -H 127.0.0.1 -p 18174 \
  >"$RUN_ROOT/multi/logs/webui.log" 2>&1 &
MULTI_WEB_PID=$!
wait_for_url http://127.0.0.1:18174/shared
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/multi/home \
  TMPDIR=/private/tmp/task-12020-37/multi/tmp \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port=18175 \
  --user-data-dir=/private/tmp/task-12020-37/multi/chrome-profile \
  --disable-extensions --disable-component-extensions-with-background-pages \
  --no-first-run --no-default-browser-check about:blank \
  >"$RUN_ROOT/multi/logs/chrome.log" 2>&1 &
MULTI_CHROME_PID=$!
wait_for_url http://127.0.0.1:18175/json/version

# Dedicated durable clone executor. TASK-12020.41 adds this WorkerSDK service;
# the UAT does not rely on FastAPI BackgroundTasks or an unobserved in-process task.
env -i PATH="$PATH" HOME=/private/tmp/task-12020-37/multi/home \
  TMPDIR=/private/tmp/task-12020-37/multi/tmp \
  PYTHONPYCACHEPREFIX=/private/tmp/task-12020-37/multi/cache/pycache \
  TLDW_CONFIG_FILE=/private/tmp/task-12020-37/multi/config.txt \
  TLDW_ENV_FILE=/private/tmp/task-12020-37/multi/.env \
  AUTH_MODE=multi_user PROFILE=multi-user-postgres \
  DATABASE_URL="postgresql://tldw_uat:$TASK_12020_37_PG_PASSWORD@127.0.0.1:55437/task1202037" \
  JWT_SECRET_KEY="$TASK_12020_37_JWT_SECRET" \
  SESSION_ENCRYPTION_KEY="$TASK_12020_37_SESSION_KEY" \
  MCP_JWT_SECRET="$TASK_12020_37_MCP_JWT_SECRET" \
  MCP_API_KEY_SALT="$TASK_12020_37_MCP_API_KEY_SALT" \
  USER_DB_BASE_DIR=/private/tmp/task-12020-37/multi/user_databases \
  JOBS_DB_PATH=/private/tmp/task-12020-37/multi/jobs.db \
  SCHEDULER_DATABASE_URL=sqlite:////private/tmp/task-12020-37/multi/scheduler.db \
  SCHEDULER_BASE_PATH=/private/tmp/task-12020-37/multi/scheduler \
  WORKFLOWS_SCHEDULER_ENABLED=false \
  WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-37/multi/watchlist-templates \
  SYSTEM_LOG_FILE_PATH=/private/tmp/task-12020-37/multi/logs/system.jsonl \
  MCP_MODULES_CONFIG=/private/tmp/task-12020-37/multi/mcp-modules.yaml \
  MCP_AUDIT_LOG_FILE=/private/tmp/task-12020-37/multi/logs/mcp-audit.log \
  AUDIT_STORAGE_MODE=per_user \
  AUDIT_SHARED_DB_PATH=/private/tmp/task-12020-37/multi/audit-shared.db \
  CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-37/multi/circuit-breakers.db \
  REDIS_ENABLED=false WORKFLOWS_DB_MAINTENANCE_ENABLED=false \
  XDG_CACHE_HOME=/private/tmp/task-12020-37/multi/cache \
  EMBEDDINGS_STORAGE_ALLOWLIST_ROOT=/private/tmp/task-12020-37/multi \
  EMBEDDINGS_MODEL_STORAGE_DIR=/private/tmp/task-12020-37/multi/cache/embedding-models \
  HF_HOME=/private/tmp/task-12020-37/multi/huggingface \
  SENTENCE_TRANSFORMERS_HOME=/private/tmp/task-12020-37/multi/sentence-transformers \
  TORCH_HOME=/private/tmp/task-12020-37/multi/torch \
  RAG_SEMANTIC_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/rag-semantic \
  RAG_FLASHRANK_CACHE_DIR=/private/tmp/task-12020-37/multi/cache/flashrank \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m tldw_Server_API.app.services.sharing_clone_jobs_worker \
  >"$RUN_ROOT/multi/logs/clone-worker.log" 2>&1 &
CLONE_WORKER_PID=$!
sleep 1
kill -0 "$CLONE_WORKER_PID"
node /private/tmp/task-12020-37/runner/research-workspace-sharing-uat.mjs \
  --web-url http://127.0.0.1:18174 --api-url http://127.0.0.1:18173 \
  --cdp-url http://127.0.0.1:18175 --run-started-at "$RUN_STARTED_AT"
node Docs/Reviews/assets/2026-07-15-research-workspace-power-user-uat/runtime-path-audit.mjs \
  after --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" \
  --baseline "$RUN_ROOT/runtime-paths-before.json" \
  --run-started-at "$RUN_STARTED_AT" \
  --output "$RUN_ROOT/runtime-paths-after.json"

# Required backend regression gate. The JUnit assertion makes a skipped
# PostgreSQL fixture or empty selection a hard failure rather than a false pass.
.venv/bin/python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py \
  tldw_Server_API/tests/Sharing/test_sharing_integration.py \
  tldw_Server_API/tests/Sharing/test_sharing_endpoints.py \
  tldw_Server_API/tests/Sharing/test_sharing_clone_jobs.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  --strict-markers --junitxml="$RUN_ROOT/pytest-sharing.xml" -q
.venv/bin/python -c 'import xml.etree.ElementTree as E; r=E.parse("/private/tmp/task-12020-37/pytest-sharing.xml").getroot(); a=r.attrib; assert int(a.get("tests", 0)) > 0 and int(a.get("failures", 0)) == 0 and int(a.get("errors", 0)) == 0 and int(a.get("skipped", 0)) == 0, a'

# Required frontend regression gate (exit code 0 and zero pending tests required).
pushd apps/packages/ui >/dev/null
bunx vitest run \
  src/components/Option/SharedWithMe.test.tsx \
  src/components/Option/__tests__/SharedWithMe.research-workspace-route.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage1.ingestion.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.shared-recipient.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ShareDialog.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage3.folders.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage5.transfer.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts \
  --reporter=json --outputFile="$RUN_ROOT/vitest-research-workspace.json"
popd >/dev/null
node -e 'const r=require("/private/tmp/task-12020-37/vitest-research-workspace.json"); if (!(r.numTotalTests > 0 && r.numFailedTests === 0 && r.numPendingTests === 0)) process.exit(1)'

# Exact static/security gates for the remediations.
.venv/bin/ruff check \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/api/v1/endpoints/sharing.py \
  tldw_Server_API/app/core/Sharing \
  tldw_Server_API/app/services/sharing_clone_jobs_worker.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py \
  tldw_Server_API/tests/Sharing
bun --cwd apps/tldw-frontend x eslint --config eslint.config.mjs \
  ../packages/ui/src/hooks/useSharing.ts \
  ../packages/ui/src/components/Option/SharedWithMe.tsx \
  ../packages/ui/src/components/Option/ResearchWorkspace/index.tsx \
  ../packages/ui/src/components/Option/ResearchWorkspace/SharedWorkspaceContext.tsx
bun --cwd apps/tldw-frontend run typecheck
.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/api/v1/endpoints/sharing.py \
  tldw_Server_API/app/core/Sharing \
  tldw_Server_API/app/services/sharing_clone_jobs_worker.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  -f json -o "$RUN_ROOT/bandit-sharing.json"
node --test Docs/Reviews/assets/2026-07-15-research-workspace-power-user-uat/*.test.mjs
git diff --check
```

The shell creates passwords that satisfy the repository password policy and keeps them only in task-owned process memory. The runner registers the outsider through `POST /api/v1/auth/register`, authenticates through form-encoded `POST /api/v1/auth/login`, and manages membership through `POST/DELETE /api/v1/orgs/{org_id}/members`. The owner creates the workspace and source through the visible Research Workspace UI before any grant. The final evidence README records the resolved PostgreSQL URL with credentials redacted, exact Chrome executable/version, and exact command lines used.

**Diagnostics contract:** Each runner segment starts with an empty ledger. The invalid-key segment must contain exactly one `GET /api/v1/health -> 200` and one `GET /api/v1/users/storage -> 401`, and the UI must visibly show `Invalid API key`. The recovery segment must repeat those methods/paths with `200` responses, visibly show the connected/healthy state, close any recovery surface, and leave no stale modal or error detail. Permission-denial segments declare an exact allowlist of `{method, path template, status, count}` before execution; every declared denial must occur and every undeclared `status >= 400`, status-0 request, `requestfailed`, page error, console error, or runtime overlay fails the segment. Public-link revocation must produce exactly one `GET /api/v1/sharing/public/{token} -> 404`. Before membership, outsider list returns `200` with zero matching items and outsider open returns exactly one `GET /api/v1/sharing/shared-with-me/{share_id}/workspace -> 403`. After owner revocation, member list returns `200` with zero matching items and member open/sources/chat/clone each returns its declared single `404` when exercised.

---

## Stage 1: Isolate And Prove The Live Contract

**Goal:** Prove the configuration boundary, authentication, provider, diagnostics, and regression baseline before persona actions begin.

**Success Criteria:** No mutable runtime write lands in repository or developer data, and known default sinks remain unchanged; llama.cpp and tldw_server both complete with the pinned provider/model; the WebUI authenticates with the task-owned key; Chrome attaches over CDP from a clean profile; focused Research Workspace tests pass with zero baseline failures.

**Tests:** Environment/path audit, provider probes, API health/auth probes, clean CDP entry probe, evidence-redactor unit test, focused Research Workspace Vitest files.

**Status:** In Progress

- [ ] Run the strict orchestration script from a nonexistent `/private/tmp/task-12020-37` root and nonexistent fixture container. Create task-owned `TLDW_CONFIG_FILE` and `TLDW_ENV_FILE`, use an allowlisted environment, and make the tested runtime-path auditor assert auth, Jobs, Scheduler, per-user databases, media/vector state, templates, logs, caches, browser profiles, and evidence were created after `RUN_STARTED_AT` under that root while repository/home default sinks and repository status remain unchanged. The only external read-only model asset is the explicitly recorded local `sentence-transformers/all-MiniLM-L6-v2` directory; copy no application data from the developer profile.
- [ ] Pin `custom-openai-api` and `Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`; first complete directly against `127.0.0.1:9099`, then complete through tldw_server with the explicit provider/model. Catalog presence alone is not proof.
- [ ] Pin the exact read-only local embedding directory shown above, complete the offline 384-dimension smoke test before server startup, and require the uploaded source projection to report `readiness.vector_ready=true` and `state=queryable`; a text-only/FTS-ready source does not satisfy the vector gate.
- [ ] Reject active frontend `.env*` files and start the WebUI under `env -i` with only the task-owned runtime variables and `NEXT_PUBLIC_API_URL=http://127.0.0.1:18170`. Launch the exact extension-free Chrome command above and attach only through `connectOverCDP`.
- [ ] Implement and test an evidence redactor before capture. Persist only allowlisted request fields and strip `Authorization`, `X-API-Key`, cookies, passwords, JWTs, API keys, share URLs/tokens, and secret query parameters.
- [ ] Run the focused Research Workspace regression baseline. Any failure blocks the live run and certification until fixed or proven unrelated with a separate passing authoritative baseline.
- [ ] Require every API, WebUI, Chrome CDP endpoint, provider observer, and clone worker to pass its readiness/process probe before use. Record exact commit, backend/config paths, provider/model, auth mode, ports, database roots, Chrome version/profile, viewport, process IDs, commands, and diagnostic policy; verify the exit trap terminates every process/container on success and induced failure.

## Stage 2: Certify Authenticated Research Work

**Goal:** Complete the core source-to-grounded-answer workflow through visible UI controls.

**Success Criteria:** A deterministic source added through the UI visibly traverses ingestion/indexing to queryable readiness, can be inspected and annotated, and produces a grounded selected-source answer whose retrieval result and citation point to that source.

**Tests:** Segmented CDP checkpoints for auth recovery, source creation, job states, readiness, preview, annotation, selection, RAG scope/retrieval, answer completion, citation inspection, reload persistence, and diagnostics.

**Status:** Not Started

- [ ] Create a deterministic document containing a unique sentinel and known answer. Upload it through the visible Add Sources flow; record the sanitized ingest request, returned media ID, workspace-source ID, and request correlation plus observable extraction/chunking/indexing/queryable states. Record a Jobs ID only when the API actually returns one; lack of a Jobs ID is not fabricated into the evidence.
- [ ] Exercise the exact invalid-key diagnostics contract above, including visible degraded copy. Clear diagnostics, recover with the valid key, require visible connected/healthy state with no stale modal/detail, and then require zero unallowlisted HTTP/runtime errors for every later segment.
- [ ] Open status details and source preview, find the sentinel, add an annotation, select the source, reload, and verify source/selection/annotation persistence.
- [ ] Ask the deterministic question. Assert the request carries the exact selected `include_media_ids`, retrieval contains the sentinel, the final answer is non-empty and correct, and the inspectable citation resolves to the same media ID/source.
- [ ] For every product failure, create or update focused Backlog and GitHub tracking with exact reproduction evidence, write a failing regression first, implement the smallest fix, and repeat the full affected segment. No failure may be left only in the UAT narrative.

## Stage 3: Certify Studio And Durable Outputs

**Goal:** Prove that selected ready sources can drive a real generated artifact and survive session resumption.

**Success Criteria:** Studio sends the selected source title/content plus the pinned provider/model through tldw_server to port `9099`, completes a non-empty artifact, and preserves it after reload.

**Tests:** Sanitized Studio request envelope, task-owned transparent provider-observer correlation, selected-source payload assertion, artifact completion and reload persistence, post-segment diagnostic gate.

**Status:** Not Started

- [ ] Generate a Summary from the deterministic selected source using a unique Studio nonce and assert the request includes its title/content, explicit `custom-openai-api` provider, and pinned model.
- [ ] Correlate the browser request through tldw_server and the transparent observer to exactly one upstream `POST /v1/chat/completions` on `127.0.0.1:9099` with the unique nonce, pinned model, successful upstream status, and matching redacted body digest; require a non-empty completed artifact rather than a guarded or catalog-only state.
- [ ] Reload and reopen the workspace; verify the artifact persists and remains attributable to the expected source.
- [ ] Close TASK-12020.24 only with this evidence. A provider or environment blocker keeps both TASK-12020.24 and RW-UAT-028 incomplete.

## Stage 4: Certify Sharing Permissions

**Goal:** Verify both tokenized read-only sharing and authenticated team or organization sharing with real authorization boundaries.

**Success Criteria:** A read-only share works in an unauthenticated clean context before revoke and fails after revoke. A real member can list/open/chat/clone according to the product contract, a non-member is denied, and the member is denied after access revocation.

**Tests:** Separate browser contexts for owner, member, true outsider, and read-only link; real PostgreSQL membership fixture; permission matrix before and after revoke; exact-denial diagnostic gates per context.

**Status:** Not Started

- [ ] Resolve the preflight blockers tracked in TASK-12020.38/#2736 (PostgreSQL sharing schema), TASK-12020.39/#2735 (recipient list response), TASK-12020.40/#2737 (canonical shared sources/chat binding), and TASK-12020.41/#2734 (durable clone status). Each receives focused RED/GREEN coverage and independent verification before this stage can start its authoritative run.
- [ ] Register `sharing_clone_jobs_worker` in the supported application worker lifecycle with an explicit enable flag, stable default, inventory visibility, and bounded shutdown. Start that same worker implementation as a dedicated process against the task-owned Jobs database for deterministic UAT, require its process probe, and prove one clone advances through persisted queued/processing/completed states (plus a controlled failed state) across recipient reload. A production path that requires an undocumented manual worker, an in-memory UUID, or a FastAPI `BackgroundTasks` callback does not satisfy this gate.
- [ ] Run the exact standalone PostgreSQL, AuthNZ initialization, backend, clean WebUI, clean Chrome, fixture-seed, session-bootstrap, and teardown commands above. The seeded `jwt_owner` owns the workspace/source, seeded `jwt_non_admin` is the member, and a separately registered user with no organization membership is the true outsider.
- [ ] As owner, create a read-only share link. Open it in a clean unauthenticated context, verify only allowed content/actions, revoke it, then verify the same separate context is denied.
- [ ] Share the workspace through the canonical team or organization mechanism. Verify the member can list and open it, then exercise chat and clone permissions exactly as advertised; verify the outsider cannot list or open it. Require recipient retrieval/chat to be constrained to workspace source media IDs so unrelated owner media cannot leak into answers.
- [ ] Revoke the member/team grant and verify the previously authorized member context can no longer list/open/chat/clone. Revoke all links and tear down users, memberships, and databases after sanitized evidence is complete.

## Stage 5: Certify Portability And Bounded Recovery

**Goal:** Prove exported state can be reconstructed in a genuinely clean client context and that workspace removal does not delete domain-owned media.

**Success Criteria:** The export is structurally valid; import into empty storage creates a distinct workspace ID with the exact state represented by the export contract after reload; removal leaves the underlying media available before Undo; Undo restores membership.

**Tests:** ZIP/JSON structural inspection, second empty-storage browser context, imported-ID inequality, exact source/selection/chat/artifact comparison, media endpoint before/after Remove and Undo, bounded RW-UAT-030 cross-reference.

**Status:** Not Started

- [ ] Export the populated workspace and inspect the downloaded ZIP/JSON schema, required files, IDs, and content before import; do not treat download existence as success.
- [ ] Create a second clean browser context with empty local/session storage and no copied workspace state. Import through the real file input and require a new workspace ID.
- [ ] Verify exact source, selection, chat, and Studio artifact state represented by the exported schema in the imported workspace, reload it, and verify the same state again. Annotations remain a Stage 2 local reload-persistence check until a separate task adds them to the export contract.
- [ ] In the original workspace, bulk Remove the disposable source. Before Undo, query the protected media endpoint by media ID and require the underlying media to remain available; then Undo and verify membership and selected-source state are restored.
- [ ] Reference the existing RW-UAT-030/TASK-12020.32/.33 evidence for broader destructive recovery and report only this task's bounded live recheck.

## Stage 6: Publish And Close Certification

**Goal:** Convert one complete successful run into durable, reviewable evidence and close stale task ownership.

**Success Criteria:** Every prior stage passes in current assets; the evidence bundle is sanitized and reproducible; RW-UAT-028 states only evidence-supported outcomes; TASK-12020.24 and TASK-12020.37 are complete; issue #2607 can be closed.

**Tests:** Full task-owned CDP runner, per-segment diagnostics, artifact integrity checks, focused Vitest/pytest, `git diff --check`, scoped lint/typecheck, scoped Bandit for Python changes, independent review.

**Status:** Not Started

- [ ] Re-run every checkpoint in one authoritative sequence. A missing gate, unexpected diagnostic, untracked product failure, or incomplete teardown keeps RW-UAT-028 `Partial` and issue #2607 open.
- [ ] Commit a bounded evidence bundle containing the exact runner, tested redactor, machine-readable checkpoints/diagnostics, representative screenshots, structural export assertions, permission matrix, and README separating assertions from visual support. Mask credential/token fields before every screenshot and run a final OCR/manual secret scan plus repository text scan over the evidence bundle.
- [ ] Update `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`, TASK-12020.37, TASK-12020.24, and issue #2607 with exact evidence and residual blockers.
- [ ] Run the exact pytest/JUnit no-skip gate, Vitest JSON no-pending gate, Ruff, ESLint, full frontend typecheck, Bandit, evidence tests, and repository hygiene commands above; request independent code review and address validated findings.
- [ ] Revoke task credentials/shares, tear down task fixtures and processes, verify no secrets or disposable databases are tracked, then commit and open a PR against current `dev` with the requester-owned Change summary left as the explicit human merge gate.

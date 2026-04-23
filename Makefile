# -----------------------------------------------------------------------------
# Default target: show help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "tldw_server — available targets"
	@echo "================================"
	@echo ""
	@echo "Getting Started:"
	@echo "  make quickstart              Docker single-user + WebUI (recommended)"
	@echo "  make setup-docker-single     Configure Docker single-user + WebUI"
	@echo "  make start-docker-single     Start Docker single-user + WebUI"
	@echo "  make verify-docker-single    Verify Docker single-user + WebUI"
	@echo "  make setup-docker-multi      Configure Docker multi-user + PostgreSQL"
	@echo "  make start-docker-multi      Start Docker multi-user + PostgreSQL"
	@echo "  make verify-docker-multi     Verify Docker multi-user + PostgreSQL"
	@echo "  make install-local           Install local Python dependencies"
	@echo "  make setup-local-single      Configure local single-user mode"
	@echo "  make start-local-single      Start local single-user server"
	@echo "  make verify-local-single     Verify local single-user server"
	@echo "  make quickstart-install      Local Python install only"
	@echo "  make quickstart-docker       Docker API-only (no WebUI)"
	@echo "  make quickstart-prereqs      Check Python, ffmpeg, and dependencies"
	@echo ""
	@echo "Server:"
	@echo "  make quickstart-local        Configure and start local server"
	@echo "  make server-up-dev           Start server in mock/dev mode"
	@echo "  make verify                  Health-check a running server"
	@echo "  make check                   Run sanity checks (health, keys, ffmpeg)"
	@echo "  make show-api-key            Print your SINGLE_USER_API_KEY"
	@echo "  make generate-secrets        Generate fresh secrets for .env"
	@echo ""
	@echo "Testing:"
	@echo "  make tooling-install         Install test/smoke extras"
	@echo "  make tooling-smoke           Run unified smoke checks"
	@echo "  make lint-changed            Lint only changed files"
	@echo "  make release                 Cut a patch release from main"
	@echo "  make release-patch           Cut a patch release from main"
	@echo "  make release-minor           Cut a minor release from main"
	@echo ""
	@echo "Docker:"
	@echo "  make quickstart-docker-webui Docker API + WebUI"
	@echo "  make monitoring-up           Start Prometheus + Grafana"
	@echo "  make monitoring-down         Stop monitoring stack"
	@echo "  make monitoring-reset        Stop monitoring + delete volumes"
	@echo ""
	@echo "See Docs/Getting_Started/README.md for full setup guides."
	@echo ""

# -----------------------------------------------------------------------------
# Quickstart targets (first-time setup)
# -----------------------------------------------------------------------------
.PHONY: setup-wizard-tools setup-docker-single start-docker-single verify-docker-single setup-docker-multi start-docker-multi verify-docker-multi install-local setup-local-single start-local-single verify-local-single quickstart quickstart-install quickstart-prereqs quickstart-local quickstart-docker quickstart-docker-bootstrap quickstart-docker-webui model-cycle verify pypi-build pypi-check tooling-install tooling-smoke show-api-key release release-patch release-minor

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python
TLDW_ENV_FILE ?= tldw_Server_API/Config_Files/.env
TLDW_ENV_TEMPLATE ?= tldw_Server_API/Config_Files/.env.example
DOCKER_BASE_COMPOSE ?= Dockerfiles/docker-compose.yml
DOCKER_SINGLE_COMPOSE ?= Dockerfiles/docker-compose.single-user.yml
DOCKER_MULTI_COMPOSE ?= Dockerfiles/docker-compose.multi-user-postgres.yml
DOCKER_WEBUI_COMPOSE ?= Dockerfiles/docker-compose.webui.yml
SETUP_VENV_DIR ?= .setup-venv
SETUP_VENV_PYTHON ?= $(SETUP_VENV_DIR)/bin/python
TLDW_SETUP ?= $(SETUP_VENV_PYTHON) -m tldw_Server_API.cli.wizard.cli
TLDW_BASE_URL ?= http://127.0.0.1:8000
TLDW_WEBUI_URL ?= http://127.0.0.1:8080
NEXT_PUBLIC_API_URL ?=
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE ?= quickstart
TLDW_INTERNAL_API_ORIGIN ?= http://app:8000
DOCKER_BUILD ?= false
DOCKER_BUILD_FLAG = $(if $(filter true TRUE 1 yes YES,$(DOCKER_BUILD)),--build,)
DOCKER_WAIT_FLAG ?= --wait
PYPI_BUILD_ARGS ?= --no-isolation
MODEL_CYCLE_FIRST ?=
MODEL_CYCLE_SECOND ?=
MODEL_CYCLE_EXCLUDED ?=postgres,redis
MODEL_CYCLE_FIRST_BOOT_WAIT ?=0
MODEL_CYCLE_SECOND_BOOT_WAIT ?=0
MODEL_CYCLE_DRY_RUN ?=false
RELEASE_DRY_RUN ?=false

quickstart-prereqs:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[quickstart] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[quickstart] Python 3.10+ is required." && exit 1)
	@$(PYTHON) -c 'import loguru, fastapi, uvicorn, dotenv' >/dev/null 2>&1 || (echo "[quickstart] Missing Python dependencies. Run: make quickstart-install" && exit 1)
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "[quickstart] Warning: ffmpeg not found; audio/video features will be limited."; \
		case "$$(uname -s)" in \
			Darwin) echo "[quickstart] Install on macOS: brew install ffmpeg" ;; \
			Linux) \
				. /etc/os-release 2>/dev/null || true; \
				case "$${ID:-}" in \
					ubuntu|debian|linuxmint|pop) echo "[quickstart] Install on Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg" ;; \
					fedora|rhel|centos|rocky|almalinux) echo "[quickstart] Install on Fedora/RHEL: sudo dnf install -y ffmpeg (RPM Fusion may be required)" ;; \
					arch) echo "[quickstart] Install on Arch: sudo pacman -S --needed ffmpeg" ;; \
					*) echo "[quickstart] Install ffmpeg via your Linux package manager." ;; \
				esac; \
				echo "[quickstart] Linux audio note: for PyAudio support install portaudio19-dev/python3-pyaudio (Debian/Ubuntu)." ;; \
		esac; \
	fi

setup-wizard-tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[setup-wizard-tools] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[setup-wizard-tools] Python 3.10+ is required." && exit 1)
	@if [ ! -x "$(SETUP_VENV_PYTHON)" ]; then \
		echo "[setup-wizard-tools] Creating lightweight wizard virtualenv at $(SETUP_VENV_DIR)"; \
		$(PYTHON) -m venv $(SETUP_VENV_DIR); \
	fi
	@$(SETUP_VENV_PYTHON) -m pip install --upgrade pip setuptools wheel >/dev/null
	@$(SETUP_VENV_PYTHON) -m pip install "typer>=0.12.0" "loguru>=0.7.0" "httpx>=0.24.0" "python-dotenv>=1.0.0" "cryptography>=41.0.0" >/dev/null

setup-docker-single: setup-wizard-tools
	@command -v docker >/dev/null 2>&1 || (echo "[setup-docker-single] docker not found. Install Docker and retry." && exit 1)
	@$(TLDW_SETUP) init --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --default --yes
	@echo "[setup-docker-single] Next: make start-docker-single"

start-docker-single:
	@command -v docker >/dev/null 2>&1 || (echo "[start-docker-single] docker not found. Install Docker and retry." && exit 1)
	docker compose --env-file "$(TLDW_ENV_FILE)" -f "$(DOCKER_SINGLE_COMPOSE)" -f "$(DOCKER_WEBUI_COMPOSE)" up -d $(DOCKER_BUILD_FLAG) $(DOCKER_WAIT_FLAG)
	@echo "[start-docker-single] API:   $(TLDW_BASE_URL)"
	@echo "[start-docker-single] WebUI: $(TLDW_WEBUI_URL)"
	@echo "[start-docker-single] Next:  make verify-docker-single"

verify-docker-single: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --webui-url "$(TLDW_WEBUI_URL)" --first-value

setup-docker-multi: setup-wizard-tools
	@command -v docker >/dev/null 2>&1 || (echo "[setup-docker-multi] docker not found. Install Docker and retry." && exit 1)
	@test -n "$$ADMIN_USERNAME" || (echo "[setup-docker-multi] Set ADMIN_USERNAME=<admin> ADMIN_PASSWORD=<password> in the shell environment for first admin bootstrap." && exit 1)
	@test -n "$$ADMIN_PASSWORD" || (echo "[setup-docker-multi] Set ADMIN_PASSWORD=<password> in the shell environment for first admin bootstrap." && exit 1)
	@ADMIN_USERNAME="$$ADMIN_USERNAME" ADMIN_PASSWORD="$$ADMIN_PASSWORD" ADMIN_EMAIL="$$ADMIN_EMAIL" $(TLDW_SETUP) init --profile docker-multi-postgres --env-file "$(TLDW_ENV_FILE)" --default --yes
	@echo "[setup-docker-multi] Next: make start-docker-multi"

start-docker-multi:
	@command -v docker >/dev/null 2>&1 || (echo "[start-docker-multi] docker not found. Install Docker and retry." && exit 1)
	@test -f "$(TLDW_ENV_FILE)" || (echo "[start-docker-multi] $(TLDW_ENV_FILE) not found. Run: make setup-docker-multi" && exit 1)
	@TLDW_ENV_FILE_ABS="$$(cd "$$(dirname "$(TLDW_ENV_FILE)")" && pwd)/$$(basename "$(TLDW_ENV_FILE)")"; \
		TLDW_ENV_FILE="$$TLDW_ENV_FILE_ABS" docker compose -f "$(DOCKER_MULTI_COMPOSE)" config >/dev/null && \
		TLDW_ENV_FILE="$$TLDW_ENV_FILE_ABS" docker compose -f "$(DOCKER_MULTI_COMPOSE)" up -d $(DOCKER_BUILD_FLAG) $(DOCKER_WAIT_FLAG)
	@echo "[start-docker-multi] API:  $(TLDW_BASE_URL)"
	@echo "[start-docker-multi] Next: make verify-docker-multi"

verify-docker-multi: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile docker-multi-postgres --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --first-value

install-local:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[install-local] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[install-local] Python 3.10+ is required." && exit 1)
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "[install-local] Creating virtualenv at $(VENV_DIR)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "[install-local] Installing Python dependencies into $(VENV_DIR)..."
	@$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(VENV_PYTHON) -m pip install -e .
	@echo "[install-local] Next: make setup-local-single"

setup-local-single: setup-wizard-tools
	@$(TLDW_SETUP) init --profile local-single --env-file "$(TLDW_ENV_FILE)" --default --yes
	@echo "[setup-local-single] Next: make start-local-single"

start-local-single:
	@echo "[start-local-single] Starting API at $(TLDW_BASE_URL)"
	TLDW_ENV_FILE="$(TLDW_ENV_FILE)" $(VENV_PYTHON) -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000

verify-local-single: setup-wizard-tools
	@$(TLDW_SETUP) verify --profile local-single --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --first-value

quickstart: setup-docker-single start-docker-single verify-docker-single

quickstart-docker-webui: quickstart

quickstart-docker: setup-docker-single
	@command -v docker >/dev/null 2>&1 || (echo "[quickstart-docker] docker not found. Install Docker and retry." && exit 1)
	docker compose --env-file "$(TLDW_ENV_FILE)" -f "$(DOCKER_SINGLE_COMPOSE)" up -d $(DOCKER_BUILD_FLAG) $(DOCKER_WAIT_FLAG)
	@$(TLDW_SETUP) verify --profile docker-single-webui --env-file "$(TLDW_ENV_FILE)" --base-url "$(TLDW_BASE_URL)" --webui-url ""

quickstart-install: install-local

quickstart-local: install-local setup-local-single start-local-single

tooling-install:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[tooling-install] $(PYTHON) not found. Install Python 3.10+ and retry." && exit 1)
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || (echo "[tooling-install] Python 3.10+ is required." && exit 1)
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "[tooling-install] Creating virtualenv at $(VENV_DIR)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "[tooling-install] Installing tooling extras into $(VENV_DIR)..."
	@$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(VENV_PYTHON) -m pip install -e ".[tooling]"

release:
	@$(MAKE) release-patch RELEASE_DRY_RUN=$(RELEASE_DRY_RUN)

release-patch:
	@$(PYTHON) Helper_Scripts/release.py --bump patch $(if $(filter true TRUE 1 yes YES,$(RELEASE_DRY_RUN)),--dry-run,)

release-minor:
	@$(PYTHON) Helper_Scripts/release.py --bump minor $(if $(filter true TRUE 1 yes YES,$(RELEASE_DRY_RUN)),--dry-run,)

quickstart-docker-bootstrap:
	@echo "[quickstart-docker-bootstrap] Ensuring $(TLDW_ENV_FILE) has safe first-use auth defaults..."
	@bash Helper_Scripts/docker_prepare_env.sh "$(TLDW_ENV_FILE)" "$(TLDW_ENV_TEMPLATE)"

model-cycle:
	@command -v docker >/dev/null 2>&1 || (echo "[model-cycle] docker not found. Install Docker and retry." && exit 1)
	@test -n "$(MODEL_CYCLE_FIRST)" || (echo "[model-cycle] Set MODEL_CYCLE_FIRST=<container_name>" && exit 1)
	@test -n "$(MODEL_CYCLE_SECOND)" || (echo "[model-cycle] Set MODEL_CYCLE_SECOND=<container_name>" && exit 1)
	@echo "[model-cycle] Cycling $(MODEL_CYCLE_FIRST) -> $(MODEL_CYCLE_SECOND) (excluded=$(MODEL_CYCLE_EXCLUDED), dry_run=$(MODEL_CYCLE_DRY_RUN))"
	$(PYTHON) Helper_Scripts/model_container_cycle.py \
		--first-container "$(MODEL_CYCLE_FIRST)" \
		--second-container "$(MODEL_CYCLE_SECOND)" \
		--excluded "$(MODEL_CYCLE_EXCLUDED)" \
		--first-boot-wait "$(MODEL_CYCLE_FIRST_BOOT_WAIT)" \
		--second-boot-wait "$(MODEL_CYCLE_SECOND_BOOT_WAIT)" \
		$(if $(filter true TRUE 1 yes YES,$(MODEL_CYCLE_DRY_RUN)),--dry-run,)

verify:
	@echo "[verify] Checking server health..."
	@curl -sf http://localhost:8000/health > /dev/null && echo "[verify] Health check PASSED" || (echo "[verify] Health check FAILED - is the server running?" && exit 1)

show-api-key:
	@echo "[show-api-key] WARNING: This prints a secret to stdout. Do not use in CI or shared terminals." >&2
	@if [ -f "$(TLDW_ENV_FILE)" ]; then \
		grep '^SINGLE_USER_API_KEY=' "$(TLDW_ENV_FILE)" | cut -d= -f2-; \
	elif command -v docker >/dev/null 2>&1; then \
		echo "[show-api-key] .env not found on host; reading from Docker container..." >&2; \
		docker compose -f $(DOCKER_BASE_COMPOSE) exec -T app cat /app/tldw_Server_API/Config_Files/.env 2>/dev/null | grep '^SINGLE_USER_API_KEY=' | cut -d= -f2- || echo "[show-api-key] Could not read key. Is the container running?" >&2; \
	else \
		echo "[show-api-key] .env file not found at $(TLDW_ENV_FILE) and docker not available." >&2; \
	fi

# -----------------------------------------------------------------------------
# PyPI packaging helpers
# -----------------------------------------------------------------------------

pypi-build:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "[pypi-build] $(PYTHON) not found." && exit 1)
	@$(PYTHON) -m pip show build >/dev/null 2>&1 || (echo "[pypi-build] Missing 'build'. Install with: $(PYTHON) -m pip install build" && exit 1)
	@echo "[pypi-build] Cleaning previous artifacts..."
	@rm -rf build dist *.egg-info tldw_server.egg-info
	@echo "[pypi-build] Building sdist + wheel ($(PYPI_BUILD_ARGS))..."
	@$(PYTHON) -m build $(PYPI_BUILD_ARGS)

pypi-check: pypi-build
	@$(PYTHON) -m pip show twine >/dev/null 2>&1 || (echo "[pypi-check] Missing 'twine'. Install with: $(PYTHON) -m pip install twine" && exit 1)
	@echo "[pypi-check] Validating distributions..."
	@$(PYTHON) -m twine check dist/*

# -----------------------------------------------------------------------------
# PostgreSQL backup/restore
# -----------------------------------------------------------------------------
.PHONY: pg-backup pg-restore

# Defaults (override on command line)
PG_BACKUP_DIR ?= ./tldw_DB_Backups/postgres
PG_LABEL ?= content
PG_DUMP_FILE ?=

pg-backup:
	@echo "[pg-backup] Writing backup to $(PG_BACKUP_DIR) (label=$(PG_LABEL))"
	@python Helper_Scripts/pg_backup_restore.py backup --backup-dir "$(PG_BACKUP_DIR)" --label "$(PG_LABEL)"

pg-restore:
	@test -n "$(PG_DUMP_FILE)" || (echo "[pg-restore] Set PG_DUMP_FILE=path/to.dump" && exit 1)
	@echo "[pg-restore] Restoring from $(PG_DUMP_FILE)"
	@python Helper_Scripts/pg_backup_restore.py restore --dump-file "$(PG_DUMP_FILE)"

# -----------------------------------------------------------------------------
# Monitoring stack (Prometheus + Grafana)
# -----------------------------------------------------------------------------
.PHONY: monitoring-up monitoring-down monitoring-reset monitoring-logs

MON_STACK := Dockerfiles/Monitoring/docker-compose.monitoring.yml

monitoring-up:
	@echo "[monitoring] Starting Prometheus + Grafana + Alertmanager"
	docker compose --env-file $(TLDW_ENV_FILE) -f $(MON_STACK) up -d
	@echo "[monitoring] Grafana: http://localhost:$${GRAFANA_PORT:-3002} (admin/admin). Prometheus: http://localhost:9090. Alertmanager: http://localhost:9093"

monitoring-down:
	@echo "[monitoring] Stopping monitoring stack"
	docker compose --env-file $(TLDW_ENV_FILE) -f $(MON_STACK) down

monitoring-reset:
	@echo "[monitoring] Stopping monitoring stack and removing volumes"
	docker compose --env-file $(TLDW_ENV_FILE) -f $(MON_STACK) down -v

monitoring-logs:
	docker compose --env-file $(TLDW_ENV_FILE) -f $(MON_STACK) logs -f

# -----------------------------------------------------------------------------
# Dev Server (mock mode)
# -----------------------------------------------------------------------------
.PHONY: server-up-dev

# Defaults (override on command line)
HOST ?= 127.0.0.1
PORT ?= 8000
API_KEY ?= REPLACE-THIS-WITH-A-SECURE-API-KEY-123

server-up-dev:
	@echo "[server] Starting uvicorn in mock mode on $(HOST):$(PORT)"
	AUTH_MODE=single_user \
	SINGLE_USER_API_KEY="$(API_KEY)" \
	DEFAULT_LLM_PROVIDER=openai \
	CHAT_FORCE_MOCK=1 \
	STREAMS_UNIFIED=1 \
	uvicorn tldw_Server_API.app.main:app --host $(HOST) --port $(PORT) --reload

# -----------------------------------------------------------------------------
# Watchlists smoke
# -----------------------------------------------------------------------------
.PHONY: watchlists-audio-smoke

WATCHLISTS_BASE_URL ?= http://127.0.0.1:8000
WATCHLISTS_API_KEY ?= $(SINGLE_USER_API_KEY)
WATCHLISTS_AUDIO_SMOKE_ARGS ?=

watchlists-audio-smoke:
	@echo "[watchlists-audio-smoke] Running watchlists audio smoke against $(WATCHLISTS_BASE_URL)"
	$(PYTHON) Helper_Scripts/watchlists/watchlists_audio_smoke.py \
		--base-url "$(WATCHLISTS_BASE_URL)" \
		--api-key "$(WATCHLISTS_API_KEY)" \
		$(WATCHLISTS_AUDIO_SMOKE_ARGS)

# -----------------------------------------------------------------------------
# Unified tooling smoke (streaming + watchlists audio)
# -----------------------------------------------------------------------------

TOOLING_SMOKE_BASE_URL ?= http://127.0.0.1:8000
TOOLING_SMOKE_API_KEY ?= $(SINGLE_USER_API_KEY)

tooling-smoke:
	@echo "[tooling-smoke] Running unified tooling smoke checks against $(TOOLING_SMOKE_BASE_URL)"
	$(PYTHON) Helper_Scripts/tooling_smoke.py \
		--base-url "$(TOOLING_SMOKE_BASE_URL)" \
		--api-key "$(TOOLING_SMOKE_API_KEY)"

# -----------------------------------------------------------------------------
# Prompt Studio tests
# -----------------------------------------------------------------------------
.PHONY: prompt-studio-test prompt-studio-test-sqlite prompt-studio-test-postgres

PS_TEST_PATH ?= tldw_Server_API/tests/prompt_studio
PS_TEST_ARGS ?= -v

prompt-studio-test:
	@echo "[prompt-studio] Running Prompt Studio tests (sqlite + postgres when available)"
	python -m pytest $(PS_TEST_PATH) $(PS_TEST_ARGS)

prompt-studio-test-sqlite:
	@echo "[prompt-studio] Running Prompt Studio tests (sqlite only)"
	python -m pytest $(PS_TEST_PATH) -k sqlite $(PS_TEST_ARGS)

prompt-studio-test-postgres:
	@echo "[prompt-studio] Running Prompt Studio tests (postgres only)"
	TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest $(PS_TEST_PATH) -k postgres $(PS_TEST_ARGS)

# -----------------------------------------------------------------------------
# Benchmarks (LLM Gateway)
# -----------------------------------------------------------------------------
.PHONY: bench-sweep bench-stream bench-rps

# Defaults (override on command line)
BASE_URL ?= http://127.0.0.1:8000
API_KEY ?= $(SINGLE_USER_API_KEY)
CONCURRENCY ?= 1 2 4 8
DURATION ?= 20
PROMPT_BYTES ?= 256
OUTDIR ?= .benchmarks

bench-sweep:
	@mkdir -p $(OUTDIR)
	@echo "[bench] Non-stream sweep: $(CONCURRENCY) for $(DURATION)s (prompt $(PROMPT_BYTES)B)"
	python Helper_Scripts/benchmarks/llm_gateway_bench.py \
		--base-url $(BASE_URL) \
		--path /api/v1/chat/completions \
		--api-key "$(API_KEY)" \
		--concurrency $(CONCURRENCY) \
		--duration $(DURATION) \
		--prompt-bytes $(PROMPT_BYTES) \
		--out $(OUTDIR)/bench_nonstream.json

bench-stream:
	@mkdir -p $(OUTDIR)
	@echo "[bench] Streaming sweep: $(CONCURRENCY) for $(DURATION)s (prompt $(PROMPT_BYTES)B)"
	python Helper_Scripts/benchmarks/llm_gateway_bench.py \
		--stream \
		--base-url $(BASE_URL) \
		--path /api/v1/chat/completions \
		--api-key "$(API_KEY)" \
		--concurrency $(CONCURRENCY) \
		--duration $(DURATION) \
		--prompt-bytes $(PROMPT_BYTES) \
		--out $(OUTDIR)/bench_stream.json

# Approximate open-loop RPS plan via Locust
RPS_PLAN ?= 10:30,20:30,40:60,20:30,10:30
TASKS_PER_USER_PER_SEC ?= 1
LOCUST_T ?= 3m

bench-rps:
	@echo "[bench-rps] RPS plan: $(RPS_PLAN) (tasks/user/sec=$(TASKS_PER_USER_PER_SEC))"
	TLDW_RPS_PLAN="$(RPS_PLAN)" \
	TLDW_TASKS_PER_USER_PER_SEC="$(TASKS_PER_USER_PER_SEC)" \
	SINGLE_USER_API_KEY="$(API_KEY)" \
	locust -f Helper_Scripts/benchmarks/locustfile.py --host $(BASE_URL) --headless -t $(LOCUST_T)

# -----------------------------------------------------------------------------
# Full run: bring up monitoring, run non-stream + stream sweeps, print links
# -----------------------------------------------------------------------------
.PHONY: bench-full

FULL_CONCURRENCY ?= 1 2 4 8
FULL_STREAM_CONCURRENCY ?= 4 8 16
FULL_DURATION ?= 20

bench-full:
	@echo "[full] Starting monitoring stack (Prometheus + Grafana)"
	$(MAKE) monitoring-up
	@echo "[full] Running non-stream sweep: $(FULL_CONCURRENCY) for $(FULL_DURATION)s"
	$(MAKE) bench-sweep CONCURRENCY="$(FULL_CONCURRENCY)" DURATION=$(FULL_DURATION)
	@echo "[full] Running stream sweep: $(FULL_STREAM_CONCURRENCY) for $(FULL_DURATION)s"
	$(MAKE) bench-stream CONCURRENCY="$(FULL_STREAM_CONCURRENCY)" DURATION=$(FULL_DURATION)
	@echo "[full] Done. Results in .benchmarks/bench_nonstream.json and .benchmarks/bench_stream.json"
	@echo "[full] Grafana: http://localhost:3000/d/tldw-llm-gateway (admin/admin)"
	@echo "[full] Prometheus: http://localhost:9090"
	@echo "[full] Tip: enable STREAMS_UNIFIED=1 on the server to populate SSE panels"
	@echo "[full] Stopping monitoring stack"
	$(MAKE) monitoring-down

# -----------------------------------------------------------------------------
# Lint (changed files only)
# -----------------------------------------------------------------------------
.PHONY: lint-changed

BASE ?=

lint-changed:
	@bash Helper_Scripts/Testing-related/lint-changed.sh $(BASE)

# -----------------------------------------------------------------------------
# Chat Streaming Load Harness (Scenario A starter)
# -----------------------------------------------------------------------------
.PHONY: load-chat-stream

LOAD_CONCURRENCY ?= 100
LOAD_STREAMS_PER_CLIENT ?= 1
LOAD_PROMPT_BYTES ?= 512
CHAT_MODEL ?= gpt-4o-mini

load-chat-stream:
	@echo "[load] Chat streaming load: concurrency=$(LOAD_CONCURRENCY) streams/client=$(LOAD_STREAMS_PER_CLIENT) prompt_bytes=$(LOAD_PROMPT_BYTES)"
	python Helper_Scripts/load_tests/chat_streaming_load.py \
		--base-url $(BASE_URL) \
		--api-key "$(API_KEY)" \
		--model "$(CHAT_MODEL)" \
		--concurrency $(LOAD_CONCURRENCY) \
		--streams-per-client $(LOAD_STREAMS_PER_CLIENT) \
		--prompt-bytes $(LOAD_PROMPT_BYTES)

.PHONY: load-chat-stream-sweep load-chat-stream-sweep-http2

LOAD_CONCURRENCY_STEPS ?= 50 100 200

load-chat-stream-sweep:
	@echo "[load] Chat streaming sweep (HTTP/1.1): $(LOAD_CONCURRENCY_STEPS)"
	python Helper_Scripts/load_tests/chat_streaming_sweep.py \
		--base-url $(BASE_URL) \
		--api-key "$(API_KEY)" \
		--model "$(CHAT_MODEL)" \
		--concurrency-steps $(LOAD_CONCURRENCY_STEPS) \
		--streams-per-client $(LOAD_STREAMS_PER_CLIENT) \
		--prompt-bytes $(LOAD_PROMPT_BYTES)

load-chat-stream-sweep-http2:
	@echo "[load] Chat streaming sweep (HTTP/2): $(LOAD_CONCURRENCY_STEPS)"
	python Helper_Scripts/load_tests/chat_streaming_sweep.py \
		--base-url $(BASE_URL) \
		--api-key "$(API_KEY)" \
		--model "$(CHAT_MODEL)" \
		--concurrency-steps $(LOAD_CONCURRENCY_STEPS) \
		--streams-per-client $(LOAD_STREAMS_PER_CLIENT) \
		--prompt-bytes $(LOAD_PROMPT_BYTES) \
		--http2

# Canonical Scenario A sweeps (short vs longer prompts)
.PHONY: scenario-a-short-http1 scenario-a-short-http2 scenario-a-long-http1 scenario-a-long-http2

SCENARIO_A_CONC_STEPS ?= 50 100 200 400 800
SCENARIO_A_SHORT_PROMPT_BYTES ?= 256
SCENARIO_A_LONG_PROMPT_BYTES ?= 1024

scenario-a-short-http1:
	@echo "[scenario-a] Short prompt, HTTP/1.1 (concurrency=$(SCENARIO_A_CONC_STEPS), prompt_bytes=$(SCENARIO_A_SHORT_PROMPT_BYTES))"
	$(MAKE) load-chat-stream-sweep \
		LOAD_CONCURRENCY_STEPS="$(SCENARIO_A_CONC_STEPS)" \
		LOAD_PROMPT_BYTES=$(SCENARIO_A_SHORT_PROMPT_BYTES)

scenario-a-short-http2:
	@echo "[scenario-a] Short prompt, HTTP/2 (concurrency=$(SCENARIO_A_CONC_STEPS), prompt_bytes=$(SCENARIO_A_SHORT_PROMPT_BYTES))"
	$(MAKE) load-chat-stream-sweep-http2 \
		LOAD_CONCURRENCY_STEPS="$(SCENARIO_A_CONC_STEPS)" \
		LOAD_PROMPT_BYTES=$(SCENARIO_A_SHORT_PROMPT_BYTES)

scenario-a-long-http1:
	@echo "[scenario-a] Longer prompt, HTTP/1.1 (concurrency=$(SCENARIO_A_CONC_STEPS), prompt_bytes=$(SCENARIO_A_LONG_PROMPT_BYTES))"
	$(MAKE) load-chat-stream-sweep \
		LOAD_CONCURRENCY_STEPS="$(SCENARIO_A_CONC_STEPS)" \
		LOAD_PROMPT_BYTES=$(SCENARIO_A_LONG_PROMPT_BYTES)

scenario-a-long-http2:
	@echo "[scenario-a] Longer prompt, HTTP/2 (concurrency=$(SCENARIO_A_CONC_STEPS), prompt_bytes=$(SCENARIO_A_LONG_PROMPT_BYTES))"
	$(MAKE) load-chat-stream-sweep-http2 \
		LOAD_CONCURRENCY_STEPS="$(SCENARIO_A_CONC_STEPS)" \
		LOAD_PROMPT_BYTES=$(SCENARIO_A_LONG_PROMPT_BYTES)

# -----------------------------------------------------------------------------
# STT Golden Adapter Validation (local/GPU-only)
# -----------------------------------------------------------------------------
.PHONY: stt-golden

STT_GOLDEN_AUDIO_DIR ?= ./test_models/stt_golden

stt-golden:
	@echo "[stt-golden] Running STT golden adapter tests against $(STT_GOLDEN_AUDIO_DIR)"
	TLDW_STT_GOLDEN_ENABLE=1 \
	TLDW_STT_GOLDEN_AUDIO_DIR="$(STT_GOLDEN_AUDIO_DIR)" \
	python -m pytest tldw_Server_API/tests/Audio/test_stt_adapters_golden.py -m "stt_golden" -v

# -----------------------------------------------------------------------------
# Sanity check (C2)
# -----------------------------------------------------------------------------
.PHONY: check

check:
	@FAIL=0; \
	echo "Running tldw sanity checks..."; \
	echo -n "  Server health:    "; \
	if curl -sf http://localhost:8000/health > /dev/null 2>&1; then echo "✓"; else echo "✗ (not running)"; FAIL=1; fi; \
	echo -n "  API key set:      "; \
	if grep -q '^SINGLE_USER_API_KEY=' $(TLDW_ENV_FILE) 2>/dev/null; then echo "✓"; else echo "✗ (check .env)"; FAIL=1; fi; \
	echo -n "  FFmpeg available:  "; \
	if command -v ffmpeg > /dev/null 2>&1; then echo "✓"; else echo "✗ (install ffmpeg)"; FAIL=1; fi; \
	echo -n "  Provider keys:    "; \
	if curl -sf -H "X-API-KEY: $$(grep '^SINGLE_USER_API_KEY=' $(TLDW_ENV_FILE) 2>/dev/null | cut -d= -f2-)" http://localhost:8000/api/v1/config/providers 2>/dev/null | grep -q '"any_configured":true'; then echo "✓"; else echo "✗ (add a provider key to .env)"; FAIL=1; fi; \
	if [ "$$FAIL" -ne 0 ]; then echo "Some checks failed."; exit 1; fi; \
	echo "All checks passed."

# -----------------------------------------------------------------------------
# Secret generation helper (C3)
# -----------------------------------------------------------------------------
.PHONY: generate-secrets

generate-secrets:
	@echo "Generated secrets (copy to your .env):"
	@echo ""
	@echo "JWT_SECRET_KEY=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
	@echo "MCP_JWT_SECRET=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
	@echo "MCP_API_KEY_SALT=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
	@echo "BYOK_ENCRYPTION_KEY=$$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
	@echo "SINGLE_USER_API_KEY=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

# -----------------------------------------------------------------------------
# Bandit (project-scoped)
# -----------------------------------------------------------------------------
.PHONY: bandit-b110-project

BANDIT_INI ?= .bandit
BANDIT_OUTPUT ?= /tmp/bandit_b110_project.json

bandit-b110-project:
	@echo "[bandit] Running project-scoped B110 scan with excludes from $(BANDIT_INI)"
	$(PYTHON) -m bandit --ini "$(BANDIT_INI)" -r . -t B110 -f json -o "$(BANDIT_OUTPUT)"
	@echo "[bandit] Wrote $(BANDIT_OUTPUT)"

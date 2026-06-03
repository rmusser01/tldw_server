# Troubleshooting Guide

Quick reference for the most common problems users encounter with tldw\_server. Each entry includes the symptoms you see, the likely cause, and step-by-step fixes.

> **Tip:** Before diving in, run `curl http://localhost:8000/health` to confirm the API is reachable. If this fails, start with [Connection Issues](#connection-issues).

---

## Table of Contents

- [Connection Issues](#connection-issues)
- [Authentication Issues](#authentication-issues)
- [Chat and LLM Issues](#chat-and-llm-issues)
- [Media Ingestion Issues](#media-ingestion-issues)
- [Transcription and Audio Issues](#transcription-and-audio-issues)
- [Docker Issues](#docker-issues)
- [Database Issues](#database-issues)
- [Multi-User and JWT Issues](#multi-user-and-jwt-issues)
- [CORS and Browser Issues](#cors-and-browser-issues)
- [Embedding and RAG Issues](#embedding-and-rag-issues)
- [MCP Issues](#mcp-issues)
- [WebUI Issues](#webui-issues)
- [Performance Issues](#performance-issues)
- [Installation Issues](#installation-issues)

---

## Connection Issues

### 1. "Connection refused" when hitting the API

**Symptoms:** `curl: (7) Failed to connect to localhost port 8000: Connection refused`

**Fix:**
1. Verify the server process is running: `ps aux | grep uvicorn` (local) or `docker compose ps` (Docker).
2. If nothing is running, start it: `make quickstart-local` (local) or `make quickstart` (Docker).
3. If the server crashed on startup, check the logs for the root cause (import error, missing config, port conflict).

### 2. Server starts but WebUI shows blank page or "can't reach your tldw server"

**Symptoms:** Browser loads but shows a connection error or empty screen.

**Fix:**
1. Confirm the API is healthy: `curl http://localhost:8000/health`
2. If the API is up but the WebUI cannot reach it, check that both are on the same origin (the Docker quickstart uses a proxy to avoid this).
3. If you run the API and WebUI separately, set `ALLOWED_ORIGINS` in `.env` to include the WebUI origin (e.g., `http://localhost:3000`).
4. Check the browser console (F12) for specific error messages.

### 3. "Port already in use"

**Symptoms:** `ERROR: [Errno 48] Address already in use` or `bind: address already in use`

**Fix:**
1. Find what is using port 8000: `lsof -i :8000` (macOS/Linux).
2. Stop the conflicting process, or start on a different port:
   ```bash
   uvicorn tldw_Server_API.app.main:app --port 8001
   ```
3. For Docker, change the port mapping in `docker-compose.yml` (e.g., `"8001:8000"`).

---

## Authentication Issues

### 4. "401 Unauthorized" on every request

**Symptoms:** API returns `{"detail": "Unauthorized"}` or `401` status.

**Fix (single-user mode):**
1. Retrieve your API key: `make show-api-key` or check `SINGLE_USER_API_KEY` in `tldw_Server_API/Config_Files/.env`.
2. Pass it in every request header: `-H "X-API-Key: YOUR_KEY_HERE"`
3. Ensure the key is at least 16 characters.

**Fix (multi-user mode):**
1. Verify your username and password are correct.
2. Obtain a JWT token via the `/api/v1/auth/login` endpoint first, then use it as `Authorization: Bearer <token>`.

### 5. "API key too short"

**Symptoms:** Server refuses to start or returns `422` with a message about key length.

**Fix:** Generate a proper key (minimum 16 characters):
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Paste the output as `SINGLE_USER_API_KEY` in `.env` and restart.

### 6. "Invalid credentials" after password change

**Symptoms:** Login worked before, now returns 401.

**Fix:**
1. If you changed `SINGLE_USER_API_KEY`, you must update any clients, scripts, or browser extensions using the old key.
2. For multi-user mode, verify the user account exists: check the users database or re-run the AuthNZ initializer.

---

## Chat and LLM Issues

### 7. "No LLM provider configured" / amber banner in WebUI

**Symptoms:** Chat page shows a warning banner. API returns `503` with `"no provider configured"`.

**Fix:**
1. Add at least one provider API key to `tldw_Server_API/Config_Files/.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Restart the server.
3. Verify providers are loaded: `curl http://localhost:8000/api/v1/llm/providers -H "X-API-Key: YOUR_KEY"`

### 8. Chat request returns 502 or "provider error"

**Symptoms:** Message sent but response is a 502 with details about the upstream provider.

**Fix:**
1. Verify your API key is valid with the provider directly:
   ```bash
   curl https://api.openai.com/v1/models -H "Authorization: Bearer sk-..."
   ```
2. Check if the provider is experiencing an outage (status pages: [OpenAI](https://status.openai.com), [Anthropic](https://status.anthropic.com)).
3. Try a different model or provider to isolate the issue.

### 9. Chat returns 429 "rate limit exceeded"

**Symptoms:** Requests work initially but fail with 429 after several calls.

**Fix:**
1. This is typically the upstream provider's rate limit, not tldw\_server's.
2. Reduce request frequency or upgrade your provider plan.
3. If using the built-in rate limiter, check `config.txt` for rate limit settings.

### 10. Streaming responses cut off or hang

**Symptoms:** Chat stream starts but stops mid-response or never completes.

**Fix:**
1. Check server logs for timeout or connection-reset errors.
2. If behind a reverse proxy (nginx, Caddy), ensure it supports SSE and has adequate timeout settings (e.g., `proxy_read_timeout 300s` in nginx).
3. Try a non-streaming request to isolate: set `"stream": false` in the request body.

---

## Media Ingestion Issues

### 11. "FFmpeg not found"

**Symptoms:** Video or audio upload fails. Logs show `ffmpeg: command not found` or the `/health` endpoint reports `ffmpeg_available: false`.

**Fix:**
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt update && sudo apt install -y ffmpeg`
- **Fedora/RHEL:** `sudo dnf install -y ffmpeg` (RPM Fusion may be required)
- **Docker:** FFmpeg is included in the official images; rebuild if using a custom image.

Verify: `ffmpeg -version`

### 12. YouTube download fails ("yt-dlp error")

**Symptoms:** URL ingestion returns an error mentioning yt-dlp, HTTP 403, or "format not available".

**Fix:**
1. Update yt-dlp to the latest version: `pip install -U yt-dlp`
2. Some sites require cookies. If authentication is needed, configure yt-dlp cookies.
3. Check if the URL is accessible: `yt-dlp --simulate "YOUR_URL"`

### 13. Upload fails or times out for large files

**Symptoms:** Upload hangs, browser times out, or server returns a 413/timeout.

**Fix:**
1. Check `max_video_file_size_mb` in `tldw_Server_API/Config_Files/config.txt` (default varies by deployment).
2. If behind a reverse proxy, increase client body size limits (e.g., nginx `client_max_body_size 2G;`).
3. For very large files, consider local ingestion via the API rather than browser upload.

### 14. PDF ingestion returns empty or garbled text

**Symptoms:** PDF is accepted but extracted text is missing or corrupted.

**Fix:**
1. The PDF may be image-based (scanned). tldw\_server uses text extraction by default; OCR requires additional setup.
2. Try a different PDF: `curl -X POST .../api/v1/media/process -F "file=@test.pdf"` to isolate the issue.
3. Check if `pymupdf` is installed: `python -c "import fitz; print(fitz.version)"`.

---

## Transcription and Audio Issues

### 15. "Transcription takes too long"

**Symptoms:** Audio processing runs for many minutes on moderate-length files.

**Fix:**
1. Use a smaller/faster model (e.g., `tiny.en` or `base.en` instead of `large-v3`).
2. If you have an NVIDIA GPU, enable CUDA acceleration (see [GPU Audio Setup](./First_Time_Audio_Setup_GPU_Accelerated.md)).
3. For long files, the system chunks automatically, but total wall time scales with file length.

### 16. CUDA / torch mismatch

**Symptoms:** `RuntimeError: CUDA error` or `torch not compiled with CUDA` or `cudnn*.dll not found`.

**Fix:**
1. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. If `False`, install the CUDA-compatible PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. On Windows, ensure `cudnn` DLLs are on your PATH. See the [NVIDIA cuDNN docs](https://developer.nvidia.com/cudnn).

### 17. "Numpy is not available" error

**Symptoms:** Runtime error about numpy during transcription model loading.

**Fix:** Pin numpy to a compatible version:
```bash
pip install "numpy<2"
```

---

## Docker Issues

### 18. Container keeps restarting

**Symptoms:** `docker compose ps` shows the container in a restart loop.

**Fix:**
1. Check logs: `docker compose logs app --tail 100`
2. Common causes:
   - **Missing MCP secrets:** Set `MCP_JWT_SECRET` and `MCP_API_KEY_SALT` in `.env` (or let the Docker entrypoint auto-generate them).
   - **Bad `DATABASE_URL`:** Verify the Postgres host is reachable from within Docker (use `postgres` as hostname, not `localhost`).
   - **Port conflict:** Another service is bound to port 8000 on the host.

### 19. "Permission denied" on database files

**Symptoms:** Logs show `PermissionError` or `sqlite3.OperationalError: unable to open database file`.

**Fix:**
1. Ensure the `Databases/` directory is writable: `chmod -R 755 Databases/`
2. In Docker, check volume mount ownership. Add `user: "1000:1000"` in `docker-compose.yml` if needed.

### 20. Docker build fails on chromadb

**Symptoms:** `pip install chromadb` fails during image build with compilation errors.

**Fix:**
1. Ensure the base image has build essentials: `apt-get install -y build-essential`.
2. If on ARM (Apple Silicon), ensure you are building for the correct platform or using a pre-built wheel.
3. See [chromadb issue #189](https://github.com/chroma-core/chroma/issues/189) for known workarounds.

### 21. "make: command not found" on Windows

**Symptoms:** `make quickstart` fails on Windows.

**Fix:** Use WSL2 or Git Bash. Alternatively, run the Docker commands directly:
```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.webui.yml up -d --build
```

---

## Database Issues

### 22. "database is locked" (SQLite)

**Symptoms:** `sqlite3.OperationalError: database is locked` under concurrent use.

**Fix:**
1. SQLite is best for single-user or low-concurrency setups. For multi-user deployments, switch to PostgreSQL.
2. Ensure WAL mode is enabled (it is by default). Check `sqlite_wal_mode = true` in `config.txt` under `[Database]`.
3. Keep database transactions short-lived; avoid long-running reads that block writers.

### 23. Database migration errors on startup

**Symptoms:** Server crashes with schema mismatch or missing column errors.

**Fix:**
1. Back up your databases first: `cp -r Databases/ Databases_backup/`
2. Re-run the server; migrations typically auto-apply on startup.
3. If migrations fail, check logs for the specific migration step that errored. Some schema changes require a fresh database (rare).

### 24. "user_databases" directory missing

**Symptoms:** Content operations fail with file-not-found errors referencing `Databases/user_databases/`.

**Fix:** Create the directory:
```bash
mkdir -p Databases/user_databases
```
The server auto-creates per-user subdirectories on first use.

---

## Multi-User and JWT Issues

### 25. Locked out with no admin account

**Symptoms:** Switched to `AUTH_MODE=multi_user` but cannot log in.

**Fix:**
1. Run the `create_admin` CLI to create (or reset) the admin user:
   ```bash
   python -m tldw_Server_API.app.core.AuthNZ.create_admin --username tldw-admin --password <new-password>
   ```
2. Or in Docker:
   ```bash
   docker compose exec app python -m tldw_Server_API.app.core.AuthNZ.create_admin \
     --username tldw-admin --password <new-password>
   ```

### 26. JWT tokens rejected after server restart

**Symptoms:** Previously valid tokens return 401 after restarting.

**Fix:**
1. Ensure `JWT_SECRET_KEY` is set explicitly in `.env` (not auto-generated on each boot).
2. Generate a persistent key:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
3. Paste it into `.env` as `JWT_SECRET_KEY=...` and restart.

### 27. "DATABASE_URL" points to wrong database in multi-user mode

**Symptoms:** Auth works in single-user but breaks when switching to multi-user.

**Fix:**
1. Multi-user mode requires a `users.db` (SQLite) or a PostgreSQL connection.
2. Verify `DATABASE_URL` in `.env`:
   - SQLite: `sqlite:///./Databases/users.db`
   - Postgres: `postgresql://user:pass@host:5432/dbname`
3. Ensure the database file or Postgres instance is accessible.

---

## CORS and Browser Issues

### 28. "Cross-origin request blocked" in browser console

**Symptoms:** API calls from the WebUI fail; browser DevTools shows a CORS error.

**Fix:**
1. Set `ALLOWED_ORIGINS` in `.env` to include your WebUI origin:
   ```
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
   ```
2. Restart the server.
3. If developing locally and you want to skip CORS entirely (not for production):
   ```
   DISABLE_CORS=true
   ```

### 29. "ALLOWED_ORIGINS is empty" crash on startup

**Symptoms:** Server refuses to start with `CORS is enabled but ALLOWED_ORIGINS is empty`.

**Fix:**
1. Add at least one origin to `.env`:
   ```
   ALLOWED_ORIGINS=http://localhost:3000
   ```
2. Or for wide-open local dev (no credentials): `ALLOWED_ORIGINS=*` with `CORS_ALLOW_CREDENTIALS=false`.
3. In production, always list explicit origins.

### 30. CORS credentials conflict

**Symptoms:** Startup error: `ALLOWED_ORIGINS cannot include '*' when credentials are enabled`.

**Fix:** You cannot use `ALLOWED_ORIGINS=*` together with `CORS_ALLOW_CREDENTIALS=true`. Either:
- List specific origins instead of `*`, or
- Set `CORS_ALLOW_CREDENTIALS=false`.

---

## Embedding and RAG Issues

### 31. "Embedding model not found" or slow first query

**Symptoms:** First RAG or embedding request takes a long time or fails with a download error.

**Fix:**
1. The default embedding model downloads on first use. Ensure internet access is available.
2. Check the model directory path in `config.txt` under `model_dir`.
3. If behind a corporate proxy, set `HTTP_PROXY`/`HTTPS_PROXY` environment variables.

### 32. ChromaDB collection errors

**Symptoms:** Errors about missing collections, dimension mismatch, or ChromaDB startup failures.

**Fix:**
1. Verify the ChromaDB path in `config.txt` (`chroma_db_path = Databases/chroma_db`).
2. A dimension mismatch means you changed the embedding model after initial indexing. Re-index or delete the old ChromaDB directory:
   ```bash
   rm -rf Databases/chroma_db
   ```
3. Restart and re-ingest content to rebuild the index.

---

## MCP Issues

### 33. MCP secrets missing or placeholder

**Symptoms:** Server fails on startup with errors about `MCP_JWT_SECRET` or `MCP_API_KEY_SALT`.

**Fix:**
1. Docker users: the entrypoint auto-generates these if they are set to placeholder values. Rebuild: `docker compose up -d --build`.
2. Non-Docker users: generate and set them manually in `.env`:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   # Set output as MCP_JWT_SECRET=... and MCP_API_KEY_SALT=...
   ```
3. Each secret must be at least 32 characters and unique.

### 34. MCP WebSocket connection rejected

**Symptoms:** WebSocket clients cannot connect to MCP endpoints.

**Fix:**
1. Check `MCP_WS_AUTH_REQUIRED` in `.env` (default: `true`). Ensure your client sends valid auth.
2. Verify `MCP_WS_ALLOWED_ORIGINS` includes your client's origin.
3. If behind a reverse proxy, ensure it supports WebSocket upgrade headers.

---

## WebUI Issues

### 35. WebUI login page loops or shows "session expired"

**Symptoms:** Logging in redirects back to the login page.

**Fix:**
1. Clear browser cookies and local storage for the site.
2. Verify that `JWT_SECRET_KEY` has not changed between server restarts (see [issue #26](#26-jwt-tokens-rejected-after-server-restart)).
3. Check browser DevTools Network tab for the specific error on the auth endpoint.

### 36. "NEXT_PUBLIC_X_API_KEY" confusion

**Symptoms:** Frontend admin features do not work, or you get 401 errors only from the WebUI.

**Fix:**
1. `NEXT_PUBLIC_X_API_KEY` is for the frontend admin API and is **not** the same as `SINGLE_USER_API_KEY`.
2. The Docker quickstart proxy handles this automatically. Only set this variable in advanced (non-proxy) deployments.
3. If needed, set it in `.env` and rebuild the WebUI.

---

## Performance Issues

### 37. Slow API response times

**Symptoms:** Requests take several seconds even for simple queries.

**Fix:**
1. If using a remote LLM provider (OpenAI, Anthropic, etc.), latency is dominated by the provider round-trip; this is expected.
2. For faster responses, consider a local model (Ollama, llama.cpp, vLLM).
3. Check if embedding model downloads or ChromaDB initialization are happening on-demand.

### 38. High memory usage

**Symptoms:** Server process consumes multiple GB of RAM, system becomes sluggish.

**Fix:**
1. Local STT models (especially `large-v3`) require significant memory. Use a smaller model (`tiny.en`, `base.en`).
2. Reduce concurrent ingestion tasks.
3. If embedding models are loaded locally, they consume RAM proportional to model size. Consider using an API-based embedding provider.

### 39. Redis connection errors (optional component)

**Symptoms:** Logs show Redis connection failures or `REDIS_URL` errors.

**Fix:**
1. Redis is optional. If you are not using it, ensure `REDIS_URL` is not set or is empty in `.env`.
2. If you want Redis, verify it is running and accessible at the configured URL.
3. Check `docker compose ps` to ensure the Redis container is healthy.

---

## Installation Issues

### 40. `ModuleNotFoundError` on startup

**Symptoms:** `ModuleNotFoundError: No module named 'loguru'` (or similar).

**Fix:**
1. Activate your virtual environment: `source .venv/bin/activate`
2. Install dependencies: `pip install -e .` or `pip install -e ".[dev]"`
3. If using Docker, rebuild: `docker compose build --no-cache`

### 41. Python version too old

**Symptoms:** `SyntaxError` or `ImportError` on startup due to Python 3.9 or earlier features.

**Fix:** tldw\_server requires Python 3.10+. Check your version:
```bash
python3 --version
```
If too old, install a newer version and specify it:
```bash
make quickstart-install PYTHON=python3.12
```

### 42. `pip install` fails on platform-specific packages

**Symptoms:** Build errors for packages like `chromadb`, `hnswlib`, or `faster-whisper` during install.

**Fix:**
1. Ensure build tools are installed:
   - **macOS:** `xcode-select --install`
   - **Ubuntu/Debian:** `sudo apt install build-essential python3-dev`
2. For Apple Silicon, ensure you are using an ARM-native Python (not Rosetta).
3. If a specific package fails, check its GitHub issues for platform-specific workarounds.

---

## Still Stuck?

If your issue is not listed here:

1. **Check the logs.** Server logs contain detailed error messages and stack traces. Look for the first error after startup.
2. **Search GitHub Issues.** Your problem may already be reported: [github.com/rmusser01/tldw\_server/issues](https://github.com/rmusser01/tldw_server/issues)
3. **Ask in Discussions.** The community forum is a good place for questions: [github.com/rmusser01/tldw\_server/discussions](https://github.com/rmusser01/tldw_server/discussions)
4. **File a bug.** Include your OS, Python version, deployment method (Docker/local), and the full error message.

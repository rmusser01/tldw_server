from pathlib import Path


DOCKERFILE = Path("tldw_Server_API/app/core/MCP_unified/docker/Dockerfile")
ENTRYPOINT = Path("tldw_Server_API/app/core/MCP_unified/docker/entrypoint.sh")


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_mcp_dockerfile_copies_real_entrypoint_and_boots_main_app() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    _ensure(
        "COPY --chown=mcp:mcp tldw_Server_API/app/core/MCP_unified/docker/entrypoint.sh /entrypoint.sh" in text,
        "Dockerfile does not copy the repo-local MCP entrypoint into the image",
    )
    _ensure(
        'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in text,
        "Dockerfile does not boot the real FastAPI app target",
    )
    _ensure(
        "mkdir -p /app /data/databases /data/logs /data/cache /data/media /data/transcripts" in text,
        "Dockerfile does not pre-create all writable /data subdirectories before dropping privileges",
    )


def test_mcp_entrypoint_script_exists_and_execs_command() -> None:
    _ensure(ENTRYPOINT.exists(), "MCP Docker entrypoint script is missing")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    _ensure('exec "$@"' in entrypoint, "Entrypoint does not exec the runtime command")
    _ensure("mkdir -p /data" not in entrypoint, "Entrypoint should not rely on runtime directory creation under /data")

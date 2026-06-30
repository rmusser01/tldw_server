from pathlib import Path


DOCKER_README = Path("tldw_Server_API/app/core/MCP_unified/docker/README.md")
DOCKERFILE = Path("tldw_Server_API/app/core/MCP_unified/docker/Dockerfile")
ENTRYPOINT = Path("tldw_Server_API/app/core/MCP_unified/docker/entrypoint.sh")
CORE_README = Path("tldw_Server_API/app/core/MCP_unified/README.md")


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_mcp_specific_docker_path_is_marked_experimental() -> None:
    _ensure(DOCKER_README.exists(), "MCP-specific Docker status README is missing")
    text = DOCKER_README.read_text(encoding="utf-8").lower()

    _ensure(
        "experimental" in text,
        "MCP-specific Docker README should mark this path experimental",
    )
    _ensure(
        "not the supported standalone gateway" in text,
        "MCP-specific Docker README should not imply this is the supported standalone gateway",
    )
    _ensure(
        "embedded in tldw server today" in text,
        "MCP-specific Docker README should point users back to the embedded TLDW Server path",
    )


def test_mcp_specific_dockerfile_does_not_claim_production_readiness() -> None:
    _ensure(DOCKERFILE.exists(), "MCP-specific Dockerfile is missing")
    text = DOCKERFILE.read_text(encoding="utf-8").lower()

    _ensure(
        "experimental" in text,
        "MCP-specific Dockerfile should mark this path experimental",
    )
    for stale_phrase in (
        "optimized for production deployment",
        "production-ready",
        "production grade",
    ):
        _ensure(
            stale_phrase not in text,
            f"MCP-specific Dockerfile should not claim {stale_phrase!r}",
        )


def test_primary_mcp_readme_flags_docker_deployment_as_experimental() -> None:
    text = CORE_README.read_text(encoding="utf-8").lower()
    docker_section = text.split("### docker deployment", 1)[1].split("##", 1)[0]

    _ensure(
        "experimental" in docker_section,
        "Core MCP README Docker section should warn that the MCP-specific image is experimental",
    )
    _ensure(
        "not the supported standalone gateway" in docker_section,
        "Core MCP README Docker section should not present this path as the supported standalone gateway",
    )


def test_mcp_entrypoint_script_exists_and_execs_command() -> None:
    _ensure(ENTRYPOINT.exists(), "MCP Docker entrypoint script is missing")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    _ensure('exec "$@"' in entrypoint, "Entrypoint does not exec the runtime command")
    _ensure("mkdir -p /data" not in entrypoint, "Entrypoint should not rely on runtime directory creation under /data")

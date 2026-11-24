import re
import shlex
from pathlib import Path

import pytest

DOCKER_README = Path("tldw_Server_API/app/core/MCP_unified/docker/README.md")
DOCKERFILE = Path("tldw_Server_API/app/core/MCP_unified/docker/Dockerfile")
ROOT_DOCKERFILE = Path("Dockerfiles/Dockerfile.prod")
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


def test_root_dockerfile_installs_standalone_mcp_package_source() -> None:
    _ensure(ROOT_DOCKERFILE.exists(), "Root production Dockerfile is missing")
    dockerfile = ROOT_DOCKERFILE.read_text(encoding="utf-8")
    copy_index = install_index = -1
    extra_name = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    for index, line in enumerate(dockerfile.replace("\\\n", " ").splitlines()):
        instruction, _, body = line.strip().partition(" ")
        if instruction == "COPY" and copy_index < 0:
            if shlex.split(body, comments=True) == ["apps/mcp-unified/src", "/app/apps/mcp-unified/src"]:
                copy_index = index
        elif instruction == "RUN" and install_index < 0:
            command_start = 0
            previous_separator = ""
            installs_root = False
            # Keep quoted/escaped spans intact; shlex decodes each simple command.
            for boundary in re.finditer(
                r"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|\#[^\n]*|(?P<separator>[;&|]+|$)""", body,
            ):
                separator = boundary.group("separator")
                if separator is None:
                    continue
                arguments = shlex.split(body[command_start:boundary.start()], comments=True, posix=True)
                command_start = boundary.end()
                if separator not in {"", "&&", "||", ";"}:
                    break
                # A final semicolon alone may leave an empty tail, including a comment.
                if not arguments and not (separator == "" and previous_separator == ";"):
                    break
                if (
                    len(arguments) == 4
                    and arguments[:3] == ["pip", "install", "--prefix=/install"]
                    and re.fullmatch(rf"\.(?:\[\s*{extra_name}(?:\s*,\s*{extra_name})*\s*\])?", arguments[3])
                ):
                    installs_root = True
                previous_separator = separator
            else:
                if installs_root:
                    install_index = index

    _ensure(copy_index >= 0, "Root Docker image must copy standalone MCP Unified source")
    _ensure(install_index >= 0, "Root Docker image must install the root package")
    _ensure(copy_index < install_index, "Standalone MCP Unified source must be copied before pip install")


@pytest.mark.parametrize(
    "requirement", [".", '"."', ".[audio]", '".[audio]"', "'.[audio,tts]'", '".[other-extra, other_extra]"'],
)
@pytest.mark.parametrize("prefix", ["pip install --upgrade pip", 'echo x"&&"', r"echo \;"])
@pytest.mark.parametrize("ending", ["", ";", "; # complete", " && true"])
def test_root_dockerfile_contract_accepts_root_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requirement: str,
    prefix: str,
    ending: str,
) -> None:
    """Quoted root requirements with arbitrary valid extras still install the root."""
    dockerfile = tmp_path / "Dockerfile.prod"
    dockerfile.write_text(
        "COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n"
        f"RUN {prefix} && \\\n"
        f"    pip install --prefix=/install {requirement}{ending}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ROOT_DOCKERFILE", dockerfile)

    test_root_dockerfile_installs_standalone_mcp_package_source()


@pytest.mark.parametrize(
    "instruction",
    [
        "RUN pip install --prefix=/install requests",
        "RUN pip install --prefix=/install ./apps/mcp-unified",
        "RUN pip install --prefix=/install ..",
        'RUN pip install --prefix=/install "..[audio]"',
        'RUN pip install --prefix=/install "../.[audio]"',
        'RUN pip install --prefix=/install ".[audio]/child"',
        'RUN pip install --prefix=/install ".[-]"',
        'RUN pip install --prefix=/install ".[.audio]"',
        'RUN pip install --prefix=/install ".[audio-]"',
        "RUN pip install .",
        "RUN pip install --prefix=/elsewhere .",
        "# RUN pip install --prefix=/install .",
        "pip install --prefix=/install .",
        "ENV EXAMPLE=pip install --prefix=/install .",
        "RUN echo pip install --prefix=/install .",
        'RUN echo "pip install --prefix=/install ."',
        'RUN echo "&&" pip install --prefix=/install .',
        'RUN echo x"&&" pip install --prefix=/install .',
        r"RUN echo \; pip install --prefix=/install .",
        "RUN true # pip install --prefix=/install .",
        "RUN true # ignored && pip install --prefix=/install .",
        "RUN pip install --prefix=/elsewhere requests && pip install .",
        "RUN pip install --prefix=/install requests && echo .",
        "RUN true &&&& pip install --prefix=/install .",
        "RUN true &&|| pip install --prefix=/install .",
        "RUN true ;; pip install --prefix=/install .",
        "RUN pip install --prefix=/install . &&&& true",
        "RUN pip install --prefix=/install . &&|| true",
        "RUN pip install --prefix=/install . ;; true",
        "RUN pip install --prefix=/install . &&&&",
        "RUN pip install --prefix=/install . &&||",
        "RUN pip install --prefix=/install . ;;",
        "RUN true && || pip install --prefix=/install .",
        "RUN true ; ; pip install --prefix=/install .",
        "RUN && pip install --prefix=/install .",
        "RUN || pip install --prefix=/install .",
        "RUN ; pip install --prefix=/install .",
        "RUN pip install --prefix=/install . &&",
        "RUN pip install --prefix=/install . ||",
    ],
)
def test_root_dockerfile_contract_rejects_non_root_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
) -> None:
    """Only a RUN command installing this root into /install satisfies the contract."""
    dockerfile = tmp_path / "Dockerfile.prod"
    dockerfile.write_text(
        f"COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n{instruction}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ROOT_DOCKERFILE", dockerfile)

    with pytest.raises(AssertionError, match="^Root Docker image must install the root package$"):
        test_root_dockerfile_installs_standalone_mcp_package_source()


@pytest.mark.parametrize(
    ("dockerfile_text", "message"),
    [
        (
            "RUN pip install --prefix=/install .\n",
            "Root Docker image must copy standalone MCP Unified source",
        ),
        (
            "# COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n"
            "RUN pip install --prefix=/install .\n",
            "Root Docker image must copy standalone MCP Unified source",
        ),
        (
            "RUN echo COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n"
            "RUN pip install --prefix=/install .\n",
            "Root Docker image must copy standalone MCP Unified source",
        ),
        (
            "COPY apps/mcp-unified/src /app/wrong-destination\n"
            "RUN pip install --prefix=/install .\n",
            "Root Docker image must copy standalone MCP Unified source",
        ),
        (
            "RUN pip install --prefix=/install .\n"
            "COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n",
            "Standalone MCP Unified source must be copied before pip install",
        ),
        (
            "# COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n"
            "RUN pip install --prefix=/install .\n"
            "COPY apps/mcp-unified/src /app/apps/mcp-unified/src\n",
            "Standalone MCP Unified source must be copied before pip install",
        ),
    ],
)
def test_root_dockerfile_contract_requires_source_copy_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dockerfile_text: str,
    message: str,
) -> None:
    """Missing, commented, or late source copies must not satisfy the contract."""
    dockerfile = tmp_path / "Dockerfile.prod"
    dockerfile.write_text(dockerfile_text, encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT_DOCKERFILE", dockerfile)

    with pytest.raises(AssertionError, match=f"^{message}$"):
        test_root_dockerfile_installs_standalone_mcp_package_source()

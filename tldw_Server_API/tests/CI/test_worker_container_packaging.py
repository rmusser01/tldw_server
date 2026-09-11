"""Validate that worker build inputs can satisfy the root package manifest."""

import shlex
from pathlib import Path, PurePosixPath

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[3]
WORKERS = ("Dockerfile.worker", "Dockerfile.audio_gpu_worker")


def _copy_inputs(dockerfile: str) -> list[tuple[Path, PurePosixPath]]:
    """Resolve repository COPY instructions in the worker's /app build context."""
    inputs = []
    for line in (ROOT / "Dockerfiles" / dockerfile).read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        tokens = shlex.split(line)[1:]
        if any(token.startswith("--from=") for token in tokens):
            continue
        tokens = [token for token in tokens if not token.startswith("--")]
        destination = PurePosixPath(tokens[-1])
        for source in tokens[:-1]:
            path = ROOT / source
            target = destination
            if tokens[-1].endswith("/") and path.is_file():
                target /= path.name
            inputs.append((path, target))
    return inputs


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", WORKERS)
def test_worker_copy_sources_exist(dockerfile: str) -> None:
    """A stale root Config_Files reference must fail before release publishing."""
    missing = [str(path.relative_to(ROOT)) for path, _ in _copy_inputs(dockerfile) if not path.exists()]
    assert not missing, f"COPY sources missing from build context: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", WORKERS)
def test_worker_build_covers_declared_local_package_inputs(dockerfile: str) -> None:
    """Every manifest-declared package and profile data file must reach /app."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    setuptools = config["tool"]["setuptools"]
    required = [Path("LICENSE"), Path("LICENSES")]
    required.extend(Path(path) for path in setuptools["packages"]["find"]["where"] if path != ".")
    for patterns in setuptools["data-files"].values():
        for pattern in patterns:
            required.extend(path.relative_to(ROOT) for path in ROOT.glob(pattern))

    copies = _copy_inputs(dockerfile)
    missing = []
    for path in required:
        target = PurePosixPath("/app") / path.as_posix()
        if not any(
            source.exists()
            and (ROOT / path).is_relative_to(source)
            and destination / (ROOT / path).relative_to(source).as_posix() == target
            for source, destination in copies
        ):
            missing.append(str(path))
    assert not missing, f"Declared package inputs omitted from worker: {missing}"

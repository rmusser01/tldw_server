from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_dependency(name: str) -> str:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_prefix = name.lower()
    for dependency in pyproject["project"]["dependencies"]:
        if dependency.lower().startswith(dependency_prefix):
            return dependency
    raise AssertionError(f"Missing direct dependency for {name!r}")


def _expect_project_dependency(name: str, expected: str) -> None:
    actual = _project_dependency(name)
    if actual != expected:
        raise AssertionError(f"Expected {name!r} dependency {expected!r}, got {actual!r}")


def test_starlette_dependency_floor_excludes_badhost_cve_versions():
    """Guard against resolving Starlette versions vulnerable to CVE-2026-48710."""
    _expect_project_dependency("starlette", "starlette>=1.0.1")


def test_fastapi_dependency_floor_supports_starlette_1_series():
    _expect_project_dependency("fastapi", "fastapi>=0.136.3")


def test_python_multipart_dependency_floor_excludes_header_dos_cve_versions():
    """Guard against resolving python-multipart versions vulnerable to CVE-2026-42561."""
    _expect_project_dependency("python-multipart", "python-multipart>=0.0.27")

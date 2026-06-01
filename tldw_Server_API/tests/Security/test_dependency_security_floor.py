import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOCK_SERVER_ROOT = PROJECT_ROOT / "mock_openai_server"
PACKAGE_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dependency_name(dependency: str) -> str:
    match = PACKAGE_NAME_PATTERN.match(dependency)
    if not match:
        raise AssertionError(f"Unable to parse dependency name from {dependency!r}")
    return _normalize_package_name(match.group(1))


def _find_dependency(dependencies: list[str], name: str) -> str:
    dependency_name = _normalize_package_name(name)
    for dependency in dependencies:
        if _dependency_name(dependency) == dependency_name:
            return dependency
    raise AssertionError(f"Missing direct dependency for {name!r}")


def _project_dependencies() -> list[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def _project_dependency(name: str) -> str:
    return _find_dependency(_project_dependencies(), name)


def _mock_pyproject() -> dict:
    return tomllib.loads((MOCK_SERVER_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _mock_setup_keywords() -> dict[str, object]:
    setup_path = MOCK_SERVER_ROOT / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            wanted_keys = {"install_requires", "python_requires"}
            return {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg in wanted_keys
            }
    raise AssertionError("mock_openai_server/setup.py does not call setup()")


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


def test_dependency_lookup_uses_exact_normalized_package_names():
    dependencies = ["pydantic-core>=2.0.0", "pydantic>=2.0.0"]
    actual = _find_dependency(dependencies, "pydantic")
    if actual != "pydantic>=2.0.0":
        raise AssertionError(f"Expected exact pydantic match, got {actual!r}")


def test_mock_server_manifests_share_multipart_security_floor():
    expected = "python-multipart>=0.0.27"
    mock_pyproject_dependencies = _mock_pyproject()["project"]["dependencies"]
    mock_requirements = (MOCK_SERVER_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    mock_setup_dependencies = _mock_setup_keywords()["install_requires"]

    for source_name, dependencies in {
        "mock pyproject": mock_pyproject_dependencies,
        "mock requirements": mock_requirements,
        "mock setup.py": mock_setup_dependencies,
    }.items():
        actual = _find_dependency(dependencies, "python-multipart")
        if actual != expected:
            raise AssertionError(f"Expected {source_name} to use {expected!r}, got {actual!r}")


def test_mock_server_python_floor_matches_multipart_security_floor():
    mock_pyproject = _mock_pyproject()
    mock_setup = _mock_setup_keywords()

    if mock_pyproject["project"]["requires-python"] != ">=3.10":
        raise AssertionError("mock pyproject must require Python >=3.10")
    if mock_setup["python_requires"] != ">=3.10":
        raise AssertionError("mock setup.py must require Python >=3.10")

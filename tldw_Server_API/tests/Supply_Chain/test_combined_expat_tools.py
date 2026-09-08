"""Fail-closed selection of the combined candidate's isolated test wheels."""

import copy
import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/combined-expat/test-tools.py"
VERSIONS = {
    "iniconfig": "2.3.0",
    "packaging": "26.3",
    "pluggy": "1.6.0",
    "pygments": "2.21.0",
    "pytest": "9.1.1",
    "pytest-asyncio": "1.4.0",
    "pytest-timeout": "2.4.0",
    "typing-extensions": "4.16.0",
}


def module():
    assert SCRIPT.is_file(), "isolated test-wheel selection is not implemented"
    spec = importlib.util.spec_from_file_location("combined_test_tools", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def locked_tools():
    return {
        "package": [
            {
                "name": name,
                "version": version,
                "source": {"registry": "https://pypi.org/simple"},
                "wheels": [
                    {
                        "url": f"https://files.pythonhosted.org/packages/aa/{name.replace('-', '_')}-{version}-py3-none-any.whl",
                        "hash": "sha256:" + "a" * 64,
                    }
                ],
            }
            for name, version in VERSIONS.items()
        ]
    }


def test_emits_only_the_fixed_wheel_cohort_without_mutating_lock():
    lock = locked_tools()
    lock["package"].append({"name": "unrelated-runtime", "version": "1.0"})
    before = copy.deepcopy(lock)
    output = module().requirements(lock)
    assert len(output.splitlines()) == 8
    assert (
        "pytest @ https://files.pythonhosted.org/packages/aa/pytest-9.1.1-py3-none-any.whl --hash=sha256:" + "a" * 64
        in output
    )
    assert "unrelated-runtime" not in output
    assert lock == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "normalized-duplicate",
        "version",
        "registry",
        "no-wheel",
        "two-wheels",
        "sdist",
        "platform-wheel",
        "http",
        "foreign-host",
        "credentials",
        "query",
        "fragment",
        "newline",
        "wrong-name",
        "wrong-wheel-version",
        "hash",
        "hash-type",
    ],
)
def test_rejects_ambiguous_or_unsafe_wheel_requirements(mutation):
    lock = locked_tools()
    package = lock["package"][-1]
    wheel = package["wheels"][0]
    if mutation == "missing":
        lock["package"].pop()
    elif mutation in {"duplicate", "normalized-duplicate"}:
        duplicate = copy.deepcopy(package)
        if mutation == "normalized-duplicate":
            duplicate["name"] = "Typing_Extensions"
        lock["package"].append(duplicate)
    elif mutation == "version":
        package["version"] = "0.0.1"
    elif mutation == "registry":
        package["source"]["registry"] = "https://example.invalid/simple"
    elif mutation == "no-wheel":
        package["wheels"] = []
    elif mutation == "two-wheels":
        package["wheels"].append(copy.deepcopy(wheel))
    elif mutation == "sdist":
        wheel["url"] = wheel["url"].replace("-py3-none-any.whl", ".tar.gz")
    elif mutation == "platform-wheel":
        wheel["url"] = wheel["url"].replace("py3-none-any", "cp312-cp312-win_amd64")
    elif mutation == "http":
        wheel["url"] = wheel["url"].replace("https:", "http:")
    elif mutation == "foreign-host":
        wheel["url"] = wheel["url"].replace("files.pythonhosted.org", "example.invalid")
    elif mutation == "credentials":
        wheel["url"] = wheel["url"].replace("https://", "https://user:pass@")
    elif mutation == "query":
        wheel["url"] += "?redirect=elsewhere"
    elif mutation == "fragment":
        wheel["url"] += "#fragment"
    elif mutation == "newline":
        wheel["url"] += "\n--extra-index-url https://example.invalid"
    elif mutation == "wrong-name":
        wheel["url"] = wheel["url"].replace("typing_extensions", "unrelated")
    elif mutation == "wrong-wheel-version":
        wheel["url"] = wheel["url"].replace("4.16.0", "4.15.0")
    elif mutation == "hash":
        wheel["hash"] = "sha256:invalid"
    elif mutation == "hash-type":
        wheel["hash"] = 123
    with pytest.raises(ValueError):
        module().requirements(lock)


def test_repository_lock_emits_exact_wheel_urls_and_hashes():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    output = module().requirements(lock)
    assert len(output.splitlines()) == 8
    for package in lock["package"]:
        if package["name"] in VERSIONS:
            wheel = package["wheels"][0]
            assert f"{package['name']} @ {wheel['url']} --hash={wheel['hash']}" in output.splitlines()


def test_cli_does_not_emit_partial_requirements_when_last_record_is_invalid(tmp_path):
    module()
    records = locked_tools()["package"]
    records[-1]["wheels"][0]["hash"] = "sha256:invalid"
    content = []
    for package in records:
        content.extend(
            [
                "[[package]]",
                f"name = {json.dumps(package['name'])}",
                f"version = {json.dumps(package['version'])}",
                'source = { registry = "https://pypi.org/simple" }',
                "[[package.wheels]]",
                f"url = {json.dumps(package['wheels'][0]['url'])}",
                f"hash = {json.dumps(package['wheels'][0]['hash'])}",
            ]
        )
    lock = tmp_path / "uv.lock"
    lock.write_text("\n".join(content))
    result = subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), str(lock)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert result.stdout == ""

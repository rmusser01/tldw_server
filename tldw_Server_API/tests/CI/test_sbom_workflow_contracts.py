from pathlib import Path

import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _get_step(steps: list[dict], name: str) -> dict:
    matching = [step for step in steps if step.get("name") == name]
    assert matching, f"{name} step missing"
    return matching[0]


def test_sbom_workflow_supports_pyproject_dependency_source() -> None:
    workflow = _load(".github/workflows/sbom.yml")
    steps = workflow["jobs"]["build-sbom"]["steps"]
    python_sbom_step = _get_step(steps, "Generate Python SBOM (CycloneDX)")
    run_script = python_sbom_step["run"]

    assert 'gen_from_requirements "$repo_root/tldw_Server_API/requirements.txt"' in run_script
    assert 'gen_from_requirements "$repo_root/requirements.txt"' in run_script
    assert 'cyclonedx-py requirements "$req_file" "${pyproject_args[@]}" -o "$out_file"' in run_script
    assert 'cyclonedx-py requirements -i "$req_file" "${pyproject_args[@]}" -o "$out_file"' in run_script
    assert "gen_requirements_from_pyproject" in run_script
    assert "tomllib" in run_script
    assert '[project].dependencies' in run_script
    assert 'gen_requirements_from_pyproject "$repo_root/pyproject.toml"' in run_script

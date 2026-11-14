"""The test-quality triage ratchet must stay green (no new or worse offenses).

Runs ``Helper_Scripts/ci/test_quality_triage.py --enforce`` in a subprocess
from the repo root so the committed baseline gates all future test-quality
drift. To clear a failure, fix the flagged test and shrink the baseline with
``--write-baseline``; never raise an entry to make the gate pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from Helper_Scripts.ci.test_quality_triage import analyze_file

REPO_ROOT = Path(__file__).resolve().parents[3]
RATCHET = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_triage.py"
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_baseline.txt"


@pytest.mark.parametrize(
    ("imports", "base"),
    [
        ("from hypothesis.stateful import RuleBasedStateMachine", "RuleBasedStateMachine"),
        ("from hypothesis.stateful import RuleBasedStateMachine as StateMachine", "StateMachine"),
        ("import hypothesis.stateful as stateful", "stateful.RuleBasedStateMachine"),
        ("import hypothesis", "hypothesis.stateful.RuleBasedStateMachine"),
        ("from hypothesis import stateful", "stateful.RuleBasedStateMachine"),
        ("from hypothesis import stateful as states", "states.RuleBasedStateMachine"),
    ],
)
def test_hypothesis_state_machine_invariants_are_not_fake_assertions(tmp_path, imports, base) -> None:
    source = tmp_path / "test_state_machine.py"
    source.write_text(
        f"{imports}\n\nclass WorkflowMachine({base}):\n"
        "    def check_store(self):\n"
        "        assert self.store.read() == self.expected\n",
        encoding="utf-8",
    )

    assert analyze_file(source, tmp_path) is None


@pytest.mark.parametrize(
    ("imports", "base"),
    [
        ("", ""),
        ("", "(RuleBasedStateMachine)"),
        ("from application import RuleBasedStateMachine", "(RuleBasedStateMachine)"),
        ("from application import stateful", "(stateful.RuleBasedStateMachine)"),
        ("from .hypothesis import stateful", "(stateful.RuleBasedStateMachine)"),
        ("from .hypothesis.stateful import RuleBasedStateMachine", "(RuleBasedStateMachine)"),
    ],
)
def test_non_test_class_assertions_remain_flagged_without_hypothesis_import(tmp_path, imports, base) -> None:
    source = tmp_path / "test_fake.py"
    source.write_text(
        f"{imports}\n\nclass FakeWorkflow{base}:\n"
        "    def run(self):\n"
        "        assert self.store.read() == self.expected\n",
        encoding="utf-8",
    )

    report = analyze_file(source, tmp_path)
    assert report is not None
    assert report.flags["tautology_suspect"] == 1


@pytest.mark.parametrize("intermediate", [False, True])
def test_pytest_contract_mixin_assertions_are_not_fake_assertions(tmp_path, intermediate) -> None:
    source = tmp_path / "test_storage_contract.py"
    parent = "Intermediate" if intermediate else "StorageContract"
    bridge = "class Intermediate(StorageContract):\n    pass\n" if intermediate else ""
    source.write_text(
        "class StorageContract:\n"
        "    def test_persisted_value(self, database):\n"
        "        assert database.read() == {'tokens': 47}\n"
        f"{bridge}class TestSQLite({parent}):\n    pass\n",
        encoding="utf-8",
    )

    assert analyze_file(source, tmp_path) is None


@pytest.mark.parametrize(
    ("body", "flag"),
    [
        ("        assert response.status_code == 200\n", "status_only"),
        (
            "        calls = []\n"
            "        def fake():\n            calls.append('called')\n"
            "        fake()\n        assert calls == ['called']\n",
            "tautology_suspect",
        ),
    ],
)
def test_pytest_contract_mixin_tests_still_receive_quality_checks(tmp_path, body, flag) -> None:
    source = tmp_path / "test_storage_contract.py"
    source.write_text(
        "class StorageContract:\n"
        "    def test_inherited_behavior(self):\n"
        f"{body}class TestSQLite(StorageContract):\n    pass\n",
        encoding="utf-8",
    )

    report = analyze_file(source, tmp_path)
    assert report is not None
    assert report.flags.get(flag) == 1


def test_uncollected_class_with_test_named_method_remains_a_fake(tmp_path) -> None:
    source = tmp_path / "test_fake_contract.py"
    source.write_text(
        "class FakeContract:\n" "    def test_value(self):\n        assert self.value == 'only-a-fake'\n",
        encoding="utf-8",
    )

    report = analyze_file(source, tmp_path)
    assert report is not None
    assert report.flags["tautology_suspect"] == 1


def test_test_quality_triage_enforces_the_committed_baseline() -> None:
    assert BASELINE.is_file(), "missing test-quality baseline; regenerate with --write-baseline"
    result = subprocess.run(
        [sys.executable, str(RATCHET), "--enforce"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (
        "test-quality ratchet failed - new or worse offenses vs the baseline "
        "(fix the flagged test and regenerate the baseline, never raise a count):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
def test_http_requests_are_not_counted_as_patch_setup(tmp_path, method):
    source = tmp_path / "test_http.py"
    source.write_text(
        "import pytest\npytestmark = pytest.mark.integration\n"
        "def test_endpoint(monkeypatch, client):\n"
        "    monkeypatch.setenv('ONE', '1')\n"
        "    monkeypatch.setenv('TWO', '2')\n"
        "    monkeypatch.setenv('THREE', '3')\n"
        f"    response = client.{method}('/resource')\n"
        "    assert response.json()['name'] == 'Rowan'\n",
        encoding="utf-8",
    )
    assert analyze_file(source, tmp_path) is None  # nosec B101 - HTTP PATCH is production dispatch, not a mock.


@pytest.mark.parametrize("patch_call", ["patch", "mock.patch", "unittest.mock.patch", "mocker.patch"])
def test_real_patch_roots_still_count_toward_integration_mock_density(tmp_path, patch_call):
    source = tmp_path / "test_patches.py"
    source.write_text(
        "import pytest\npytestmark = pytest.mark.integration\n"
        "def test_endpoint(monkeypatch, client):\n"
        "    monkeypatch.setenv('ONE', '1')\n"
        "    monkeypatch.setenv('TWO', '2')\n"
        "    monkeypatch.setenv('THREE', '3')\n"
        f"    with {patch_call}('provider.call'):\n"
        "        response = client.get('/resource')\n"
        "    assert response.json()['name'] == 'Rowan'\n",
        encoding="utf-8",
    )
    report = analyze_file(source, tmp_path)
    assert report is not None and report.flags == {"mock_density": 1}  # nosec B101 - the genuine four-patch gate stays intact.


@pytest.mark.parametrize("unrelated_fake", [False, True])
def test_real_dependency_bindings_are_not_fake_injection(tmp_path, unrelated_fake):
    source = tmp_path / "test_http_bindings.py"
    source.write_text(
        "from application import User\n"
        + ("class UnusedFake:\n    pass\n" if unrelated_fake else "")
        + "def test_endpoint(app, database):\n"
        "    def user():\n        return User(id=1)\n"
        "    app.dependency_overrides[auth] = user\n"
        "    app.dependency_overrides[store] = lambda: database\n"
        "    assert database.read() == {'saved': 'Selene'}\n",
        encoding="utf-8",
    )
    assert analyze_file(source, tmp_path) is None  # nosec B101 - real auth/store wiring never fabricates the persistence result.


@pytest.mark.parametrize(
    "provider",
    [
        "lambda: FakeStore()",
        "provide_fake",
        "lambda: fake",
        "lambda: StoreAlias()",
        "lambda fake=FakeStore(): fake",
    ],
)
def test_defined_fake_dependency_providers_remain_flagged(tmp_path, provider):
    source = tmp_path / "test_fake_dependency.py"
    source.write_text(
        "class FakeStore:\n    pass\n"
        "def test_endpoint(app):\n"
        "    fake = FakeStore()\n"
        "    StoreAlias = FakeStore\n"
        "    def provide_fake():\n        return fake\n"
        f"    app.dependency_overrides[first] = {provider}\n"
        f"    app.dependency_overrides[second] = {provider}\n"
        "    assert app.result() == 'saved'\n",
        encoding="utf-8",
    )
    report = analyze_file(source, tmp_path)
    assert report is not None and report.flags == {"stub_injection": 1}  # nosec B101 - aliases cannot hide fabricated dependency behavior.


def test_defined_fake_attribute_injection_remains_flagged(tmp_path):
    source = tmp_path / "test_fake_attribute.py"
    source.write_text(
        "class FakeStore:\n    pass\n"
        "def test_endpoint(app):\n"
        "    fake = FakeStore()\n"
        "    app.store = FakeStore()\n"
        "    assert app.result() == 'saved'\n",
        encoding="utf-8",
    )
    report = analyze_file(source, tmp_path)
    assert report is not None and report.flags == {"stub_injection": 1}  # nosec B101 - the original direct fake attribute gate remains intact.


def test_nonreturned_fake_decoy_does_not_taint_real_provider(tmp_path):
    source = tmp_path / "test_provider.py"
    source.write_text(
        "class FakeStore:\n    pass\n"
        "def test_endpoint(app, database):\n"
        "    def provider():\n        decoy = FakeStore()\n        return database\n"
        "    app.dependency_overrides[first] = provider\n"
        "    app.dependency_overrides[second] = provider\n"
        "    assert database.read() == {'saved': 'Selene'}\n",
        encoding="utf-8",
    )
    assert analyze_file(source, tmp_path) is None  # nosec B101 - provenance follows returned values, not unrelated calls.

@pytest.mark.parametrize(
    "nested_body",
    [
        "    def unused():\n        return FakeStore()\n",
        "    def unused():\n        database = FakeStore()\n        return database\n",
    ],
)
def test_unused_nested_provider_scope_does_not_taint_real_binding(tmp_path, nested_body):
    source = tmp_path / "test_provider_scope.py"
    source.write_text(
        "class FakeStore:\n    pass\n"
        "def test_endpoint(app, database):\n"
        "    def provider():\n"
        + "".join("    " + line for line in nested_body.splitlines(keepends=True))
        + "        return database\n"
        "    app.dependency_overrides[first] = provider\n"
        "    app.dependency_overrides[second] = provider\n"
        "    assert database.read() == {'saved': 'Selene'}\n",
        encoding="utf-8",
    )
    assert analyze_file(source, tmp_path) is None  # nosec B101 - uncalled nested code has no provider-return authority.

def test_module_defined_fake_provider_remains_flagged(tmp_path):
    source = tmp_path / "test_module_provider.py"
    source.write_text(
        "class FakeStore:\n    pass\n"
        "def provider():\n    return FakeStore()\n"
        "def test_endpoint(app):\n"
        "    app.dependency_overrides[first] = provider\n"
        "    app.dependency_overrides[second] = provider\n"
        "    assert app.result() == 'saved'\n",
        encoding="utf-8",
    )
    report = analyze_file(source, tmp_path)
    assert report is not None and report.flags == {"stub_injection": 1}  # nosec B101 - moving a fake provider out of a test must not hide it.

@pytest.mark.parametrize(
    "provider_source",
    [
        "def provider():\n    return database\n",
        "",
    ],
)
def test_provider_uses_its_defining_scope_not_test_shadow(tmp_path, provider_source):
    source = tmp_path / "test_defining_scope.py"
    provider = "provider" if provider_source else "lambda database=real_db: database"
    source.write_text(
        "from application import database\n"
        "class FakeStore:\n    pass\n" + provider_source + "def test_endpoint(app, real_db):\n"
        "    database = FakeStore()\n"
        f"    app.dependency_overrides[first] = {provider}\n"
        f"    app.dependency_overrides[second] = {provider}\n"
        "    assert real_db.read() == {'saved': 'Selene'}\n",
        encoding="utf-8",
    )
    assert analyze_file(source, tmp_path) is None  # nosec B101 - provider closure and argument authority do not inherit an unrelated test local.

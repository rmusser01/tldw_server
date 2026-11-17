"""Configured Admin stores must not share another profile's feature flags."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 -- Fixed Python child verifies normal import-time configuration.
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.services import admin_system_ops_service as service

ROOT = Path(__file__).resolve().parents[3]
CHILD = """
import json,sys
from pathlib import Path
from tldw_Server_API.app.services import admin_system_ops_service as service
expected=Path(sys.argv[1])
if service._STORE_PATH != expected:
    raise AssertionError('Configured store was not selected before write')
if not Path(service.__file__).is_relative_to(Path(sys.argv[3])):
    raise AssertionError('Child imported a different source checkout')
service.upsert_feature_flag(key=sys.argv[2],scope='user',enabled=True,description=None,
 org_id=None,user_id=7,target_user_ids=None,rollout_percent=100,variant_value=None,actor='test',note=None)
flags=service.list_feature_flags(scope='user',user_id=7)
print(json.dumps([flag['key'] for flag in flags]))
"""


def test_separate_configured_stores_do_not_share_user_flags(tmp_path: Path) -> None:
    """Use real module initialization and JSON operations in separate processes."""
    first_bytes = None
    for name in ("alpha", "beta"):
        path = tmp_path / name / "system_ops.json"
        key = "synthetic." + name
        env = {key: value for key, value in os.environ.items() if key in {"HOME", "USER", "PATH", "LANG", "LC_ALL"}}
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", SYSTEM_OPS_STORE_PATH=str(path))
        child = subprocess.run(  # nosec B603 -- Fixed interpreter/script and controlled private fixture paths.
            [sys.executable, "-B", "-c", CHILD, str(path), key, str(ROOT)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if child.returncode != 0:
            raise AssertionError("Real configured-store child failed: " + child.stderr[-2000:])
        if json.loads(child.stdout) != [key]:
            raise AssertionError("A profile observed another profile's flag")
        if name == "alpha":
            first_bytes = path.read_bytes()
        elif (tmp_path / "alpha/system_ops.json").read_bytes() != first_bytes:
            raise AssertionError("The second profile changed the first store")


@pytest.mark.parametrize("value", [None, "", "  "])
def test_absent_or_blank_setting_preserves_legacy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SYSTEM_OPS_STORE_PATH", raising=False)
    else:
        monkeypatch.setenv("SYSTEM_OPS_STORE_PATH", value)
    monkeypatch.setattr(service, "get_database_dir", lambda: str(tmp_path))
    if service._default_store_path() != tmp_path / "system_ops.json":
        raise AssertionError("Legacy Admin store path changed")


def test_configured_store_does_not_consult_legacy_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "separate/system_ops.json"
    monkeypatch.setenv("SYSTEM_OPS_STORE_PATH", "  " + str(path) + "  ")

    def unexpected_legacy_root() -> str:
        raise AssertionError("Configured profile consulted the shared legacy root")

    monkeypatch.setattr(service, "get_database_dir", unexpected_legacy_root)
    if service._default_store_path() != path:
        raise AssertionError("Configured Admin store path changed")

"""Shared Jobs facts stay canonical, compatible, and inert below both layers."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CORE = "tldw_Server_API.app.core"
_MODULE_PAIRS = (
    ("Jobs.operations.contracts", "Utils.jobs_operation_contracts"),
    ("Jobs.operations.workflow_contracts", "Utils.jobs_workflow_contracts"),
)


@pytest.mark.parametrize("legacy_path,canonical_path", _MODULE_PAIRS)
def test_legacy_paths_share_every_export_and_private_helper(legacy_path: str, canonical_path: str) -> None:
    """A second DTO definition or a dropped direct import breaks compatibility."""
    canonical = importlib.import_module(f"{_CORE}.{canonical_path}")
    legacy = importlib.import_module(f"{_CORE}.{legacy_path}")
    names = {name for name in vars(canonical) if not name.startswith("__")}
    assert {name for name in vars(legacy) if not name.startswith("__")} == names
    for name in names:
        assert getattr(legacy, name) is getattr(canonical, name), name
    assert getattr(legacy, "__all__", None) is getattr(canonical, "__all__", None)


@pytest.mark.parametrize("legacy_path,canonical_path", _MODULE_PAIRS)
def test_canonical_imports_are_inert_without_loading_domain_or_database_modules(
    legacy_path: str, canonical_path: str, tmp_path: Path
) -> None:
    """A clean interpreter must not load Jobs/config/DB code or perform I/O."""
    probe = textwrap.dedent(
        """
        import importlib
        import json
        import os
        import sys
        import threading

        sys.path.insert(0, sys.argv[1])
        before_threads = {thread.ident for thread in threading.enumerate()}

        def audit(event, args):
            if event == "open":
                flags = args[2]
                if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                    raise RuntimeError("contract import attempted a file write")
            if event.startswith(("socket.", "sqlite3.connect", "subprocess.", "os.mkdir", "os.remove", "os.rename")):
                raise RuntimeError("contract import attempted I/O: " + event)

        sys.addaudithook(audit)
        importlib.import_module(sys.argv[2])
        allowed = {
            "tldw_Server_API", "tldw_Server_API.app", "tldw_Server_API.app.core",
            "tldw_Server_API.app.core.testing",
            "tldw_Server_API.app.core.Utils",
            "tldw_Server_API.app.core.Utils.coercion",
            "tldw_Server_API.app.core.Utils.jobs_operation_contracts",
            "tldw_Server_API.app.core.Utils.jobs_workflow_contracts",
        }
        unexpected = sorted(name for name in sys.modules if name.startswith("tldw_Server_API") and name not in allowed)
        print(json.dumps({
            "unexpected_modules": unexpected,
            "new_threads": sorted(thread.ident for thread in threading.enumerate() if thread.ident not in before_threads),
        }))
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            probe,
            str(Path(__file__).resolve().parents[3]),
            f"{_CORE}.{canonical_path}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == {"unexpected_modules": [], "new_threads": []}
    assert list(tmp_path.iterdir()) == []


def test_canonical_definitions_live_in_foundation_not_jobs() -> None:
    operations = importlib.import_module(f"{_CORE}.Utils.jobs_operation_contracts")
    workflows = importlib.import_module(f"{_CORE}.Utils.jobs_workflow_contracts")
    assert operations.CreateJobCommand.__module__ == f"{_CORE}.Utils.jobs_operation_contracts"
    assert workflows.WorkflowControls.__module__ == f"{_CORE}.Utils.jobs_workflow_contracts"
    assert workflows.encode_workflow_receipt.__module__ == f"{_CORE}.Utils.jobs_workflow_contracts"


def test_cross_path_controls_preserve_enum_identity_and_default() -> None:
    operations = importlib.import_module(f"{_CORE}.Utils.jobs_operation_contracts")
    workflows = importlib.import_module(f"{_CORE}.Utils.jobs_workflow_contracts")
    legacy = importlib.import_module(f"{_CORE}.Jobs.operations.contracts")
    controls = workflows.WorkflowControls(3, None, legacy.ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT)
    assert controls.expired_lease_policy is operations.ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT
    assert controls.no_progress_limit == 5
    command = operations.CreateJobCommand("test", "queue", "root", {}, None)
    assert command.expired_lease_policy is legacy.ExpiredLeasePolicy.CONSUME_RETRY
    with pytest.raises(ValueError):
        workflows.WorkflowControls(3, None, operations.ExpiredLeasePolicy.CONSUME_RETRY)


def test_receipt_bytes_and_decoded_type_match_across_paths() -> None:
    canonical = importlib.import_module(f"{_CORE}.Utils.jobs_workflow_contracts")
    legacy = importlib.import_module(f"{_CORE}.Jobs.operations.workflow_contracts")
    receipt = legacy.WorkflowReceipt(
        "a" * 64,
        "b" * 64,
        "00000000-0000-4000-8000-000000000001",
        canonical.WorkflowAction.COMPLETE,
        "completed",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    wire = (
        b'{"action":"complete","applied_at":"2026-01-02T00:00:00+00:00","effect_key":"'
        b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","facts_fingerprint":"'
        b'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","job_uuid":"'
        b'00000000-0000-4000-8000-000000000001","released":false,"resulting_status":"completed"}'
    )
    assert canonical.encode_workflow_receipt(receipt) == wire
    decoded = legacy.decode_workflow_receipt(wire)
    assert type(decoded) is canonical.WorkflowReceipt
    assert decoded == receipt


def test_shared_scope_still_rejects_reserved_admin_webhook_queue() -> None:
    operations = importlib.import_module(f"{_CORE}.Utils.jobs_operation_contracts")
    workflows = importlib.import_module(f"{_CORE}.Utils.jobs_workflow_contracts")
    with pytest.raises(ValueError, match="reserved"):
        workflows.WorkflowScope(
            operations.ADMIN_WEBHOOK_DELIVERY_DOMAIN,
            operations.ADMIN_WEBHOOK_DELIVERY_QUEUE,
            operations.ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
        )

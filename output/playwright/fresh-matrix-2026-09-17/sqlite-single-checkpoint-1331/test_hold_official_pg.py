"""Hold the existing official auth/content fixtures until the parent releases them."""

import json
import os
import signal
import time
from pathlib import Path

import pytest
from pg_role_adapter import qualify_runtime, runtime_server, write_private


@pytest.fixture(scope="session")
def pg_server(pg_server):
    """Adapt the official provisioning server; dependent official DBs clean up first."""
    root = Path(os.environ["MATRIX_HOLDER_ROOT"]).resolve()
    with runtime_server(pg_server, record_path=root / "created-role.private.json") as runtime:
        yield runtime


def test_hold_official_auth_and_content_databases(pg_server, pg_temp_db, pg_temp_db_session):
    root = Path(os.environ["MATRIX_HOLDER_ROOT"]).resolve()
    name = os.environ["MATRIX_CELL"]
    assert name in ("pg-single", "pg-multi")
    receipt_path = root / f"{name}.pg-receipt.private.json"
    release_path = root / f"{name}.release-holder"
    assert not receipt_path.exists(), "Refusing to replace a holder receipt"
    assert not release_path.exists(), "Refusing to reuse a released fixture"
    assert pg_temp_db["database"] != pg_temp_db_session["database"]
    runtime_role = qualify_runtime(pg_server, pg_temp_db, pg_temp_db_session)
    provisioner = pg_server["_matrix_provisioner"]
    write_private(
        root / "runtime.pg-config.private.json",
        {
            **{key: pg_server[key] for key in ("host", "port", "user", "password")},
            "purpose": "matrix-runtime",
            "cell": name,
            "run_id": os.environ["MATRIX_RUN_ID"],
            "container": os.environ["MATRIX_PG_CONTAINER"],
            "defaultDb": "postgres",
        },
    )
    interrupted = False

    def release(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, release)
    signal.signal(signal.SIGINT, release)
    receipt = {
        "profile": name,
        "run_id": os.environ["MATRIX_RUN_ID"],
        "source_root": os.environ["MATRIX_SOURCE_ROOT"],
        "source_commit": os.environ["MATRIX_SOURCE_COMMIT"],
        "python_venv": os.environ["MATRIX_PYTHON_VENV"],
        "pid": os.getpid(),
        "status": "held",
        "provisioning": {key: provisioner[key] for key in ("host", "port", "user")},
        "runtime_role": runtime_role,
        "auth_fixture": "pg_temp_db",
        "content_fixture": "pg_temp_db_session",
        "auth": pg_temp_db,
        "content": pg_temp_db_session,
        "created_at": time.time(),
    }

    def write_receipt():
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(receipt, output, indent=2)
        receipt_path.chmod(0o600)

    write_receipt()
    try:
        while not interrupted and not release_path.exists():
            time.sleep(0.5)
    finally:
        receipt["status"] = "released"
        receipt["released_at"] = time.time()
        write_receipt()
    # Returning lets the official fixtures clean up only their own temporary DBs.

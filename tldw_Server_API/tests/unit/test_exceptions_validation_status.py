from __future__ import annotations

import os
import subprocess
import sys

import pytest
from fastapi import status

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.AuthNZ import membership_writer, user_provider_secrets

pytestmark = pytest.mark.unit


def test_exceptions_import_does_not_emit_deprecated_422_warning() -> None:
    script = "import tldw_Server_API.app.core.exceptions"
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "always"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined
    assert "HTTP_422_UNPROCESSABLE_ENTITY' is deprecated" not in combined


def test_api_validation_error_uses_resolved_default_status() -> None:
    expected = (
        status.HTTP_422_UNPROCESSABLE_CONTENT
        if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT")
        else status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    exc = core_exceptions.APIValidationError(detail="bad input")
    assert exc.status_code == expected


def test_user_profile_exceptions_are_centrally_owned_and_compatibly_reexported() -> None:
    membership_exception_names = (
        "MembershipWriterContractError",
        "OfflineMigrationContextRejected",
        "MembershipWriteError",
        "MembershipReadError",
        "MembershipAuthorizationError",
        "MembershipScopeNotFound",
        "MembershipTargetNotFound",
        "MembershipParentRequired",
        "MembershipPreflightChanged",
        "_MembershipScopeDeletionRetry",
    )

    for name in membership_exception_names:
        central_exception = getattr(core_exceptions, name)
        assert getattr(membership_writer, name) is central_exception
        assert central_exception.__module__ == core_exceptions.__name__

    assert (
        user_provider_secrets.ProviderCredentialAliasConflictError
        is core_exceptions.ProviderCredentialAliasConflictError
    )
    assert (
        core_exceptions.ProviderCredentialAliasConflictError.__module__
        == core_exceptions.__name__
    )

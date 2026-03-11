from __future__ import annotations

import tldw_Server_API.app.core.Sandbox.macos_diagnostics as diagnostics_module
from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import HelperPingReply
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult


def _patch_macos_host(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostics_module,
        "vz_host_facts",
        lambda: {
            "os": "darwin",
            "arch": "arm64",
            "apple_silicon": True,
        },
    )
    monkeypatch.setattr(diagnostics_module.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))


def _sample_diagnostics_payload() -> dict:
    return {
        "host": {
            "os": "darwin",
            "arch": "arm64",
            "apple_silicon": True,
            "macos_version": "15.0",
            "supported": True,
            "reasons": [],
        },
        "helper": {
            "configured": True,
            "path": "/tmp/helper",
            "exists": True,
            "executable": True,
            "ready": True,
            "transport": "fake",
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "reasons": [],
        },
        "templates": {
            "vz_linux": {
                "configured": True,
                "ready": True,
                "source": "/tmp/vz-linux.img",
                "reasons": [],
            },
        },
        "runtimes": {
            "vz_linux": {
                "available": True,
                "supported_trust_levels": ["trusted", "standard", "untrusted"],
                "reasons": [],
                "execution_mode": "fake",
                "remediation": None,
            }
        },
    }


def test_collect_macos_diagnostics_reports_missing_helper_and_templates(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC", raising=False)

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["host"]["supported"] is True
    assert data["helper"]["configured"] is False
    assert data["helper"]["path"] is None
    assert data["helper"]["ready"] is False
    assert data["helper"]["protocol_version"] is None
    assert data["helper"]["helper_version"] is None
    assert data["templates"]["vz_linux"]["configured"] is False
    assert data["templates"]["vz_linux"]["source"] is None
    assert data["templates"]["vz_linux"]["ready"] is False
    assert "macos_helper_missing" in data["runtimes"]["vz_linux"]["reasons"]
    assert data["runtimes"]["vz_linux"]["execution_mode"] == "none"


def test_collect_macos_diagnostics_reports_real_vz_linux_execution_mode(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FakeHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            assert request["runtime"] == "vz_linux"
            return {
                "template_id": "vz_linux:ubuntu-24.04",
                "source": request["template"],
                "ready": True,
                "reasons": [],
            }

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=True,
                reasons=[],
                execution_mode="real",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["runtimes"]["vz_linux"]["available"] is True
    assert data["runtimes"]["vz_linux"]["execution_mode"] == "real"
    assert data["runtimes"]["vz_linux"]["reasons"] == []
    assert data["helper"]["ready"] is True
    assert data["helper"]["transport"] == "unix"
    assert data["helper"]["protocol_version"] == "1"
    assert data["helper"]["helper_version"] == "0.1.0"
    assert data["templates"]["vz_linux"]["ready"] is True
    assert data["templates"]["vz_linux"]["reasons"] == []


def test_collect_macos_diagnostics_separates_policy_from_host_readiness(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["runtimes"]["seatbelt"]["supported_trust_levels"] == ["trusted"]
    assert data["runtimes"]["seatbelt"]["available"] in (True, False)


def test_collect_macos_diagnostics_uses_optional_operator_metadata_env(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_PATH", "/tmp/macos-helper")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["helper"]["path"] == "/tmp/macos-helper"
    assert data["templates"]["vz_linux"]["source"] == "/tmp/vz-linux.img"


def test_collect_macos_diagnostics_reports_helper_validated_template_failure(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FailingTemplateHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": None,
                "source": request["template"],
                "ready": False,
                "reasons": ["template_invalid"],
            }

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FailingTemplateHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["template_invalid"],
                execution_mode="none",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["templates"]["vz_linux"]["configured"] is True
    assert data["templates"]["vz_linux"]["ready"] is False
    assert data["templates"]["vz_linux"]["reasons"] == ["template_invalid"]


def test_collect_macos_diagnostics_does_not_trust_ready_env_without_reachable_helper(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_PATH", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", raising=False)

    class _FailingHelper:
        def ping(self):
            raise diagnostics_module.MacOSVirtualizationHelperUnavailable(
                "macos_virtualization_helper_unavailable"
            )

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FailingHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["helper"]["configured"] is False
    assert data["helper"]["ready"] is False
    assert "macos_virtualization_helper_unavailable" in data["helper"]["reasons"]


def test_service_macos_diagnostics_returns_probe_payload(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    expected = _sample_diagnostics_payload()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.collect_macos_diagnostics",
        lambda: expected,
    )

    svc = SandboxService()

    assert svc.macos_diagnostics() == expected


def test_admin_schema_accepts_macos_diagnostics_payload() -> None:
    from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
        SandboxAdminMacOSDiagnosticsResponse,
    )

    model = SandboxAdminMacOSDiagnosticsResponse.model_validate(_sample_diagnostics_payload())

    assert model.host.supported is True
    assert model.runtimes["vz_linux"].execution_mode == "fake"

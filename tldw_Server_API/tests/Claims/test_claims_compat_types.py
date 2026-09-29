from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.Claims_Extraction import adjudicator, claims_engine, fva_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import VerificationStatus


def test_verification_status_includes_contested() -> None:
    assert VerificationStatus.CONTESTED.value == "contested"  # nosec B101


def test_claims_compat_types_shim_removed() -> None:
    shim_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "core"
        / "Claims_Extraction"
        / "compat_types.py"
    )
    assert not shim_path.exists()  # nosec B101


def test_modules_share_canonical_verification_status() -> None:
    assert claims_engine.VerificationStatus is VerificationStatus  # nosec B101
    assert adjudicator.VerificationStatus is VerificationStatus  # nosec B101
    # fva_pipeline dropped its unused import in 19bd4c83cc; it must still not fork one.
    assert getattr(fva_pipeline, "VerificationStatus", VerificationStatus) is VerificationStatus  # nosec B101

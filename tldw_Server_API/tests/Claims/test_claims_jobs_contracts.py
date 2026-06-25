import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_contracts as contracts


pytestmark = pytest.mark.unit


def test_rebuild_payload_validation_accepts_id_only_payload() -> None:
    payload = contracts.validate_rebuild_media_payload(
        {"version": 1, "owner_user_id": "1", "media_id": 42}
    )

    assert payload == {"version": 1, "owner_user_id": "1", "media_id": 42}  # nosec B101


@pytest.mark.parametrize("media_id", [True, False, 3.7, "", "abc", 0, -1])
def test_rebuild_payload_validation_rejects_non_integral_media_id(media_id: object) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {"version": 1, "owner_user_id": "1", "media_id": media_id}
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


def test_payload_validation_rejects_paths_and_synthetic_owner() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {
                "version": 1,
                "owner_user_id": "0",
                "media_id": 42,
                "db_path": "/tmp/Media_DB_v2.db",  # nosec B108
            }
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


def test_review_payload_sorts_and_dedupes_notification_ids() -> None:
    payload = contracts.validate_review_notification_payload(
        {"version": 1, "owner_user_id": "7", "notification_ids": [3, "2", 3]}
    )

    assert payload["notification_ids"] == [2, 3]  # nosec B101


def test_review_payload_reports_missing_owner_before_invalid_notification_ids() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_review_notification_payload(
            {"version": 1, "owner_user_id": "0", "notification_ids": [0]}
        )

    assert excinfo.value.failure_code == "claims_missing_owner"  # nosec B101


def test_alert_payload_rejects_unsupported_channel() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_alert_delivery_payload(
            {
                "version": 1,
                "owner_user_id": "7",
                "event_id": 55,
                "alert_id": 9,
                "channel": "email",
            }
        )

    assert excinfo.value.failure_code == "claims_unsupported_channel"  # nosec B101


def test_alert_payload_reports_missing_owner_before_unsupported_channel() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_alert_delivery_payload(
            {
                "version": 1,
                "owner_user_id": "0",
                "event_id": 55,
                "alert_id": 9,
                "channel": "email",
            }
        )

    assert excinfo.value.failure_code == "claims_missing_owner"  # nosec B101


def test_claims_job_error_exposes_worker_sdk_attributes() -> None:
    exc = contracts.ClaimsJobError(
        "locked",
        retryable=True,
        failure_code="claims_db_locked",
        backoff_seconds=13,
    )

    assert str(exc) == "locked"  # nosec B101
    assert exc.retryable is True  # nosec B101
    assert exc.failure_code == "claims_db_locked"  # nosec B101
    assert exc.backoff_seconds == 13  # nosec B101

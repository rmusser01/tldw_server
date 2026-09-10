from collections.abc import Callable

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_contracts as contracts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CLAIMS_ANALYTICS_EXPORT_WORKSPACE_ID_MAX_CHARS", 19),
        ("CLAIMS_ANALYTICS_EXPORT_EVENT_TYPE_MAX_CHARS", 128),
        ("CLAIMS_ANALYTICS_EXPORT_SEVERITY_MAX_CHARS", 64),
        ("CLAIMS_ANALYTICS_EXPORT_PROVIDER_MAX_CHARS", 128),
        ("CLAIMS_ANALYTICS_EXPORT_MODEL_MAX_CHARS", 256),
        ("CLAIMS_ANALYTICS_EXPORT_TIMESTAMP_MAX_CHARS", 64),
    ],
)
def test_analytics_export_limits_remain_available_from_job_contracts(
    name: str,
    expected: int,
) -> None:
    assert getattr(contracts, name) == expected


def test_rebuild_payload_validation_accepts_id_only_payload() -> None:
    payload = contracts.validate_rebuild_media_payload(
        {"version": 1, "owner_user_id": "1", "media_id": 42}
    )

    assert payload == {"version": 1, "owner_user_id": "1", "media_id": 42}  # nosec B101


def test_rebuild_payload_validation_accepts_integer_owner_user_id() -> None:
    payload = contracts.validate_rebuild_media_payload(
        {"version": 1, "owner_user_id": 7, "media_id": 42}
    )

    assert payload["owner_user_id"] == "7"  # nosec B101


@pytest.mark.parametrize("owner_user_id", [True, 3.7, {"id": "7"}, ["7"]])
def test_rebuild_payload_validation_rejects_non_scalar_owner_user_id(
    owner_user_id: object,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {"version": 1, "owner_user_id": owner_user_id, "media_id": 42}
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_missing_owner"  # nosec B101


@pytest.mark.parametrize("owner_user_id", ["", "0", "007", " 7", "7 ", "abc"])
def test_rebuild_payload_validation_rejects_noncanonical_owner_user_id(
    owner_user_id: str,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {"version": 1, "owner_user_id": owner_user_id, "media_id": 42}
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_missing_owner"  # nosec B101


@pytest.mark.parametrize("media_id", [True, False, 3.7, "", "abc", 0, -1])
def test_rebuild_payload_validation_rejects_non_integral_media_id(media_id: object) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {"version": 1, "owner_user_id": "1", "media_id": media_id}
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        ("{not json", "claims job payload is not valid JSON"),
        ("[]", "claims job payload must be an object"),
    ],
)
def test_payload_validation_rejects_malformed_or_non_object_json_strings(
    value: str,
    expected_message: str,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(value)

    exc = excinfo.value
    assert str(exc) == expected_message  # nosec B101
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


@pytest.mark.parametrize(
    ("validator", "payload", "unknown_key", "unknown_value"),
    [
        (
            contracts.validate_rebuild_media_payload,
            {"version": 1, "owner_user_id": "7", "media_id": 42},
            "body",
            "claim text",
        ),
        (
            contracts.validate_review_notification_payload,
            {"version": 1, "owner_user_id": "7", "notification_ids": [3]},
            "url",
            "https://example.test/hook",
        ),
        (
            contracts.validate_review_notification_payload,
            {"version": 1, "owner_user_id": "7", "notification_ids": [3]},
            "headers",
            {"Authorization": "Bearer token"},
        ),
        (
            contracts.validate_alert_delivery_payload,
            {
                "version": 1,
                "owner_user_id": "7",
                "event_id": 55,
                "alert_id": 9,
                "channel": "webhook",
            },
            "access_token",
            "secret",
        ),
        (
            contracts.validate_alert_delivery_payload,
            {
                "version": 1,
                "owner_user_id": "7",
                "event_id": 55,
                "alert_id": 9,
                "channel": "webhook",
            },
            "payload",
            {"unsafe": True},
        ),
    ],
)
def test_payload_validation_rejects_unknown_top_level_keys(
    validator: Callable[[object], dict[str, object]],
    payload: dict[str, object],
    unknown_key: str,
    unknown_value: object,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        validator({**payload, unknown_key: unknown_value})

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


def test_payload_validation_rejects_paths() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {
                "version": 1,
                "owner_user_id": "7",
                "media_id": 42,
                "db_path": "/tmp/Media_DB_v2.db",  # nosec B108
            }
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_invalid_payload"  # nosec B101


def test_analytics_export_payload_accepts_exact_id_only_contract() -> None:
    export_id = "0123456789abcdef0123456789abcdef"

    payload = contracts.validate_analytics_export_payload(
        {"version": 1, "owner_user_id": "123", "export_id": export_id}
    )

    assert payload == {  # nosec B101
        "version": 1,
        "owner_user_id": "123",
        "export_id": export_id,
    }


def test_analytics_export_payload_enforces_routable_owner_range() -> None:
    maximum = "9223372036854775807"
    export_id = "0123456789abcdef0123456789abcdef"

    assert contracts.validate_analytics_export_payload(
        {"version": 1, "owner_user_id": maximum, "export_id": export_id}
    )["owner_user_id"] == maximum

    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_analytics_export_payload(
            {
                "version": 1,
                "owner_user_id": "9223372036854775808",
                "export_id": export_id,
            }
        )

    assert excinfo.value.failure_code == "claims_missing_owner"  # nosec B101


def test_payload_owner_validation_rejects_huge_integer_with_stable_error() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {"version": 1, "owner_user_id": 10**5000, "media_id": 42}
        )

    assert excinfo.value.failure_code == "claims_missing_owner"  # nosec B101


@pytest.mark.parametrize(
    "export_id",
    [
        "0123456789ABCDEF0123456789ABCDEF",
        "01234567-89ab-cdef-0123-456789abcdef",
        "0123456789abcdef0123456789abcde",
        "0123456789abcdef0123456789abcdef0",
        "g123456789abcdef0123456789abcdef",
        "",
        None,
        123,
    ],
)
def test_analytics_export_payload_rejects_noncanonical_export_ids(
    export_id: object,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_analytics_export_payload(
            {"version": 1, "owner_user_id": "123", "export_id": export_id}
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_export_invalid_payload"  # nosec B101


@pytest.mark.parametrize(
    "owner_user_id",
    [123, True, "", "0", "0123", " 123", "123 ", "+123", "abc"],
)
def test_analytics_export_payload_rejects_noncanonical_owner_strings(
    owner_user_id: object,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_analytics_export_payload(
            {
                "version": 1,
                "owner_user_id": owner_user_id,
                "export_id": "0123456789abcdef0123456789abcdef",
            }
        )

    assert excinfo.value.failure_code == "claims_missing_owner"  # nosec B101


@pytest.mark.parametrize("version", [0, 2, "2"])
def test_analytics_export_payload_rejects_unsupported_versions(version: object) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_analytics_export_payload(
            {
                "version": version,
                "owner_user_id": "123",
                "export_id": "0123456789abcdef0123456789abcdef",
            }
        )

    assert excinfo.value.failure_code == (  # nosec B101
        "claims_unsupported_payload_version"
        if version in {2, "2"}
        else "claims_invalid_payload"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("filters", {"provider": "private-provider"}),
        ("pagination", {"limit": 100}),
        ("events", [{"claim": "private claim"}]),
        ("payload_json", "private-json"),
        ("payload_csv", "private,csv"),
        ("content", "private-content"),
        ("workspace_id", "private-workspace"),
        ("db_path", "/private/media.db"),
        ("database_path", "/private/media.db"),
        ("file_path", "/private/export.csv"),
        ("credentials", {"username": "u", "password": "p"}),
        ("api_key", "private-api-key"),
        ("unexpected", "private-unknown-value"),
    ],
)
def test_analytics_export_payload_rejects_every_extra_key_without_echoing_values(
    key: str,
    value: object,
) -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_analytics_export_payload(
            {
                "version": 1,
                "owner_user_id": "123",
                "export_id": "0123456789abcdef0123456789abcdef",
                key: value,
            }
        )

    exc = excinfo.value
    assert exc.retryable is False  # nosec B101
    assert exc.failure_code == "claims_export_invalid_payload"  # nosec B101
    assert repr(value) not in str(exc)  # nosec B101


def test_analytics_export_body_keys_are_in_sensitive_payload_coverage() -> None:
    assert {  # nosec B101
        "filters",
        "pagination",
        "events",
        "payload_json",
        "payload_csv",
        "content",
        "workspace_id",
        "database_path",
        "file_path",
        "credentials",
    }.issubset(contracts.SENSITIVE_PAYLOAD_KEYS)


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


def test_review_payload_reports_missing_owner_before_unknown_keys() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_review_notification_payload(
            {
                "version": 1,
                "owner_user_id": "0",
                "notification_ids": [3],
                "body": "x",
            }
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


def test_alert_payload_reports_missing_owner_before_unknown_keys() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_alert_delivery_payload(
            {
                "version": 1,
                "owner_user_id": "0",
                "event_id": 55,
                "alert_id": 9,
                "channel": "webhook",
                "body": "x",
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


def test_result_helpers_keep_reserved_fields_authoritative() -> None:
    assert contracts.ok_result(outcome="bad", count=2) == {  # nosec B101
        "count": 2,
        "outcome": "ok",
    }
    assert contracts.skipped_result("reason", outcome="bad", reason="other") == {  # nosec B101
        "outcome": "skipped",
        "reason": "reason",
    }

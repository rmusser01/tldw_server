import pytest

from tldw_Server_API.app.api.v1.utils import cache, http_errors, request_parsing
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError as ChaChaConflictError,
    InputError as ChaChaInputError,
    SchemaError as ChaChaSchemaError,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError as KanbanConflictError,
    InputError as KanbanInputError,
    KanbanDBError,
    NotFoundError as KanbanNotFoundError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.Meetings_DB import (
    InputError as MeetingsInputError,
    MeetingsDatabaseError,
    SchemaError as MeetingsSchemaError,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    ConflictError as PromptsConflictError,
    DatabaseError as PromptsDatabaseError,
    InputError as PromptsInputError,
    SchemaError as PromptsSchemaError,
)
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError as SlidesConflictError,
    InputError as SlidesInputError,
    SchemaError as SlidesSchemaError,
    SlidesDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_errors import (
    ConflictError as UnifiedConflictError,
    DataIntegrityError as UnifiedDataIntegrityError,
    DatabaseError as UnifiedDatabaseError,
    InputError as UnifiedInputError,
    NotFoundError as UnifiedNotFoundError,
    SchemaError as UnifiedSchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)

pytestmark = pytest.mark.unit


class _FakeRedis:
    def __init__(self):
        self.calls = []
        self.kv = {}
        self.sets = {}

    # Basic KV
    def setex(self, key, ttl, value):
        self.calls.append(("setex", key, ttl, value))
        self.kv[key] = value

    def get(self, key):
        self.calls.append(("get", key))
        return self.kv.get(key)

    def delete(self, *keys):
        self.calls.append(("delete", keys))
        deleted = 0
        for k in keys:
            k = k.decode() if isinstance(k, (bytes, bytearray)) else k
            if k in self.kv:
                del self.kv[k]
                deleted += 1
        return deleted

    # Set ops
    def sadd(self, key, member):
        self.calls.append(("sadd", key, member))
        self.sets.setdefault(key, set()).add(member)

    def smembers(self, key):
        self.calls.append(("smembers", key))
        return set(self.sets.get(key, set()))

    def expire(self, key, ttl):
        self.calls.append(("expire", key, ttl))
        return True

    # Scan fallback
    def scan(self, cursor=0, match=None, count=None):
        self.calls.append(("scan", cursor, match, count))
        return 0, []


def test_build_cache_key_is_stable_and_drops_token():
    params1 = {"page": "1", "token": "secret"}  # nosec B105 - cache-key test fixture, not a credential
    params2 = {"token": "secret", "page": "1"}  # nosec B105 - cache-key test fixture, not a credential

    key1 = cache.build_cache_key("/api/v1/media", params1)
    key2 = cache.build_cache_key("/api/v1/media", params2)

    assert key1 == key2
    assert "token" not in key1

    key3 = cache.build_cache_key("/api/v1/media", {"page": "2"})
    assert key3 != key1


def test_cache_response_and_get_cached_response_roundtrip():
    fake = _FakeRedis()
    payload = {"ok": True, "items": [1, 2, 3]}
    key = "cache:/api/v1/media/123:abc"

    etag = cache.cache_response(key, payload, client=fake, media_id=123)
    assert etag

    # Ensure index key updated
    idx_key = "cacheidx:/api/v1/media/123"
    assert idx_key in fake.sets
    assert key in fake.sets[idx_key]

    cached = cache.get_cached_response(key, client=fake)
    assert cached is not None
    cached_etag, cached_payload = cached
    assert cached_etag == etag
    assert cached_payload == payload


def test_invalidate_media_cache_uses_index_and_scan():
    class _ScanRedis(_FakeRedis):
        def scan(self, cursor=0, match=None, count=None):
            # Expose one extra key on first call when index already seeded
            if cursor == 0 and match:
                self.kv["cache:/api/v1/media/123:extra"] = "v"
                return 0, ["cache:/api/v1/media/123:extra"]
            return 0, []

    fake = _ScanRedis()
    key = "cache:/api/v1/media/123:abc123"
    fake.setex(key, cache.CACHE_TTL, "v")
    fake.sadd("cacheidx:/api/v1/media/123", key)

    cache.invalidate_media_cache(123, client=fake)

    # Both indexed and scanned keys should be removed
    assert not any(k.startswith("cache:/api/v1/media/123:") for k in fake.kv)


def test_etag_and_if_none_match_parsing():
    payload = {"a": 1, "b": 2}
    etag = cache.generate_etag(payload)

    header = f'W/"{etag}", "other"'
    assert cache.is_not_modified(etag, header)

    header_miss = '"somethingelse"'
    assert not cache.is_not_modified(etag, header_miss)


def test_request_parsing_to_bool_and_to_int():
    assert request_parsing.to_bool("yes") is True
    assert request_parsing.to_bool("No") is False
    assert request_parsing.to_bool(None, default=True) is True

    assert request_parsing.to_int("10") == 10
    assert request_parsing.to_int("  ", default=5) == 5
    assert request_parsing.to_int("not-a-number", default=None) is None


def test_request_parsing_normalize_str_list_and_urls():
    assert request_parsing.normalize_str_list("a, b; c d") == ["a", "b", "c", "d"]
    assert request_parsing.normalize_str_list([" a ", ""]) == ["a"]

    urls = request_parsing.normalize_urls([" https://a ", "https://b", "https://a"])
    assert urls == ["https://a", "https://b"]


def test_http_error_mapping_for_db_exceptions():
    exc = InputError("bad")
    http_exc = http_errors.map_db_error_to_http(exc)
    assert http_exc.status_code == 400

    exc = ConflictError()
    http_exc = http_errors.map_db_error_to_http(exc)
    assert http_exc.status_code == 409

    exc = SchemaError("schema")
    http_exc = http_errors.map_db_error_to_http(exc)
    assert http_exc.status_code == 500
    assert "schema" in str(exc)

    exc = DatabaseError("db")
    http_exc = http_errors.map_db_error_to_http(exc)
    assert http_exc.status_code == 500


def test_http_error_mapping_can_promote_matching_input_errors_to_413():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Attachment exceeds maximum size"),
        payload_too_large_substrings=("exceeds maximum size",),
    )

    assert http_exc.status_code == 413
    assert http_exc.detail == "Attachment exceeds maximum size"


def test_http_error_mapping_keeps_non_matching_input_errors_at_400():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Attachment metadata is invalid"),
        payload_too_large_substrings=("exceeds maximum size",),
    )

    assert http_exc.status_code == 400
    assert http_exc.detail == "Attachment metadata is invalid"


def test_http_error_mapping_can_promote_matching_input_errors_to_404():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Dictionary revision not found"),
        not_found_substrings=("not found",),
    )

    assert http_exc.status_code == 404
    assert http_exc.detail == "Dictionary revision not found"


def test_http_error_mapping_can_sanitize_promoted_404_input_detail():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Dictionary revision not found at /private/tmp/db.sqlite"),
        not_found_substrings=("not found",),
        not_found_detail="Dictionary revision not found",
    )

    assert http_exc.status_code == 404
    assert http_exc.detail == "Dictionary revision not found"


def test_http_error_mapping_keeps_non_matching_not_found_input_errors_at_400():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Revision must be positive"),
        not_found_substrings=("not found",),
    )

    assert http_exc.status_code == 400
    assert http_exc.detail == "Revision must be positive"


def test_http_error_mapping_can_use_sanitized_input_detail_attribute():
    http_exc = http_errors.map_db_error_to_http(
        PromptsInputError("raw prompt failure", safe_message="sanitized prompt failure"),
        input_detail_attr="safe_message",
    )

    assert http_exc.status_code == 400
    assert http_exc.detail == "sanitized prompt failure"


def test_http_error_mapping_falls_back_to_string_when_input_detail_attribute_missing():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("raw chacha failure"),
        input_detail_attr="safe_message",
    )

    assert http_exc.status_code == 400
    assert http_exc.detail == "raw chacha failure"


def test_http_error_mapping_can_force_input_error_status_code():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaInputError("Project not found"),
        input_status_code=404,
    )

    assert http_exc.status_code == 404
    assert http_exc.detail == "Project not found"


def test_http_error_mapping_can_force_database_error_status_code():
    http_exc = http_errors.map_db_error_to_http(
        CharactersRAGDBError("Import backend unavailable"),
        default_detail="Import backend unavailable",
        database_status_code=400,
    )

    assert http_exc.status_code == 400
    assert http_exc.detail == "Import backend unavailable"


def test_http_error_mapping_can_force_conflict_error_status_code():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaConflictError("clip not found"),
        conflict_status_code=404,
    )

    assert http_exc.status_code == 404
    assert http_exc.detail == "clip not found"


def test_http_error_mapping_can_override_conflict_error_detail():
    http_exc = http_errors.map_db_error_to_http(
        ChaChaConflictError("stale write"),
        conflict_detail="Conflict during deletion",
    )

    assert http_exc.status_code == 409
    assert http_exc.detail == "Conflict during deletion"


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (UnifiedInputError("bad"), 400),
        (UnifiedConflictError("conflict"), 409),
        (UnifiedNotFoundError("missing"), 404),
        (UnifiedDataIntegrityError("bad data"), 422),
        (UnifiedSchemaError("schema"), 500),
        (KanbanInputError("bad"), 400),
        (KanbanConflictError("conflict"), 409),
        (KanbanNotFoundError("missing"), 404),
        (ChaChaInputError("bad"), 400),
        (ChaChaConflictError("conflict"), 409),
        (ChaChaSchemaError("schema"), 500),
        (PromptsInputError("bad"), 400),
        (PromptsConflictError("conflict"), 409),
        (PromptsSchemaError("schema"), 500),
        (MeetingsInputError("bad"), 400),
        (MeetingsSchemaError("schema"), 500),
        (SlidesInputError("bad"), 400),
        (SlidesConflictError("conflict"), 409),
        (SlidesSchemaError("schema"), 500),
    ],
)
def test_http_error_mapping_handles_cross_module_db_errors(exc, expected_status):
    http_exc = http_errors.map_db_error_to_http(exc, default_detail="db fallback")
    assert http_exc.status_code == expected_status
    if isinstance(exc, UnifiedDataIntegrityError):
        assert http_exc.detail == "Data integrity violation"


@pytest.mark.parametrize(
    "exc",
    [
        UnifiedDatabaseError("db"),
        BackendDatabaseError("db"),
        KanbanDBError("db"),
        CharactersRAGDBError("db"),
        PromptsDatabaseError("db"),
        MeetingsDatabaseError("db"),
        SlidesDatabaseError("db"),
    ],
)
def test_http_error_mapping_uses_default_detail_for_database_base_errors(exc):
    http_exc = http_errors.map_db_error_to_http(exc, default_detail="db fallback")
    assert http_exc.status_code == 500
    assert http_exc.detail == "db fallback"


def test_http_error_mapping_allows_safe_public_overrides():
    conflict_exc = ConflictError(
        entity="Media",
        identifier=9,
    )
    conflict_http_exc = http_errors.map_db_error_to_http(
        conflict_exc,
        conflict_detail="Media was modified concurrently",
    )

    assert conflict_http_exc.status_code == 409
    assert conflict_http_exc.detail == "Media was modified concurrently"

    input_exc = InputError("Cannot update keywords: Media ID 9 not found or deleted.")
    input_http_exc = http_errors.map_db_error_to_http(
        input_exc,
        input_status=404,
        input_detail="Media not found or deleted",
    )

    assert input_http_exc.status_code == 404
    assert input_http_exc.detail == "Media not found or deleted"


def test_http_error_mapping_logs_database_errors_with_context(monkeypatch):
    logged_calls = []

    def _fake_error(message, *args, **kwargs):
        logged_calls.append((message, args, kwargs))

    monkeypatch.setattr(http_errors.logger, "error", _fake_error)

    http_exc = http_errors.map_db_error_to_http(
        DatabaseError("write failed"),
        default_detail="Database error moving media to trash",
        log_context="delete_media_item media_id=42",
    )

    assert http_exc.status_code == 500
    assert http_exc.detail == "Database error moving media to trash"
    assert logged_calls
    assert "delete_media_item media_id=42" in logged_calls[0][0]


def test_http_error_mapping_can_skip_database_error_logging(monkeypatch):
    logged_calls = []

    def _fake_error(message, *args, **kwargs):
        logged_calls.append((message, args, kwargs))

    monkeypatch.setattr(http_errors.logger, "error", _fake_error)

    http_exc = http_errors.map_db_error_to_http(
        DatabaseError("write failed"),
        default_detail="Database error moving media to trash",
        log_error=False,
    )

    assert http_exc.status_code == 500
    assert http_exc.detail == "Database error moving media to trash"
    assert logged_calls == []

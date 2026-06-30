import pytest


class _FakeMetadataDB:
    def __init__(self, index_error=None):
        from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

        self.backend_type = BackendType.SQLITE
        self.index_error = index_error
        self.queries = []

    def execute_query(self, query, params, connection=None):
        self.queries.append((query, params, connection))
        if "DocumentVersionIdentifiers" in query and self.index_error is not None:
            raise self.index_error
        return None


def test_normalize_safe_metadata_doi_valid():


    from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

    sm = normalize_safe_metadata({"DOI": "10.1000/xyz.ABC-123"})
    assert sm.get("doi") == "10.1000/xyz.ABC-123"


def test_normalize_safe_metadata_doi_invalid_raises():


    from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

    with pytest.raises(ValueError):
        normalize_safe_metadata({"doi": "not-a-doi"})


def test_normalize_safe_metadata_pmcid_normalizes():


    from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

    sm = normalize_safe_metadata({"PMCID": "PMC123456"})
    assert sm.get("pmcid") == "123456"


def test_normalize_safe_metadata_pmid_digits():


    from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

    sm = normalize_safe_metadata({"pmid": "PMID 987654"})
    assert sm.get("pmid") == "987654"


def test_normalize_safe_metadata_arxiv_pass_through():


    from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

    sm = normalize_safe_metadata({"arXiv": "1706.03762v2"})
    assert sm.get("arxiv_id") == "1706.03762v2"


def test_update_version_safe_metadata_propagates_unexpected_identifier_index_failure():
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.Utils.metadata_utils import update_version_safe_metadata_in_transaction

    db = _FakeMetadataDB(index_error=DatabaseError("constraint failed"))

    with pytest.raises(DatabaseError, match="constraint failed"):
        update_version_safe_metadata_in_transaction(
            db=db,
            dv_id=12,
            safe_metadata_json='{"doi":"10.1000/example"}',
            merged_metadata={"doi": "10.1000/example"},
            connection=object(),
        )


def test_update_version_safe_metadata_propagates_unrelated_unsupported_identifier_failure():
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.Utils.metadata_utils import update_version_safe_metadata_in_transaction

    db = _FakeMetadataDB(index_error=DatabaseError("unsupported column type"))

    with pytest.raises(DatabaseError, match="unsupported column type"):
        update_version_safe_metadata_in_transaction(
            db=db,
            dv_id=12,
            safe_metadata_json='{"doi":"10.1000/example"}',
            merged_metadata={"doi": "10.1000/example"},
            connection=object(),
        )


def test_update_version_safe_metadata_skips_missing_identifier_table_for_legacy_schema():
    import sqlite3

    from tldw_Server_API.app.core.Utils.metadata_utils import update_version_safe_metadata_in_transaction

    db = _FakeMetadataDB(index_error=sqlite3.OperationalError("no such table: DocumentVersionIdentifiers"))

    update_version_safe_metadata_in_transaction(
        db=db,
        dv_id=12,
        safe_metadata_json='{"doi":"10.1000/example"}',
        merged_metadata={"doi": "10.1000/example"},
        connection=object(),
    )

    assert len(db.queries) == 2

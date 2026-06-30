"""
ChromaDB Tenant Isolation Verification Tests (Gap 5.1)

These tests verify that ChromaDB tenant isolation works correctly.
They prove that:
1. Cross-tenant search isolation: Tenant A's data is invisible to Tenant B
2. Filesystem directory separation: Each tenant gets their own ChromaDB directory
3. Collection listing isolation: Tenant B cannot see Tenant A's collections
4. User ID validation: Path traversal attacks are rejected
5. Default collection naming: Collection names always include user_id

These are VERIFICATION tests -- they should PASS because isolation already exists
in ChromaDBManager. If any test fails, it reveals a real security gap.
"""

import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import (
    ChromaDBManager,
    validate_user_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(user_id: str, base_dir: str) -> ChromaDBManager:
    """Create a ChromaDBManager with the in-memory stub backend for testing."""
    return ChromaDBManager(
        user_id=user_id,
        user_embedding_config={
            "USER_DB_BASE_DIR": base_dir,
            "embedding_config": {},
            "chroma_client_settings": {"backend": "stub"},
        },
    )


def _random_embeddings(n: int, dim: int = 8) -> list[list[float]]:
    """Generate n random embedding vectors of given dimension."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim))
    # L2-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).tolist()


@pytest.fixture(autouse=True)
def _clear_stub_clients():
    yield
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
    cdl._TEST_STUB_CLIENTS.clear()


@pytest.fixture()
def isolated_base_dir(tmp_path):
    """Provide a temporary base directory for ChromaDB tenant tests."""
    base = tmp_path / "chroma_tenant_test"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


# ---------------------------------------------------------------------------
# 1. Cross-tenant search isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCrossTenantSearchIsolation:
    """Verify that one tenant's stored embeddings are invisible to another."""

    def test_tenant_b_gets_zero_results_from_tenant_a_data(self, isolated_base_dir):
        """Tenant A stores embeddings; Tenant B searches and gets nothing."""
        manager_a = _make_manager("tenantA", isolated_base_dir)
        manager_b = _make_manager("tenantB", isolated_base_dir)

        collection_name = "shared_name_collection"
        texts = ["document alpha", "document beta"]
        ids = ["id1", "id2"]
        embeddings = _random_embeddings(2, dim=8)
        metadatas = [{"source": "tenantA"}, {"source": "tenantA"}]

        # Tenant A stores data
        manager_a.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        # Verify Tenant A can see its own data
        coll_a = manager_a.get_or_create_collection(collection_name)
        assert coll_a.count() == 2, "Tenant A should see its own 2 documents"

        # Tenant B creates/gets same-named collection -- it should be empty
        coll_b = manager_b.get_or_create_collection(collection_name)
        assert coll_b.count() == 0, (
            "Tenant B must NOT see Tenant A's documents in a same-named collection"
        )

    def test_tenant_b_query_returns_empty_for_tenant_a_data(self, isolated_base_dir):
        """Query by Tenant B against a same-named collection yields no results."""
        manager_a = _make_manager("tenantA", isolated_base_dir)
        manager_b = _make_manager("tenantB", isolated_base_dir)

        collection_name = "query_isolation_test"
        embeddings = _random_embeddings(3, dim=8)
        texts = ["alpha", "beta", "gamma"]
        ids = ["a1", "a2", "a3"]
        metadatas = [{"idx": i} for i in range(3)]

        manager_a.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        # Tenant B queries with precomputed embedding
        query_emb = _random_embeddings(1, dim=8)
        results = manager_b.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=query_emb,
            n_results=10,
        )
        # results should have empty id lists
        result_ids = results.get("ids", [[]])[0] if results else []
        assert len(result_ids) == 0, (
            "Tenant B query must return zero results from Tenant A's collection"
        )

    def test_separate_tenants_store_independently(self, isolated_base_dir):
        """Both tenants store data in same-named collection; each sees only their own."""
        manager_a = _make_manager("userAlpha", isolated_base_dir)
        manager_b = _make_manager("userBeta", isolated_base_dir)

        coll_name = "independent_data"
        emb_a = _random_embeddings(2, dim=8)
        emb_b = _random_embeddings(3, dim=8)

        manager_a.store_in_chroma(
            collection_name=coll_name,
            texts=["a1", "a2"],
            embeddings=emb_a,
            ids=["a-1", "a-2"],
            metadatas=[{"tenant": "alpha"}] * 2,
        )
        manager_b.store_in_chroma(
            collection_name=coll_name,
            texts=["b1", "b2", "b3"],
            embeddings=emb_b,
            ids=["b-1", "b-2", "b-3"],
            metadatas=[{"tenant": "beta"}] * 3,
        )

        coll_a = manager_a.get_or_create_collection(coll_name)
        coll_b = manager_b.get_or_create_collection(coll_name)

        assert coll_a.count() == 2, "Tenant alpha should see exactly 2 items"
        assert coll_b.count() == 3, "Tenant beta should see exactly 3 items"


# ---------------------------------------------------------------------------
# 2. Filesystem directory separation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilesystemDirectorySeparation:
    """Verify each tenant gets their own ChromaDB directory on disk."""

    def test_different_users_get_different_chroma_dirs(self, isolated_base_dir):
        """Two users must have distinct user_chroma_path values."""
        manager_a = _make_manager("user1", isolated_base_dir)
        manager_b = _make_manager("user2", isolated_base_dir)

        path_a = pathlib.Path(str(manager_a.user_chroma_path)).resolve()
        path_b = pathlib.Path(str(manager_b.user_chroma_path)).resolve()

        assert path_a != path_b, (
            f"Users must have different chroma dirs, got same: {path_a}"
        )

    def test_chroma_dir_contains_user_id(self, isolated_base_dir):
        """The chroma directory path must include the user_id component."""
        manager = _make_manager("myuser42", isolated_base_dir)
        chroma_path_str = str(manager.user_chroma_path)

        assert "myuser42" in chroma_path_str, (
            f"Chroma path should contain user_id 'myuser42', got: {chroma_path_str}"
        )

    def test_chroma_dir_under_base_dir(self, isolated_base_dir):
        """The user's chroma directory must be under the configured base directory."""
        manager = _make_manager("tenant99", isolated_base_dir)
        resolved_base = pathlib.Path(isolated_base_dir).resolve()
        resolved_chroma = pathlib.Path(str(manager.user_chroma_path)).resolve()

        assert str(resolved_chroma).startswith(str(resolved_base)), (
            f"Chroma dir {resolved_chroma} must be under base dir {resolved_base}"
        )

    def test_chroma_dir_is_created_on_init(self, isolated_base_dir):
        """The chroma storage directory should exist after manager init."""
        manager = _make_manager("newuser", isolated_base_dir)
        assert os.path.isdir(str(manager.user_chroma_path)), (
            f"Chroma directory should be created on init: {manager.user_chroma_path}"
        )

    def test_chroma_dir_includes_chroma_storage_subdir(self, isolated_base_dir):
        """Path should end with the 'chroma_storage' subdirectory."""
        manager = _make_manager("subdir_check", isolated_base_dir)
        chroma_path = pathlib.Path(str(manager.user_chroma_path))

        assert chroma_path.name == "chroma_storage", (
            f"Chroma dir should end with 'chroma_storage', got: {chroma_path.name}"
        )


# ---------------------------------------------------------------------------
# 3. Collection listing isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCollectionListingIsolation:
    """Verify that listing collections only shows the current tenant's collections."""

    def test_tenant_b_cannot_see_tenant_a_collections(self, isolated_base_dir):
        """Tenant B's list_collections must not include Tenant A's collections."""
        manager_a = _make_manager("listUserA", isolated_base_dir)
        manager_b = _make_manager("listUserB", isolated_base_dir)

        # Tenant A creates some collections
        manager_a.get_or_create_collection("collection_x")
        manager_a.get_or_create_collection("collection_y")

        # Tenant B should see zero collections
        b_collections = manager_b.list_collections()
        b_names = [c.name if hasattr(c, "name") else str(c) for c in b_collections]

        assert "collection_x" not in b_names, (
            "Tenant B must not see Tenant A's collection_x"
        )
        assert "collection_y" not in b_names, (
            "Tenant B must not see Tenant A's collection_y"
        )

    def test_tenant_sees_only_own_collections(self, isolated_base_dir):
        """Each tenant should see only the collections they created."""
        manager_a = _make_manager("colOwnerA", isolated_base_dir)
        manager_b = _make_manager("colOwnerB", isolated_base_dir)

        manager_a.get_or_create_collection("a_private")
        manager_b.get_or_create_collection("b_private")

        a_names = [
            c.name if hasattr(c, "name") else str(c)
            for c in manager_a.list_collections()
        ]
        b_names = [
            c.name if hasattr(c, "name") else str(c)
            for c in manager_b.list_collections()
        ]

        assert "a_private" in a_names, "Tenant A should see its own collection"
        assert "b_private" not in a_names, "Tenant A must not see Tenant B's collection"
        assert "b_private" in b_names, "Tenant B should see its own collection"
        assert "a_private" not in b_names, "Tenant B must not see Tenant A's collection"

    def test_empty_listing_for_new_tenant(self, isolated_base_dir):
        """A brand-new tenant should have zero collections."""
        # First, create a tenant with collections to ensure they exist somewhere
        manager_existing = _make_manager("existingTenant", isolated_base_dir)
        manager_existing.get_or_create_collection("some_col")

        # New tenant should see nothing
        manager_new = _make_manager("freshTenant", isolated_base_dir)
        collections = manager_new.list_collections()
        assert len(collections) == 0, (
            "A fresh tenant must see zero collections"
        )


# ---------------------------------------------------------------------------
# 4. User ID validation (path traversal attacks)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserIdValidation:
    """Verify that dangerous user IDs are rejected by validate_user_id()."""

    @pytest.mark.parametrize(
        "bad_id,description",
        [
            ("../evil", "parent directory traversal"),
            ("..\\evil", "backslash parent traversal"),
            ("/etc/passwd", "absolute path with slash"),
            ("user/../other", "embedded parent traversal"),
            ("user/sub", "forward slash in id"),
            ("user\\sub", "backslash in id"),
            ("user\x00evil", "null byte injection"),
            ("user\nevil", "newline injection"),
            ("user\revil", "carriage return injection"),
            ("", "empty string"),
        ],
    )
    def test_validate_user_id_rejects_dangerous_input(self, bad_id, description):
        """validate_user_id() must raise ValueError for: {description}."""
        with pytest.raises(ValueError):
            validate_user_id(bad_id)

    def test_validate_user_id_rejects_path_traversal_when_audit_times_out(self, monkeypatch):
        """Audit timeouts must not mask the path traversal validation error."""
        def _timeout(**_kwargs):
            raise TimeoutError("audit write timed out")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Embeddings.ChromaDB_Library.log_security_violation",
            _timeout,
        )

        with pytest.raises(ValueError, match="forbidden characters"):
            validate_user_id("../evil")

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../evil",
            "..\\evil",
            "/etc/passwd",
            "user/../other",
            "user\x00evil",
            "..\\escape",
            "user/slash",
            "user\\back",
            "user\nevil",
            "user\revil",
            "",
        ],
    )
    def test_chromadb_manager_rejects_dangerous_user_id(self, bad_id, isolated_base_dir):
        """ChromaDBManager constructor must reject dangerous user_ids."""
        with pytest.raises(ValueError):
            _make_manager(bad_id, isolated_base_dir)

    @pytest.mark.parametrize(
        "valid_id",
        [
            "user123",
            "test_user",
            "tenant-42",
            "UPPERCASE",
            "MiXeD_CaSe-123",
        ],
    )
    def test_validate_user_id_accepts_valid_input(self, valid_id):
        """validate_user_id() must accept safe identifiers."""
        result = validate_user_id(valid_id)
        assert result == valid_id

    def test_user_id_max_length_enforced(self):
        """User IDs exceeding 255 characters must be rejected."""
        long_id = "a" * 256
        with pytest.raises(ValueError, match="maximum length"):
            validate_user_id(long_id)

    def test_user_id_at_max_length_accepted(self):
        """A user ID of exactly 255 characters should be accepted."""
        exact_id = "a" * 255
        result = validate_user_id(exact_id)
        assert result == exact_id

    @pytest.mark.parametrize(
        "bad_id",
        [
            "user$name",
            "user@domain",
            "user name",
            "user;cmd",
            "user$(whoami)",
            "user`ls`",
            "user|pipe",
        ],
    )
    def test_special_characters_rejected(self, bad_id):
        """User IDs with shell or special characters must be rejected."""
        with pytest.raises(ValueError):
            validate_user_id(bad_id)


# ---------------------------------------------------------------------------
# 5. Default collection naming includes user_id
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDefaultCollectionNaming:
    """Verify that default collection names incorporate the user_id."""

    def test_default_collection_name_contains_user_id(self, isolated_base_dir):
        """get_user_default_collection_name() must include the user_id."""
        manager = _make_manager("myuser", isolated_base_dir)
        default_name = manager.get_user_default_collection_name()

        assert "myuser" in default_name, (
            f"Default collection name '{default_name}' must contain user_id 'myuser'"
        )

    def test_default_collection_name_has_prefix(self, isolated_base_dir):
        """Default collection name should start with the standard prefix."""
        manager = _make_manager("prefixtest", isolated_base_dir)
        default_name = manager.get_user_default_collection_name()

        expected_prefix = ChromaDBManager.DEFAULT_COLLECTION_NAME_PREFIX
        assert default_name.startswith(expected_prefix), (
            f"Default name '{default_name}' should start with '{expected_prefix}'"
        )

    def test_different_users_get_different_default_collection_names(self, isolated_base_dir):
        """Two different users must get different default collection names."""
        manager_a = _make_manager("alice", isolated_base_dir)
        manager_b = _make_manager("bob", isolated_base_dir)

        name_a = manager_a.get_user_default_collection_name()
        name_b = manager_b.get_user_default_collection_name()

        assert name_a != name_b, (
            f"Default collection names must differ: alice='{name_a}', bob='{name_b}'"
        )

    def test_default_collection_name_format(self, isolated_base_dir):
        """Default collection name should be exactly PREFIX + user_id."""
        user_id = "formatcheck"
        manager = _make_manager(user_id, isolated_base_dir)
        default_name = manager.get_user_default_collection_name()
        expected = f"{ChromaDBManager.DEFAULT_COLLECTION_NAME_PREFIX}{user_id}"

        assert default_name == expected, (
            f"Expected '{expected}', got '{default_name}'"
        )

    def test_get_or_create_uses_default_name_when_none(self, isolated_base_dir):
        """get_or_create_collection(None) should use the default user-scoped name."""
        manager = _make_manager("defaultcol", isolated_base_dir)
        collection = manager.get_or_create_collection(None)
        expected_name = manager.get_user_default_collection_name()

        assert collection.name == expected_name, (
            f"Collection name should be '{expected_name}', got '{collection.name}'"
        )


# ---------------------------------------------------------------------------
# Additional edge-case isolation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsolationEdgeCases:
    """Additional edge cases for tenant isolation."""

    def test_delete_in_one_tenant_does_not_affect_other(self, isolated_base_dir):
        """Deleting a collection in Tenant A must not affect Tenant B."""
        manager_a = _make_manager("delA", isolated_base_dir)
        manager_b = _make_manager("delB", isolated_base_dir)

        coll_name = "deletable"
        emb = _random_embeddings(1, dim=8)

        # Both tenants create same-named collection with data
        manager_a.store_in_chroma(
            collection_name=coll_name,
            texts=["doc_a"],
            embeddings=emb,
            ids=["id_a"],
            metadatas=[{"t": "a"}],
        )
        manager_b.store_in_chroma(
            collection_name=coll_name,
            texts=["doc_b"],
            embeddings=emb,
            ids=["id_b"],
            metadatas=[{"t": "b"}],
        )

        # Delete Tenant A's collection
        manager_a.delete_collection(coll_name)

        # Tenant B's collection must still exist with its data
        coll_b = manager_b.get_or_create_collection(coll_name)
        assert coll_b.count() == 1, (
            "Deleting Tenant A's collection must not affect Tenant B's data"
        )

    def test_manager_user_id_stored_correctly(self, isolated_base_dir):
        """The manager's user_id attribute must match what was provided."""
        manager = _make_manager("verify_id", isolated_base_dir)
        assert manager.user_id == "verify_id"

    def test_stub_clients_are_per_user_and_base_dir(self, isolated_base_dir):
        """In-memory stub clients must be scoped per (user_id, base_dir) pair."""
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

        base_dir_2 = tempfile.mkdtemp(prefix="chroma_tenant_test2_")
        try:
            m1 = _make_manager("sameuser", isolated_base_dir)
            m2 = _make_manager("sameuser", base_dir_2)

            # They should have different client instances because base_dir differs
            assert m1.client is not m2.client, (
                "Same user_id with different base_dirs must get different stub clients"
            )
        finally:
            shutil.rmtree(base_dir_2, ignore_errors=True)

import contextlib
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends import pg_rls_policies as rls_module
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    build_chacha_rls_sql,
    ensure_chacha_rls,
    ensure_prompt_studio_rls,
)


class _FailingCursor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _sql: str) -> None:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")


class _TxnConn:
    def __init__(self) -> None:
        self.cursor_obj = _FailingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Backend:
    backend_type = SimpleNamespace(name="POSTGRESQL")

    def __init__(self, conn: _TxnConn) -> None:
        self._conn = conn

    def transaction(self):
        @contextlib.contextmanager
        def _ctx():
            yield self._conn

        return _ctx()


@pytest.mark.parametrize(
    "backend",
    [
        SimpleNamespace(backend_type=SimpleNamespace(name="SQLITE")),
        object(),
    ],
)
def test_ensure_prompt_studio_rls_returns_false_for_non_postgres_backends(backend):
    if ensure_prompt_studio_rls(backend) is not False:
        pytest.fail("expected non-PostgreSQL backends to be ignored")


def test_ensure_prompt_studio_rls_raises_on_partial_failure():
    conn = _TxnConn()

    with pytest.raises(DatabaseError, match="prompt_studio"):
        ensure_prompt_studio_rls(_Backend(conn))

    if conn.committed is not False:
        pytest.fail("transaction should not commit after a partial failure")
    if conn.rolled_back is not True:
        pytest.fail("transaction should roll back after a partial failure")


def test_chacha_rls_includes_workspace_resource_memberships_tenant_policy():
    sql = "\n".join(build_chacha_rls_sql())

    assert "ALTER TABLE IF EXISTS workspace_resource_memberships ENABLE ROW LEVEL SECURITY" in sql
    assert "ALTER TABLE IF EXISTS workspace_resource_memberships FORCE ROW LEVEL SECURITY" in sql
    assert "DROP POLICY IF EXISTS workspace_resource_memberships_tenant_isolation" in sql
    assert "CREATE POLICY workspace_resource_memberships_tenant_isolation" in sql
    assert "ON workspace_resource_memberships" in sql
    assert "client_id = current_setting('app.current_user_id', true)" in sql


def test_shared_workspace_chat_rls_is_guarded_forced_and_canonical() -> None:
    policy_statements = rls_module.build_shared_workspace_chat_rls_sql()
    policy_sql = " ".join("\n".join(policy_statements).split())
    canonical_sql = "\n".join(build_chacha_rls_sql())

    for table in (
        "shared_workspace_chat_threads",
        "shared_workspace_chat_requests",
    ):
        assert f"to_regclass('{table}')" in policy_sql
        assert f"ALTER TABLE IF EXISTS {table} ENABLE ROW LEVEL SECURITY" in policy_sql
        assert f"ALTER TABLE IF EXISTS {table} FORCE ROW LEVEL SECURITY" in policy_sql
        assert f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}" in policy_sql
        assert f"CREATE POLICY {table}_tenant_isolation ON {table}" in policy_sql
    for statement in policy_statements:
        assert canonical_sql.count(statement) == 1


def test_shared_workspace_chat_rls_checks_recipient_conversation_thread_and_messages() -> None:
    sql = " ".join("\n".join(rls_module.build_shared_workspace_chat_rls_sql()).split())
    owner = "current_setting('app.current_user_id', true)"
    thread_policy = sql.split(
        "CREATE POLICY shared_workspace_chat_threads_tenant_isolation "
        "ON shared_workspace_chat_threads",
        1,
    )[1].split("$thread_policy$", 1)[0]
    request_policy = sql.split(
        "CREATE POLICY shared_workspace_chat_requests_tenant_isolation "
        "ON shared_workspace_chat_requests",
        1,
    )[1].split("$request_policy$", 1)[0]

    assert "USING (" in thread_policy
    assert "WITH CHECK (" in thread_policy
    for clause in (
        f"shared_workspace_chat_threads.recipient_user_id = {owner}",
        "conversation.id = shared_workspace_chat_threads.conversation_id",
        f"conversation.client_id = {owner}",
        "conversation.client_id = shared_workspace_chat_threads.recipient_user_id",
        "conversation.deleted = false",
    ):
        assert thread_policy.count(clause) == 2

    assert "USING (" in request_policy
    assert "WITH CHECK (" in request_policy
    for clause in (
        f"shared_workspace_chat_requests.recipient_user_id = {owner}",
        "thread.recipient_user_id = shared_workspace_chat_requests.recipient_user_id",
        "thread.share_id = shared_workspace_chat_requests.share_id",
        "thread.conversation_id = shared_workspace_chat_requests.conversation_id",
        "conversation.id = shared_workspace_chat_requests.conversation_id",
        f"conversation.client_id = {owner}",
        "conversation.client_id = shared_workspace_chat_requests.recipient_user_id",
        "conversation.deleted = false",
        "user_message.id = shared_workspace_chat_requests.user_message_id",
        "user_message.conversation_id = shared_workspace_chat_requests.conversation_id",
        f"user_message.client_id = {owner}",
        "user_message.client_id = shared_workspace_chat_requests.recipient_user_id",
        "assistant_message.id = shared_workspace_chat_requests.assistant_message_id",
        "assistant_message.conversation_id = shared_workspace_chat_requests.conversation_id",
        f"assistant_message.client_id = {owner}",
        "assistant_message.client_id = shared_workspace_chat_requests.recipient_user_id",
    ):
        assert request_policy.count(clause) == 2


def test_chacha_rls_scopes_graph_projection_state_and_allows_unresolved_targets():
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())

    assert "CREATE POLICY note_graph_note_state_tenant_isolation" in sql
    state_policy = sql.split(
        "CREATE POLICY note_graph_note_state_tenant_isolation ON note_graph_note_state",
        1,
    )[1].split(";", 1)[0]
    assert "owner_user_id = current_setting('app.current_user_id', true)" in state_policy
    policy = sql.split(
        "CREATE POLICY note_wikilink_edges_tenant_isolation ON note_wikilink_edges",
        1,
    )[1].split(";", 1)[0]
    assert "source_note.id = note_wikilink_edges.source_note_id" in policy
    assert "source_note.client_id = current_setting('app.current_user_id', true)" in policy
    assert "target_note.id = note_wikilink_edges.target_note_id" not in policy


def test_chacha_rls_includes_web_clipper_owner_read_and_write_policies():
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    owner = "client_id = current_setting('app.current_user_id', true)"

    for table in (
        "note_clipper_documents",
        "note_clipper_workspace_placements",
    ):
        assert f"ALTER TABLE IF EXISTS {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE IF EXISTS {table} FORCE ROW LEVEL SECURITY" in sql
        policy = sql.split(
            f"CREATE POLICY {table}_tenant_isolation ON {table}", 1
        )[1].split(";", 1)[0]
        assert "USING (" in policy
        assert "WITH CHECK (" in policy
        assert policy.count(owner) >= 2


def test_chacha_web_clipper_rls_derives_owner_from_every_endpoint():
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    owner = "current_setting('app.current_user_id', true)"
    expected_endpoint_checks = {
        "note_clipper_documents": (
            "note.id = note_clipper_documents.note_id",
            f"note.client_id = {owner}",
        ),
        "note_clipper_workspace_placements": (
            "document.client_id = note_clipper_workspace_placements.client_id",
            "document.clip_id = note_clipper_workspace_placements.clip_id",
            "document.note_id = note_clipper_workspace_placements.source_note_id",
            "workspace.id = note_clipper_workspace_placements.workspace_id",
            f"workspace.client_id = {owner}",
            "note.id = note_clipper_workspace_placements.source_note_id",
            f"note.client_id = {owner}",
        ),
    }

    for table, checks in expected_endpoint_checks.items():
        policy = sql.split(
            f"CREATE POLICY {table}_tenant_isolation ON {table}", 1
        )[1].split(";", 1)[0]
        using, with_check = policy.split("WITH CHECK", 1)
        for check in checks:
            assert check in using
            assert check in with_check


def test_chacha_rls_covers_every_notes_organization_resource_and_derived_table():
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    tables = (
        "chacha_keywords",
        "keyword_collections",
        "note_folders",
        "note_keywords",
        "conversation_keywords",
        "collection_keywords",
        "note_folder_memberships",
        "note_folder_source_memberships",
        "note_folder_source_keys",
        "note_folder_sync_suppressions",
    )

    for table in tables:
        assert f"ALTER TABLE IF EXISTS {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE IF EXISTS {table} FORCE ROW LEVEL SECURITY" in sql
        assert f"CREATE POLICY {table}_tenant_isolation ON {table}" in sql

    for table in ("chacha_keywords", "keyword_collections", "note_folders"):
        policy = sql.split(
            f"CREATE POLICY {table}_tenant_isolation ON {table}", 1
        )[1].split(";", 1)[0]
        assert "USING (client_id = current_setting('app.current_user_id', true))" in policy
        assert "WITH CHECK (client_id = current_setting('app.current_user_id', true))" in policy


def test_chacha_notes_organization_link_rls_derives_owner_from_every_endpoint():
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    owner = "current_setting('app.current_user_id', true)"
    expected_endpoint_checks = {
        "note_keywords": (
            "note.id = note_keywords.note_id",
            f"note.client_id = {owner}",
            "keyword.id = note_keywords.keyword_id",
            f"keyword.client_id = {owner}",
        ),
        "conversation_keywords": (
            "conversation.id = conversation_keywords.conversation_id",
            f"conversation.client_id = {owner}",
            "keyword.id = conversation_keywords.keyword_id",
            f"keyword.client_id = {owner}",
        ),
        "collection_keywords": (
            "collection.id = collection_keywords.collection_id",
            f"collection.client_id = {owner}",
            "keyword.id = collection_keywords.keyword_id",
            f"keyword.client_id = {owner}",
        ),
        "note_folder_memberships": (
            "note.id = note_folder_memberships.note_id",
            f"note.client_id = {owner}",
            "folder.id = note_folder_memberships.folder_id",
            f"folder.client_id = {owner}",
        ),
        "note_folder_source_memberships": (
            "note.id = note_folder_source_memberships.note_id",
            f"note.client_id = {owner}",
            "folder.id = note_folder_source_memberships.folder_id",
            f"folder.client_id = {owner}",
        ),
        "note_folder_source_keys": (
            "folder.id = note_folder_source_keys.folder_id",
            f"folder.client_id = {owner}",
        ),
        "note_folder_sync_suppressions": (
            "note.id = note_folder_sync_suppressions.note_id",
            f"note.client_id = {owner}",
            "folder.id = note_folder_sync_suppressions.folder_id",
            f"folder.client_id = {owner}",
        ),
    }

    for table, checks in expected_endpoint_checks.items():
        policy = sql.split(
            f"CREATE POLICY {table}_tenant_isolation ON {table}", 1
        )[1].split(";", 1)[0]
        assert "USING (" in policy
        assert "WITH CHECK (" in policy
        for check in checks:
            assert check in policy


def test_chacha_notes_link_rls_checks_owner_and_both_note_endpoints() -> None:
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    policy = sql.split(
        "CREATE POLICY note_edges_tenant_isolation ON note_edges", 1
    )[1].split(";", 1)[0]
    owner = "current_setting('app.current_user_id', true)"

    assert "USING (" in policy
    assert "WITH CHECK (" in policy
    for clause in (
        f"note_edges.user_id = {owner}",
        "source_note.id = note_edges.from_note_id",
        f"source_note.client_id = {owner}",
        "target_note.id = note_edges.to_note_id",
        f"target_note.client_id = {owner}",
    ):
        assert policy.count(clause) == 2


def test_chacha_note_attachment_rls_checks_registry_and_note_owner_for_reads_and_writes() -> None:
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    policy = sql.split(
        "CREATE POLICY note_attachments_tenant_isolation ON note_attachments", 1
    )[1].split(";", 1)[0]
    owner = "current_setting('app.current_user_id', true)"

    assert "ALTER TABLE IF EXISTS note_attachments ENABLE ROW LEVEL SECURITY" in sql
    assert "ALTER TABLE IF EXISTS note_attachments FORCE ROW LEVEL SECURITY" in sql
    assert "USING (" in policy
    assert "WITH CHECK (" in policy
    for clause in (
        f"note_attachments.client_id = {owner}",
        "note.id = note_attachments.note_id",
        f"note.client_id = {owner}",
        "note.client_id = note_attachments.client_id",
    ):
        assert policy.count(clause) == 2


def test_chacha_note_task_graph_rls_checks_scope_and_owned_parents_for_reads_and_writes() -> None:
    sql = " ".join("\n".join(build_chacha_rls_sql()).split())
    owner = "current_setting('app.current_user_id', true)"
    dataset = "current_setting('app.current_dataset_id', true)"

    for table in (
        "note_tasks",
        "task_note_projections",
        "task_events",
        "task_event_read_state",
        "note_task_reconciliation_state",
        "task_projection_drifts",
    ):
        assert f"ALTER TABLE IF EXISTS {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE IF EXISTS {table} FORCE ROW LEVEL SECURITY" in sql
        policy = sql.split(
            f"CREATE POLICY {table}_tenant_isolation ON {table}", 1
        )[1].split(";", 1)[0]
        assert "USING (" in policy
        assert "WITH CHECK (" in policy
        assert policy.count(f"{table}.owner_user_id = {owner}") == 2
        assert policy.count(f"{table}.dataset_id = {dataset}") == 2

    authority_policy = sql.split(
        "CREATE POLICY note_task_scope_authority_tenant_isolation "
        "ON note_task_scope_authority",
        1,
    )[1].split(";", 1)[0]
    assert authority_policy.count(
        f"note_task_scope_authority.owner_user_id = {owner}"
    ) == 2
    assert "current_dataset_id" not in authority_policy

    task_policy = sql.split(
        "CREATE POLICY note_tasks_tenant_isolation ON note_tasks", 1
    )[1].split(";", 1)[0]
    for clause in (
        "note.id = note_tasks.note_id",
        "note.client_id = note_tasks.owner_user_id",
        f"note.client_id = {owner}",
    ):
        assert task_policy.count(clause) == 2

    read_state_policy = sql.split(
        "CREATE POLICY task_event_read_state_tenant_isolation ON task_event_read_state", 1
    )[1].split(";", 1)[0]
    for clause in (
        "task_event_read_state.user_id = task_event_read_state.owner_user_id",
        "event.id = task_event_read_state.event_id",
        "event.owner_user_id = task_event_read_state.owner_user_id",
        "event.dataset_id = task_event_read_state.dataset_id",
    ):
        assert read_state_policy.count(clause) == 2


def test_chacha_rls_includes_source_review_read_and_write_policies():
    sql = "\n".join(build_chacha_rls_sql())

    for table_name in ("source_review_plans", "source_review_occurrences"):
        assert f"to_regclass('{table_name}')" in sql
        assert f"ALTER TABLE IF EXISTS {table_name} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE IF EXISTS {table_name} FORCE ROW LEVEL SECURITY" in sql
        assert f"CREATE POLICY {table_name}_tenant_isolation ON {table_name}" in sql
    assert sql.count("WITH CHECK") >= 2


def test_chacha_rls_includes_guarded_active_workspace_source_saved_view_policy():
    sql = "\n".join(build_chacha_rls_sql())

    assert "to_regclass('workspace_source_saved_views')" in sql
    assert "ALTER TABLE IF EXISTS workspace_source_saved_views ENABLE ROW LEVEL SECURITY" in sql
    assert "ALTER TABLE IF EXISTS workspace_source_saved_views FORCE ROW LEVEL SECURITY" in sql
    assert "DROP POLICY IF EXISTS workspace_source_saved_views_tenant_isolation" in sql
    assert "CREATE POLICY workspace_source_saved_views_tenant_isolation" in sql
    assert sql.count("owner_user_id = current_setting('app.current_user_id', true)") >= 2
    assert sql.count("w.id = workspace_source_saved_views.workspace_id") >= 2
    assert sql.count("w.client_id = current_setting('app.current_user_id', true)") >= 2
    assert sql.count("w.deleted = false") >= 2
    assert "WITH CHECK" in sql


def test_ensure_chacha_rls_uses_the_guarded_saved_view_policy_block(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = _TxnConn()
    conn.cursor_obj.execute = lambda _sql: None
    monkeypatch.setattr(rls_module, "_ensure_chacha_schema", lambda _backend: None)

    assert ensure_chacha_rls(_Backend(conn)) is True
    assert conn.committed is True


def test_ensure_chacha_rls_migrates_schema_before_installing_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(backend_type=SimpleNamespace(name="POSTGRESQL"))
    calls: list[str] = []

    monkeypatch.setattr(
        rls_module,
        "_ensure_chacha_schema",
        lambda seen_backend: calls.append("schema") if seen_backend is backend else None,
        raising=False,
    )
    monkeypatch.setattr(
        rls_module,
        "_ensure_rls_policy_set",
        lambda seen_backend, **_kwargs: calls.append("policies") or seen_backend is backend,
    )

    assert rls_module.ensure_chacha_rls(backend) is True
    assert calls == ["schema", "policies"]


def test_run_pg_rls_auto_ensure_logs_success_only_after_both_installers_pass(monkeypatch):
    import tldw_Server_API.app.main as main_mod

    monkeypatch.setattr(main_mod, "ensure_prompt_studio_rls", lambda _backend: True, raising=False)
    monkeypatch.setattr(main_mod, "ensure_chacha_rls", lambda _backend: True, raising=False)

    logged_messages: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _LoggerStub:
        def info(self, message: str, *args: object, **kwargs: object) -> None:
            logged_messages.append((message, args, kwargs))

    monkeypatch.setattr(main_mod, "logger", _LoggerStub(), raising=False)

    main_mod._run_pg_rls_auto_ensure(object())

    if not logged_messages:
        pytest.fail("expected startup helper to log the combined RLS result")
    if "PG RLS ensure invoked" not in logged_messages[0][0]:
        pytest.fail("expected startup helper to log the combined RLS result")

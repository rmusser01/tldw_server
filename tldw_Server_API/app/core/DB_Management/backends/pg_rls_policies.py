"""
Helpers to apply Postgres RLS policies for per-tenant isolation (Prompt Studio & ChaChaNotes).

Usage (programmatic):
  from .pg_rls_policies import ensure_prompt_studio_rls
  ensure_prompt_studio_rls(backend)  # idempotent

This module builds and applies SQL similar to Docs/Deployment/Database/postgres-rls-policies.sql.
"""
from __future__ import annotations

import contextlib

try:
    from .base import DatabaseBackend, DatabaseError
except Exception:  # pragma: no cover
    DatabaseBackend = object  # type: ignore
    class DatabaseError(Exception): ...  # type: ignore


def build_prompt_studio_rls_sql() -> list[str]:
    stmts: list[str] = []

    def add(sql: str) -> None:
        stmts.append(sql.strip())

    # Projects
    add("ALTER TABLE IF EXISTS prompt_studio_projects ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_projects FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_projects_tenant_isolation ON prompt_studio_projects;")
    add(
        """
        CREATE POLICY ps_projects_tenant_isolation ON prompt_studio_projects
          USING (user_id = current_setting('app.current_user_id', true));
        """
    )
    # Prompts
    add("ALTER TABLE IF EXISTS prompt_studio_prompts ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_prompts FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_prompts_tenant_isolation ON prompt_studio_prompts;")
    add(
        """
        CREATE POLICY ps_prompts_tenant_isolation ON prompt_studio_prompts
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_prompts.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Signatures
    add("ALTER TABLE IF EXISTS prompt_studio_signatures ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_signatures FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_signatures_tenant_isolation ON prompt_studio_signatures;")
    add(
        """
        CREATE POLICY ps_signatures_tenant_isolation ON prompt_studio_signatures
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_signatures.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Test cases
    add("ALTER TABLE IF EXISTS prompt_studio_test_cases ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_test_cases FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_test_cases_tenant_isolation ON prompt_studio_test_cases;")
    add(
        """
        CREATE POLICY ps_test_cases_tenant_isolation ON prompt_studio_test_cases
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_test_cases.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Test runs
    add("ALTER TABLE IF EXISTS prompt_studio_test_runs ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_test_runs FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_test_runs_tenant_isolation ON prompt_studio_test_runs;")
    add(
        """
        CREATE POLICY ps_test_runs_tenant_isolation ON prompt_studio_test_runs
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_test_runs.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Evaluations
    add("ALTER TABLE IF EXISTS prompt_studio_evaluations ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_evaluations FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_evals_tenant_isolation ON prompt_studio_evaluations;")
    add(
        """
        CREATE POLICY ps_evals_tenant_isolation ON prompt_studio_evaluations
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_evaluations.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Optimizations
    add("ALTER TABLE IF EXISTS prompt_studio_optimizations ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_optimizations FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_opts_tenant_isolation ON prompt_studio_optimizations;")
    add(
        """
        CREATE POLICY ps_opts_tenant_isolation ON prompt_studio_optimizations
          USING (
            EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_optimizations.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Optimization iterations
    add("ALTER TABLE IF EXISTS prompt_studio_optimization_iterations ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_optimization_iterations FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_iter_tenant_isolation ON prompt_studio_optimization_iterations;")
    add(
        """
        CREATE POLICY ps_iter_tenant_isolation ON prompt_studio_optimization_iterations
          USING (
            EXISTS (
              SELECT 1
              FROM prompt_studio_optimizations o
              JOIN prompt_studio_projects p ON p.id = o.project_id
              WHERE o.id = prompt_studio_optimization_iterations.optimization_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Job queue
    add("ALTER TABLE IF EXISTS prompt_studio_job_queue ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_job_queue FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_jobq_tenant_isolation ON prompt_studio_job_queue;")
    add(
        """
        CREATE POLICY ps_jobq_tenant_isolation ON prompt_studio_job_queue
          USING (
            (client_id = current_setting('app.current_user_id', true))
            OR EXISTS (
              SELECT 1 FROM prompt_studio_projects p
              WHERE p.id = prompt_studio_job_queue.project_id
                AND p.user_id = current_setting('app.current_user_id', true)
            )
          );
        """
    )
    # Idempotency (own + NULL scope)
    add("ALTER TABLE IF EXISTS prompt_studio_idempotency ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS prompt_studio_idempotency FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS ps_idem_tenant_isolation ON prompt_studio_idempotency;")
    add(
        """
        CREATE POLICY ps_idem_tenant_isolation ON prompt_studio_idempotency
          USING (
            user_id = current_setting('app.current_user_id', true)
            OR user_id IS NULL
          );
        """
    )
    return stmts


def build_source_review_rls_sql() -> list[str]:
    """Build tenant policies for source-review plans and occurrences."""
    return [
        """
        DO $source_review_plans_rls$
        BEGIN
          IF to_regclass('source_review_plans') IS NULL THEN
            RETURN;
          END IF;
          EXECUTE 'ALTER TABLE IF EXISTS source_review_plans ENABLE ROW LEVEL SECURITY';
          EXECUTE 'ALTER TABLE IF EXISTS source_review_plans FORCE ROW LEVEL SECURITY';
          EXECUTE 'DROP POLICY IF EXISTS source_review_plans_tenant_isolation ON source_review_plans';
          EXECUTE $plan_policy$
            CREATE POLICY source_review_plans_tenant_isolation ON source_review_plans
              USING (client_id = current_setting('app.current_user_id', true))
              WITH CHECK (client_id = current_setting('app.current_user_id', true))
          $plan_policy$;
        END
        $source_review_plans_rls$;
        """.strip(),
        """
        DO $source_review_occurrences_rls$
        BEGIN
          IF to_regclass('source_review_occurrences') IS NULL THEN
            RETURN;
          END IF;
          EXECUTE 'ALTER TABLE IF EXISTS source_review_occurrences ENABLE ROW LEVEL SECURITY';
          EXECUTE 'ALTER TABLE IF EXISTS source_review_occurrences FORCE ROW LEVEL SECURITY';
          EXECUTE 'DROP POLICY IF EXISTS source_review_occurrences_tenant_isolation ON source_review_occurrences';
          EXECUTE $occurrence_policy$
            CREATE POLICY source_review_occurrences_tenant_isolation ON source_review_occurrences
              USING (
                client_id = current_setting('app.current_user_id', true)
                AND EXISTS (
                  SELECT 1 FROM source_review_plans p
                  WHERE p.id = source_review_occurrences.plan_id
                    AND p.client_id = current_setting('app.current_user_id', true)
                )
              )
              WITH CHECK (
                client_id = current_setting('app.current_user_id', true)
                AND EXISTS (
                  SELECT 1 FROM source_review_plans p
                  WHERE p.id = source_review_occurrences.plan_id
                    AND p.client_id = current_setting('app.current_user_id', true)
                )
              )
          $occurrence_policy$;
        END
        $source_review_occurrences_rls$;
        """.strip(),
    ]


def build_workspace_source_saved_view_rls_sql() -> list[str]:
    """Build the guarded owner-and-active-workspace saved-view policy."""
    return [
        """
        DO $workspace_source_saved_views_rls$
        BEGIN
          IF to_regclass('workspace_source_saved_views') IS NULL THEN
            RETURN;
          END IF;
          EXECUTE 'ALTER TABLE IF EXISTS workspace_source_saved_views ENABLE ROW LEVEL SECURITY';
          EXECUTE 'ALTER TABLE IF EXISTS workspace_source_saved_views FORCE ROW LEVEL SECURITY';
          EXECUTE 'DROP POLICY IF EXISTS workspace_source_saved_views_tenant_isolation ON workspace_source_saved_views';
          EXECUTE $saved_view_policy$
            CREATE POLICY workspace_source_saved_views_tenant_isolation
              ON workspace_source_saved_views
              USING (
                owner_user_id = current_setting('app.current_user_id', true)
                AND EXISTS (
                  SELECT 1 FROM workspaces w
                  WHERE w.id = workspace_source_saved_views.workspace_id
                    AND w.client_id = current_setting('app.current_user_id', true)
                    AND w.deleted = false
                )
              )
              WITH CHECK (
                owner_user_id = current_setting('app.current_user_id', true)
                AND EXISTS (
                  SELECT 1 FROM workspaces w
                  WHERE w.id = workspace_source_saved_views.workspace_id
                    AND w.client_id = current_setting('app.current_user_id', true)
                    AND w.deleted = false
                )
              )
          $saved_view_policy$;
        END
        $workspace_source_saved_views_rls$;
        """.strip()
    ]


def build_web_clipper_rls_sql() -> list[str]:
    """Owner policies for shared PostgreSQL Web Clipper sidecars."""

    document_owner = """
        client_id = current_setting('app.current_user_id', true)
        AND EXISTS (
          SELECT 1 FROM notes AS note
          WHERE note.id = note_clipper_documents.note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
    """.strip()
    placement_owner = """
        client_id = current_setting('app.current_user_id', true)
        AND EXISTS (
          SELECT 1 FROM note_clipper_documents AS document
          WHERE document.client_id = note_clipper_workspace_placements.client_id
            AND document.clip_id = note_clipper_workspace_placements.clip_id
            AND document.note_id = note_clipper_workspace_placements.source_note_id
        )
        AND EXISTS (
          SELECT 1 FROM workspaces AS workspace
          WHERE workspace.id = note_clipper_workspace_placements.workspace_id
            AND workspace.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM notes AS note
          WHERE note.id = note_clipper_workspace_placements.source_note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
    """.strip()
    return [
        "ALTER TABLE IF EXISTS note_clipper_documents ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE IF EXISTS note_clipper_documents FORCE ROW LEVEL SECURITY;",
        "DROP POLICY IF EXISTS note_clipper_documents_tenant_isolation ON note_clipper_documents;",
        "CREATE POLICY note_clipper_documents_tenant_isolation ON "
        f"note_clipper_documents USING ({document_owner}) WITH CHECK ({document_owner});",
        "ALTER TABLE IF EXISTS note_clipper_workspace_placements ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE IF EXISTS note_clipper_workspace_placements FORCE ROW LEVEL SECURITY;",
        "DROP POLICY IF EXISTS note_clipper_workspace_placements_tenant_isolation "
        "ON note_clipper_workspace_placements;",
        "CREATE POLICY note_clipper_workspace_placements_tenant_isolation ON "
        "note_clipper_workspace_placements "
        f"USING ({placement_owner}) WITH CHECK ({placement_owner});",
    ]


def build_shared_workspace_chat_rls_sql() -> list[str]:
    """Build guarded recipient policies for shared workspace chat state."""

    thread_owner = """
        shared_workspace_chat_threads.recipient_user_id =
          current_setting('app.current_user_id', true)
        AND EXISTS (
          SELECT 1 FROM conversations AS conversation
          WHERE conversation.id = shared_workspace_chat_threads.conversation_id
            AND conversation.client_id = current_setting('app.current_user_id', true)
            AND conversation.client_id = shared_workspace_chat_threads.recipient_user_id
            AND conversation.deleted = false
        )
    """.strip()
    request_owner = """
        shared_workspace_chat_requests.recipient_user_id =
          current_setting('app.current_user_id', true)
        AND EXISTS (
          SELECT 1 FROM shared_workspace_chat_threads AS thread
          WHERE thread.recipient_user_id = shared_workspace_chat_requests.recipient_user_id
            AND thread.share_id = shared_workspace_chat_requests.share_id
            AND thread.conversation_id = shared_workspace_chat_requests.conversation_id
        )
        AND EXISTS (
          SELECT 1 FROM conversations AS conversation
          WHERE conversation.id = shared_workspace_chat_requests.conversation_id
            AND conversation.client_id = current_setting('app.current_user_id', true)
            AND conversation.client_id = shared_workspace_chat_requests.recipient_user_id
            AND conversation.deleted = false
        )
        AND (
          shared_workspace_chat_requests.user_message_id IS NULL
          OR EXISTS (
            SELECT 1 FROM messages AS user_message
            WHERE user_message.id = shared_workspace_chat_requests.user_message_id
              AND user_message.conversation_id = shared_workspace_chat_requests.conversation_id
              AND user_message.client_id = current_setting('app.current_user_id', true)
              AND user_message.client_id = shared_workspace_chat_requests.recipient_user_id
          )
        )
        AND (
          shared_workspace_chat_requests.assistant_message_id IS NULL
          OR EXISTS (
            SELECT 1 FROM messages AS assistant_message
            WHERE assistant_message.id = shared_workspace_chat_requests.assistant_message_id
              AND assistant_message.conversation_id = shared_workspace_chat_requests.conversation_id
              AND assistant_message.client_id = current_setting('app.current_user_id', true)
              AND assistant_message.client_id = shared_workspace_chat_requests.recipient_user_id
          )
        )
    """.strip()
    return [
        f"""
        DO $shared_workspace_chat_threads_rls$
        BEGIN
          IF to_regclass('shared_workspace_chat_threads') IS NULL THEN
            RETURN;
          END IF;
          EXECUTE 'ALTER TABLE IF EXISTS shared_workspace_chat_threads ENABLE ROW LEVEL SECURITY';
          EXECUTE 'ALTER TABLE IF EXISTS shared_workspace_chat_threads FORCE ROW LEVEL SECURITY';
          EXECUTE 'DROP POLICY IF EXISTS shared_workspace_chat_threads_tenant_isolation ON shared_workspace_chat_threads';
          EXECUTE $thread_policy$
            CREATE POLICY shared_workspace_chat_threads_tenant_isolation
              ON shared_workspace_chat_threads
              USING ({thread_owner})
              WITH CHECK ({thread_owner})
          $thread_policy$;
        END;
        $shared_workspace_chat_threads_rls$;
        """.strip(),
        f"""
        DO $shared_workspace_chat_requests_rls$
        BEGIN
          IF to_regclass('shared_workspace_chat_requests') IS NULL THEN
            RETURN;
          END IF;
          EXECUTE 'ALTER TABLE IF EXISTS shared_workspace_chat_requests ENABLE ROW LEVEL SECURITY';
          EXECUTE 'ALTER TABLE IF EXISTS shared_workspace_chat_requests FORCE ROW LEVEL SECURITY';
          EXECUTE 'DROP POLICY IF EXISTS shared_workspace_chat_requests_tenant_isolation ON shared_workspace_chat_requests';
          EXECUTE $request_policy$
            CREATE POLICY shared_workspace_chat_requests_tenant_isolation
              ON shared_workspace_chat_requests
              USING ({request_owner})
              WITH CHECK ({request_owner})
          $request_policy$;
        END;
        $shared_workspace_chat_requests_rls$;
        """.strip(),
    ]


def build_chacha_rls_sql() -> list[str]:
    """RLS for ChaChaNotes (notes, character_cards) using client_id scoping."""
    stmts: list[str] = []

    def add(sql: str) -> None:
        stmts.append(sql.strip())

    def add_tenant_policy(table: str, predicate: str) -> None:
        add(f"ALTER TABLE IF EXISTS {table} ENABLE ROW LEVEL SECURITY;")
        add(f"ALTER TABLE IF EXISTS {table} FORCE ROW LEVEL SECURITY;")
        add(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table};")
        add(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table}
              USING ({predicate})
              WITH CHECK ({predicate});
            """
        )

    # Notes
    add("ALTER TABLE IF EXISTS notes ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS notes FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS notes_tenant_isolation ON notes;")
    add(
        """
        CREATE POLICY notes_tenant_isolation ON notes
          USING (client_id = current_setting('app.current_user_id', true))
          WITH CHECK (client_id = current_setting('app.current_user_id', true));
        """
    )

    note_attachment_owner = """
    note_attachments.client_id = current_setting('app.current_user_id', true)
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = note_attachments.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = note_attachments.client_id
    )
    """.strip()
    add_tenant_policy("note_attachments", note_attachment_owner)

    task_scope = """
    note_tasks.owner_user_id = current_setting('app.current_user_id', true)
    AND note_tasks.dataset_id = current_setting('app.current_dataset_id', true)
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = note_tasks.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = note_tasks.owner_user_id
    )
    """.strip()
    add_tenant_policy("note_tasks", task_scope)

    projection_scope = """
    task_note_projections.owner_user_id = current_setting('app.current_user_id', true)
    AND task_note_projections.dataset_id = current_setting('app.current_dataset_id', true)
    AND EXISTS (
      SELECT 1 FROM note_tasks AS task
      WHERE task.owner_user_id = task_note_projections.owner_user_id
        AND task.dataset_id = task_note_projections.dataset_id
        AND task.id = task_note_projections.task_id
        AND task.note_id = task_note_projections.note_id
    )
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = task_note_projections.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = task_note_projections.owner_user_id
    )
    """.strip()
    add_tenant_policy("task_note_projections", projection_scope)

    event_scope = """
    task_events.owner_user_id = current_setting('app.current_user_id', true)
    AND task_events.dataset_id = current_setting('app.current_dataset_id', true)
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = task_events.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = task_events.owner_user_id
    )
    AND (
      task_events.task_id IS NULL
      OR EXISTS (
        SELECT 1 FROM note_tasks AS task
        WHERE task.owner_user_id = task_events.owner_user_id
          AND task.dataset_id = task_events.dataset_id
          AND task.id = task_events.task_id
          AND task.note_id = task_events.note_id
      )
    )
    """.strip()
    add_tenant_policy("task_events", event_scope)

    read_state_scope = """
    task_event_read_state.owner_user_id = current_setting('app.current_user_id', true)
    AND task_event_read_state.dataset_id = current_setting('app.current_dataset_id', true)
    AND task_event_read_state.user_id = task_event_read_state.owner_user_id
    AND EXISTS (
      SELECT 1 FROM task_events AS event
      WHERE event.owner_user_id = task_event_read_state.owner_user_id
        AND event.dataset_id = task_event_read_state.dataset_id
        AND event.id = task_event_read_state.event_id
    )
    """.strip()
    add_tenant_policy("task_event_read_state", read_state_scope)

    reconciliation_scope = """
    note_task_reconciliation_state.owner_user_id = current_setting('app.current_user_id', true)
    AND note_task_reconciliation_state.dataset_id = current_setting('app.current_dataset_id', true)
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = note_task_reconciliation_state.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = note_task_reconciliation_state.owner_user_id
    )
    """.strip()
    add_tenant_policy("note_task_reconciliation_state", reconciliation_scope)

    drift_scope = """
    task_projection_drifts.owner_user_id = current_setting('app.current_user_id', true)
    AND task_projection_drifts.dataset_id = current_setting('app.current_dataset_id', true)
    AND EXISTS (
      SELECT 1 FROM note_tasks AS task
      WHERE task.owner_user_id = task_projection_drifts.owner_user_id
        AND task.dataset_id = task_projection_drifts.dataset_id
        AND task.id = task_projection_drifts.task_id
        AND task.note_id = task_projection_drifts.note_id
    )
    AND EXISTS (
      SELECT 1 FROM notes AS note
      WHERE note.id = task_projection_drifts.note_id
        AND note.client_id = current_setting('app.current_user_id', true)
        AND note.client_id = task_projection_drifts.owner_user_id
    )
    """.strip()
    add_tenant_policy("task_projection_drifts", drift_scope)

    scope_authority_owner = """
    note_task_scope_authority.owner_user_id =
      current_setting('app.current_user_id', true)
    """.strip()
    add_tenant_policy("note_task_scope_authority", scope_authority_owner)

    note_edge_owner = """
    note_edges.user_id = current_setting('app.current_user_id', true)
    AND EXISTS (
      SELECT 1 FROM notes source_note
      WHERE source_note.id = note_edges.from_note_id
        AND source_note.client_id = current_setting('app.current_user_id', true)
    )
    AND EXISTS (
      SELECT 1 FROM notes target_note
      WHERE target_note.id = note_edges.to_note_id
        AND target_note.client_id = current_setting('app.current_user_id', true)
    )
    """.strip()
    add_tenant_policy("note_edges", note_edge_owner)

    for table in (
        "note_graph_dirty",
        "note_graph_note_state",
        "note_graph_projection_state",
        "note_graph_revisions",
    ):
        add_tenant_policy(
            table,
            "owner_user_id = current_setting('app.current_user_id', true)",
        )
    add_tenant_policy(
        "note_wikilink_edges",
        """
        owner_user_id = current_setting('app.current_user_id', true)
        AND EXISTS (
          SELECT 1 FROM notes source_note
          WHERE source_note.id = note_wikilink_edges.source_note_id
            AND source_note.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )

    for table in ("chacha_keywords", "keyword_collections", "note_folders"):
        add_tenant_policy(
            table,
            "client_id = current_setting('app.current_user_id', true)",
        )

    add_tenant_policy(
        "note_keywords",
        """
        EXISTS (
          SELECT 1 FROM notes note
          WHERE note.id = note_keywords.note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM chacha_keywords keyword
          WHERE keyword.id = note_keywords.keyword_id
            AND keyword.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )
    add_tenant_policy(
        "conversation_keywords",
        """
        EXISTS (
          SELECT 1 FROM conversations conversation
          WHERE conversation.id = conversation_keywords.conversation_id
            AND conversation.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM chacha_keywords keyword
          WHERE keyword.id = conversation_keywords.keyword_id
            AND keyword.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )
    add_tenant_policy(
        "collection_keywords",
        """
        EXISTS (
          SELECT 1 FROM keyword_collections collection
          WHERE collection.id = collection_keywords.collection_id
            AND collection.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM chacha_keywords keyword
          WHERE keyword.id = collection_keywords.keyword_id
            AND keyword.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )
    folder_endpoint_tables = (
        "note_folder_memberships",
        "note_folder_source_memberships",
        "note_folder_sync_suppressions",
    )
    folder_endpoint_predicates = (
        """
        EXISTS (
          SELECT 1 FROM notes note
          WHERE note.id = note_folder_memberships.note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM note_folders folder
          WHERE folder.id = note_folder_memberships.folder_id
            AND folder.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
        """
        EXISTS (
          SELECT 1 FROM notes note
          WHERE note.id = note_folder_source_memberships.note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM note_folders folder
          WHERE folder.id = note_folder_source_memberships.folder_id
            AND folder.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
        """
        EXISTS (
          SELECT 1 FROM notes note
          WHERE note.id = note_folder_sync_suppressions.note_id
            AND note.client_id = current_setting('app.current_user_id', true)
        )
        AND EXISTS (
          SELECT 1 FROM note_folders folder
          WHERE folder.id = note_folder_sync_suppressions.folder_id
            AND folder.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )
    for table, predicate in zip(
        folder_endpoint_tables,
        folder_endpoint_predicates,
        strict=True,
    ):
        add_tenant_policy(
            table,
            predicate,
        )
    add_tenant_policy(
        "note_folder_source_keys",
        """
        EXISTS (
          SELECT 1 FROM note_folders folder
          WHERE folder.id = note_folder_source_keys.folder_id
            AND folder.client_id = current_setting('app.current_user_id', true)
        )
        """.strip(),
    )

    stmts.extend(build_web_clipper_rls_sql())

    # Character cards
    add("ALTER TABLE IF EXISTS character_cards ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS character_cards FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS chars_tenant_isolation ON character_cards;")
    add(
        """
        CREATE POLICY chars_tenant_isolation ON character_cards
          USING (client_id = current_setting('app.current_user_id', true));
        """
    )

    # Workspace resource memberships
    add("ALTER TABLE IF EXISTS workspace_resource_memberships ENABLE ROW LEVEL SECURITY;")
    add("ALTER TABLE IF EXISTS workspace_resource_memberships FORCE ROW LEVEL SECURITY;")
    add("DROP POLICY IF EXISTS workspace_resource_memberships_tenant_isolation ON workspace_resource_memberships;")
    add(
        """
        CREATE POLICY workspace_resource_memberships_tenant_isolation ON workspace_resource_memberships
          USING (client_id = current_setting('app.current_user_id', true));
        """
    )
    stmts.extend(build_source_review_rls_sql())
    stmts.extend(build_workspace_source_saved_view_rls_sql())
    stmts.extend(build_shared_workspace_chat_rls_sql())
    return stmts


def _ensure_rls_policy_set(
    backend: DatabaseBackend,
    *,
    name: str,
    statements: list[str],
) -> bool:
    try:
        if not hasattr(backend, "backend_type") or backend.backend_type.name != "POSTGRESQL":
            return False
    except Exception:
        return False

    with backend.transaction() as conn:
        cur = conn.cursor()
        index = 0
        try:
            for statement in statements:
                index += 1
                cur.execute(statement)
            conn.commit()
            return True
        except Exception as exc:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise DatabaseError(f"{name} RLS statement {index} failed: {exc}") from exc


def ensure_prompt_studio_rls(backend: DatabaseBackend) -> bool:
    """Apply Prompt Studio RLS statements if running against PostgreSQL.

    Returns True when the full policy set succeeds; False only for explicit no-op cases.
    """
    return _ensure_rls_policy_set(
        backend,
        name="prompt_studio",
        statements=build_prompt_studio_rls_sql(),
    )


def _ensure_chacha_schema(backend: DatabaseBackend) -> None:
    """Run ChaChaNotes migrations before standalone policy installation."""

    try:
        if not hasattr(backend, "backend_type") or backend.backend_type.name != "POSTGRESQL":
            return
    except Exception:
        return
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    db.close_connection()


def ensure_chacha_rls(backend: DatabaseBackend) -> bool:
    _ensure_chacha_schema(backend)
    return _ensure_rls_policy_set(
        backend,
        name="chacha",
        statements=build_chacha_rls_sql(),
    )

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
          USING (client_id = current_setting('app.current_user_id', true));
        """
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


def ensure_chacha_rls(backend: DatabaseBackend) -> bool:
    return _ensure_rls_policy_set(
        backend,
        name="chacha",
        statements=build_chacha_rls_sql(),
    )

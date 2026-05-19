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


def build_chacha_rls_sql() -> list[str]:
    """RLS for ChaChaNotes (notes, character_cards) using client_id scoping."""
    stmts: list[str] = []

    def add(sql: str) -> None:
        stmts.append(sql.strip())

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
        try:
            for index, statement in enumerate(statements, start=1):
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

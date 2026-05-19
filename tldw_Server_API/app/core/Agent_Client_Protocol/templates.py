"""Inheritable ACP config template system.

Three-tier scoping: **system** -> **persona** -> **session**.  Templates stored
in the ``config_templates`` table are resolved via inheritance chains and merged
using :func:`merge_config` from ``merge_utils``.

When no DB templates are found the module falls back to the flat
:data:`PERMISSION_POLICY_TEMPLATES` dict from ``config.py`` so existing
deployments continue to work without migration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.merge_utils import merge_config

# Re-export merge_config for convenience of callers that import from templates.
__all__ = [
    "ACPConfigTemplate",
    "resolve_template_chain",
    "resolve_for_session",
    "seed_system_templates",
]

_MAX_INHERITANCE_DEPTH = 20


@dataclass
class ACPConfigTemplate:
    """A single config template record."""

    id: int | None = None
    name: str = ""
    description: str = ""
    scope: str = "system"  # system | persona | session
    scope_id: str | None = None
    base_template_id: int | None = None
    schema_version: str = "1"
    config: dict[str, Any] = field(default_factory=dict)

    # Timestamps (populated from DB rows)
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------


def _row_to_template(row: dict[str, Any]) -> ACPConfigTemplate:
    """Convert a DB row dict to an :class:`ACPConfigTemplate`."""
    config_raw = row.get("config_json", "{}")
    if isinstance(config_raw, str):
        try:
            config = json.loads(config_raw)
        except (json.JSONDecodeError, TypeError):
            config = {}
    elif isinstance(config_raw, dict):
        config = config_raw
    else:
        config = {}

    return ACPConfigTemplate(
        id=row.get("id"),
        name=str(row.get("name", "")),
        description=str(row.get("description", "")),
        scope=str(row.get("scope", "system")),
        scope_id=row.get("scope_id"),
        base_template_id=row.get("base_template_id"),
        schema_version=str(row.get("schema_version", "1")),
        config=config,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# Inheritance resolution
# ---------------------------------------------------------------------------


def _resolve_inheritance(
    db: Any,
    template: ACPConfigTemplate,
) -> list[ACPConfigTemplate]:
    """Walk the ``base_template_id`` chain, returning templates from root to *template*.

    Raises :class:`ValueError` on circular references.
    """
    chain: list[ACPConfigTemplate] = [template]
    seen_ids: set[int] = set()
    if template.id is not None:
        seen_ids.add(template.id)

    current = template
    for _ in range(_MAX_INHERITANCE_DEPTH):
        base_id = current.base_template_id
        if base_id is None:
            break
        if base_id in seen_ids:
            raise ValueError(
                f"Circular template inheritance detected: "
                f"template {current.id} references ancestor {base_id} "
                f"which is already in the chain"
            )
        seen_ids.add(base_id)
        parent_row = db.get_config_template(base_id)
        if parent_row is None:
            logger.warning(
                "Template {} references missing base_template_id {}",
                current.id,
                base_id,
            )
            break
        parent = _row_to_template(parent_row)
        chain.append(parent)
        current = parent

    chain.reverse()  # root-first
    return chain


# ---------------------------------------------------------------------------
# Chain merging
# ---------------------------------------------------------------------------


def resolve_template_chain(templates: list[ACPConfigTemplate]) -> dict[str, Any]:
    """Merge a list of templates (least-to-most specific) using :func:`merge_config`.

    Returns the fully merged config dict.
    """
    merged: dict[str, Any] = {}
    for tpl in templates:
        merged = merge_config(merged, tpl.config)
    return merged


# ---------------------------------------------------------------------------
# Session resolution (full three-tier)
# ---------------------------------------------------------------------------


def resolve_for_session(
    db: Any,
    session_id: str | None,
    persona_id: str | None,
    template_name: str | None,
) -> dict[str, Any] | None:
    """Load and merge templates for a session following system -> persona -> session scoping.

    Returns the merged config dict, or ``None`` if no templates were found
    (caller should fall back to flat ``PERMISSION_POLICY_TEMPLATES``).
    """
    layers: list[ACPConfigTemplate] = []

    # 1. System-scope template (by name)
    if template_name:
        system_rows = db.list_config_templates(scope="system", name=template_name)
        if system_rows:
            system_tpl = _row_to_template(system_rows[0])
            chain = _resolve_inheritance(db, system_tpl)
            layers.extend(chain)

    # 2. Persona-scope override (by persona_id + template_name)
    if persona_id and template_name:
        persona_rows = db.list_config_templates(
            scope="persona", scope_id=persona_id, name=template_name,
        )
        if persona_rows:
            persona_tpl = _row_to_template(persona_rows[0])
            chain = _resolve_inheritance(db, persona_tpl)
            layers.extend(chain)

    # 3. Session-scope override (by session_id + template_name)
    if session_id and template_name:
        session_rows = db.list_config_templates(
            scope="session", scope_id=session_id, name=template_name,
        )
        if session_rows:
            session_tpl = _row_to_template(session_rows[0])
            chain = _resolve_inheritance(db, session_tpl)
            layers.extend(chain)

    if not layers:
        return None

    return resolve_template_chain(layers)


# ---------------------------------------------------------------------------
# Seeding from flat PERMISSION_POLICY_TEMPLATES
# ---------------------------------------------------------------------------


def seed_system_templates(db: Any) -> int:
    """Populate the config_templates table from the flat PERMISSION_POLICY_TEMPLATES dict.

    Idempotent: existing templates with the same name and scope='system' are
    skipped.  Returns the number of templates actually inserted.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
        PERMISSION_POLICY_TEMPLATES,
    )

    inserted = 0
    for name, config in PERMISSION_POLICY_TEMPLATES.items():
        existing = db.list_config_templates(scope="system", name=name)
        if existing:
            continue
        db.create_config_template(
            name=name,
            description=config.get("description", ""),
            scope="system",
            scope_id=None,
            base_template_id=None,
            schema_version="1",
            config_json=json.dumps(config),
        )
        inserted += 1
        logger.debug("Seeded system template: {}", name)

    return inserted

"""Inheritable config template system for ACP.

Three-tier scoping: system -> persona -> session.
Templates can inherit from a base template via base_template_id.
Resolution walks the chain most-specific-first, then merges bottom-up.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Agent_Client_Protocol.merge_utils import merge_config


@dataclass
class ACPConfigTemplate:
    id: str
    name: str
    description: str = ""
    scope: str = "system"           # "system" | "persona" | "session"
    scope_id: str | None = None     # persona_id or session_id
    base_template_id: str | None = None
    schema_version: str = "1"
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "base_template_id": self.base_template_id,
            "schema_version": self.schema_version,
            "config": self.config,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def resolve_template_chain(
    templates: list[ACPConfigTemplate],
) -> dict[str, Any]:
    """Merge a chain of templates from least-specific to most-specific.

    Templates should be ordered [system, persona, session].
    Each template's config is merged on top of the previous result.
    """
    if not templates:
        return {}
    result: dict[str, Any] = {}
    for template in templates:
        result = merge_config(result, template.config)
    return result


def resolve_for_session(
    db: Any,
    *,
    session_id: str | None = None,
    persona_id: str | None = None,
    template_name: str | None = None,
) -> dict[str, Any]:
    """Resolve the effective config for a session by loading and merging templates.

    Resolution order (least to most specific):
    1. System template (by name, or "system-default")
    2. Persona-scoped template (if persona_id provided)
    3. Session-scoped template (if session_id provided)
    """
    chain: list[ACPConfigTemplate] = []

    # 1. System template
    system_name = template_name or "system-default"
    system_templates = db.list_config_templates(scope="system", name=system_name)
    if system_templates:
        chain.append(_row_to_template(system_templates[0]))

    # 2. Persona-scoped template
    if persona_id:
        persona_templates = db.list_config_templates(scope="persona", scope_id=persona_id)
        if persona_templates:
            chain.append(_row_to_template(persona_templates[0]))

    # 3. Session-scoped template
    if session_id:
        session_templates = db.list_config_templates(scope="session", scope_id=session_id)
        if session_templates:
            chain.append(_row_to_template(session_templates[0]))

    # Resolve inheritance chains for each template
    resolved_chain: list[ACPConfigTemplate] = []
    for tpl in chain:
        resolved_chain.extend(_resolve_inheritance(db, tpl))

    return resolve_template_chain(resolved_chain)


def _resolve_inheritance(db: Any, template: ACPConfigTemplate) -> list[ACPConfigTemplate]:
    """Walk the inheritance chain (base_template_id) and return [base, ..., template]."""
    chain = [template]
    current = template
    seen = {current.id}
    while current.base_template_id:
        if current.base_template_id in seen:
            break  # Prevent infinite loops
        base_row = db.get_config_template(current.base_template_id)
        if not base_row:
            break
        base = _row_to_template(base_row)
        chain.insert(0, base)  # Base goes first (least specific)
        seen.add(base.id)
        current = base
    return chain


def _row_to_template(row: dict[str, Any]) -> ACPConfigTemplate:
    config = row.get("config_json", "{}")
    if isinstance(config, str):
        config = json.loads(config) if config else {}
    return ACPConfigTemplate(
        id=row["id"],
        name=row.get("name", ""),
        description=row.get("description", ""),
        scope=row.get("scope", "system"),
        scope_id=row.get("scope_id"),
        base_template_id=row.get("base_template_id"),
        schema_version=row.get("schema_version", "1"),
        config=config,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def seed_system_templates(db: Any) -> int:
    """Seed the DB with system templates from PERMISSION_POLICY_TEMPLATES.

    Only creates templates that don't already exist (by name).
    Returns count of templates created.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import PERMISSION_POLICY_TEMPLATES

    created = 0
    for name, template_data in PERMISSION_POLICY_TEMPLATES.items():
        existing = db.list_config_templates(scope="system", name=name)
        if existing:
            continue
        config = {k: v for k, v in template_data.items() if k != "description"}
        db.create_config_template(
            name=name,
            description=template_data.get("description", ""),
            scope="system",
            config_json=json.dumps(config),
        )
        created += 1
    return created

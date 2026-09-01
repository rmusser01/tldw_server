from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Path]:
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    return admin_system_ops_service, store_path


async def _create_incident(service: Any) -> dict[str, Any]:
    return await service.create_incident(
        title="Queue backlog",
        status="investigating",
        severity="high",
        summary="Jobs delayed",
        tags=["queue"],
        actor="alice_admin",
    )


async def test_create_incident_includes_authoritative_workflow_defaults(monkeypatch, tmp_path):
    service, _ = _configure_store(monkeypatch, tmp_path)

    incident = await service.create_incident(
        title="Queue backlog",
        status="investigating",
        severity="high",
        summary="Jobs delayed",
        tags=["queue"],
        actor="alice_admin",
    )

    assert incident["assigned_to_user_id"] is None
    assert incident["assigned_to_label"] is None
    assert incident["root_cause"] is None
    assert incident["impact"] is None
    assert incident["action_items"] == []


async def test_list_incidents_backfills_missing_authoritative_workflow_fields(monkeypatch, tmp_path):
    service, store_path = _configure_store(monkeypatch, tmp_path)

    store_path.write_text(
        """
{
  "maintenance": {
    "enabled": false,
    "message": "",
    "allowlist_user_ids": [],
    "allowlist_emails": [],
    "updated_at": null,
    "updated_by": null
  },
  "feature_flags": [],
  "incidents": [
    {
      "id": "inc_legacy",
      "title": "Legacy incident",
      "status": "open",
      "severity": "medium",
      "summary": null,
      "tags": [],
      "created_at": "2026-03-12T00:00:00+00:00",
      "updated_at": "2026-03-12T00:00:00+00:00",
      "resolved_at": null,
      "created_by": "legacy",
      "updated_by": "legacy",
      "timeline": []
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    incidents, total = service.list_incidents(
        status=None,
        severity=None,
        tag=None,
        limit=20,
        offset=0,
    )

    assert total == 1
    assert incidents[0]["version"] == 1
    assert incidents[0]["assigned_to_user_id"] is None
    assert incidents[0]["assigned_to_label"] is None
    assert incidents[0]["root_cause"] is None
    assert incidents[0]["impact"] is None
    assert incidents[0]["action_items"] == []

    updated = await service.update_incident(
        incident_id="inc_legacy",
        title=None,
        status=None,
        severity="high",
        summary=None,
        tags=None,
        update_message=None,
        actor="alice_admin",
    )

    assert updated["version"] == 2
    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert persisted["incidents"][0]["version"] == 2


async def test_update_incident_persists_assignment_and_can_clear_it(monkeypatch, tmp_path):
    service, _ = _configure_store(monkeypatch, tmp_path)
    incident = await _create_incident(service)

    assigned = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status=None,
        severity=None,
        summary=None,
        tags=None,
        assigned_to_user_id=7,
        assigned_to_label="Alice Admin",
        update_message="Assigned to Alice Admin",
        actor="alice_admin",
    )

    assert assigned["assigned_to_user_id"] == 7
    assert assigned["assigned_to_label"] == "Alice Admin"
    assert assigned["timeline"][-1]["message"] == "Assigned to Alice Admin"

    cleared = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status=None,
        severity=None,
        summary=None,
        tags=None,
        assigned_to_user_id=None,
        assigned_to_label=None,
        update_message="Assignment cleared",
        actor="alice_admin",
    )

    assert cleared["assigned_to_user_id"] is None
    assert cleared["assigned_to_label"] is None
    assert cleared["timeline"][-1]["message"] == "Assignment cleared"


async def test_update_incident_persists_postmortem_fields_and_normalizes_blank_action_items(monkeypatch, tmp_path):
    service, _ = _configure_store(monkeypatch, tmp_path)
    incident = await _create_incident(service)

    updated = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status=None,
        severity=None,
        summary=None,
        tags=None,
        root_cause="Connection pool exhaustion",
        impact="Writes failed for 4 minutes",
        action_items=[
            {"id": "ai_keep", "text": " Add pool saturation alert ", "done": False},
            {"id": "ai_blank", "text": "   ", "done": True},
        ],
        update_message="Post-mortem updated",
        actor="alice_admin",
    )

    assert updated["root_cause"] == "Connection pool exhaustion"
    assert updated["impact"] == "Writes failed for 4 minutes"
    assert updated["action_items"] == [
        {"id": "ai_keep", "text": "Add pool saturation alert", "done": False},
    ]
    assert updated["timeline"][-1]["message"] == "Post-mortem updated"


async def test_update_incident_distinguishes_omitted_fields_from_explicit_null(monkeypatch, tmp_path):
    service, _ = _configure_store(monkeypatch, tmp_path)
    incident = await _create_incident(service)

    seeded = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status=None,
        severity=None,
        summary=None,
        tags=None,
        root_cause="Connection pool exhaustion",
        impact="Writes failed for 4 minutes",
        action_items=[{"id": "ai_keep", "text": "Add pool saturation alert", "done": False}],
        update_message="Workflow seeded",
        actor="alice_admin",
    )

    preserved = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status="mitigating",
        severity=None,
        summary=None,
        tags=None,
        update_message="Status updated",
        actor="alice_admin",
    )

    assert preserved["status"] == "mitigating"
    assert preserved["root_cause"] == seeded["root_cause"]
    assert preserved["impact"] == seeded["impact"]
    assert preserved["action_items"] == seeded["action_items"]

    cleared = await service.update_incident(
        incident_id=incident["id"],
        title=None,
        status=None,
        severity=None,
        summary=None,
        tags=None,
        root_cause=None,
        impact=None,
        action_items=None,
        update_message="Workflow cleared",
        actor="alice_admin",
    )

    assert cleared["root_cause"] is None
    assert cleared["impact"] is None
    assert cleared["action_items"] == []


async def test_update_incident_does_not_append_timeline_or_persist_workflow_on_failed_validation(monkeypatch, tmp_path):
    service, _ = _configure_store(monkeypatch, tmp_path)
    incident = await _create_incident(service)

    original_timeline = list(incident["timeline"])

    with pytest.raises(ValueError, match="invalid_severity"):
        await service.update_incident(
            incident_id=incident["id"],
            title=None,
            status=None,
            severity="severe",
            summary=None,
            tags=None,
            root_cause="Connection pool exhaustion",
            impact="Writes failed for 4 minutes",
            action_items=[{"id": "ai_keep", "text": "Add pool saturation alert", "done": False}],
            update_message="Should not persist",
            actor="alice_admin",
        )

    incidents, total = service.list_incidents(
        status=None,
        severity=None,
        tag=None,
        limit=20,
        offset=0,
    )

    assert total == 1
    reloaded = incidents[0]
    assert reloaded["severity"] == "high"
    assert reloaded["root_cause"] is None
    assert reloaded["impact"] is None
    assert reloaded["action_items"] == []
    assert reloaded["timeline"] == original_timeline

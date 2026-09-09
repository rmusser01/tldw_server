"""Independent native artwork snapshots and freshly authorized attachments."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.api.v1.schemas.buddies import BuddyCreate
from tldw_Server_API.app.core.DB_Management.Buddy_DB import BuddyRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, NotFoundError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import BuddyNotFoundError
from tldw_Server_API.app.core.Persona.visual_asset_constraints import VISUAL_MIME_EXTENSIONS
from tldw_Server_API.app.core.Persona.visual_manifest_assets import remap_visual_manifest_assets
from tldw_Server_API.app.core.Persona.visual_service import MAX_VISUAL_UPLOAD_BYTES, PersonaVisualService
from tldw_Server_API.app.core.Persona.visual_starter_catalog import PersonaVisualStarterCatalogService
from tldw_Server_API.app.core.Persona.visuals import validate_visual_manifest

MAX_BUDDY_ASSETS = 256
MAX_BUDDY_ART_BYTES = 64 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BuddyService:
    """Use only the authenticated principal's selected content database."""

    def __init__(self, db: CharactersRAGDB, user_id: str) -> None:
        self.db = db
        self.user_id = user_id
        self.repository = BuddyRepository(db, user_id)

    def _persona_available(self, persona_id: str | None) -> bool:
        if not persona_id:
            return False
        profile = self.db.get_persona_profile(persona_id, user_id=self.user_id)
        return bool(profile and profile.get("is_active", True) and not profile.get("deleted"))

    def _require_persona(self, persona_id: str | None) -> None:
        if persona_id and not self._persona_available(persona_id):
            raise BuddyNotFoundError("Persona not found")

    def get(self, buddy_id: str) -> dict[str, Any]:
        profile = self.repository.get(buddy_id)
        if profile is None:
            raise BuddyNotFoundError("Buddy not found")
        return self._response(profile)

    def list_profiles(self, *, limit: int, offset: int) -> dict[str, Any]:
        return {"buddies": [self._response(row) for row in self.repository.list_profiles(limit=limit, offset=offset)]}

    def _response(self, profile: dict[str, Any]) -> dict[str, Any]:
        buddy_id = profile["id"]
        assets = []
        for asset in self.repository.assets(buddy_id):
            assets.append(
                {
                    **{
                        key: asset[key]
                        for key in ("id", "mime_type", "byte_size", "width", "height", "checksum_sha256")
                    },
                    "content_url": f"/api/v1/buddies/{buddy_id}/assets/{asset['id']}/content",
                }
            )
        return {
            **{
                key: profile[key]
                for key in ("id", "name", "optional_persona_id", "display_mode", "version", "manifest", "attribution")
            },
            "optional_persona_available": self._persona_available(profile["optional_persona_id"]),
            "assets": assets,
        }

    def _asset_path(self, buddy_id: str, asset_id: str, mime_type: str) -> Path:
        if not all(re.fullmatch(r"[a-f0-9]{32}", value) for value in (buddy_id, asset_id)):
            raise ValueError("Invalid artwork identity")
        user_root = DatabasePaths.get_user_base_directory(self.user_id).resolve()
        path = (user_root / "buddies" / buddy_id / f"{asset_id}{VISUAL_MIME_EXTENSIONS[mime_type]}").resolve()
        if not path.is_relative_to(user_root):
            raise ValueError("Invalid artwork storage path")
        return path

    def read_asset(self, buddy_id: str, asset_id: str) -> tuple[bytes, str]:
        if self.repository.get(buddy_id) is None:
            raise BuddyNotFoundError("Buddy not found")
        asset = next((row for row in self.repository.assets(buddy_id) if row["id"] == asset_id), None)
        if asset is None:
            raise BuddyNotFoundError("Artwork not found")
        path = self._asset_path(buddy_id, asset_id, asset["mime_type"])
        try:
            with path.open("rb") as handle:
                content = handle.read(MAX_VISUAL_UPLOAD_BYTES + 1)
        except OSError as exc:
            raise BuddyNotFoundError("Artwork unavailable") from exc
        if len(content) != asset["byte_size"] or hashlib.sha256(content).hexdigest() != asset["checksum_sha256"]:
            raise BuddyNotFoundError("Artwork unavailable")
        return content, asset["mime_type"]

    def create(self, request: BuddyCreate) -> dict[str, Any]:
        self._require_persona(request.optional_persona_id)
        visual_service = PersonaVisualService(self.db)
        source_assets: list[tuple[str, bytes, str]] = []
        if request.source.kind == "starter":
            catalog = PersonaVisualStarterCatalogService(self.db)
            starter = catalog.get_starter_pack(request.source.starter_id)
            manifest = starter["manifest"]
            attribution = {
                "source_kind": "starter",
                "starter_id": request.source.starter_id,
                "title": starter["title"],
                "license_label": starter["license_label"],
            }
            for asset in starter["assets"]:
                content, mime_type = catalog.get_starter_asset_content(request.source.starter_id, asset["asset_key"])
                source_assets.append((asset["asset_key"], content, mime_type))
        else:
            source = request.source
            self._require_persona(source.persona_id)
            pack = self.db.get_persona_visual_pack(
                pack_id=source.pack_id, persona_id=source.persona_id, user_id=self.user_id
            )
            if pack is None:
                raise BuddyNotFoundError("Source artwork not found")
            manifest = pack["manifest"]
            attribution = {
                "source_kind": "persona_pack",
                "persona_id": source.persona_id,
                "pack_id": source.pack_id,
                "pack_version": pack["version"],
                "title": pack["title"],
                "provenance": pack["provenance"],
            }
            # Preserve existing authored metadata without interpreting it as policy.
            for key in ("attribution", "license_label"):
                if key in pack:
                    attribution[key] = pack[key]
            assets = self.db.list_persona_visual_assets(
                pack_id=source.pack_id, persona_id=source.persona_id, user_id=self.user_id
            )
            if len(assets) > MAX_BUDDY_ASSETS or sum(asset["byte_size"] for asset in assets) > MAX_BUDDY_ART_BYTES:
                raise ValueError("Artwork exceeds Buddy snapshot limits")
            for asset in assets:
                path = visual_service._asset_storage_path(user_id=self.user_id, storage_key=asset["storage_key"])
                try:
                    with path.open("rb") as handle:
                        content = handle.read(MAX_VISUAL_UPLOAD_BYTES + 1)
                except OSError as exc:
                    raise BuddyNotFoundError("Source artwork unavailable") from exc
                if (
                    len(content) != asset["byte_size"]
                    or hashlib.sha256(content).hexdigest() != asset["checksum_sha256"]
                ):
                    raise ValueError("Source artwork checksum mismatch")
                source_assets.append((asset["id"], content, asset["mime_type"]))
        if (
            not source_assets
            or len(source_assets) > MAX_BUDDY_ASSETS
            or sum(len(item[1]) for item in source_assets) > MAX_BUDDY_ART_BYTES
        ):
            raise ValueError("Artwork exceeds Buddy snapshot limits")

        buddy_id = uuid.uuid4().hex
        assets = []
        mapped_ids = {}
        for source_id, content, mime_type in source_assets:
            mime_type = visual_service._normalize_mime_type(mime_type)
            if len(content) > MAX_VISUAL_UPLOAD_BYTES:
                raise ValueError("Artwork asset exceeds size limit")
            width, height = visual_service._validate_image_bytes(content, mime_type=mime_type)
            asset_id = uuid.uuid4().hex
            mapped_ids[source_id] = asset_id
            assets.append(
                {
                    "id": asset_id,
                    "mime_type": mime_type,
                    "byte_size": len(content),
                    "width": width,
                    "height": height,
                    "checksum_sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        validation = validate_visual_manifest(
            remap_visual_manifest_assets(manifest, mapped_ids),
            available_asset_ids=set(mapped_ids.values()),
            available_asset_dimensions={asset["id"]: (asset["width"], asset["height"]) for asset in assets},
            require_activatable=True,
        )
        directory: Path | None = None
        try:
            for asset, (_, content, _) in zip(assets, source_assets, strict=True):
                path = self._asset_path(buddy_id, asset["id"], asset["mime_type"])
                directory = path.parent
                directory.mkdir(parents=True, mode=0o700, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(content)
            self.repository.create(
                {
                    "id": buddy_id,
                    "name": request.name,
                    "optional_persona_id": request.optional_persona_id,
                    "display_mode": request.display_mode,
                    "manifest": validation.manifest,
                    "attribution": attribution,
                    "created_at": _now(),
                },
                assets,
            )
        except Exception:
            if directory is not None:
                shutil.rmtree(directory)
            raise
        return self.get(buddy_id)

    def update(self, buddy_id: str, *, expected_version: int, changes: dict[str, Any]) -> dict[str, Any]:
        self.get(buddy_id)
        if "optional_persona_id" in changes:
            self._require_persona(changes["optional_persona_id"])
        self.repository.update(buddy_id, expected_version=expected_version, changes=changes, timestamp=_now())
        return self.get(buddy_id)

    def delete(self, buddy_id: str, *, expected_version: int) -> None:
        self.repository.update(buddy_id, expected_version=expected_version, changes={"deleted": 1}, timestamp=_now())

    def resolve_target(self, scope_type: str, scope_id: str) -> dict[str, Any]:
        """Resolve existing private targets; never open a shared owner's DB."""
        if scope_type == "workspace":
            workspace = self.db.get_workspace(scope_id)
            if workspace is None or workspace.get("deleted") or str(workspace.get("client_id")) != self.user_id:
                raise BuddyNotFoundError("Target unavailable")
            return {"title": workspace.get("name") or "Workspace", "workspace_id": scope_id}
        conversation = self._conversation(scope_id)
        return {
            "title": conversation.get("title") or "Conversation",
            "workspace_id": conversation.get("workspace_id") if conversation.get("scope_type") == "workspace" else None,
        }

    def _conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.db.get_conversation_by_id(conversation_id)
        if conversation is None or conversation.get("deleted") or str(conversation.get("client_id")) != self.user_id:
            raise BuddyNotFoundError("Target unavailable")
        workspace_id = conversation.get("workspace_id") if conversation.get("scope_type") == "workspace" else None
        if workspace_id:
            self.resolve_target("workspace", workspace_id)
        elif conversation.get("scope_type") == "workspace":
            raise BuddyNotFoundError("Target unavailable")
        return conversation

    def conversation_summary(self, conversation_id: str) -> dict[str, Any]:
        """Resolve current scope and Persona identity without changing behavior."""
        conversation = self._conversation(conversation_id)
        persona = None
        if conversation.get("assistant_kind") == "persona":
            persona = self.db.get_persona_profile(conversation.get("assistant_id") or "", user_id=self.user_id)
        return {
            "id": conversation["id"],
            "title": conversation.get("title") or "Conversation",
            "created_at": conversation.get("created_at"),
            "scope_type": conversation.get("scope_type") or "global",
            "workspace_id": conversation.get("workspace_id") if conversation.get("scope_type") == "workspace" else None,
            "version": conversation["version"],
            "assistant_kind": conversation.get("assistant_kind"),
            "assistant_id": conversation.get("assistant_id"),
            "assistant_name": persona.get("name") if persona else None,
        }

    def resolve_reply_completion(self, conversation_id: str) -> dict[str, Any]:
        """Resolve saved completion settings from the principal's exact conversation.

        Args:
            conversation_id: Existing conversation owned by this service's user.

        Returns:
            Provider and model identifiers, trimmed or None, plus effective
            sampling settings. Each identifier prefers the roleplay resume
            completion over raw conversation settings; no server defaults apply.

        Raises:
            BuddyNotFoundError: If the conversation is missing, deleted or foreign.
            CharactersRAGDBError: If reading the conversation state fails.
        """
        try:
            resume = self.db.get_roleplay_resume_state(conversation_id, owner_client_id=self.user_id)
        except NotFoundError as exc:
            raise BuddyNotFoundError("Target unavailable") from exc
        settings = resume.get("settings") or {}
        effective = resume.get("effective_completion") or {}
        completion = {"sampling": effective.get("sampling") or {}}
        for key in ("provider", "model"):
            value = effective.get(key) or settings.get(key)
            completion[key] = value.strip() if isinstance(value, str) and value.strip() else None
        return completion

    def conversation_reply_settings(self, client_slot: str, conversation_id: str) -> dict[str, str | None]:
        """Read reply identifiers for a currently attached and authorized target.

        Args:
            client_slot: Principal-local preference slot containing the attachment.
            conversation_id: Exact conversation to check against that attachment.

        Returns:
            Only provider and model, each a trimmed identifier or None, using the
            same effective-completion fallback as Buddy turn acceptance.

        Raises:
            BuddyNotFoundError: If the Buddy, attachment or target is unavailable,
                or the conversation is outside the attached conversation/workspace.
            CharactersRAGDBError: If reading attachment or conversation state fails.
        """
        with self.db.transaction():
            attachment = self.attachment(client_slot)["attachment"]
            if attachment is None:
                raise BuddyNotFoundError("Attach a Buddy to an available target first")
            resolved = self.resolve_target("conversation", conversation_id)
            if (attachment["scope_type"] == "conversation" and attachment["scope_id"] != conversation_id) or (
                attachment["scope_type"] == "workspace" and attachment["scope_id"] != resolved["workspace_id"]
            ):
                raise BuddyNotFoundError("Conversation is outside the attached target")
            completion = self.resolve_reply_completion(conversation_id)
            return {key: completion[key] for key in ("provider", "model")}

    def attachment(self, client_slot: str) -> dict[str, Any]:
        row = self.repository.attachment(client_slot)
        response = {
            "client_slot": client_slot,
            "version": row["version"],
            "attachment": None,
            "target": None,
            "unavailable_reason": None,
        }
        if row["buddy_id"] is None:
            return response
        if self.repository.get(row["buddy_id"]) is None:
            response["unavailable_reason"] = "buddy_unavailable"
            return response
        try:
            response["target"] = self.resolve_target(row["scope_type"], row["scope_id"])
        except BuddyNotFoundError:
            response["unavailable_reason"] = "target_unavailable"
            return response
        response["attachment"] = {key: row[key] for key in ("buddy_id", "scope_type", "scope_id")}
        return response

    def set_attachment(
        self, client_slot: str, *, expected_version: int, attachment: dict[str, Any] | None
    ) -> dict[str, Any]:
        if attachment is not None:
            self.get(attachment["buddy_id"])
            self.resolve_target(attachment["scope_type"], attachment["scope_id"])
        self.repository.set_attachment(client_slot, expected_version=expected_version, attachment=attachment)
        return self.attachment(client_slot)

    def conversations(self, client_slot: str, *, limit: int, offset: int) -> dict[str, Any]:
        attachment = self.attachment(client_slot)["attachment"]
        rows = []
        if attachment is not None:
            if attachment["scope_type"] == "conversation":
                if offset == 0:
                    rows = [self.db.get_conversation_by_id(attachment["scope_id"])]
            else:
                rows = self.db.get_conversations_for_user(
                    self.user_id,
                    scope_type="workspace",
                    workspace_id=attachment["scope_id"],
                    limit=limit,
                    offset=offset,
                )
        summaries = []
        for row in rows:
            if row is None:
                continue
            try:
                target = self.conversation_summary(row["id"])
            except BuddyNotFoundError:
                continue
            if attachment["scope_type"] == "workspace" and target["workspace_id"] != attachment["scope_id"]:
                continue
            summaries.append(target)
        return {"conversations": summaries, "limit": limit, "offset": offset}

    def activity(self, client_slot: str, *, limit: int, offset: int) -> dict[str, Any]:
        """Project persisted results; execution status belongs to its real owner."""
        summaries = self.conversations(client_slot, limit=limit, offset=offset)["conversations"]
        rows = self.repository.latest_results(client_slot, [summary["id"] for summary in summaries])
        results = {row["conversation_id"]: row for row in rows}
        items = []
        for summary in summaries:
            row = results.get(summary["id"])
            if row is None:
                continue
            items.append(
                {
                    "conversation_id": summary["id"],
                    "title": summary["title"],
                    "workspace_id": summary["workspace_id"],
                    "result": {"id": row["id"], "created_at": str(row["created_at"]), "content": row["content"]},
                    "acknowledged": bool(row["acknowledged"]),
                }
            )
        return {"items": items, "limit": limit, "offset": offset}

    def acknowledge(self, client_slot: str, *, conversation_id: str, message_id: str) -> dict[str, bool]:
        self.repository.acknowledge(
            client_slot, conversation_id=conversation_id, message_id=message_id, timestamp=_now()
        )
        return {"acknowledged": True}

"""Service helpers for Notes Studio derive/fetch/regenerate/diagram flows."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_note_upsert,
    capture_plan,
    stable_note_id,
)
from tldw_Server_API.app.core.Notes.studio_markdown import (
    NOTE_STUDIO_RENDER_VERSION,
    build_derived_studio_payload,
    render_studio_markdown,
    stable_content_hash,
    studio_payload_from_markdown,
)

NoteStudioAdapter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]

_LOCAL_STUDIO_PROVIDER = "tldw"
_LOCAL_DERIVE_MODEL = "notes-studio-deterministic-v1"
_LOCAL_DIAGRAM_MODEL = "diagram-deterministic-v1"


async def _run_notes_studio_generate_adapter(request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    from tldw_Server_API.app.core.Workflows.adapters.content import run_notes_studio_generate_adapter

    return await run_notes_studio_generate_adapter(request, context)


async def _run_diagram_generate_adapter(request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    from tldw_Server_API.app.core.Workflows.adapters.content import run_diagram_generate_adapter

    return await run_diagram_generate_adapter(request, context)


@dataclass(slots=True)
class NotesStudioService:
    """Focused orchestration for Notes Studio state and sidecar persistence."""

    db: CharactersRAGDB
    user_id: int | str
    generation_adapter: NoteStudioAdapter = _run_notes_studio_generate_adapter
    diagram_adapter: NoteStudioAdapter = _run_diagram_generate_adapter

    @staticmethod
    def _derive_execution_identity(
        generated: dict[str, Any],
        *,
        provider: str | None,
        model: str | None,
    ) -> tuple[str, str]:
        if generated.get("source") == "llm":
            if not provider or not model:
                raise InputError("Notes Studio LLM execution identity is incomplete.")  # noqa: TRY003
            return str(provider), str(model)
        return _LOCAL_STUDIO_PROVIDER, _LOCAL_DERIVE_MODEL

    @staticmethod
    def _diagram_execution_identity(
        *, provider: str | None, model: str | None
    ) -> tuple[str, str]:
        if provider and model:
            return str(provider), str(model)
        return _LOCAL_STUDIO_PROVIDER, _LOCAL_DIAGRAM_MODEL

    async def derive_from_excerpt(
        self,
        *,
        source_note_id: str,
        excerpt_text: str,
        template_type: str,
        handwriting_mode: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        coordinator = active_coordinator(self.db, user_id=self.user_id)
        request_fields = {
            "source_note_id": source_note_id,
            "excerpt_text": str(excerpt_text or "").strip(),
            "template_type": template_type,
            "handwriting_mode": handwriting_mode,
            "provider": provider,
            "model": model,
        }
        request_fingerprint = (
            coordinator.request_fingerprint("notes-studio.derive", request_fields)
            if coordinator is not None
            else None
        )
        capture_key = request_fingerprint
        if coordinator is not None and capture_key is not None:
            replay = coordinator.replay_request_plan(
                source="notes-studio",
                idempotency_key=capture_key,
                request_fingerprint=request_fingerprint,
                result_domain="notes.note",
            )
            if replay is not None:
                replayed_note = capture_plan(
                    coordinator,
                    replay,
                    source="notes-studio",
                    key=capture_key,
                )
                replayed_note_id = str(replayed_note["id"])
                studio_document = self.db.get_note_studio_document(replayed_note_id)
                if studio_document is not None:
                    return self._build_state(
                        note_id=replayed_note_id,
                        studio_document=studio_document,
                    )
                replayed_markdown = str(replayed_note.get("content") or "")
                replayed_payload = studio_payload_from_markdown(
                    replayed_markdown,
                    template_type=template_type,
                    handwriting_mode=handwriting_mode,
                    render_version=NOTE_STUDIO_RENDER_VERSION,
                    fallback_title=str(replayed_note.get("title") or "Untitled Study Notes"),
                    source_note_id=source_note_id,
                    preserve_existing_sections_when_empty=False,
                )
                repaired_studio_document = self._ensure_studio_document(
                    note_id=replayed_note_id,
                    payload_json=replayed_payload,
                    template_type=template_type,
                    handwriting_mode=handwriting_mode,
                    source_note_id=source_note_id,
                    excerpt_snapshot=request_fields["excerpt_text"],
                    excerpt_hash=stable_content_hash(str(request_fields["excerpt_text"])),
                    diagram_manifest_json=None,
                    companion_content_hash=stable_content_hash(replayed_markdown),
                    render_version=NOTE_STUDIO_RENDER_VERSION,
                    provenance_kind="derive",
                    provenance_provider=_LOCAL_STUDIO_PROVIDER,
                    provenance_model=_LOCAL_DERIVE_MODEL,
                )
                return self._build_state(
                    note_id=replayed_note_id,
                    studio_document=repaired_studio_document,
                )

        source_note = self._require_note(source_note_id)
        excerpt_snapshot = self._validate_excerpt(source_note=source_note, excerpt_text=excerpt_text)
        derived_title = self._build_derived_title(source_note.get("title"))

        generated = await self.generation_adapter(
            {
                "source_note_id": str(source_note["id"]),
                "source_title": source_note.get("title"),
                "derived_title": derived_title,
                "excerpt_text": excerpt_snapshot,
                "template_type": template_type,
                "handwriting_mode": handwriting_mode,
                "provider": provider,
                "model": model,
            },
            {"source_note": source_note},
        )
        payload = generated.get("payload")
        if not isinstance(payload, dict) or not payload:
            raise InputError("Notes Studio generation failed to return a canonical payload.")  # noqa: TRY003
        try:
            payload = build_derived_studio_payload(
                payload,
                template_type=template_type,
                handwriting_mode=handwriting_mode,
                render_version=NOTE_STUDIO_RENDER_VERSION,
                fallback_title=derived_title,
                source_note_id=str(source_note["id"]),
            )
        except (TypeError, ValueError) as exc:
            raise InputError(
                "Notes Studio generation did not return canonical sections."
            ) from exc
        executed_provider, executed_model = self._derive_execution_identity(
            generated,
            provider=provider,
            model=model,
        )

        markdown = render_studio_markdown(payload)
        note_title = str(payload.get("meta", {}).get("title") or derived_title).strip() or derived_title
        if coordinator is not None:
            note_id = stable_note_id("notes-studio", capture_key)
            capture_note_upsert(
                coordinator,
                note_id=note_id,
                title=note_title,
                content=markdown,
                source="notes-studio",
                key=capture_key,
                request_fingerprint=request_fingerprint,
            )
            studio_document = self._ensure_studio_document(
                note_id=str(note_id),
                payload_json=payload,
                template_type=template_type,
                handwriting_mode=handwriting_mode,
                source_note_id=str(source_note["id"]),
                excerpt_snapshot=excerpt_snapshot,
                excerpt_hash=stable_content_hash(excerpt_snapshot),
                diagram_manifest_json=None,
                companion_content_hash=stable_content_hash(markdown),
                render_version=NOTE_STUDIO_RENDER_VERSION,
                provenance_kind="derive",
                provenance_provider=executed_provider,
                provenance_model=executed_model,
            )
        else:
            with self.db.transaction() as conn:
                note_id = self.db.add_note(title=note_title, content=markdown, conn=conn)
                if note_id is None:
                    raise InputError("Failed to create derived note.")  # noqa: TRY003

                studio_document = self.db.create_note_studio_document(
                    note_id=str(note_id),
                    payload_json=payload,
                    template_type=template_type,
                    handwriting_mode=handwriting_mode,
                    source_note_id=str(source_note["id"]),
                    excerpt_snapshot=excerpt_snapshot,
                    excerpt_hash=stable_content_hash(excerpt_snapshot),
                    diagram_manifest_json=None,
                    companion_content_hash=stable_content_hash(markdown),
                    render_version=NOTE_STUDIO_RENDER_VERSION,
                    provenance_kind="derive",
                    provenance_provider=executed_provider,
                    provenance_model=executed_model,
                    conn=conn,
                )
        return self._build_state(note_id=str(note_id), studio_document=studio_document)

    async def get_note_studio_state(self, *, note_id: str) -> dict[str, Any]:
        studio_document = self._require_studio_document(note_id)
        return self._build_state(note_id=note_id, studio_document=studio_document)

    async def regenerate_note_markdown(
        self,
        *,
        note_id: str,
        expected_version: int,
        current_markdown: str | None = None,
    ) -> dict[str, Any]:
        coordinator = active_coordinator(self.db, user_id=self.user_id)
        request_fields = {
            "note_id": note_id,
            "expected_version": expected_version,
            "current_markdown": current_markdown,
        }
        request_fingerprint = (
            coordinator.request_fingerprint("notes-studio.regenerate", request_fields)
            if coordinator is not None
            else None
        )
        capture_key = request_fingerprint
        if coordinator is not None and capture_key is not None:
            replay = coordinator.replay_request_plan(
                source="notes-studio",
                idempotency_key=capture_key,
                request_fingerprint=request_fingerprint,
                result_domain="notes.note",
            )
            if replay is not None:
                replayed_note = capture_plan(
                    coordinator,
                    replay,
                    source="notes-studio",
                    key=capture_key,
                )
                replayed_note_id = str(replayed_note["id"])
                studio_document = self._require_studio_document(replayed_note_id)
                if studio_document.get("companion_content_hash") == stable_content_hash(
                    str(replayed_note.get("content") or "")
                ):
                    return self._build_state(
                        note_id=replayed_note_id,
                        studio_document=studio_document,
                    )
                payload, markdown, _rebuilt_title = self._rebuild_studio_markdown(
                    note=replayed_note,
                    studio_document=studio_document,
                    current_markdown=current_markdown,
                )
                repaired_studio_document = self.db.upsert_note_studio_document(
                    note_id=replayed_note_id,
                    payload_json=payload,
                    template_type=studio_document["template_type"],
                    handwriting_mode=studio_document["handwriting_mode"],
                    source_note_id=studio_document.get("source_note_id"),
                    excerpt_snapshot=studio_document.get("excerpt_snapshot"),
                    excerpt_hash=studio_document.get("excerpt_hash"),
                    diagram_manifest_json=studio_document.get("diagram_manifest_json"),
                    companion_content_hash=stable_content_hash(markdown),
                    render_version=int(
                        studio_document.get("render_version") or NOTE_STUDIO_RENDER_VERSION
                    ),
                    provenance_kind="regenerate",
                    provenance_provider=None,
                    provenance_model=None,
                )
                return self._build_state(
                    note_id=replayed_note_id,
                    studio_document=repaired_studio_document,
                )

        note = self._require_note(note_id)
        current_version = int(note.get("version") or 0)
        if current_version != expected_version:
            raise ConflictError(
                f"Note ID {note_id} regenerate failed: version mismatch "
                f"(db has {current_version}, client expected {expected_version}).",
                entity="notes",
                entity_id=note_id,
            )  # noqa: TRY003
        studio_document = self._require_studio_document(note_id)
        payload, markdown, rebuilt_title = self._rebuild_studio_markdown(
            note=note,
            studio_document=studio_document,
            current_markdown=current_markdown,
        )

        if coordinator is not None:
            capture_note_upsert(
                coordinator,
                note_id=note_id,
                title=rebuilt_title,
                content=markdown,
                conversation_id=note.get("conversation_id"),
                message_id=note.get("message_id"),
                expected_version=expected_version,
                source="notes-studio",
                key=capture_key,
                request_fingerprint=request_fingerprint,
            )
            updated_studio_document = self.db.upsert_note_studio_document(
                note_id=note_id,
                payload_json=payload,
                template_type=studio_document["template_type"],
                handwriting_mode=studio_document["handwriting_mode"],
                source_note_id=studio_document.get("source_note_id"),
                excerpt_snapshot=studio_document.get("excerpt_snapshot"),
                excerpt_hash=studio_document.get("excerpt_hash"),
                diagram_manifest_json=studio_document.get("diagram_manifest_json"),
                companion_content_hash=stable_content_hash(markdown),
                render_version=int(studio_document.get("render_version") or NOTE_STUDIO_RENDER_VERSION),
                provenance_kind="regenerate",
                provenance_provider=None,
                provenance_model=None,
            )
        else:
            with self.db.transaction() as conn:
                self.db.update_note(
                    note_id=note_id,
                    update_data={"title": rebuilt_title, "content": markdown},
                    expected_version=expected_version,
                    conn=conn,
                )
                updated_studio_document = self.db.upsert_note_studio_document(
                    note_id=note_id,
                    payload_json=payload,
                    template_type=studio_document["template_type"],
                    handwriting_mode=studio_document["handwriting_mode"],
                    source_note_id=studio_document.get("source_note_id"),
                    excerpt_snapshot=studio_document.get("excerpt_snapshot"),
                    excerpt_hash=studio_document.get("excerpt_hash"),
                    diagram_manifest_json=studio_document.get("diagram_manifest_json"),
                    companion_content_hash=stable_content_hash(markdown),
                    render_version=int(studio_document.get("render_version") or NOTE_STUDIO_RENDER_VERSION),
                    provenance_kind="regenerate",
                    provenance_provider=None,
                    provenance_model=None,
                    conn=conn,
                )
        return self._build_state(note_id=note_id, studio_document=updated_studio_document)

    @staticmethod
    def _rebuild_studio_markdown(
        *,
        note: dict[str, Any],
        studio_document: dict[str, Any],
        current_markdown: str | None,
    ) -> tuple[dict[str, Any], str, str]:
        has_current_markdown_override = isinstance(current_markdown, str)
        markdown_source = current_markdown if has_current_markdown_override else str(note.get("content") or "")
        existing_payload = studio_document.get("payload_json")
        if not isinstance(existing_payload, dict):
            raise InputError("Studio document payload is invalid.")  # noqa: TRY003

        payload = studio_payload_from_markdown(
            markdown_source,
            template_type=str(studio_document["template_type"]),
            handwriting_mode=str(studio_document["handwriting_mode"]),
            render_version=int(studio_document.get("render_version") or NOTE_STUDIO_RENDER_VERSION),
            fallback_title=str(note.get("title") or "Untitled Study Notes"),
            source_note_id=str(studio_document.get("source_note_id") or "").strip() or None,
            existing_payload=existing_payload,
            preserve_existing_sections_when_empty=not has_current_markdown_override,
        )
        markdown = render_studio_markdown(payload)
        rebuilt_title = str(
            payload.get("meta", {}).get("title") or note.get("title") or "Untitled Study Notes"
        ).strip() or "Untitled Study Notes"
        return payload, markdown, rebuilt_title

    def _ensure_studio_document(self, **fields: Any) -> dict[str, Any]:
        """Create one sidecar, or accept an identical prior write on retry."""

        note_id = str(fields["note_id"])
        existing = self.db.get_note_studio_document(note_id)
        if existing is None:
            return self.db.create_note_studio_document(**fields)
        comparable = {
            key: existing.get(key)
            for key in fields
            if key not in {"note_id", "conn"}
        }
        expected = {
            key: value
            for key, value in fields.items()
            if key not in {"note_id", "conn"}
        }
        if comparable != expected:
            raise ConflictError(
                f"Note Studio document for note ID '{note_id}' conflicts with the captured retry.",
                entity="note_studio",
                entity_id=note_id,
            )
        return existing

    async def update_diagram_manifest(
        self,
        *,
        note_id: str,
        diagram_type: str,
        source_section_ids: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        studio_document = self._require_studio_document(note_id)
        payload = studio_document.get("payload_json")
        if not isinstance(payload, dict):
            raise InputError("Studio document payload is invalid.")  # noqa: TRY003

        requested_section_ids = [str(section_id).strip() for section_id in (source_section_ids or []) if str(section_id).strip()]
        selected_sections = self._select_sections(payload=payload, requested_section_ids=requested_section_ids)
        diagram_context = self._build_diagram_context(selected_sections)
        expected_companion_content_hash = studio_document.get("companion_content_hash")
        expected_render_version = int(studio_document.get("render_version") or NOTE_STUDIO_RENDER_VERSION)
        expected_last_modified = studio_document.get("last_modified")

        diagram_result = await self.diagram_adapter(
            {
                "content": diagram_context["text"],
                "diagram_type": diagram_type,
                "format": "mermaid",
                "provider": provider,
                "model": model,
            },
            {"note_id": note_id, "sections": selected_sections},
        )

        diagram_code = str(diagram_result.get("diagram") or "").strip()
        if not diagram_code or diagram_result.get("error"):
            raise InputError("Notes Studio diagram generation returned no accepted diagram.")  # noqa: TRY003
        executed_provider, executed_model = self._diagram_execution_identity(
            provider=provider,
            model=model,
        )
        render_hash = stable_content_hash(f"{diagram_type}\n{diagram_context['text']}\n{diagram_code}")
        manifest = {
            "diagram_type": diagram_type,
            "source_section_ids": [section["id"] for section in selected_sections],
            "source_graph": diagram_context["source_graph"],
            "canonical_source": diagram_context["source_graph"],
            "diagram": diagram_code,
            "cached_svg": self._build_svg_preview(diagram_type=diagram_type, text=diagram_context["text"]),
            "render_hash": render_hash,
            "generation_status": "ready",
            "status": "ready",
            "format": str(diagram_result.get("format") or "mermaid"),
        }

        updated_studio_document = self.db.update_note_studio_diagram_manifest(
            note_id=note_id,
            diagram_manifest_json=manifest,
            expected_companion_content_hash=expected_companion_content_hash,
            expected_render_version=expected_render_version,
            expected_last_modified=expected_last_modified,
            provenance_kind="diagram",
            provenance_provider=executed_provider,
            provenance_model=executed_model,
        )
        return self._build_state(note_id=note_id, studio_document=updated_studio_document)

    def _require_note(self, note_id: str) -> dict[str, Any]:
        note = self.db.get_note_by_id(note_id=note_id, include_studio_summary=True)
        if not note:
            raise ConflictError(f"Note ID '{note_id}' not found.", entity="notes", entity_id=note_id)  # noqa: TRY003
        return note

    def _require_studio_document(self, note_id: str) -> dict[str, Any]:
        studio_document = self.db.get_note_studio_document(note_id)
        if not studio_document:
            raise ConflictError(
                f"Note Studio document for note ID '{note_id}' not found.",
                entity="note_studio",
                entity_id=note_id,
            )  # noqa: TRY003
        return studio_document

    def _validate_excerpt(self, *, source_note: dict[str, Any], excerpt_text: str) -> str:
        excerpt_snapshot = str(excerpt_text or "").strip()
        if not excerpt_snapshot:
            raise InputError("excerpt_text cannot be empty.")  # noqa: TRY003

        source_content = str(source_note.get("content") or "")
        if excerpt_snapshot not in source_content:
            raise InputError("excerpt_text must match content from the source note.")  # noqa: TRY003
        return excerpt_snapshot

    @staticmethod
    def _build_derived_title(source_title: Any) -> str:
        title = str(source_title or "").strip() or "Untitled"
        return f"{title} Study Notes"

    def _build_state(self, *, note_id: str, studio_document: dict[str, Any]) -> dict[str, Any]:
        note = self._require_note(note_id)
        stale_reason = self._get_stale_reason(note=note, studio_document=studio_document)
        compatibility_document = dict(studio_document)
        payload = studio_document.get("payload_json")
        if isinstance(payload, dict):
            compatibility_payload = dict(payload)
            compatibility_payload["meta"] = {
                "title": str(note.get("title") or ""),
                "source_note_id": studio_document.get("source_note_id"),
            }
            compatibility_payload["layout"] = {
                "template_type": studio_document["template_type"],
                "handwriting_mode": studio_document["handwriting_mode"],
                "render_version": int(studio_document["render_version"]),
            }
            compatibility_document["payload_json"] = compatibility_payload
        manifest = studio_document.get("diagram_manifest_json")
        if isinstance(manifest, dict):
            compatibility_manifest = dict(manifest)
            compatibility_manifest["canonical_source"] = manifest.get("source_graph")
            compatibility_manifest["generation_status"] = manifest.get("status")
            source_text = "\n".join(
                str(item.get("content") or "")
                for item in (manifest.get("source_graph") or [])
                if isinstance(item, dict)
            )
            compatibility_manifest["cached_svg"] = self._build_svg_preview(
                diagram_type=str(manifest.get("diagram_type") or "flowchart"),
                text=source_text,
            )
            compatibility_document["diagram_manifest_json"] = compatibility_manifest
        return {
            "note": note,
            "studio_document": compatibility_document,
            "is_stale": stale_reason is not None,
            "stale_reason": stale_reason,
        }

    @staticmethod
    def _get_stale_reason(*, note: dict[str, Any], studio_document: dict[str, Any]) -> str | None:
        current_hash = stable_content_hash(str(note.get("content") or ""))
        stored_hash = str(studio_document.get("companion_content_hash") or "").strip()
        if not stored_hash:
            return "missing_companion_content_hash"
        if current_hash != stored_hash:
            return "companion_content_hash_mismatch"
        return None

    @staticmethod
    def _select_sections(*, payload: dict[str, Any], requested_section_ids: list[str]) -> list[dict[str, Any]]:
        sections = payload.get("sections")
        if not isinstance(sections, list):
            return []

        normalized_sections = [dict(section) for section in sections if isinstance(section, dict)]
        if not requested_section_ids:
            return normalized_sections

        requested = set(requested_section_ids)
        available = {str(section.get("id") or "") for section in normalized_sections}
        missing = [section_id for section_id in requested_section_ids if section_id not in available]
        if missing:
            raise InputError(f"Unknown Studio section ID(s): {', '.join(missing)}")  # noqa: TRY003
        selected = [section for section in normalized_sections if str(section.get("id") or "") in requested]
        return selected

    @staticmethod
    def _build_diagram_context(selected_sections: list[dict[str, Any]]) -> dict[str, Any]:
        canonical_sections: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for section in selected_sections:
            section_id = str(section.get("id") or "").strip()
            title = str(section.get("title") or "").strip()
            kind = str(section.get("kind") or "").strip()
            if kind == "cue":
                content = "\n".join(str(item).strip() for item in section.get("items") or [] if str(item).strip())
            else:
                content = str(section.get("content") or "").strip()
            canonical_sections.append(
                {
                    "id": section_id,
                    "title": title,
                    "kind": kind,
                    "content": content,
                }
            )
            if title:
                text_parts.append(title)
            if content:
                text_parts.append(content)

        combined_text = "\n".join(part for part in text_parts if part).strip()
        return {
            "source_graph": canonical_sections,
            "text": combined_text or "Notes Studio diagram",
        }

    @staticmethod
    def _build_svg_preview(*, diagram_type: str, text: str) -> str:
        preview_text = (text or diagram_type).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        preview_text = preview_text[:180]
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="160" viewBox="0 0 640 160">'
            '<rect width="640" height="160" fill="#f8fafc" stroke="#cbd5e1" rx="12" ry="12"/>'
            f'<text x="24" y="44" font-size="18" font-family="Arial, sans-serif" fill="#0f172a">{diagram_type.title()} Diagram</text>'
            f'<text x="24" y="84" font-size="14" font-family="Arial, sans-serif" fill="#334155">{preview_text}</text>'
            "</svg>"
        )

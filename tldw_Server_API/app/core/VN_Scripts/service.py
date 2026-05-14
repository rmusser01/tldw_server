"""Service layer for VN script authoring and publishing."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    LOCAL_DEFAULT_POLICY_DEFINITION,
    STORY_DEFAULT_GENERATION_DEFINITION,
    STRICT_HOSTED_POLICY_DEFINITION,
    VNProfileSnapshotRepository,
)
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Platform.idempotency import canonical_payload_hash
from tldw_Server_API.app.core.VN_Policy.service import evaluate_character_safety_definition
from tldw_Server_API.app.core.VN_Scripts.authoring_catalog import list_authoring_catalog
from tldw_Server_API.app.core.VN_Scripts.authoring_graph import (
    MAX_SUPPLIED_DRAFT_BYTES,
    build_script_authoring_graph,
)
from tldw_Server_API.app.core.VN_Scripts.playtest import build_script_playtest
from tldw_Server_API.app.core.VN_Scripts.snippet_patcher import SnippetPatchResult, apply_snippet_patch
from tldw_Server_API.app.core.VN_Scripts.templates import instantiate_template, list_template_catalog
from tldw_Server_API.app.core.VN_Scripts.validator import VNScriptValidationContext, validate_script_program

ManifestResolver = Callable[[int], Mapping[str, Any]]
AudioRefResolver = Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]
ProfileRow = Mapping[str, Any]

_POLICY_DEFINITIONS = {
    "local_default": LOCAL_DEFAULT_POLICY_DEFINITION,
    "strict_hosted": STRICT_HOSTED_POLICY_DEFINITION,
}

_GENERATION_DEFINITIONS = {
    "story_default": STORY_DEFAULT_GENERATION_DEFINITION,
}


class VNScriptService:
    """Coordinate VN script metadata, draft validation, and publishing."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        owner_user_id: int,
        manifest_resolver: ManifestResolver | None = None,
        audio_ref_resolver: AudioRefResolver | None = None,
    ) -> None:
        self.repo = VNScriptsRepository.initialized(db)
        self.profile_snapshots = VNProfileSnapshotRepository.initialized(db)
        self.db = db
        self.owner_user_id = owner_user_id
        self.manifest_resolver = manifest_resolver or self._default_manifest
        self.audio_ref_resolver = audio_ref_resolver or _empty_audio_refs

    def create_script(
        self,
        *,
        title: str,
        primary_asset_pack_id: int,
        policy_profile_id: str = "local_default",
        generation_profile_id: str = "story_default",
        generation_profiles: Mapping[str, str] | None = None,
        description: str | None = None,
        content_rating: str = "general",
        initial_draft: Mapping[str, Any] | None = None,
        initial_diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a script shell and its empty draft."""
        return self.repo.create_script(
            owner_user_id=self.owner_user_id,
            title=title,
            description=description,
            primary_asset_pack_id=primary_asset_pack_id,
            policy_profile_id=policy_profile_id,
            generation_profile_id=generation_profile_id,
            generation_profiles=generation_profiles,
            content_rating=content_rating,
            initial_draft=initial_draft,
            initial_diagnostics=initial_diagnostics,
        )

    def list_scripts(self, *, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """List scripts owned by the current user."""
        return self.repo.list_scripts(owner_user_id=self.owner_user_id, limit=limit, offset=offset)

    def list_templates(self) -> list[dict[str, Any]]:
        """List built-in starter templates as preview-safe catalog entries."""
        return list_template_catalog()

    def get_authoring_catalog(self) -> dict[str, Any]:
        """Return preview-safe script authoring metadata and snippet catalog."""
        return list_authoring_catalog()

    def build_snippet_patch(
        self,
        script_id: int,
        snippet_id: str,
        anchor: Mapping[str, Any],
        parameters: Mapping[str, Any],
        *,
        draft: Mapping[str, Any] | None = None,
        draft_revision: int | None = None,
        if_revision: int | None = None,
    ) -> dict[str, Any]:
        """Build a side-effect-free snippet patch against a stored or supplied draft."""
        script = self._require_script(script_id)
        draft_row = self.get_draft(script_id)
        stored_revision = int(draft_row["revision"])
        if if_revision is not None and int(if_revision) != stored_revision:
            raise ValueError("draft_revision_conflict")
        if draft is not None:
            if draft_revision is None:
                raise ValueError("draft_revision_required")
            if int(draft_revision) != stored_revision:
                raise ValueError("draft_revision_conflict")
            base_revision = int(draft_revision)
            base_draft = draft
        else:
            base_revision = stored_revision
            base_draft = draft_row["draft"]
        patch = apply_snippet_patch(base_draft, snippet_id, anchor, parameters)
        return {
            "script": script,
            "base_revision": base_revision,
            "snippet_id": snippet_id,
            "patch": patch,
        }

    def preview_snippet_patch(
        self,
        script_id: int,
        snippet_id: str,
        base_revision: int,
        patch: SnippetPatchResult,
        *,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Validate a patched draft without storing the draft or diagnostics."""
        script = self._require_script(script_id)
        diagnostics = self.validate_draft_payload(
            script,
            patch.draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        return {
            "script_id": script_id,
            "base_revision": int(base_revision),
            "snippet_id": snippet_id,
            "draft": patch.draft,
            "diagnostics": diagnostics,
            "patch_summary": patch.patch_summary,
            "warnings": list(diagnostics.get("warnings") or []),
        }

    def apply_snippet_patch_result(
        self,
        script_id: int,
        snippet_id: str,
        base_revision: int,
        patch: SnippetPatchResult,
        *,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Validate and persist a patched draft using optimistic revision control."""
        script = self._require_script(script_id)
        draft_row = self.get_draft(script_id)
        if int(draft_row["revision"]) != int(base_revision):
            raise ValueError("draft_revision_conflict")
        diagnostics = self.validate_draft_payload(
            script,
            patch.draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        updated_draft = self.repo.replace_draft(
            script_id,
            owner_user_id=self.owner_user_id,
            if_revision=int(base_revision),
            draft=patch.draft,
            diagnostics=diagnostics,
        )
        return {
            "script_id": script_id,
            "revision": int(updated_draft["revision"]),
            "snippet_id": snippet_id,
            "draft": updated_draft["draft"],
            "diagnostics": updated_draft["diagnostics"],
            "patch_summary": patch.patch_summary,
        }

    def create_script_from_template(
        self,
        template_id: str,
        *,
        title: str,
        primary_asset_pack_id: int,
        policy_profile_id: str = "local_default",
        generation_profile_id: str = "story_default",
        generation_profiles: Mapping[str, str] | None = None,
        description: str | None = None,
        content_rating: str = "general",
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        draft: Mapping[str, Any] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        resolved_generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Create a normal script and store a validated template draft."""
        template_draft = (
            dict(draft)
            if draft is not None
            else instantiate_template(
                template_id,
                title=title,
                primary_asset_pack_id=primary_asset_pack_id,
                generation_profile_id=generation_profile_id,
            )
        )
        script_metadata = _script_metadata_payload(
            primary_asset_pack_id=primary_asset_pack_id,
            policy_profile_id=policy_profile_id,
            generation_profile_id=generation_profile_id,
            generation_profiles=generation_profiles,
            content_rating=content_rating,
        )
        validation = self.validate_draft_payload(
            script_metadata,
            template_draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=resolved_generation_profiles,
        )
        script = self.create_script(
            title=title,
            description=description,
            primary_asset_pack_id=primary_asset_pack_id,
            policy_profile_id=policy_profile_id,
            generation_profile_id=generation_profile_id,
            generation_profiles=generation_profiles,
            content_rating=content_rating,
            initial_draft=template_draft,
            initial_diagnostics=validation,
        )
        draft_response = self.get_draft(int(script["id"]))
        return {"script": script, "draft": draft_response}

    def get_script(self, script_id: int) -> dict[str, Any]:
        """Return an owned script or raise not found."""
        return self._require_script(script_id)

    def update_script(self, script_id: int, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Patch script metadata."""
        self._require_script(script_id)
        updated = self.repo.update_script(script_id, fields, owner_user_id=self.owner_user_id)
        if updated is None:
            raise ValueError("script_not_found")
        return updated

    def delete_script(self, script_id: int) -> None:
        """Soft-delete an owned script."""
        self._require_script(script_id)
        self.repo.soft_delete_script(script_id, owner_user_id=self.owner_user_id)

    def get_draft(self, script_id: int) -> dict[str, Any]:
        """Return the mutable script draft."""
        self._require_script(script_id)
        draft = self.repo.get_draft(script_id, owner_user_id=self.owner_user_id)
        if draft is None:
            raise ValueError("script_not_found")
        return draft

    def get_draft_graph(
        self,
        script_id: int,
        *,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Return a computed authoring graph for the stored draft without persistence."""
        script = self._require_script(script_id)
        draft_row = self.get_draft(script_id)
        validation = self.validate_draft_payload(
            script,
            draft_row["draft"],
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        return build_script_authoring_graph(
            draft_row["draft"],
            source="stored_draft",
            script_id=script_id,
            base_revision=int(draft_row["revision"]),
            validation_diagnostics=validation,
            validation_context_source="current_draft_context",
        )

    def preview_draft_graph(
        self,
        script_id: int,
        draft: Mapping[str, Any],
        *,
        draft_revision: int | None = None,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Return a computed authoring graph for a supplied draft without persistence."""
        script = self._require_script(script_id)
        if not isinstance(draft, Mapping):
            raise ValueError("supplied_draft_invalid_shape")
        if _payload_size_bytes(draft) > MAX_SUPPLIED_DRAFT_BYTES:
            raise ValueError("supplied_draft_too_large")
        draft_row = self.get_draft(script_id)
        current_revision = int(draft_row["revision"])
        validation = self.validate_draft_payload(
            script,
            draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        result = build_script_authoring_graph(
            draft,
            source="supplied_draft",
            script_id=script_id,
            base_revision=current_revision,
            validation_diagnostics=validation,
            validation_context_source="current_draft_context",
        )
        if draft_revision is not None and int(draft_revision) != current_revision:
            result["diagnostics"]["warnings"].append(
                {
                    "code": "graph_preview_revision_stale",
                    "severity": "warning",
                    "message": "Supplied draft revision does not match the current stored draft revision.",
                    "path": "$.draft_revision",
                    "details": {
                        "supplied_draft_revision": int(draft_revision),
                        "current_revision": current_revision,
                    },
                }
            )
        return result

    def playtest_draft(
        self,
        script_id: int,
        *,
        max_steps: int = 500,
        max_paths: int = 100,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic playtest traversal for the stored draft."""
        script = self._require_script(script_id)
        draft_row = self.get_draft(script_id)
        validation = self.validate_draft_payload(
            script,
            draft_row["draft"],
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        return build_script_playtest(
            draft_row["draft"],
            source="stored_draft",
            script_id=script_id,
            base_revision=int(draft_row["revision"]),
            validation_diagnostics=validation,
            validation_context_source="current_draft_context",
            max_steps=max_steps,
            max_paths=max_paths,
        )

    def preview_draft_playtest(
        self,
        script_id: int,
        draft: Mapping[str, Any],
        *,
        draft_revision: int | None = None,
        max_steps: int = 500,
        max_paths: int = 100,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic playtest traversal for a supplied draft."""
        script = self._require_script(script_id)
        if not isinstance(draft, Mapping):
            raise ValueError("supplied_draft_invalid_shape")
        if _payload_size_bytes(draft) > MAX_SUPPLIED_DRAFT_BYTES:
            raise ValueError("supplied_draft_too_large")
        draft_row = self.get_draft(script_id)
        current_revision = int(draft_row["revision"])
        validation = self.validate_draft_payload(
            script,
            draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        result = build_script_playtest(
            draft,
            source="supplied_draft",
            script_id=script_id,
            base_revision=current_revision,
            validation_diagnostics=validation,
            validation_context_source="current_draft_context",
            max_steps=max_steps,
            max_paths=max_paths,
        )
        if draft_revision is not None and int(draft_revision) != current_revision:
            result["diagnostics"]["warnings"].append(
                {
                    "code": "playtest_preview_revision_stale",
                    "severity": "warning",
                    "message": "Supplied draft revision does not match the current stored draft revision.",
                    "path": "$.draft_revision",
                    "details": {
                        "supplied_draft_revision": int(draft_revision),
                        "current_revision": current_revision,
                    },
                }
            )
        return result

    def replace_draft(
        self,
        script_id: int,
        *,
        if_revision: int,
        draft: Mapping[str, Any],
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Replace a whole script draft using optimistic revision control."""
        script = self._require_script(script_id)
        validation = self.validate_draft_payload(
            script,
            draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        return self.repo.replace_draft(
            script_id,
            owner_user_id=self.owner_user_id,
            if_revision=if_revision,
            draft=draft,
            diagnostics=validation,
        )

    def validate_draft(
        self,
        script_id: int,
        draft: Mapping[str, Any] | None = None,
        *,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Validate a supplied draft or the currently saved draft."""
        script = self._require_script(script_id)
        if draft is None:
            draft = self.get_draft(script_id)["draft"]
        validation = self.validate_draft_payload(
            script,
            draft,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
        self.repo.store_diagnostics(script_id, owner_user_id=self.owner_user_id, diagnostics=validation)
        return validation

    def validate_draft_payload(
        self,
        script: Mapping[str, Any],
        draft: Mapping[str, Any],
        *,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Validate a draft with service-resolved manifest and profile context."""
        manifest = self._manifest_for_script(script)
        resolved_policy_profile = self._policy_profile(str(script["policy_profile_id"]), policy_profile)
        resolved_generation_profile = self._generation_profile(str(script["generation_profile_id"]), generation_profile)
        resolved_generation_profiles = self._generation_profiles(
            script,
            generation_profiles,
            default_resolved_profile=generation_profile,
        )
        resolved_audio_refs = audio_refs if audio_refs is not None else self.audio_ref_resolver(draft)
        context = VNScriptValidationContext(
            approved_slot_keys=_approved_slot_keys(manifest),
            audio_refs=_normalize_audio_refs(resolved_audio_refs),
            generation_profile={
                "profile_id": resolved_generation_profile["profile_id"],
                **resolved_generation_profile["definition"],
            },
            available_generation_profiles={
                profile_key: {"profile_id": profile["profile_id"], **profile["definition"]}
                for profile_key, profile in resolved_generation_profiles.items()
            },
            content_rating=str(script.get("content_rating") or "general"),
            owner_user_id=self.owner_user_id,
        )
        policy_decision = self._evaluate_publish_policy(script, policy_profile=resolved_policy_profile)
        policy_errors, policy_warnings = _policy_profile_validation_issues(policy_decision)
        return _merge_validation_issues(
            validate_script_program(draft, context).to_dict(),
            [*_script_metadata_consistency_errors(script, draft), *policy_errors],
            policy_warnings,
        )

    def publish_script(
        self,
        script_id: int,
        *,
        draft_revision: int,
        label: str | None,
        idempotency_key: str,
        acknowledgements: list[str] | None = None,
        audio_refs: Mapping[str, Mapping[str, Any]] | None = None,
        policy_profile: ProfileRow | None = None,
        generation_profile: ProfileRow | None = None,
        generation_profiles: Mapping[str, ProfileRow] | None = None,
    ) -> dict[str, Any]:
        """Validate and publish an immutable script version."""
        script = self._require_script(script_id)
        request_payload = _publish_request_payload(
            script_id=script_id,
            draft_revision=draft_revision,
            label=label,
            acknowledgements=acknowledgements,
        )
        legacy_payload_hash = canonical_payload_hash(request_payload)
        existing = self.repo.get_publish_request_by_key(
            owner_user_id=self.owner_user_id,
            script_id=script_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            existing_request_payload = existing.get("request_payload")
            if isinstance(existing_request_payload, Mapping):
                matches_existing = dict(existing_request_payload) == request_payload
            else:
                matches_existing = existing["payload_hash"] == legacy_payload_hash
            if not matches_existing:
                raise ValueError("idempotency_key_conflict")
            return dict(existing["response"])

        resolved_generation_profiles = self._generation_profiles(
            script,
            generation_profiles,
            default_resolved_profile=generation_profile,
        )
        resolved_generation_profile_ids = {
            profile_key: str(profile["profile_id"])
            for profile_key, profile in sorted(resolved_generation_profiles.items())
        }
        payload_hash = canonical_payload_hash(
            {
                **request_payload,
                "generation_profile_ids": resolved_generation_profile_ids,
                "generation_profile_versions": {
                    profile_key: int(profile["version"])
                    for profile_key, profile in sorted(resolved_generation_profiles.items())
                },
            }
        )

        draft_row = self.get_draft(script_id)
        if int(draft_row["revision"]) != int(draft_revision):
            raise ValueError("draft_revision_conflict")
        program = draft_row["draft"]
        resolved_policy_profile = self._policy_profile(str(script["policy_profile_id"]), policy_profile)
        resolved_generation_profile = resolved_generation_profiles["default"]
        validation = self.validate_draft_payload(
            script,
            program,
            audio_refs=audio_refs,
            policy_profile=resolved_policy_profile,
            generation_profile=resolved_generation_profile,
            generation_profiles=resolved_generation_profiles,
        )
        policy_decision = self._evaluate_publish_policy(script, policy_profile=resolved_policy_profile)
        if policy_decision["decision"] == "block":
            raise ValueError("script_publish_policy_blocked")
        if not validation["valid"]:
            raise ValueError("script_publish_validation_failed")
        missing_acknowledgements = _required_acknowledgement_codes(policy_decision) - set(acknowledgements or [])
        if missing_acknowledgements:
            raise ValueError("script_publish_acknowledgement_required")

        manifest = self._manifest_for_script(script)
        manifest_hash = canonical_payload_hash(manifest)
        published = self.repo.publish_version_with_request(
            owner_user_id=self.owner_user_id,
            script_id=script_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            request_payload=request_payload,
            label=label,
            draft_revision=draft_revision,
            program=program,
            asset_pack_id=int(script["primary_asset_pack_id"]),
            manifest=manifest,
            manifest_hash=manifest_hash,
            policy_profile=resolved_policy_profile,
            generation_profile=resolved_generation_profile,
            generation_profiles=resolved_generation_profiles,
            script_defaults=_script_defaults(program, script),
            validation=validation,
        )
        return dict(published["response"])

    def get_publish_request_by_key(
        self,
        script_id: int,
        *,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Return an existing publish request for endpoint idempotency checks."""
        self._require_script(script_id)
        return self.repo.get_publish_request_by_key(
            owner_user_id=self.owner_user_id,
            script_id=script_id,
            idempotency_key=idempotency_key,
        )

    def list_versions(self, script_id: int, *, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """List published versions for an owned script."""
        self._require_script(script_id)
        return self.repo.list_versions(script_id, owner_user_id=self.owner_user_id, limit=limit, offset=offset)

    def get_version(self, script_id: int, version_id: int) -> dict[str, Any]:
        """Return a published script version."""
        self._require_script(script_id)
        version = self.repo.get_version(script_id, version_id, owner_user_id=self.owner_user_id)
        if version is None:
            raise ValueError("script_version_not_found")
        return version

    def get_version_graph(self, script_id: int, version_id: int) -> dict[str, Any]:
        """Return a computed authoring graph for an immutable published version."""
        version = self.get_version(script_id, version_id)
        return build_script_authoring_graph(
            version["program"],
            source="published_version",
            script_id=script_id,
            version_id=version_id,
            validation_diagnostics=_stored_version_validation(version),
            validation_context_source="published_version_snapshot",
        )

    def playtest_version(self, script_id: int, version_id: int, *, max_steps: int = 500, max_paths: int = 100) -> dict[str, Any]:
        """Return a deterministic playtest traversal for an immutable published version."""
        version = self.get_version(script_id, version_id)
        return build_script_playtest(
            version["program"],
            source="published_version",
            script_id=script_id,
            version_id=version_id,
            validation_diagnostics=_stored_version_validation(version),
            validation_context_source="published_version_snapshot",
            max_steps=max_steps,
            max_paths=max_paths,
        )

    def get_manifest_snapshot(self, script_id: int, version_id: int) -> dict[str, Any]:
        """Return the manifest snapshot pinned to a version."""
        self._require_script(script_id)
        snapshot = self.repo.get_manifest_snapshot_for_version(
            script_id=script_id,
            version_id=version_id,
            owner_user_id=self.owner_user_id,
        )
        if snapshot is None:
            raise ValueError("script_version_not_found")
        return snapshot

    def evaluate_version_policy(
        self,
        script_id: int,
        version_id: int,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a published script version with the stored policy profile id."""
        version = self.get_version(script_id, version_id)
        policy_snapshot = self.profile_snapshots.get_profile_snapshot(
            int(version["policy_snapshot_id"]),
            owner_user_id=self.owner_user_id,
        )
        if policy_snapshot is None:
            raise ValueError("policy_snapshot_not_found")
        evaluation_context = dict(context or {})
        character_safety = evaluation_context.get("character_safety")
        metadata_status = "missing"
        if isinstance(character_safety, Mapping):
            metadata_status = str(character_safety.get("metadata_status") or "missing")
        return evaluate_character_safety_definition(
            profile_definition=policy_snapshot["definition"],
            policy_profile_id=str(policy_snapshot["profile_id"]),
            content_rating=str(
                evaluation_context.get("content_rating")
                or version.get("script_defaults", {}).get("content_rating")
                or version.get("program", {}).get("content_rating")
                or "general"
            ),
            metadata_status=metadata_status,
        )

    def _require_script(self, script_id: int) -> dict[str, Any]:
        script = self.repo.get_script(script_id, owner_user_id=self.owner_user_id)
        if script is None:
            raise ValueError("script_not_found")
        return script

    def _manifest_for_script(self, script: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.manifest_resolver(int(script["primary_asset_pack_id"]))

    def _default_manifest(self, asset_pack_id: int) -> Mapping[str, Any]:
        manifest = VNAssetPackService(self.db, owner_user_id=self.owner_user_id).build_manifest(asset_pack_id)
        return manifest.model_dump(mode="json")

    def _policy_profile(self, profile_id: str, resolved_profile: ProfileRow | None = None) -> dict[str, Any]:
        return _resolved_profile(
            profile_id,
            resolved_profile,
            _POLICY_DEFINITIONS,
            missing_reason="policy_profile_not_found",
        )

    def _generation_profile(self, profile_id: str, resolved_profile: ProfileRow | None = None) -> dict[str, Any]:
        return _resolved_profile(
            profile_id,
            resolved_profile,
            _GENERATION_DEFINITIONS,
            missing_reason="generation_profile_not_found",
        )

    def _generation_profiles(
        self,
        script: Mapping[str, Any],
        resolved_profiles: Mapping[str, ProfileRow] | None = None,
        *,
        default_resolved_profile: ProfileRow | None = None,
    ) -> dict[str, dict[str, Any]]:
        profile_ids = dict(script.get("generation_profiles") or {"default": script["generation_profile_id"]})
        profile_ids["default"] = str(script["generation_profile_id"])
        resolved: dict[str, dict[str, Any]] = {}
        provided = {"default": default_resolved_profile, **dict(resolved_profiles or {})}
        for profile_key, profile_id in profile_ids.items():
            profile_row = provided.get(profile_key)
            resolved[profile_key] = self._generation_profile(str(profile_id), profile_row)
        return resolved

    def _evaluate_publish_policy(
        self,
        script: Mapping[str, Any],
        *,
        policy_profile: ProfileRow | None = None,
    ) -> dict[str, Any]:
        resolved_profile = self._policy_profile(str(script["policy_profile_id"]), policy_profile)
        return evaluate_character_safety_definition(
            profile_definition=resolved_profile["definition"],
            policy_profile_id=resolved_profile["profile_id"],
            content_rating=str(script.get("content_rating") or "general"),
            metadata_status=self._primary_character_safety_status(script),
        )

    def _primary_character_safety_status(self, script: Mapping[str, Any]) -> str:
        pack = VNAssetPacksRepository.initialized(self.db).get_pack(int(script["primary_asset_pack_id"]))
        if pack is None:
            return "missing"
        character_id = pack.get("primary_character_id")
        if not isinstance(character_id, int) or isinstance(character_id, bool):
            return "missing"
        character = self.db.get_character_card_by_id(character_id)
        if not isinstance(character, Mapping):
            return "missing"
        return _character_safety_status(character)


def _approved_slot_keys(manifest: Mapping[str, Any]) -> set[str]:
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping):
        return set()
    slot_keys: set[str] = set()
    for collection in assets.values():
        if not isinstance(collection, list):
            continue
        for item in collection:
            if isinstance(item, Mapping) and isinstance(item.get("slot_key"), str):
                slot_keys.add(str(item["slot_key"]))
    return slot_keys


def _empty_audio_refs(program: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return {}


def _payload_size_bytes(payload: Mapping[str, Any]) -> int:
    """Return the canonical JSON byte size used for supplied-draft limits."""
    return len(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _stored_version_validation(version: Mapping[str, Any]) -> dict[str, Any]:
    """Return validation diagnostics pinned to a published script version."""
    validation = version.get("validation")
    if isinstance(validation, Mapping):
        return {
            "valid": bool(validation.get("valid")),
            "errors": list(validation.get("errors") or []),
            "warnings": list(validation.get("warnings") or []),
        }
    return {"valid": False, "errors": [], "warnings": []}


def _script_metadata_payload(
    *,
    primary_asset_pack_id: int,
    policy_profile_id: str,
    generation_profile_id: str,
    generation_profiles: Mapping[str, str] | None,
    content_rating: str,
) -> dict[str, Any]:
    return {
        "primary_asset_pack_id": primary_asset_pack_id,
        "policy_profile_id": policy_profile_id,
        "generation_profile_id": generation_profile_id,
        "generation_profiles": dict(generation_profiles or {}),
        "content_rating": content_rating,
    }


def _normalize_audio_refs(raw_refs: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_refs, Mapping):
        return {}
    return {
        str(media_ref): {**dict(metadata), "owner_user_id": metadata.get("owner_user_id")}
        for media_ref, metadata in raw_refs.items()
        if isinstance(media_ref, str) and isinstance(metadata, Mapping)
    }


def _resolved_profile(
    profile_id: str,
    resolved_profile: ProfileRow | None,
    builtin_definitions: Mapping[str, Mapping[str, Any]],
    *,
    missing_reason: str,
) -> dict[str, Any]:
    if resolved_profile is None:
        definition = builtin_definitions.get(profile_id)
        if definition is None:
            raise ValueError(missing_reason)
        return {
            "profile_id": profile_id,
            "version": 1,
            "definition": dict(definition),
        }

    resolved_profile_id = resolved_profile.get("profile_id")
    definition = resolved_profile.get("definition")
    if str(resolved_profile_id) != profile_id or not isinstance(definition, Mapping):
        raise ValueError(missing_reason)
    try:
        version = int(resolved_profile.get("version") or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(missing_reason) from exc
    return {
        "profile_id": str(resolved_profile_id),
        "version": version,
        "definition": dict(definition),
    }


def _character_safety_status(character: Mapping[str, Any]) -> str:
    metadata = _character_safety_metadata(character)
    if isinstance(metadata, Mapping):
        status = metadata.get("age_status") or metadata.get("status")
        if isinstance(status, str) and status.strip():
            return _normalize_safety_status(status)

    explicit_status = character.get("age_status") or character.get("safety_status")
    if isinstance(explicit_status, str) and explicit_status.strip():
        return _normalize_safety_status(explicit_status)

    minor_flag = _truthy_flag(character.get("is_minor")) or _truthy_flag(character.get("minor"))
    adult_flag = _truthy_flag(character.get("is_adult")) or _truthy_flag(character.get("adult"))
    if minor_flag and adult_flag:
        return "conflicting"
    if minor_flag:
        return "minor"
    if adult_flag:
        return "adult"

    age = character.get("age_years", character.get("age"))
    if isinstance(age, int) and not isinstance(age, bool):
        return "adult" if age >= 18 else "minor"
    return "missing"


def _character_safety_metadata(character: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = character.get("safety_metadata")
    if isinstance(metadata, Mapping):
        return metadata
    extensions = character.get("extensions")
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except json.JSONDecodeError:
            extensions = None
    if not isinstance(extensions, Mapping):
        return None
    for key in ("safety_metadata", "vn_safety_metadata", "character_safety"):
        metadata = extensions.get(key)
        if isinstance(metadata, Mapping):
            return metadata
    return None


def _normalize_safety_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"adult", "18_plus", "18plus", "of_age"}:
        return "adult"
    if normalized in {"minor", "under_18", "under18"}:
        return "minor"
    if normalized in {"missing", "unspecified", "not_provided"}:
        return "missing"
    if normalized in {"unknown", "ambiguous", "unknown_ambiguous", "unknown_or_ambiguous"}:
        return "unknown_or_ambiguous"
    if normalized in {"conflict", "conflicting"}:
        return "conflicting"
    if normalized in {"imported_untrusted", "untrusted_import", "imported_without_trusted_provenance"}:
        return "imported_untrusted"
    return "unknown_or_ambiguous"


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _script_metadata_consistency_errors(script: Mapping[str, Any], program: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    program_pack_id = program.get("primary_asset_pack_id")
    if isinstance(program_pack_id, int) and not isinstance(program_pack_id, bool):
        if int(program_pack_id) != int(script["primary_asset_pack_id"]):
            errors.append(
                _validation_error(
                    "primary_asset_pack_mismatch",
                    "Program primary_asset_pack_id must match script metadata.",
                    "$.primary_asset_pack_id",
                    {"program_asset_pack_id": program_pack_id, "script_asset_pack_id": script["primary_asset_pack_id"]},
                )
            )
    generation_defaults = program.get("generation_defaults")
    if isinstance(generation_defaults, Mapping):
        profile_id = generation_defaults.get("profile_id")
        if isinstance(profile_id, str) and profile_id != str(script["generation_profile_id"]):
            errors.append(
                _validation_error(
                    "generation_profile_mismatch",
                    "Program generation_defaults.profile_id must match script metadata.",
                    "$.generation_defaults.profile_id",
                    {"program_profile_id": profile_id, "script_profile_id": script["generation_profile_id"]},
                )
            )
    return errors


def _policy_profile_validation_issues(
    policy_decision: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    reasons = policy_decision.get("reasons")
    if not isinstance(reasons, list):
        return errors, warnings
    for reason in reasons:
        if not isinstance(reason, Mapping):
            continue
        issue = {
            "code": f"policy_{reason.get('code')}",
            "message": str(reason.get("message") or "Policy profile requires acknowledgement."),
            "path": "$.policy_profile_id",
            "details": {
                "profile_id": policy_decision.get("profile_id"),
                "decision": policy_decision.get("decision"),
                "severity": reason.get("severity"),
                "requires_acknowledgement": bool(reason.get("requires_acknowledgement")),
            },
        }
        if policy_decision.get("decision") == "block" or reason.get("severity") == "error":
            errors.append(issue)
        else:
            warnings.append(issue)
    return errors, warnings


def _merge_validation_issues(
    validation: dict[str, Any],
    extra_errors: list[dict[str, Any]],
    extra_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not extra_errors and not extra_warnings:
        return validation
    merged = dict(validation)
    if extra_errors:
        merged["errors"] = [*list(validation.get("errors") or []), *extra_errors]
        merged["valid"] = False
    if extra_warnings:
        merged["warnings"] = [*list(validation.get("warnings") or []), *extra_warnings]
    return merged


def _validation_error(code: str, message: str, path: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "path": path, "details": dict(details or {})}


def _script_defaults(program: Mapping[str, Any], script: Mapping[str, Any]) -> dict[str, Any]:
    defaults = dict(program.get("generation_defaults") or {})
    defaults["content_rating"] = str(script.get("content_rating") or "general")
    defaults["policy_profile_id"] = str(script["policy_profile_id"])
    defaults["generation_profile_id"] = str(script["generation_profile_id"])
    return defaults


def _publish_request_payload(
    *,
    script_id: int,
    draft_revision: int,
    label: str | None,
    acknowledgements: list[str] | None,
) -> dict[str, Any]:
    return {
        "script_id": script_id,
        "draft_revision": draft_revision,
        "label": label,
        "acknowledgements": sorted(acknowledgements or []),
    }


def _required_acknowledgement_codes(policy_decision: Mapping[str, Any]) -> set[str]:
    reasons = policy_decision.get("reasons")
    if not isinstance(reasons, list):
        return set()
    return {
        str(reason["code"])
        for reason in reasons
        if isinstance(reason, Mapping) and reason.get("requires_acknowledgement") and reason.get("code")
    }

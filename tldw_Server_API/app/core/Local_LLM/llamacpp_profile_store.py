"""JSON-backed store for managed llama.cpp runtime profiles."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppProfileConflictError,
    LlamaCppProfileNotFoundError,
    LlamaCppProfileStoreError,
)
from tldw_Server_API.app.core.Setup import setup_manager


DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "Default llama.cpp server"


def default_profile_store_path() -> Path:
    return setup_manager.get_config_file_path().expanduser().resolve().with_name("llamacpp_profiles.json")


class JsonLlamaCppProfileStore:
    """Persist managed llama.cpp profiles in a small JSON document."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def list_profiles(self) -> list[LlamaCppProfile]:
        return self._read_profiles()

    def get(self, profile_id: str) -> LlamaCppProfile | None:
        for profile in self._read_profiles():
            if profile.profile_id == profile_id:
                return profile
        return None

    def upsert(self, profile: LlamaCppProfile) -> LlamaCppProfile:
        profiles = self._read_profiles()
        replaced = False
        next_profiles: list[LlamaCppProfile] = []
        for existing in profiles:
            if existing.profile_id == profile.profile_id:
                next_profiles.append(profile)
                replaced = True
            else:
                next_profiles.append(existing)
        if not replaced:
            next_profiles.append(profile)
        self._validate_unique_explicit_ports(next_profiles)
        self._write_profiles(next_profiles)
        return profile

    def delete(self, profile_id: str) -> bool:
        profiles = self._read_profiles()
        next_profiles = [profile for profile in profiles if profile.profile_id != profile_id]
        if len(next_profiles) == len(profiles):
            return False
        self._write_profiles(next_profiles)
        return True

    def ensure_default_profile(
        self,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        server_args: dict[str, object] | None = None,
    ) -> LlamaCppProfile:
        existing = self.get(DEFAULT_PROFILE_ID)
        if existing is not None:
            return existing
        profile = LlamaCppProfile(
            profile_id=DEFAULT_PROFILE_ID,
            name=DEFAULT_PROFILE_NAME,
            enabled=True,
            model_id=model_id,
            model_path=model_path,
            host=host,
            port=port,
            port_policy=LlamaCppPortPolicy.EXPLICIT,
            server_args=dict(server_args or {}),
        )
        return self.upsert(profile)

    def _read_profiles(self) -> list[LlamaCppProfile]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise LlamaCppProfileStoreError(f"Unable to read llama.cpp profile store: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LlamaCppProfileStoreError(f"Invalid llama.cpp profile store JSON: {exc}") from exc

        if isinstance(raw, dict):
            raw_profiles = raw.get("profiles", [])
        else:
            raw_profiles = raw
        if not isinstance(raw_profiles, list):
            raise LlamaCppProfileStoreError("Invalid llama.cpp profile store: profiles must be a list.")
        try:
            return [LlamaCppProfile.model_validate(item) for item in raw_profiles]
        except ValidationError as exc:
            raise LlamaCppProfileStoreError(f"Invalid llama.cpp profile entry: {exc}") from exc

    def _write_profiles(self, profiles: list[LlamaCppProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"profiles": [profile.model_dump(mode="json") for profile in profiles]}
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise LlamaCppProfileStoreError(f"Unable to write llama.cpp profile store: {exc}") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _validate_unique_explicit_ports(profiles: list[LlamaCppProfile]) -> None:
        seen: dict[tuple[str, int], str] = {}
        for profile in profiles:
            if not profile.enabled or profile.port_policy != LlamaCppPortPolicy.EXPLICIT:
                continue
            key = (profile.host, profile.port)
            other_profile_id = seen.get(key)
            if other_profile_id is not None:
                raise LlamaCppProfileConflictError(
                    f"Enabled explicit llama.cpp profiles have a duplicate host/port: "
                    f"{profile.host}:{profile.port} ({other_profile_id}, {profile.profile_id})."
                )
            seen[key] = profile.profile_id


__all__ = [
    "DEFAULT_PROFILE_ID",
    "DEFAULT_PROFILE_NAME",
    "JsonLlamaCppProfileStore",
    "LlamaCppProfileConflictError",
    "LlamaCppProfileNotFoundError",
    "LlamaCppProfileStoreError",
    "default_profile_store_path",
]

"""Ephemeral voice readiness, fenced by connection and preparation identity."""

from threading import RLock
from uuid import uuid4

from tldw_Server_API.app.core.exceptions import PersonaVoiceInputLimitError  # noqa: F401 - compatibility re-export


class PersonaLiveVoiceRegistry:
    """Project prepared runtimes without granting permission or persisting state."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], tuple[str, bool]] = {}
        self._lock = RLock()

    def begin_preparation(self, *, user_id: str, session_id: str, connection_id: str) -> str:
        """Replace a connection's readiness with a unique pending preparation."""
        token = uuid4().hex
        with self._lock:
            self._entries[(user_id, session_id, connection_id)] = (token, False)
        return token

    def complete_preparation(self, *, user_id: str, session_id: str, connection_id: str, token: str) -> bool:
        """Publish readiness only while the original preparation still owns it."""
        key = (user_id, session_id, connection_id)
        with self._lock:
            if self._entries.get(key) != (token, False):
                return False
            self._entries[key] = (token, True)
            return True

    def is_preparing(self, *, user_id: str, session_id: str, connection_id: str, token: str) -> bool:
        """Check that an expensive preparation still owns its pending slot."""
        with self._lock:
            return self._entries.get((user_id, session_id, connection_id)) == (token, False)

    def is_ready(self, *, user_id: str, session_id: str, connection_id: str | None = None) -> bool:
        """Check one connection's runtime, or project any prepared connection."""
        with self._lock:
            return any(
                uid == user_id and sid == session_id and (connection_id is None or cid == connection_id) and ready
                for (uid, sid, cid), (_, ready) in self._entries.items()
            )

    def fail_preparation(self, *, user_id: str, session_id: str, connection_id: str, token: str) -> bool:
        """Revoke this pending attempt without changing a newer attempt's ownership."""
        key = (user_id, session_id, connection_id)
        with self._lock:
            if self._entries.get(key) != (token, False):
                return False
            self._entries.pop(key, None)
            return True

    def clear(self, *, user_id: str, session_id: str, connection_id: str | None = None) -> None:
        """Revoke pending and ready runtimes, including late initialization results."""
        with self._lock:
            for key in list(self._entries):
                uid, sid, cid = key
                if uid == user_id and sid == session_id and (connection_id is None or cid == connection_id):
                    self._entries.pop(key, None)


persona_live_voice_registry = PersonaLiveVoiceRegistry()

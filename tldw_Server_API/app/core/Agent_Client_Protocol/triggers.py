"""ACP webhook trigger manager.

Routes inbound webhook events to acp_run task submissions with
provider-specific signature verification.

Security features:
- Multi-provider HMAC verification (GitHub, Slack, generic)
- Encrypted secrets at rest (Fernet, requires ``cryptography`` package)
- Per-trigger rate limiting (in-memory, per-process)
- Replay attack prevention for all providers:
  - Slack: 5-minute timestamp window
  - GitHub: ``X-GitHub-Delivery`` idempotency (TTL-based dedup)
  - Generic: ``X-Webhook-Timestamp`` required, included in signed base string,
    5-minute window enforced
- Timing-safe HMAC comparison (hmac.compare_digest)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Secret encryption helper
# ---------------------------------------------------------------------------


class TriggerSecretManager:
    """Encrypts/decrypts webhook secrets at rest using Fernet (AES-128-CBC).

    Requires the ``cryptography`` package and an explicit encryption key.
    The key must be provided either as a constructor argument or via the
    ``ACP_TRIGGER_ENCRYPTION_KEY`` environment variable.  If neither is
    set, ``__init__`` raises ``ValueError`` to fail fast.
    """

    def __init__(self, encryption_key: str | None = None) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for webhook trigger secrets. "
                "Install it with: pip install cryptography"
            )

        key = encryption_key or os.getenv("ACP_TRIGGER_ENCRYPTION_KEY")
        if not key:
            raise ValueError(
                "ACP_TRIGGER_ENCRYPTION_KEY environment variable is required for webhook triggers. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )

        # Ensure key is Fernet-compatible (url-safe base64, 32 bytes)
        try:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception:
            # If key is not valid Fernet format, derive one via SHA-256
            derived = base64.urlsafe_b64encode(
                hashlib.sha256(key.encode()).digest()
            )
            self._fernet = Fernet(derived)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return a string safe for DB storage."""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a previously encrypted value."""
        return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key suitable for ``ACP_TRIGGER_ENCRYPTION_KEY``."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------


class WebhookVerifier:
    """Provider-specific webhook signature verification.

    All comparison functions use ``hmac.compare_digest`` for timing-safe
    equality checks.  Replay protection is enforced for every provider:

    - **Slack**: 5-minute timestamp window (built into signature base string).
    - **GitHub**: ``X-GitHub-Delivery`` idempotency key with TTL-based dedup.
    - **Generic**: ``X-Webhook-Timestamp`` included in signed base string and
      subject to a 5-minute freshness window.
    """

    _REPLAY_WINDOW_SEC = 300  # 5 minutes
    _DELIVERY_CACHE_MAX = 10_000

    def __init__(self) -> None:
        # GitHub delivery-ID dedup cache: {delivery_id: expiry_timestamp}
        self._github_deliveries: dict[str, float] = {}

    def verify_github(
        self, payload_body: bytes, signature: str, secret: str,
        delivery_id: str | None = None,
    ) -> bool:
        """Verify GitHub ``X-Hub-Signature-256`` header.

        Replay prevention uses the ``X-GitHub-Delivery`` UUID.  Each
        delivery ID is accepted at most once within a 5-minute window.
        """
        # Dedup via delivery ID
        if delivery_id:
            now = time.time()
            self._prune_delivery_cache(now)
            if delivery_id in self._github_deliveries:
                logger.warning("GitHub webhook rejected: duplicate delivery_id (replay)")
                return False
            # Record *after* signature check below
        else:
            logger.warning("GitHub webhook missing X-GitHub-Delivery header (replay protection weakened)")

        expected = "sha256=" + hmac.new(
            secret.encode("utf-8"), payload_body, hashlib.sha256
        ).hexdigest()
        sig_ok = hmac.compare_digest(expected, signature)

        if sig_ok and delivery_id:
            self._github_deliveries[delivery_id] = time.time() + self._REPLAY_WINDOW_SEC

        return sig_ok

    def _prune_delivery_cache(self, now: float) -> None:
        """Remove expired delivery IDs and cap cache size."""
        expired = [k for k, exp in self._github_deliveries.items() if exp <= now]
        for k in expired:
            del self._github_deliveries[k]
        # Hard cap to prevent unbounded growth
        if len(self._github_deliveries) > self._DELIVERY_CACHE_MAX:
            oldest = sorted(self._github_deliveries, key=self._github_deliveries.get)  # type: ignore[arg-type]
            for k in oldest[: len(self._github_deliveries) - self._DELIVERY_CACHE_MAX]:
                del self._github_deliveries[k]

    @staticmethod
    def verify_slack(
        payload_body: bytes,
        timestamp: str,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify Slack request signing.

        Rejects requests with timestamps older than 5 minutes (replay
        prevention).
        """
        try:
            ts_float = float(timestamp)
        except (ValueError, TypeError):
            return False
        if abs(time.time() - ts_float) > WebhookVerifier._REPLAY_WINDOW_SEC:
            logger.warning("Slack webhook rejected: timestamp outside window (replay prevention)")
            return False
        sig_basestring = f"v0:{timestamp}:{payload_body.decode('utf-8', errors='replace')}"
        expected = "v0=" + hmac.new(
            secret.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def verify_generic(
        payload_body: bytes, signature: str, secret: str,
        timestamp: str | None = None,
    ) -> bool:
        """Verify generic HMAC-SHA256 via ``X-Webhook-Signature`` header.

        Requires ``X-Webhook-Timestamp``.  The timestamp is included in the
        signed base string (``<timestamp>.<body>``) and must be within a
        5-minute window to prevent replay attacks.
        """
        if not timestamp:
            logger.warning("Generic webhook rejected: missing X-Webhook-Timestamp")
            return False
        try:
            ts_float = float(timestamp)
        except (ValueError, TypeError):
            return False
        if abs(time.time() - ts_float) > WebhookVerifier._REPLAY_WINDOW_SEC:
            logger.warning("Generic webhook rejected: timestamp outside window (replay prevention)")
            return False

        # Include timestamp in signed payload to bind signature to the time
        signed_payload = f"{timestamp}.".encode("utf-8") + payload_body
        expected = hmac.new(
            secret.encode("utf-8"), signed_payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Trigger configuration dataclass
# ---------------------------------------------------------------------------


class TriggerConfig:
    """In-memory representation of a webhook trigger row."""

    __slots__ = (
        "id",
        "name",
        "source_type",
        "secret_encrypted",
        "owner_user_id",
        "agent_config",
        "prompt_template",
        "enabled",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        id: str,
        name: str,
        source_type: str,
        secret_encrypted: str,
        owner_user_id: int,
        agent_config: dict[str, Any],
        prompt_template: str,
        enabled: bool = True,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.source_type = source_type
        self.secret_encrypted = secret_encrypted
        self.owner_user_id = owner_user_id
        self.agent_config = agent_config
        self.prompt_template = prompt_template
        self.enabled = enabled
        self.created_at = created_at
        self.updated_at = updated_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "owner_user_id": self.owner_user_id,
            "agent_config": self.agent_config,
            "prompt_template": self.prompt_template,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# ACPTriggerManager -- the main orchestrator
# ---------------------------------------------------------------------------


class ACPTriggerManager:
    """Manages webhook triggers and routes events to acp_run.

    Responsibilities:
    - CRUD for trigger configurations (stored in ``webhook_triggers`` table)
    - Per-trigger rate limiting (in-memory sliding window)
    - Signature verification dispatch by provider
    - Building and submitting ``acp_run`` payloads
    """

    def __init__(self, db: Any, secret_manager: TriggerSecretManager) -> None:
        self._db = db
        self._secret_mgr = secret_manager
        self._verifier = WebhookVerifier()
        # In-memory rate limit state: trigger_id -> list of timestamps
        self._rate_limits: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(self, trigger_id: str, max_per_minute: int = 60) -> bool:
        """Return ``True`` if the trigger is within its rate limit.

        Uses a sliding 60-second window tracked in memory.
        """
        now = time.time()
        window = self._rate_limits.setdefault(trigger_id, [])
        # Prune timestamps older than 60 seconds
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= max_per_minute:
            return False
        window.append(now)
        return True

    # ------------------------------------------------------------------
    # Signature verification dispatch
    # ------------------------------------------------------------------

    def _verify_signature(
        self,
        source_type: str,
        payload_body: bytes,
        headers: dict[str, str],
        secret: str,
    ) -> bool:
        """Dispatch to the correct verifier based on *source_type*.

        Error responses are deliberately vague ("verification failed") to
        avoid leaking which specific header was missing or invalid.
        """
        if source_type == "github":
            sig = headers.get("x-hub-signature-256", "")
            if not sig:
                logger.debug("GitHub webhook: missing required signature header")
                return False
            delivery_id = headers.get("x-github-delivery", "")
            return self._verifier.verify_github(
                payload_body, sig, secret,
                delivery_id=delivery_id or None,
            )

        if source_type == "slack":
            timestamp = headers.get("x-slack-request-timestamp", "")
            sig = headers.get("x-slack-signature", "")
            if not timestamp or not sig:
                logger.debug("Slack webhook: missing required headers")
                return False
            return self._verifier.verify_slack(payload_body, timestamp, sig, secret)

        # Default: generic HMAC (timestamp required for replay prevention)
        sig = headers.get("x-webhook-signature", "")
        timestamp = headers.get("x-webhook-timestamp", "")
        if not sig:
            logger.debug("Generic webhook: missing required signature header")
            return False
        return self._verifier.verify_generic(
            payload_body, sig, secret,
            timestamp=timestamp or None,
        )

    # ------------------------------------------------------------------
    # Prompt template rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_prompt(template: str, payload_body: bytes, headers: dict[str, str]) -> str:
        """Render a prompt template with event placeholders.

        Supported placeholders (using ``{variable}`` syntax):
        - ``{payload}``    -- raw payload as string
        - ``{event_type}`` -- inferred from headers or "webhook"

        Uses ``str.format_map`` with a restricted set of known keys
        to prevent format string injection.  Unknown placeholders
        are replaced via a safe fallback.
        """
        try:
            payload_str = payload_body.decode("utf-8", errors="replace")
        except Exception:
            payload_str = "<binary payload>"

        # Try to determine event type from common header patterns
        event_type = (
            headers.get("x-github-event", "")
            or headers.get("x-event-type", "")
            or "webhook"
        )

        safe_vars = {"payload": payload_str, "event_type": event_type}
        try:
            return template.format_map(safe_vars)
        except (KeyError, ValueError, IndexError):
            # If template has unknown placeholders, fall back to simple replacement
            result = template
            for key, val in safe_vars.items():
                result = result.replace(f"{{{key}}}", val)
            return result

    # ------------------------------------------------------------------
    # Core webhook handler
    # ------------------------------------------------------------------

    async def handle_webhook(
        self,
        trigger_id: str,
        payload_body: bytes,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Verify and process an inbound webhook.

        Steps:
        1. Load trigger config from DB
        2. Check enabled flag
        3. Check per-trigger rate limit
        4. Decrypt webhook secret
        5. Verify HMAC signature (provider-specific)
        6. Build acp_run payload from agent_config + prompt_template
        7. Submit acp_run task via the scheduler
        8. Return ``{task_id, status}``
        """
        # 1. Load trigger (sync DB call wrapped for async safety)
        trigger = await asyncio.to_thread(self.get_trigger, trigger_id)
        if trigger is None:
            logger.warning("Webhook received for unknown trigger_id={}", trigger_id)
            return {"error": "trigger_not_found", "status": "rejected"}

        # 2. Enabled?
        if not trigger["enabled"]:
            logger.info("Webhook for disabled trigger_id={}", trigger_id)
            return {"error": "trigger_disabled", "status": "rejected"}

        # 3. Rate limit
        if not self.check_rate_limit(trigger_id):
            logger.warning("Rate limit exceeded for trigger_id={}", trigger_id)
            return {"error": "rate_limit_exceeded", "status": "rejected"}

        # 4. Decrypt secret
        try:
            secret = self._secret_mgr.decrypt(trigger["secret_encrypted"])
        except Exception as exc:
            logger.error("Failed to decrypt secret for trigger_id={}: {}", trigger_id, exc)
            return {"error": "secret_decryption_failed", "status": "rejected"}

        # 5. Verify signature
        source_type = trigger.get("source_type", "generic")
        # Normalize headers to lowercase for consistent lookup
        norm_headers = {k.lower(): v for k, v in headers.items()}
        if not self._verify_signature(source_type, payload_body, norm_headers, secret):
            logger.warning("Webhook verification failed for trigger_id={}", trigger_id)
            return {"error": "verification_failed", "status": "rejected"}

        # 6. Build acp_run payload
        agent_config = trigger.get("agent_config", {})
        prompt_template = trigger.get("prompt_template", "")

        if prompt_template:
            prompt_text = self._render_prompt(prompt_template, payload_body, norm_headers)
        else:
            # Default prompt: pass the payload as context
            try:
                payload_str = payload_body.decode("utf-8", errors="replace")
            except Exception:
                payload_str = "<binary payload>"
            prompt_text = f"Process the following webhook event:\n\n{payload_str}"

        acp_payload: dict[str, Any] = {
            "user_id": trigger["owner_user_id"],
            "prompt": prompt_text,
        }
        # Merge agent_config fields into the payload
        for key in ("cwd", "agent_type", "persona_id", "workspace_id",
                     "workspace_group_id", "scope_snapshot_id", "model",
                     "token_budget"):
            if key in agent_config and agent_config[key] is not None:
                acp_payload[key] = agent_config[key]

        # 7. Submit acp_run task
        try:
            task_id = await self._submit_acp_run(acp_payload)
        except Exception:
            logger.exception("Failed to submit acp_run for trigger_id={}", trigger_id)
            return {"error": "submission_failed", "status": "error"}

        logger.info(
            "Webhook trigger_id={} submitted acp_run task_id={}",
            trigger_id, task_id,
        )
        return {"task_id": task_id, "status": "accepted"}

    async def _submit_acp_run(self, payload: dict[str, Any]) -> str:
        """Submit an ``acp_run`` task via the global scheduler."""
        from tldw_Server_API.app.core.Scheduler import get_global_scheduler

        scheduler = get_global_scheduler()
        task_id = await scheduler.submit(
            handler="acp_run",
            payload=payload,
            queue_name="acp",
        )
        return task_id

    # ------------------------------------------------------------------
    # CRUD operations (delegating to DB)
    # ------------------------------------------------------------------

    def create_trigger(
        self,
        name: str,
        source_type: str,
        secret: str,
        owner_user_id: int,
        agent_config: dict[str, Any] | None = None,
        prompt_template: str = "",
        enabled: bool = True,
    ) -> str:
        """Create a new webhook trigger and return its id."""
        trigger_id = str(uuid.uuid4())
        encrypted_secret = self._secret_mgr.encrypt(secret)
        self._db.create_webhook_trigger(
            trigger_id=trigger_id,
            name=name,
            source_type=source_type,
            secret_encrypted=encrypted_secret,
            owner_user_id=owner_user_id,
            agent_config_json=json.dumps(agent_config or {}),
            prompt_template=prompt_template,
            enabled=enabled,
        )
        logger.info("Created webhook trigger id={} name={!r}", trigger_id, name)
        return trigger_id

    def list_triggers(self, user_id: int) -> list[dict[str, Any]]:
        """List all triggers owned by *user_id*."""
        rows = self._db.list_webhook_triggers(user_id)
        result = []
        for row in rows:
            d = dict(row) if not isinstance(row, dict) else row
            # Parse agent_config_json
            if "agent_config_json" in d:
                try:
                    d["agent_config"] = json.loads(d.pop("agent_config_json"))
                except (json.JSONDecodeError, TypeError):
                    d["agent_config"] = {}
            d["enabled"] = bool(d.get("enabled", True))
            # Never expose the encrypted secret in list results
            d.pop("secret_encrypted", None)
            result.append(d)
        return result

    def get_trigger(self, trigger_id: str) -> dict[str, Any] | None:
        """Get a single trigger by id (includes secret_encrypted for internal use)."""
        row = self._db.get_webhook_trigger(trigger_id)
        if row is None:
            return None
        d = dict(row) if not isinstance(row, dict) else row
        if "agent_config_json" in d:
            try:
                d["agent_config"] = json.loads(d.pop("agent_config_json"))
            except (json.JSONDecodeError, TypeError):
                d["agent_config"] = {}
        d["enabled"] = bool(d.get("enabled", True))
        return d

    def update_trigger(self, trigger_id: str, **kwargs: Any) -> bool:
        """Update trigger fields.

        Accepted kwargs: name, source_type, secret (plaintext -- will be
        encrypted), agent_config (dict), prompt_template, enabled.
        """
        updates: dict[str, Any] = {}
        if "name" in kwargs:
            updates["name"] = kwargs["name"]
        if "source_type" in kwargs:
            updates["source_type"] = kwargs["source_type"]
        if "secret" in kwargs:
            updates["secret_encrypted"] = self._secret_mgr.encrypt(kwargs["secret"])
        if "agent_config" in kwargs:
            updates["agent_config_json"] = json.dumps(kwargs["agent_config"])
        if "prompt_template" in kwargs:
            updates["prompt_template"] = kwargs["prompt_template"]
        if "enabled" in kwargs:
            updates["enabled"] = 1 if kwargs["enabled"] else 0

        if not updates:
            return False
        return self._db.update_webhook_trigger(trigger_id, updates)

    def delete_trigger(self, trigger_id: str) -> bool:
        """Delete a trigger."""
        ok = self._db.delete_webhook_trigger(trigger_id)
        if ok:
            # Clean up rate limit state
            self._rate_limits.pop(trigger_id, None)
            logger.info("Deleted webhook trigger id={}", trigger_id)
        return ok

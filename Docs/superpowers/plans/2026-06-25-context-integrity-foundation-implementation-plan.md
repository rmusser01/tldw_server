# Context Integrity Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first enforceable Context Integrity foundation for prompt-bearing assets, proving signed manifests, anti-rollback policy, quarantine resolution, startup warnings, Skills enforcement, and config prompt-loader enforcement.

**Backlog:** Design `TASK-2363`; implementation planning `TASK-2365`; create or reuse a new implementation Backlog task before code edits.

**Architecture:** Add a focused `tldw_Server_API.app.core.Context_Integrity` package with pure canonicalization, manifest, verifier, resolver, and inventory modules. Wire it into startup through the existing `StartupWarningRegistry`, expose admin status/findings through the existing admin router, and add thin resolver checks to Skills and prompt loading. This first slice protects filesystem skills and config prompt files; DB prompt-version and MCP prompt-catalog enforcement are represented by interfaces and explicit follow-up tests so they can plug into the same resolver without changing the core model.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, Loguru, SQLite-backed existing DB layers, pytest, Bandit.

---

## File Structure

Create:

- `tldw_Server_API/app/core/Context_Integrity/__init__.py` - package exports for the context integrity subsystem.
- `tldw_Server_API/app/core/Context_Integrity/models.py` - dataclasses and enums for asset descriptors, manifests, findings, resolver decisions, and boot state.
- `tldw_Server_API/app/core/Context_Integrity/canonicalization.py` - deterministic filesystem and DB-prompt canonical hashing helpers.
- `tldw_Server_API/app/core/Context_Integrity/manifest.py` - signed manifest serialization, HMAC signing provider, signature verification, and anti-rollback checks.
- `tldw_Server_API/app/core/Context_Integrity/inventory.py` - filesystem inventory adapters for user skills and config prompt files.
- `tldw_Server_API/app/core/Context_Integrity/verifier.py` - manifest-versus-inventory comparison and finding generation.
- `tldw_Server_API/app/core/Context_Integrity/resolver.py` - in-memory resolver used by runtime call sites.
- `tldw_Server_API/app/services/startup_context_integrity.py` - startup producer that builds inventory, verifies manifest state, sets `app.state.context_integrity_resolver`, and emits startup warnings.
- `tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py` - admin status and findings endpoint.
- `tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py`
- `tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py`
- `tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py`
- `tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py`
- `tldw_Server_API/tests/Services/test_startup_context_integrity.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py`

Modify:

- `tldw_Server_API/app/services/lifespan_startup_sequence.py` - call the Context Integrity startup producer after core initialization and before startup blockers are evaluated.
- `tldw_Server_API/app/services/lifespan_shutdown_sequence.py` - clear global Context Integrity resolver state during shutdown.
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` - include the admin Context Integrity router.
- `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` - add admin response schemas for current-process Context Integrity status.
- `tldw_Server_API/app/core/Skills/skills_service.py` - add resolver dependency, filter quarantined skills from listings/context, and block direct content/execution access.
- `tldw_Server_API/app/api/v1/endpoints/skills.py` - map quarantine errors to deterministic HTTP errors without leaking asset content.
- `tldw_Server_API/app/core/Utils/prompt_loader.py` - enforce resolver checks for loaded prompt-file bytes with single-read semantics.
- `tldw_Server_API/tests/Skills/unit/test_skills_service.py` - cover quarantined skill filtering and blocking.
- `tldw_Server_API/tests/Skills/integration/test_skills_api.py` - cover API error mapping for quarantined skills.
- `tldw_Server_API/tests/Utils/test_prompt_loader_paths.py` - cover prompt-loader quarantine blocking and at-use verified bytes behavior.

## Pre-Execution Review Amendments

The following review fixes supersede any narrower snippets later in this plan:

- New code must preserve Python 3.10 support. Use `datetime.now(timezone.utc)`, not `datetime.UTC`.
- Startup must distinguish "no manifest loaded" from "valid empty manifest." No manifest means `degraded=True`, a `degraded_integrity` finding, and runtime context/execution reads fail closed.
- Runtime resolver checks must fail closed for:
  - any non-`audit_only` degraded state except static admin inspection paths;
  - assets with no approved digest in the verified boot state;
  - at-use digest mismatches.
- Skills discovery/context paths must rehash the exact current skill file map and call `require_digest_allowed()`. A skill added or modified after startup must not appear in `available_skills`, skill tool availability, or system-message context.
- Skills write paths (`create_skill`, `update_skill`, `import_skill`, `import_from_zip`) must still succeed under enforcement by returning a write-response snapshot that is not eligible for injection/execution until approved. Do not route write responses through the normal guarded `get_skill()` path.
- Startup must discover existing user skill roots from `DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True)` in addition to test-injected roots.
- Inventory must not follow symlinks outside the asset root. Unreadable files, path escapes, and permission errors should emit `verification_error` findings and quarantine the narrowest asset/scope, not crash startup.
- API and chat/tool paths that expose or execute full skill content must map `ContextIntegrityBlocked` to stable content-free failures, including direct get, execute, export, `context_integration.handle_skill_tool_call()`, `add_skill_tool_to_tools_list()`, and chat slash-command execution.
- Environment prompt override files are protected only after they are inventoried. This slice should inventory currently configured `TLDW_PROMPT_FILE_*` paths; otherwise they must be explicitly excluded from enforcement. The plan chooses inventory.
- The global resolver must be cleared during lifespan shutdown/test cleanup. App-state resolver injection remains preferred for request services; the global resolver is only a compatibility bridge for current non-request prompt loading.

## Task 1: Canonical Asset Models And Hashing

**Files:**
- Create: `tldw_Server_API/app/core/Context_Integrity/__init__.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/models.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/canonicalization.py`
- Create: `tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py`

- [ ] **Step 1: Write failing canonicalization tests**

Create `tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py`:

```python
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_filesystem_digest_is_stable_for_sorted_paths() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    first = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "SKILL.md": b"hello\r\n",
            "refs/notes.md": b"reference",
        },
        metadata={"context": "inline"},
    )
    second = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "refs/notes.md": b"reference",
            "SKILL.md": b"hello\r\n",
        },
        metadata={"context": "inline"},
    )

    assert first == second
    assert first.startswith("sha256:")


def test_filesystem_digest_detects_formatting_edits() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    original = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n"},
    )
    edited = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n# changed\n"},
    )

    assert original != edited


def test_db_prompt_digest_normalizes_unicode_and_line_endings() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    composed = canonical_db_prompt_digest(
        {
            "uuid": "prompt-1",
            "version": 3,
            "name": "Cafe",
            "system": "caf\u00e9\r\nline",
            "user": "body",
            "structured": {"b": 2, "a": 1},
        }
    )
    decomposed = canonical_db_prompt_digest(
        {
            "structured": {"a": 1, "b": 2},
            "user": "body",
            "system": "cafe\u0301\nline",
            "name": "Cafe",
            "version": 3,
            "uuid": "prompt-1",
        }
    )

    assert composed == decomposed
    payload = json.loads(composed.canonical_json)
    assert payload["system"] == "caf\u00e9\nline"
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Context_Integrity'`.

- [ ] **Step 3: Add models and hashing implementation**

Create `tldw_Server_API/app/core/Context_Integrity/models.py`:

```python
"""Shared models for context integrity verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ContextAssetSource = Literal["skill_file", "prompt_file", "db_prompt"]
ContextAssetState = Literal[
    "trusted",
    "changed_approved_executable",
    "changed_approved_non_executable",
    "new_unapproved",
    "missing_required",
    "missing_optional",
    "signature_invalid",
    "manifest_rollback_detected",
    "verification_error",
    "degraded_integrity",
    "quarantined",
]


@dataclass(frozen=True, slots=True)
class CanonicalDigest:
    """Canonical digest plus optional JSON payload used to produce it."""

    digest: str
    canonical_json: str = ""

    def __str__(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class ContextAssetDescriptor:
    """One prompt-bearing asset discovered by an inventory adapter."""

    asset_id: str
    source_type: ContextAssetSource
    digest: str
    display_name: str
    executable: bool = False
    required: bool = False
    owner_scope: str = "system"
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextIntegrityFinding:
    """Verification finding for one asset or source scope."""

    asset_id: str
    state: ContextAssetState
    severity: Literal["info", "warning", "error"]
    summary: str
    remediation: str
    source_type: ContextAssetSource | Literal["manifest"]
    current_digest: str | None = None
    approved_digest: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class ContextIntegrityBootState:
    """Current-process context integrity verification result."""

    mode: Literal["audit_only", "enforce", "hardened"]
    degraded: bool
    manifest_sequence: int | None
    manifest_digest: str | None
    approved_digests_by_asset_id: dict[str, str] = field(default_factory=dict)
    findings: tuple[ContextIntegrityFinding, ...] = ()
```

Create `tldw_Server_API/app/core/Context_Integrity/canonicalization.py`:

```python
"""Canonical hashing helpers for prompt-bearing assets."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Mapping

from tldw_Server_API.app.core.Context_Integrity.models import CanonicalDigest


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _normalize_text(str(value))


def _stable_json(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_json_value(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_filesystem_digest(
    *,
    source_type: str,
    asset_id: str,
    files: Mapping[str, bytes],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash raw file bytes plus deterministic identity metadata."""
    hasher = hashlib.sha256()
    identity = _stable_json(
        {
            "asset_id": asset_id,
            "source_type": source_type,
            "metadata": dict(metadata or {}),
        }
    ).encode("utf-8")
    hasher.update(len(identity).to_bytes(8, "big"))
    hasher.update(identity)
    for relative_path in sorted(files):
        path_bytes = relative_path.replace("\\", "/").encode("utf-8")
        content = files[relative_path]
        hasher.update(len(path_bytes).to_bytes(8, "big"))
        hasher.update(path_bytes)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return "sha256:" + hasher.hexdigest()


def canonical_db_prompt_digest(record: Mapping[str, Any]) -> CanonicalDigest:
    """Hash a stable prompt-version JSON representation."""
    canonical_json = _stable_json(dict(record))
    return CanonicalDigest(
        digest=_sha256(canonical_json.encode("utf-8")),
        canonical_json=canonical_json,
    )
```

Create `tldw_Server_API/app/core/Context_Integrity/__init__.py`:

```python
"""Context integrity controls for prompt-bearing assets."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Context_Integrity/__init__.py \
  tldw_Server_API/app/core/Context_Integrity/models.py \
  tldw_Server_API/app/core/Context_Integrity/canonicalization.py \
  tldw_Server_API/tests/Context_Integrity/unit/test_canonicalization.py
git commit -m "feat: add context integrity canonical hashing"
```

## Task 2: Manifest Signing And Anti-Rollback Policy

**Files:**
- Create: `tldw_Server_API/app/core/Context_Integrity/manifest.py`
- Modify: `tldw_Server_API/app/core/Context_Integrity/models.py`
- Create: `tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py`:

```python
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _entry(asset_id: str, digest: str = "sha256:a") -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "source_type": "skill_file",
        "digest": digest,
        "display_name": asset_id,
        "executable": True,
        "required": False,
        "owner_scope": "user:1",
    }


def test_manifest_roundtrip_verifies_signature() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)

    verified = verify_signed_manifest(signed, signer=signer)

    assert verified.sequence == 1
    assert verified.entries[0]["asset_id"] == "skill:user:1/demo"


def test_manifest_tamper_is_rejected() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        ManifestSignatureError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=1, entries=[_entry("skill:user:1/demo")], signer=signer)
    signed["manifest"]["entries"][0]["digest"] = "sha256:evil"

    with pytest.raises(ManifestSignatureError):
        verify_signed_manifest(signed, signer=signer)


def test_anti_rollback_anchor_rejects_older_valid_manifest() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        AntiRollbackAnchor,
        HmacManifestSigner,
        ManifestRollbackError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=2, entries=[_entry("skill:user:1/demo")], signer=signer)
    anchor = AntiRollbackAnchor(sequence=3, manifest_digest="sha256:newer")

    with pytest.raises(ManifestRollbackError):
        verify_signed_manifest(signed, signer=signer, anti_rollback_anchor=anchor)


def test_anti_rollback_anchor_rejects_same_sequence_with_different_digest() -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        AntiRollbackAnchor,
        HmacManifestSigner,
        ManifestRollbackError,
        create_signed_manifest,
        verify_signed_manifest,
    )

    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed = create_signed_manifest(sequence=3, entries=[_entry("skill:user:1/demo")], signer=signer)
    anchor = AntiRollbackAnchor(sequence=3, manifest_digest="sha256:different")

    with pytest.raises(ManifestRollbackError):
        verify_signed_manifest(signed, signer=signer, anti_rollback_anchor=anchor)
```

- [ ] **Step 2: Run failing manifest tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing manifest functions.

- [ ] **Step 3: Add manifest implementation**

Create `tldw_Server_API/app/core/Context_Integrity/manifest.py`:

```python
"""Signed context integrity manifest helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping


class ManifestSignatureError(ValueError):
    """Raised when a manifest signature cannot be verified."""


class ManifestRollbackError(ValueError):
    """Raised when a valid manifest is older than the anti-rollback anchor."""


@dataclass(frozen=True, slots=True)
class AntiRollbackAnchor:
    """Last accepted manifest identity from a non-DB trust anchor."""

    sequence: int
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    """Verified signed manifest payload."""

    sequence: int
    manifest_digest: str
    key_id: str
    entries: tuple[dict[str, Any], ...]


class HmacManifestSigner:
    """Test and deployment signer backed by an externally supplied secret."""

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if not key_id:
            raise ValueError("key_id is required")
        if not secret:
            raise ValueError("secret is required")
        self.key_id = key_id
        self._secret = secret

    def sign(self, payload: bytes) -> str:
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


def _stable_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _manifest_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def create_signed_manifest(
    *,
    sequence: int,
    entries: list[dict[str, Any]],
    signer: HmacManifestSigner,
    schema_version: int = 1,
) -> dict[str, Any]:
    manifest = {
        "schema_version": schema_version,
        "sequence": sequence,
        "entries": sorted(entries, key=lambda item: str(item["asset_id"])),
    }
    payload = _stable_json(manifest)
    return {
        "manifest": manifest,
        "signature": {
            "alg": "hmac-sha256",
            "key_id": signer.key_id,
            "value": signer.sign(payload),
        },
        "manifest_digest": _manifest_digest(payload),
    }


def verify_signed_manifest(
    signed_manifest: Mapping[str, Any],
    *,
    signer: HmacManifestSigner,
    anti_rollback_anchor: AntiRollbackAnchor | None = None,
) -> VerifiedManifest:
    manifest = signed_manifest.get("manifest")
    signature = signed_manifest.get("signature")
    if not isinstance(manifest, dict) or not isinstance(signature, dict):
        raise ManifestSignatureError("signed manifest is malformed")
    if signature.get("key_id") != signer.key_id:
        raise ManifestSignatureError("manifest key id mismatch")
    payload = _stable_json(manifest)
    expected_digest = _manifest_digest(payload)
    if signed_manifest.get("manifest_digest") != expected_digest:
        raise ManifestSignatureError("manifest digest mismatch")
    signature_value = signature.get("value")
    if not isinstance(signature_value, str) or not signer.verify(payload, signature_value):
        raise ManifestSignatureError("manifest signature mismatch")
    sequence = int(manifest.get("sequence") or 0)
    if anti_rollback_anchor and (
        sequence < anti_rollback_anchor.sequence
        or (sequence == anti_rollback_anchor.sequence and expected_digest != anti_rollback_anchor.manifest_digest)
    ):
        raise ManifestRollbackError("manifest rollback detected")
    entries = manifest.get("entries") or []
    if not isinstance(entries, list):
        raise ManifestSignatureError("manifest entries must be a list")
    return VerifiedManifest(
        sequence=sequence,
        manifest_digest=expected_digest,
        key_id=signer.key_id,
        entries=tuple(dict(item) for item in entries),
    )
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Context_Integrity/manifest.py \
  tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py
git commit -m "feat: add context integrity signed manifests"
```

## Task 3: Verifier And Runtime Resolver

**Files:**
- Create: `tldw_Server_API/app/core/Context_Integrity/verifier.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/resolver.py`
- Modify: `tldw_Server_API/app/core/Context_Integrity/models.py`
- Create: `tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py`

- [ ] **Step 1: Write failing verifier and resolver tests**

Create `tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py`:

```python
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _asset(asset_id: str, digest: str, *, executable: bool = False):
    from tldw_Server_API.app.core.Context_Integrity.models import ContextAssetDescriptor

    return ContextAssetDescriptor(
        asset_id=asset_id,
        source_type="skill_file",
        digest=digest,
        display_name=asset_id,
        executable=executable,
        owner_scope="user:1",
    )


def test_verifier_detects_changed_executable_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/demo", "sha256:current", executable=True)],
        approved_entries=[
            {
                "asset_id": "skill:user:1/demo",
                "source_type": "skill_file",
                "digest": "sha256:approved",
                "display_name": "demo",
                "executable": True,
                "required": False,
                "owner_scope": "user:1",
            }
        ],
    )

    assert len(findings) == 1
    assert findings[0].state == "changed_approved_executable"
    assert findings[0].severity == "error"


def test_verifier_detects_new_unapproved_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory

    findings = verify_inventory(
        current_assets=[_asset("skill:user:1/new", "sha256:new")],
        approved_entries=[],
    )

    assert findings[0].state == "new_unapproved"


def test_resolver_blocks_quarantined_asset() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    finding = ContextIntegrityFinding(
        asset_id="skill:user:1/demo",
        state="changed_approved_executable",
        severity="error",
        summary="changed",
        remediation="review",
        source_type="skill_file",
    )
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(finding,),
        )
    )

    with pytest.raises(ContextIntegrityBlocked, match="quarantined"):
        resolver.require_allowed("skill:user:1/demo", purpose="skill_execution")


def test_resolver_detects_live_digest_mismatch_at_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"skill:user:1/demo": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked, match="quarantined"):
        resolver.require_digest_allowed(
            "skill:user:1/demo",
            current_digest="sha256:changed",
            purpose="skill_execution",
        )


def test_resolver_blocks_unknown_asset_in_enforce_mode() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"skill:user:1/known": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked) as exc_info:
        resolver.require_allowed("skill:user:1/live-added", purpose="skill_discovery")

    assert exc_info.value.state == "new_unapproved"


def test_resolver_blocks_degraded_injection_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityBlocked,
        ContextIntegrityResolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            approved_digests_by_asset_id={"prompt_file:demo.prompts.md": "sha256:approved"},
        )
    )

    with pytest.raises(ContextIntegrityBlocked) as exc_info:
        resolver.require_allowed("prompt_file:demo.prompts.md", purpose="prompt_load")

    assert exc_info.value.state == "degraded_integrity"


def test_resolver_allows_matching_digest_at_use() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={"prompt_file:demo.prompts.md": "sha256:approved"},
        )
    )

    resolver.require_digest_allowed(
        "prompt_file:demo.prompts.md",
        current_digest="sha256:approved",
        purpose="prompt_load",
    )


def test_global_resolver_can_be_set_and_cleared() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        get_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
        )
    )
    set_global_context_integrity_resolver(resolver)
    assert get_global_context_integrity_resolver() is resolver
    clear_global_context_integrity_resolver()
    assert get_global_context_integrity_resolver() is None
```

- [ ] **Step 2: Run failing verifier tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py -v
```

Expected: FAIL with missing verifier/resolver modules.

- [ ] **Step 3: Add verifier and resolver**

Create `tldw_Server_API/app/core/Context_Integrity/verifier.py`:

```python
"""Verification of current context assets against approved manifest entries."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetDescriptor,
    ContextIntegrityFinding,
)


def _entry_by_asset_id(entries: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(entry["asset_id"]): entry for entry in entries}


def verify_inventory(
    *,
    current_assets: Iterable[ContextAssetDescriptor],
    approved_entries: Iterable[Mapping[str, Any]],
) -> tuple[ContextIntegrityFinding, ...]:
    """Compare current inventory against approved manifest entries."""
    approved = _entry_by_asset_id(approved_entries)
    current = {asset.asset_id: asset for asset in current_assets}
    findings: list[ContextIntegrityFinding] = []

    for asset_id, asset in current.items():
        entry = approved.get(asset_id)
        if entry is None:
            findings.append(
                ContextIntegrityFinding(
                    asset_id=asset_id,
                    state="new_unapproved",
                    severity="warning",
                    summary=f"Unapproved context asset detected: {asset.display_name}",
                    remediation="Review and approve the asset before model use.",
                    source_type=asset.source_type,
                    current_digest=asset.digest,
                )
            )
            continue
        if str(entry.get("digest")) != asset.digest:
            state = "changed_approved_executable" if asset.executable else "changed_approved_non_executable"
            findings.append(
                ContextIntegrityFinding(
                    asset_id=asset_id,
                    state=state,
                    severity="error" if asset.executable else "warning",
                    summary=f"Approved context asset changed: {asset.display_name}",
                    remediation="Review the diff and approve a new manifest version or restore the asset.",
                    source_type=asset.source_type,
                    current_digest=asset.digest,
                    approved_digest=str(entry.get("digest")),
                )
            )

    for asset_id, entry in approved.items():
        if asset_id in current:
            continue
        required = bool(entry.get("required", False))
        findings.append(
            ContextIntegrityFinding(
                asset_id=asset_id,
                state="missing_required" if required else "missing_optional",
                severity="error" if required else "warning",
                summary=f"Approved context asset is missing: {entry.get('display_name') or asset_id}",
                remediation="Restore the asset or approve a manifest that removes it.",
                source_type=str(entry.get("source_type", "skill_file")),  # type: ignore[arg-type]
                approved_digest=str(entry.get("digest")),
            )
        )

    return tuple(findings)
```

Create `tldw_Server_API/app/core/Context_Integrity/resolver.py`:

```python
"""Runtime resolver for context integrity enforcement."""

from __future__ import annotations

from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextIntegrityBootState,
    ContextIntegrityFinding,
)


class ContextIntegrityBlocked(RuntimeError):
    """Raised when a prompt-bearing asset is quarantined or unavailable."""

    def __init__(self, asset_id: str, state: str) -> None:
        self.asset_id = asset_id
        self.state = state
        super().__init__("Asset is quarantined pending admin review.")


class ContextIntegrityResolver:
    """Current-process resolver backed by verified boot state."""

    def __init__(self, boot_state: ContextIntegrityBootState) -> None:
        self.boot_state = boot_state
        self._findings_by_asset_id: dict[str, ContextIntegrityFinding] = {
            finding.asset_id: finding for finding in boot_state.findings
        }

    def finding_for(self, asset_id: str) -> ContextIntegrityFinding | None:
        return self._findings_by_asset_id.get(asset_id)

    def require_allowed(self, asset_id: str, *, purpose: str) -> None:
        if self.boot_state.mode != "audit_only" and self.boot_state.degraded:
            if not purpose.startswith("admin_review"):
                raise ContextIntegrityBlocked(asset_id=asset_id, state="degraded_integrity")
        finding = self.finding_for(asset_id)
        if finding is None:
            if self.boot_state.mode == "audit_only":
                return
            if asset_id not in self.boot_state.approved_digests_by_asset_id:
                raise ContextIntegrityBlocked(asset_id=asset_id, state="new_unapproved")
            return
        if self.boot_state.mode == "audit_only":
            return
        raise ContextIntegrityBlocked(asset_id=asset_id, state=finding.state)

    def require_digest_allowed(
        self,
        asset_id: str,
        *,
        current_digest: str,
        purpose: str,
        changed_state: str = "changed_approved_executable",
    ) -> None:
        self.require_allowed(asset_id, purpose=purpose)
        if self.boot_state.mode == "audit_only":
            return
        approved_digest = self.boot_state.approved_digests_by_asset_id.get(asset_id)
        if approved_digest is None:
            raise ContextIntegrityBlocked(asset_id=asset_id, state="new_unapproved")
        if approved_digest != current_digest:
            raise ContextIntegrityBlocked(asset_id=asset_id, state=changed_state)


_global_resolver: ContextIntegrityResolver | None = None


def set_global_context_integrity_resolver(resolver: ContextIntegrityResolver | None) -> None:
    global _global_resolver
    _global_resolver = resolver


def get_global_context_integrity_resolver() -> ContextIntegrityResolver | None:
    return _global_resolver


def clear_global_context_integrity_resolver() -> None:
    set_global_context_integrity_resolver(None)
```

- [ ] **Step 4: Run verifier/resolver tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Context_Integrity/verifier.py \
  tldw_Server_API/app/core/Context_Integrity/resolver.py \
  tldw_Server_API/tests/Context_Integrity/unit/test_verifier_resolver.py
git commit -m "feat: add context integrity verifier resolver"
```

## Task 4: Filesystem Inventory Adapters

**Files:**
- Create: `tldw_Server_API/app/core/Context_Integrity/inventory.py`
- Create: `tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py`

- [ ] **Step 1: Write failing inventory tests**

Create `tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py`:

```python
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_inventory_user_skill_directory_includes_supporting_files(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill_dir / "ref.md").write_text("reference", encoding="utf-8")

    assets = inventory_user_skills(user_id=1, skills_root=tmp_path / "skills")

    assert len(assets) == 1
    assert assets[0].asset_id == "skill:user:1/demo"
    assert assets[0].executable is True
    assert assets[0].source_type == "skill_file"


def test_inventory_prompt_files_finds_supported_extensions(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_prompt_files

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: prompt", encoding="utf-8")
    (prompts / "ignore.bin").write_bytes(b"no")

    assets = inventory_prompt_files(prompts_dir=prompts)

    assert [asset.asset_id for asset in assets] == ["prompt_file:rag.prompts.yaml"]
    assert assets[0].executable is False


def test_inventory_env_prompt_overrides_finds_configured_files(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_env_prompt_overrides

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")

    assets = inventory_env_prompt_overrides(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert [asset.asset_id for asset in assets] == [
        "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    ]
    assert assets[0].metadata["source_label"] == "env:TLDW_PROMPT_FILE_CHAT__SYSTEM"
```

- [ ] **Step 2: Run failing inventory tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py -v
```

Expected: FAIL with missing `inventory.py`.

- [ ] **Step 3: Add inventory implementation**

Create `tldw_Server_API/app/core/Context_Integrity/inventory.py`:

```python
"""Filesystem inventory adapters for Context Integrity."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.models import ContextAssetDescriptor

_PROMPT_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".txt"}
_SKILL_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"}


def _read_file_map(root: Path, suffixes: set[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = path.read_bytes()
    return files


def inventory_user_skills(*, user_id: int, skills_root: Path) -> list[ContextAssetDescriptor]:
    """Inventory per-user skill directories."""
    if not skills_root.exists():
        return []
    assets: list[ContextAssetDescriptor] = []
    for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        files = _read_file_map(skill_dir, _SKILL_TEXT_SUFFIXES)
        asset_id = f"skill:user:{user_id}/{skill_dir.name}"
        digest = canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=asset_id,
            files=files,
            metadata={"skill_name": skill_dir.name},
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="skill_file",
                digest=digest,
                display_name=skill_dir.name,
                executable=True,
                owner_scope=f"user:{user_id}",
                path=str(skill_dir),
            )
        )
    return assets


def inventory_prompt_files(*, prompts_dir: Path) -> list[ContextAssetDescriptor]:
    """Inventory config prompt files under a Prompts directory."""
    if not prompts_dir.exists():
        return []
    assets: list[ContextAssetDescriptor] = []
    for path in sorted(prompts_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _PROMPT_SUFFIXES:
            continue
        relative = path.name
        asset_id = f"prompt_file:{relative}"
        digest = canonical_filesystem_digest(
            source_type="prompt_file",
            asset_id=asset_id,
            files={relative: path.read_bytes()},
            metadata={"path": relative},
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="prompt_file",
                digest=digest,
                display_name=relative,
                executable=False,
                owner_scope="system",
                path=str(path),
            )
        )
    return assets


def inventory_env_prompt_overrides(
    *,
    environ: Mapping[str, str] | None = None,
) -> list[ContextAssetDescriptor]:
    """Inventory configured prompt override files from TLDW_PROMPT_FILE_* vars."""
    source = environ if environ is not None else os.environ
    assets: list[ContextAssetDescriptor] = []
    for env_name, raw_path in sorted(source.items()):
        if not env_name.startswith("TLDW_PROMPT_FILE_") or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
        source_label = f"env:{env_name}"
        asset_id = f"prompt_file:{source_label}:{path.name}"
        digest = canonical_filesystem_digest(
            source_type="prompt_file",
            asset_id=asset_id,
            files={path.name: path.read_bytes()},
            metadata={"path": str(path), "source_label": source_label},
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="prompt_file",
                digest=digest,
                display_name=env_name,
                executable=False,
                owner_scope="system",
                path=str(path),
                metadata={"source_label": source_label},
            )
        )
    return assets
```

- [ ] **Step 4: Run inventory tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Context_Integrity/inventory.py \
  tldw_Server_API/tests/Context_Integrity/unit/test_inventory.py
git commit -m "feat: add context integrity filesystem inventory"
```

## Task 5: Startup Verification Producer

**Status**: Complete. Implemented in `9727db3f28` with no-create discovery review fix in `a14388fd51`.

**Files:**
- Create: `tldw_Server_API/app/services/startup_context_integrity.py`
- Modify: `tldw_Server_API/app/services/lifespan_startup_sequence.py`
- Modify: `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
- Create: `tldw_Server_API/tests/Services/test_startup_context_integrity.py`
- Modify: `tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py`

- [ ] **Step 1: Write failing startup producer tests**

Create `tldw_Server_API/tests/Services/test_startup_context_integrity.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_startup_context_integrity_sets_resolver_and_warning(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import StartupWarningRegistry

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: changed", encoding="utf-8")

    registry = StartupWarningRegistry(startup_id="boot-1")
    app_state = type("State", (), {})()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
        mode="enforce",
    )

    assert len(findings) == 2
    assert registry.summary(component_prefix="context_integrity")["total"] == 2
    assert app_state.context_integrity_boot_state.degraded is True
    assert any(finding.state == "degraded_integrity" for finding in findings)
    assert app_state.context_integrity_resolver.finding_for("prompt_file:rag.prompts.yaml") is not None
```

- [ ] **Step 2: Run failing startup producer tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_context_integrity.py -v
```

Expected: FAIL with missing `startup_context_integrity.py`.

- [ ] **Step 3: Add startup producer**

Create `tldw_Server_API/app/services/startup_context_integrity.py`:

```python
"""Startup producer for Context Integrity warnings and resolver state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Context_Integrity.inventory import (
    inventory_env_prompt_overrides,
    inventory_prompt_files,
    inventory_user_skills,
)
from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetDescriptor,
    ContextIntegrityBootState,
    ContextIntegrityFinding,
)
from tldw_Server_API.app.core.Context_Integrity.manifest import (
    HmacManifestSigner,
    verify_signed_manifest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityResolver,
    set_global_context_integrity_resolver,
)
from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory
from tldw_Server_API.app.core.config_paths import resolve_prompts_dir
from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord
from tldw_Server_API.app.services.startup_warning_registry import StartupWarningRegistry


def _load_startup_manifest_from_env() -> tuple[
    tuple[Mapping[str, object], ...],
    int | None,
    str | None,
    bool,
    ContextIntegrityFinding | None,
]:
    """Load an operator-provided signed manifest from environment configuration."""
    manifest_path = os.getenv("CONTEXT_INTEGRITY_MANIFEST_PATH")
    secret = os.getenv("CONTEXT_INTEGRITY_HMAC_SECRET")
    key_id = os.getenv("CONTEXT_INTEGRITY_HMAC_KEY_ID") or "local-hmac"
    if not manifest_path or not secret:
        return (), None, None, False, None
    try:
        signed_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        verified = verify_signed_manifest(
            signed_manifest,
            signer=HmacManifestSigner(key_id=key_id, secret=secret.encode("utf-8")),
        )
        return verified.entries, verified.sequence, verified.manifest_digest, True, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            (),
            None,
            None,
            False,
            ContextIntegrityFinding(
                asset_id="manifest:env",
                state="signature_invalid",
                severity="error",
                summary="Configured Context Integrity manifest could not be verified.",
                remediation="Fix the manifest path/key or import a valid admin-held manifest.",
                source_type="manifest",
                details={"error_type": exc.__class__.__name__},
            ),
        )


def _discover_user_skill_roots() -> list[tuple[int, Path]]:
    """Discover existing per-user skill roots without creating directories."""
    roots: list[tuple[int, Path]] = []
    try:
        base_dir = DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True)
    except Exception as exc:
        logger.warning("Context Integrity could not discover user skill roots: {}", exc)
        return roots
    if not base_dir.exists():
        return roots
    for user_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        if not user_dir.name.isdigit():
            continue
        skills_root = user_dir / "skills"
        if skills_root.is_dir():
            roots.append((int(user_dir.name), skills_root))
    return roots


def _finding_to_warning(finding: ContextIntegrityFinding) -> StartupWarningRecord:
    action = "warn"
    return StartupWarningRecord(
        component=f"context_integrity.{finding.source_type}",
        severity=finding.severity,
        startup_action=action,
        code=finding.state,
        summary=finding.summary,
        remediation=finding.remediation,
        details={
            "asset_id": finding.asset_id,
            "current_digest": finding.current_digest,
            "approved_digest": finding.approved_digest,
        },
        detected_at=finding.detected_at,
    )


def produce_context_integrity_startup_warnings(
    *,
    app_state: object,
    registry: StartupWarningRegistry,
    prompts_dir: Path | None = None,
    user_skill_roots: Iterable[tuple[int, Path]] | None = None,
    approved_entries: Iterable[Mapping[str, object]] | None = None,
    manifest_loaded: bool | None = None,
    manifest_sequence: int | None = None,
    manifest_digest: str | None = None,
    mode: str = "enforce",
) -> tuple[ContextIntegrityFinding, ...]:
    """Build startup inventory, register warnings, and attach resolver state."""
    current_assets: list[ContextAssetDescriptor] = []
    inventory_findings: list[ContextIntegrityFinding] = []
    try:
        current_assets.extend(inventory_prompt_files(prompts_dir=prompts_dir or resolve_prompts_dir()))
        current_assets.extend(inventory_env_prompt_overrides())
    except OSError as exc:
        inventory_findings.append(
            ContextIntegrityFinding(
                asset_id="prompt_file:*",
                state="verification_error",
                severity="error",
                summary="Prompt file inventory failed.",
                remediation="Fix file permissions or remove unsafe prompt paths, then restart.",
                source_type="prompt_file",
                details={"error_type": exc.__class__.__name__},
            )
        )
    roots = list(user_skill_roots) if user_skill_roots is not None else _discover_user_skill_roots()
    for user_id, skills_root in roots:
        try:
            current_assets.extend(inventory_user_skills(user_id=user_id, skills_root=skills_root))
        except OSError as exc:
            inventory_findings.append(
                ContextIntegrityFinding(
                    asset_id=f"skill:user:{user_id}:*",
                    state="verification_error",
                    severity="error",
                    summary=f"Skill inventory failed for user {user_id}.",
                    remediation="Fix file permissions or remove unsafe skill paths, then restart.",
                    source_type="skill_file",
                    details={"error_type": exc.__class__.__name__},
                )
            )

    manifest_finding: ContextIntegrityFinding | None = None
    if approved_entries is None:
        (
            approved_entries_tuple,
            manifest_sequence,
            manifest_digest,
            loaded_from_env,
            manifest_finding,
        ) = _load_startup_manifest_from_env()
        manifest_loaded = loaded_from_env
    else:
        approved_entries_tuple = tuple(approved_entries)
        manifest_loaded = bool(manifest_loaded)
    findings_list = list(inventory_findings)
    if manifest_finding is not None:
        findings_list.append(manifest_finding)
    findings_list.extend(
        verify_inventory(
            current_assets=current_assets,
            approved_entries=approved_entries_tuple,
        )
    )
    degraded = not manifest_loaded
    if degraded:
        findings_list.append(
            ContextIntegrityFinding(
                asset_id="manifest:none",
                state="degraded_integrity",
                severity="error",
                summary="No approved Context Integrity manifest was loaded.",
                remediation="Import or approve a signed manifest before protected assets are used.",
                source_type="manifest",
            )
        )
    findings = tuple(findings_list)
    approved_digests_by_asset_id = {
        str(entry["asset_id"]): str(entry["digest"])
        for entry in approved_entries_tuple
        if "asset_id" in entry and "digest" in entry
    }
    boot_state = ContextIntegrityBootState(
        mode=mode,  # type: ignore[arg-type]
        degraded=degraded,
        manifest_sequence=manifest_sequence,
        manifest_digest=manifest_digest,
        approved_digests_by_asset_id=approved_digests_by_asset_id,
        findings=findings,
    )
    resolver = ContextIntegrityResolver(boot_state)
    setattr(app_state, "context_integrity_resolver", resolver)
    setattr(app_state, "context_integrity_boot_state", boot_state)
    set_global_context_integrity_resolver(resolver)

    for finding in findings:
        registry.add_warning(_finding_to_warning(finding))
        logger.warning("Context integrity startup finding: {} {}", finding.state, finding.asset_id)
    return findings
```

- [ ] **Step 4: Wire lifespan startup**

Modify `tldw_Server_API/app/services/lifespan_startup_sequence.py`:

```python
def _run_startup_warning_producers(*, app: Any, startup_core_handles: Any) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = app.state.startup_warning_registry
    produce_context_integrity_startup_warnings(
        app_state=app.state,
        registry=registry,
    )
    produce_sandbox_startup_warnings(
        orchestrator=getattr(startup_core_handles, "startup_sandbox_orchestrator", None),
        registry=registry,
    )
```

- [ ] **Step 5: Clear resolver on shutdown**

Modify `tldw_Server_API/app/services/lifespan_shutdown_sequence.py` near the start of `run_lifespan_shutdown_sequence`:

```python
    try:
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            clear_global_context_integrity_resolver,
        )

        clear_global_context_integrity_resolver()
        if hasattr(app.state, "context_integrity_resolver"):
            delattr(app.state, "context_integrity_resolver")
        if hasattr(app.state, "context_integrity_boot_state"):
            delattr(app.state, "context_integrity_boot_state")
    except startup_guard_exceptions:
        pass
```

- [ ] **Step 6: Update lifespan test expectation**

Modify `test_startup_initializes_registry_and_runs_sandbox_producer` in `tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py` to monkeypatch the new producer:

```python
from tldw_Server_API.app.services import startup_context_integrity

context_calls: list[dict[str, object]] = []

def _fake_produce_context_integrity_startup_warnings(**kwargs):
    context_calls.append(kwargs)
    return []

monkeypatch.setattr(
    startup_context_integrity,
    "produce_context_integrity_startup_warnings",
    _fake_produce_context_integrity_startup_warnings,
)

assert context_calls == [{"app_state": app.state, "registry": registry}]
```

- [ ] **Step 7: Run startup tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Services/test_startup_context_integrity.py \
  tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py \
  tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add tldw_Server_API/app/services/startup_context_integrity.py \
  tldw_Server_API/app/services/lifespan_startup_sequence.py \
  tldw_Server_API/app/services/lifespan_shutdown_sequence.py \
  tldw_Server_API/tests/Services/test_startup_context_integrity.py \
  tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py \
  tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py
git commit -m "feat: wire context integrity startup checks"
```

## Task 6: Skills Enforcement

**Status**: Complete. Implemented with an additional manual review correction: default `SkillsService` instances now honor degraded global resolver state, and symlinked skill directories are skipped/rejected at runtime.

**Verification**:
- RED run for review regressions: `.venv/bin/python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_degraded_global_resolver_blocks_default_service tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_symlinked_skill_directory_is_not_read_without_resolver -q` failed before the correction with both tests failing.
- Focused Task 6 suite: `.venv/bin/python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q` passed with `133 passed, 6 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Skills/skills_service.py tldw_Server_API/app/core/Skills/context_integration.py tldw_Server_API/app/core/Chat/command_router.py tldw_Server_API/app/api/v1/endpoints/skills.py -f json -o /tmp/bandit_context_integrity_task6.json` exited 0 with zero findings.
- Formatter/whitespace: `.venv/bin/python -m black ...` completed; `git diff --check` clean.

**Files:**
- Modify: `tldw_Server_API/app/core/Skills/skills_service.py`
- Modify: `tldw_Server_API/app/core/Skills/context_integration.py`
- Modify: `tldw_Server_API/app/core/Chat/command_router.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify: `tldw_Server_API/tests/Skills/unit/test_skills_service.py`
- Modify: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- Modify: `tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`

- [ ] **Step 1: Add failing service tests for quarantined skills**

Append to `TestSkillsService` in `tldw_Server_API/tests/Skills/unit/test_skills_service.py`:

```python
    @pytest.mark.asyncio
    async def test_quarantined_skill_is_filtered_from_context(self, service):
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
            ContextIntegrityFinding,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

        await service.create_skill("blocked-skill", "---\ndescription: Blocked\n---\nBody")
        finding = ContextIntegrityFinding(
            asset_id="skill:user:1/blocked-skill",
            state="changed_approved_executable",
            severity="error",
            summary="changed",
            remediation="review",
            source_type="skill_file",
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                findings=(finding,),
            )
        )

        payload = service.get_context_payload()

        assert payload["available_skills"] == []
        assert "blocked-skill" not in payload["context_text"]

    @pytest.mark.asyncio
    async def test_quarantined_skill_get_is_blocked(self, service):
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
            ContextIntegrityFinding,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        await service.create_skill("blocked-skill", "Body")
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                findings=(
                    ContextIntegrityFinding(
                        asset_id="skill:user:1/blocked-skill",
                        state="changed_approved_executable",
                        severity="error",
                        summary="changed",
                        remediation="review",
                        source_type="skill_file",
                    ),
                ),
            )
        )

        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("blocked-skill")

    @pytest.mark.asyncio
    async def test_live_skill_edit_after_boot_is_blocked(self, service):
        from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills
        from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        await service.create_skill("live-skill", "---\ndescription: Live\n---\nOriginal")
        asset = inventory_user_skills(user_id=1, skills_root=service.skills_dir)[0]
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={asset.asset_id: asset.digest},
            )
        )
        (service.skills_dir / "live-skill" / "SKILL.md").write_text(
            "---\ndescription: Live\n---\nModified",
            encoding="utf-8",
        )

        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("live-skill")

    @pytest.mark.asyncio
    async def test_live_skill_edit_after_boot_is_filtered_from_context(self, service):
        from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills
        from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
        from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityResolver

        await service.create_skill("context-skill", "---\ndescription: Safe\n---\nOriginal")
        asset = inventory_user_skills(user_id=1, skills_root=service.skills_dir)[0]
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={asset.asset_id: asset.digest},
            )
        )
        (service.skills_dir / "context-skill" / "SKILL.md").write_text(
            "---\ndescription: Backdoored\n---\nModified",
            encoding="utf-8",
        )

        payload = service.get_context_payload()

        assert payload["available_skills"] == []
        assert "Backdoored" not in payload["context_text"]

    @pytest.mark.asyncio
    async def test_create_skill_returns_write_response_under_enforce(self, service):
        from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={},
            )
        )

        created = await service.create_skill("pending-skill", "---\ndescription: Pending\n---\nBody")

        assert created["name"] == "pending-skill"
        assert created["description"] == "Pending"
        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("pending-skill")
```

- [ ] **Step 2: Run failing skills service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_quarantined_skill_is_filtered_from_context \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_quarantined_skill_get_is_blocked \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_live_skill_edit_after_boot_is_blocked \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_live_skill_edit_after_boot_is_filtered_from_context \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_create_skill_returns_write_response_under_enforce -v
```

Expected: FAIL because `SkillsService` does not have integrity resolver logic.

- [ ] **Step 3: Add resolver support to SkillsService**

Modify `SkillsService.__init__` in `tldw_Server_API/app/core/Skills/skills_service.py`:

```python
from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    ContextIntegrityResolver,
    get_global_context_integrity_resolver,
)
```

Add parameter and attribute:

```python
        integrity_resolver: ContextIntegrityResolver | None = None,
```

Inside `__init__`:

```python
        self.integrity_resolver = integrity_resolver or get_global_context_integrity_resolver()
```

Add helper methods:

```python
    def _skill_asset_id(self, name: str) -> str:
        return f"skill:user:{self.user_id}/{name}"

    def _read_skill_file_map(self, skill_dir: Path) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"):
                continue
            files[path.relative_to(skill_dir).as_posix()] = path.read_bytes()
        return files

    def _skill_digest(self, name: str, skill_dir: Path, files: dict[str, bytes]) -> str:
        return canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=self._skill_asset_id(name),
            files=files,
            metadata={"skill_name": name},
        )

    def _is_skill_allowed(self, name: str, *, purpose: str) -> bool:
        if self.integrity_resolver is None:
            return True
        try:
            skill_dir = self._get_skill_dir(name)
            files = self._read_skill_file_map(skill_dir)
            current_digest = self._skill_digest(name, skill_dir, files)
            self.integrity_resolver.require_digest_allowed(
                self._skill_asset_id(name),
                current_digest=current_digest,
                purpose=purpose,
            )
            return True
        except (ContextIntegrityBlocked, OSError, UnicodeDecodeError):
            return False

    def _require_skill_allowed(
        self,
        name: str,
        *,
        purpose: str,
        current_digest: str | None = None,
    ) -> None:
        if self.integrity_resolver is not None:
            asset_id = self._skill_asset_id(name)
            if current_digest is None:
                self.integrity_resolver.require_allowed(asset_id, purpose=purpose)
            else:
                self.integrity_resolver.require_digest_allowed(
                    asset_id,
                    current_digest=current_digest,
                    purpose=purpose,
                )

    def _parse_unchecked_skill_directory(
        self,
        name: str,
        skill_dir: Path,
        files: dict[str, bytes] | None = None,
    ):
        if files is None:
            files = self._read_skill_file_map(skill_dir)
        if "SKILL.md" not in files:
            raise SkillNotFoundError(name, detail="SKILL.md not found")
        raw_skill = files["SKILL.md"].decode("utf-8")
        parsed = self._parser.parse_content(raw_skill, default_name=name)
        parsed.supporting_files = {
            relative_path: content.decode("utf-8")
            for relative_path, content in files.items()
            if relative_path != "SKILL.md"
        }
        return parsed

    def _parse_verified_skill_directory(self, name: str, skill_dir: Path):
        files = self._read_skill_file_map(skill_dir)
        current_digest = self._skill_digest(name, skill_dir, files)
        self._require_skill_allowed(name, purpose="skill_read", current_digest=current_digest)
        return self._parse_unchecked_skill_directory(name, skill_dir, files=files)
```

Add an `enforce_integrity: bool = True` keyword-only parameter to `get_skill`. When `True`, replace `self._parser.parse_directory(skill_dir)` with `self._parse_verified_skill_directory(name, skill_dir)`. When `False`, use `_parse_unchecked_skill_directory(name, skill_dir)` and return the response only to the write caller.

Update `create_skill`, `update_skill`, `import_skill`, and `import_from_zip` write-return paths to call `await self.get_skill(name, enforce_integrity=False)` after a successful write. Direct API reads, execution, exports, context injection, and tool use must keep the default `enforce_integrity=True`.

Filter `list_skills`, `get_context_payload`, and `get_context_payload_async` so blocked skills are excluded:

```python
        return [
            self._metadata_from_row(row)
            for row in rows
            if self._is_skill_allowed(str(row.get("name") or ""), purpose="skill_discovery")
        ]
```

- [ ] **Step 4: Map blocked skills in API endpoints**

Modify `tldw_Server_API/app/api/v1/endpoints/skills.py` imports:

```python
from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
```

In `get_skill`, `export_skill`, and `execute_skill`, add:

```python
    except ContextIntegrityBlocked as e:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=str(e),
        ) from e
```

- [ ] **Step 5: Add API regression test**

Append to `tldw_Server_API/tests/Skills/integration/test_skills_api.py`:

```python
    def test_get_quarantined_skill_returns_423(self, client, monkeypatch):
        from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
        from tldw_Server_API.app.core.Skills.skills_service import SkillsService

        async def _blocked(self, name):
            raise ContextIntegrityBlocked(asset_id=f"skill:user:1/{name}", state="changed_approved_executable")

        monkeypatch.setattr(SkillsService, "get_skill", _blocked)

        response = client.get(f"{SKILLS_PREFIX}/blocked-skill")

        assert response.status_code == 423
        assert response.json()["detail"] == "Asset is quarantined pending admin review."

    def test_export_quarantined_skill_returns_423(self, client, monkeypatch):
        from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
        from tldw_Server_API.app.core.Skills.skills_service import SkillsService

        async def _blocked(self, name):
            raise ContextIntegrityBlocked(asset_id=f"skill:user:1/{name}", state="changed_approved_executable")

        monkeypatch.setattr(SkillsService, "export_skill", _blocked)

        response = client.get(f"{SKILLS_PREFIX}/blocked-skill/export")

        assert response.status_code == 423
        assert response.json()["detail"] == "Asset is quarantined pending admin review."
```

- [ ] **Step 6: Run focused Skills tests**

Append to `tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py`:

```python
@pytest.mark.asyncio
async def test_handle_skill_tool_call_returns_content_free_integrity_error(tmp_path):
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        set_global_context_integrity_resolver,
        clear_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Skills.context_integration import handle_skill_tool_call

    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(
                ContextIntegrityFinding(
                    asset_id="skill:user:1/blocked-skill",
                    state="changed_approved_executable",
                    severity="error",
                    summary="changed",
                    remediation="review",
                    source_type="skill_file",
                ),
            ),
        )
    )
    set_global_context_integrity_resolver(resolver)
    try:
        result = await handle_skill_tool_call(
            skill_name="blocked-skill",
            args="",
            user_id=1,
            base_path=tmp_path,
            db=None,
        )
    finally:
        clear_global_context_integrity_resolver()

    assert result == {
        "success": False,
        "error": "context_integrity_blocked",
    }
```

Modify `tldw_Server_API/app/core/Skills/context_integration.py`:

```python
from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
```

Add `except ContextIntegrityBlocked` before the existing `SkillsError` handler in `handle_skill_tool_call()`:

```python
    except ContextIntegrityBlocked:
        logger.warning("Blocked quarantined skill invocation: {}", skill_name)
        return {
            "success": False,
            "error": "context_integrity_blocked",
        }
```

`add_skill_tool_to_tools_list()` already calls `service.get_context_payload()`; after `_is_skill_allowed()` rehashes current files, quarantined or live-edited skills must not cause the `Skill` tool to be advertised.

Append to `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`:

```python
@pytest.mark.asyncio
async def test_skill_command_reports_context_integrity_block(monkeypatch):
    async def fake_exec(ctx, skill_name, skill_args):
        return {"success": False, "error": "context_integrity_blocked"}

    monkeypatch.setattr(command_router, "_execute_skill", fake_exec)
    ctx = command_router.CommandContext(user_id="u1", auth_user_id=1)

    res = await command_router.async_dispatch_command(ctx, "skill", "blocked-skill x")

    assert not res.ok
    assert "quarantined pending admin review" in res.content
    assert res.metadata["error"] == "context_integrity_blocked"
```

Modify `tldw_Server_API/app/core/Chat/command_router.py`:

```python
from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
```

Catch `ContextIntegrityBlocked` in `_execute_skill()` before `SkillsError`:

```python
    except ContextIntegrityBlocked:
        return {"success": False, "error": "context_integrity_blocked"}
```

Handle the stable error in `_skill_handler()`:

```python
        elif error == "context_integrity_blocked":
            message = "Skill is quarantined pending admin review."
```

- [ ] **Step 6: Run focused Skills tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add tldw_Server_API/app/core/Skills/skills_service.py \
  tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/app/core/Chat/command_router.py \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py
git commit -m "feat: enforce context integrity for skills"
```

## Task 7: Prompt Loader Enforcement With Single-Read Semantics

**Status**: Complete. Implemented single-read prompt-file verification for Markdown, YAML, JSON, and environment override prompt files.

**Verification**:
- RED run: `.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py -q` failed before implementation with the three new prompt-loader integrity tests failing.
- Focused Task 7 suite: `.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py -q` passed with `10 passed, 6 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Utils/prompt_loader.py -f json -o /tmp/bandit_context_integrity_task7.json` exited 0 with zero findings.
- Formatter/whitespace: `.venv/bin/python -m black ...` completed; `git diff --check` clean.

**Files:**
- Modify: `tldw_Server_API/app/core/Utils/prompt_loader.py`
- Modify: `tldw_Server_API/tests/Utils/test_prompt_loader_paths.py`

- [ ] **Step 1: Add failing prompt loader tests**

Append to `tldw_Server_API/tests/Utils/test_prompt_loader_paths.py`:

```python
def test_load_prompt_blocks_quarantined_prompt_file(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        set_global_context_integrity_resolver,
        clear_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "demo.prompts.md").write_text("# Existing Key\n```\nfrom-md\n```\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(
                ContextIntegrityFinding(
                    asset_id="prompt_file:demo.prompts.md",
                    state="changed_approved_non_executable",
                    severity="warning",
                    summary="changed",
                    remediation="review",
                    source_type="prompt_file",
                ),
            ),
        )
    )
    set_global_context_integrity_resolver(resolver)
    try:
        assert pl.load_prompt("demo", "Existing Key") is None
    finally:
        clear_global_context_integrity_resolver()


def test_load_prompt_uses_verified_bytes_without_second_read(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    prompt_file = prompts / "demo.prompts.md"
    prompt_file.write_text("# Existing Key\n```\nfrom-md\n```\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))

    read_count = {"count": 0}
    original_read_text = pl._read_prompt_file_text

    def _counting_read(path):
        read_count["count"] += 1
        return original_read_text(path)

    monkeypatch.setattr(pl, "_read_prompt_file_text", _counting_read)

    assert pl.load_prompt("demo", "Existing Key") == "from-md"
    assert read_count["count"] == 1


def test_load_prompt_blocks_live_edit_after_boot(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_prompt_files
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        clear_global_context_integrity_resolver,
        ContextIntegrityResolver,
        set_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    prompt_file = prompts / "demo.prompts.md"
    prompt_file.write_text("# Existing Key\n```\nfrom-md\n```\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    asset = inventory_prompt_files(prompts_dir=prompts)[0]
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={asset.asset_id: asset.digest},
        )
    )
    prompt_file.write_text("# Existing Key\n```\nmodified\n```\n", encoding="utf-8")
    set_global_context_integrity_resolver(resolver)
    try:
        assert pl.load_prompt("demo", "Existing Key") is None
    finally:
        clear_global_context_integrity_resolver()
```

- [ ] **Step 2: Run failing prompt loader tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py -v
```

Expected: FAIL because `_read_prompt_file_text` and resolver checks do not exist.

- [ ] **Step 3: Add prompt-loader read helper and resolver guard**

Modify `tldw_Server_API/app/core/Utils/prompt_loader.py` imports:

```python
from pathlib import Path

from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    get_global_context_integrity_resolver,
)
```

Add helper:

```python
def _prompt_asset_id(path: str, *, source_label: str | None = None) -> str:
    filename = Path(path).name
    if source_label:
        return f"prompt_file:{source_label}:{filename}"
    return f"prompt_file:{filename}"


def _read_prompt_file_text(path: str, *, source_label: str | None = None) -> str:
    prompt_path = Path(path)
    asset_id = _prompt_asset_id(path, source_label=source_label)
    with open(path, "rb") as f:
        raw = f.read()
    metadata = {"path": prompt_path.name}
    if source_label is not None:
        metadata = {"path": str(prompt_path), "source_label": source_label}
    current_digest = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id=asset_id,
        files={prompt_path.name: raw},
        metadata=metadata,
    )
    resolver = get_global_context_integrity_resolver()
    if resolver is not None:
        resolver.require_digest_allowed(
            asset_id,
            current_digest=current_digest,
            purpose="prompt_load",
            changed_state="changed_approved_non_executable",
        )
    return raw.decode("utf-8")
```

Update `_load_env_prompt_file`, `_load_yaml`, `_load_json`, and markdown loading to read once through `_read_prompt_file_text`. Environment override files are prompt-bearing sources too; if enforcement is enabled and no manifest entry approves the override path, the resolver blocks them as `new_unapproved`.

For environment override files:

```python
    try:
        return _read_prompt_file_text(path, source_label=f"env:{env_name}").strip()
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked) as exc:
        logger.warning(
            "Prompt override file read failed for env '{}' (module='{}', key='{}', error_type='{}')",
            env_name,
            module,
            key,
            exc.__class__.__name__,
        )
        return None
```

For YAML:

```python
raw = _read_prompt_file_text(path)
data = yaml.safe_load(raw)
```

For JSON:

```python
raw = _read_prompt_file_text(path)
data = json.loads(raw)
```

For markdown:

```python
try:
    text = _read_prompt_file_text(md_path)
except (OSError, UnicodeDecodeError, ContextIntegrityBlocked):
    text = ""
```

Catch `ContextIntegrityBlocked` and `UnicodeDecodeError` in YAML/JSON helpers and return `None` without logging untrusted content. Because `_read_prompt_file_text` reads bytes once, computes the digest over those bytes, and returns the same bytes decoded to text, parsing cannot use a different file version than the resolver checked.

- [ ] **Step 4: Run prompt loader tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add tldw_Server_API/app/core/Utils/prompt_loader.py \
  tldw_Server_API/tests/Utils/test_prompt_loader_paths.py
git commit -m "feat: enforce context integrity for prompt loader"
```

## Task 8: Admin Status And Findings Endpoint

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Create: `tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py`

- [ ] **Step 1: Write failing admin endpoint tests**

Create `tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app(*, roles: list[str]) -> FastAPI:
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    app = FastAPI()
    app.include_router(admin_router, prefix="/api/v1")

    async def _principal_override(request=None):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="adminuser",
            token_type="access",
            jti=None,
            roles=roles,
            permissions=["system.configure"],
            is_admin="admin" in roles,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    return app


def test_admin_context_integrity_returns_boot_state() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )

    app = _build_app(roles=["admin"])
    app.state.context_integrity_boot_state = ContextIntegrityBootState(
        mode="enforce",
        degraded=False,
        manifest_sequence=7,
        manifest_digest="sha256:manifest",
        findings=(
            ContextIntegrityFinding(
                asset_id="prompt_file:rag.prompts.yaml",
                state="new_unapproved",
                severity="warning",
                summary="new",
                remediation="review",
                source_type="prompt_file",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/context-integrity")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == "current_process"
    assert body["mode"] == "enforce"
    assert body["manifest_sequence"] == 7
    assert body["findings"][0]["asset_id"] == "prompt_file:rag.prompts.yaml"


def test_admin_context_integrity_is_admin_only() -> None:
    app = _build_app(roles=["user"])

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/context-integrity")

    assert response.status_code == 403
```

- [ ] **Step 2: Run failing admin tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py -v
```

Expected: FAIL with 404 or missing schemas.

- [ ] **Step 3: Add schemas**

Append to `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` near the startup warning schemas:

```python
class AdminContextIntegrityFinding(BaseModel):
    """One current-process Context Integrity finding."""

    asset_id: str
    state: str
    severity: str
    summary: str
    remediation: str
    source_type: str
    current_digest: str | None = None
    approved_digest: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminContextIntegrityResponse(BaseModel):
    """Current-process Context Integrity state for admin inspection."""

    scope: Literal["current_process"] = "current_process"
    mode: str
    degraded: bool
    manifest_sequence: int | None = None
    manifest_digest: str | None = None
    findings_present: bool
    findings: list[AdminContextIntegrityFinding] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 4: Add endpoint and include router**

Create `tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Request

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminContextIntegrityFinding,
    AdminContextIntegrityResponse,
)

router = APIRouter()


@router.get(
    "/context-integrity",
    response_model=AdminContextIntegrityResponse,
)
async def get_context_integrity_status(request: Request) -> AdminContextIntegrityResponse:
    """Return current-process Context Integrity boot state."""
    boot_state = getattr(request.app.state, "context_integrity_boot_state", None)
    if boot_state is None:
        return AdminContextIntegrityResponse(
            mode="uninitialized",
            degraded=True,
            manifest_sequence=None,
            manifest_digest=None,
            findings_present=False,
            findings=[],
        )
    findings = [
        AdminContextIntegrityFinding.model_validate(finding)
        for finding in boot_state.findings
    ]
    return AdminContextIntegrityResponse(
        mode=str(boot_state.mode),
        degraded=bool(boot_state.degraded),
        manifest_sequence=boot_state.manifest_sequence,
        manifest_digest=boot_state.manifest_digest,
        findings_present=bool(findings),
        findings=findings,
    )
```

Modify `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`:

```python
from . import context_integrity as context_integrity_endpoints
```

and include:

```python
router.include_router(context_integrity_endpoints.router)
```

- [ ] **Step 5: Run admin tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_startup_warnings_sqlite.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py
git commit -m "feat: expose context integrity admin status"
```

## Task 9: Focused Integration And Security Verification

**Files:**
- Modify only files touched by prior tasks if verification reveals defects.

- [ ] **Step 1: Run core Context Integrity tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit -v
```

Expected: PASS.

- [ ] **Step 2: Run service and endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Services/test_startup_context_integrity.py \
  tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py \
  tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py \
  tldw_Server_API/tests/Utils/test_prompt_loader_paths.py \
  tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py -v
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Context_Integrity \
  tldw_Server_API/app/services/startup_context_integrity.py \
  tldw_Server_API/app/services/lifespan_startup_sequence.py \
  tldw_Server_API/app/services/lifespan_shutdown_sequence.py \
  tldw_Server_API/app/core/Skills/skills_service.py \
  tldw_Server_API/app/core/Skills/context_integration.py \
  tldw_Server_API/app/core/Chat/command_router.py \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/app/core/Utils/prompt_loader.py \
  tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py \
  -f json -o /tmp/bandit_context_integrity_foundation.json
```

Expected: exit 0 or only pre-existing accepted findings outside changed code. Fix any new findings in touched code before continuing.

- [ ] **Step 4: Commit verification fixes if needed**

If Step 1, 2, or 3 required code fixes:

```bash
git add <fixed-files>
git commit -m "fix: stabilize context integrity foundation"
```

If no fixes were needed, do not create an empty commit.

## Follow-Up Plan Boundaries

This plan intentionally makes DB prompt-version and MCP prompt-catalog enforcement pluggable but does not fully enforce them. After this foundation lands, create separate implementation plans for:

1. DB prompt-version inventory and approval workflow in `Prompts_DB` and prompt APIs.
2. MCP prompt-catalog resolver checks in `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py`.
3. Bundled/plugin/repo skill inventory adapters, including repo prompt-skill files such as `Docs/Prompts/Skills/*`.
4. OS/hardware-backed key provider integration beyond the first HMAC signer.
5. Admin approval/import/export endpoints for signed manifest lifecycle.
6. Frontend review UI for source-grouped baseline approval and canonical diffs.

## Plan Self-Review

Spec coverage:

- Threat model and degraded integrity: Tasks 2, 3, 5, 8, and 9.
- Canonical hashing: Task 1.
- Signed manifest and anti-rollback: Task 2.
- Resolver and TOCTOU-safe runtime use: Tasks 3, 6, and 7.
- Startup warnings: Task 5.
- Skills enforcement: Task 6.
- Prompt loader enforcement: Task 7.
- Admin status/audit visibility: Task 8.
- Security verification: Task 9.
- DB/MCP/full manifest lifecycle: explicitly bounded follow-up plans so the first slice remains reviewable.

Red-flag term scan:

- No banned planning-token sections are intentionally present.
- Commands include exact paths and expected outcomes.

Type consistency:

- `ContextIntegrityBootState`, `ContextIntegrityFinding`, `ContextIntegrityResolver`, `ContextIntegrityBlocked`, and `ContextAssetDescriptor` are introduced before later tasks use them.
- Asset IDs use `skill:user:{user_id}/{skill_name}` and `prompt_file:{filename}` consistently across tests, resolver checks, and inventory adapters.

Verification for this plan artifact:

- Documentation-only plan creation. Run `rg -n "TB[D]|TO[D]O|FIXM[E]|placeholde[r]|implement late[r]|fill in detail[s]|add appropriat[e]|handle edge case[s]|Write tests for the abov[e]|Similar t[o]" Docs/superpowers/plans/2026-06-25-context-integrity-foundation-implementation-plan.md` after saving.
- Bandit is not applicable to this plan artifact itself; implementation tasks include Bandit for touched Python code.

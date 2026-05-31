# OCI-Aware Image Store Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add metadata-only OCI/source provenance fields to the sandbox image store while preserving the current `tldw_bundle` boot path.

**Architecture:** Keep `SandboxImageStore` as the local inventory and provenance authority, not a registry or boot validator. Add optional bounded OCI metadata to template manifests, set `artifact_format="tldw_bundle"` for canonical bundle registration, and surface the resulting template provenance through read-only macOS diagnostics. Do not change helper boot, networking, guest execution, runtime admission, or run-clone planning semantics.

**Tech Stack:** Python dataclasses, JSON manifests, Pydantic response schemas, pytest, existing macOS diagnostics and image-store tests.

---

## Scope Guardrails

This PR is intentionally metadata-only.

In scope:

- Persist `artifact_format` on template records.
- Persist optional OCI/source fields on template records.
- Keep old manifests loadable.
- Keep `register_bundle()` callers working unchanged while writing `artifact_format="tldw_bundle"`.
- Allow future callers to pass bounded OCI metadata into `register_template()`.
- Expose template metadata through existing admin macOS image-store diagnostics.
- Update tests and docs.

Out of scope:

- No helper protocol changes.
- No VM boot path changes.
- No `validate_template` changes.
- No networking or vmnet work.
- No guest-agent changes.
- No runtime admission or trust-level changes.
- No Apple `container` or `containerization` package dependency.
- No OCI image pull/import implementation.

## File Map

- Modify: `tldw_Server_API/app/core/Sandbox/image_store.py`
  - Add template metadata fields, validation helpers, manifest read/write support, and backward-compatible defaults.
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
  - Include registered template metadata in `probe_image_store()` output.
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
  - Add Pydantic models for image-store template diagnostics.
- Modify: `tldw_Server_API/tests/sandbox/test_macos_image_store.py`
  - Add coverage for persisted artifact format, OCI metadata, reload, and validation.
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
  - Add coverage for diagnostics surfacing template provenance.
- Modify: `tools/vz-linux-image/README.md`
  - Document `artifact_format="tldw_bundle"` and optional future OCI provenance fields.
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
  - Mention that image-store manifests are OCI-aware metadata scaffolding while helper boot remains bundle-backed.

## Metadata Contract

Add these fields to `TemplateRecord`:

```python
artifact_format: str = "unknown"
oci_image_ref: str | None = None
oci_platform: str | None = None
oci_manifest_digest: str | None = None
oci_config_digest: str | None = None
oci_layer_digests: list[str] = field(default_factory=list)
registry: str | None = None
imported_at: str | None = None
```

Allowed `artifact_format` values:

- `tldw_bundle`
- `raw_artifacts`
- `oci_image`
- `unknown`

Default behavior:

- `register_bundle()` writes `artifact_format="tldw_bundle"`.
- `register_template()` writes `artifact_format="raw_artifacts"` unless the caller passes a different allowed value.
- Existing manifests that do not contain `artifact_format` reload as `artifact_format="unknown"`.
- Existing manifests that do not contain OCI fields reload with `None` or `[]` defaults.

Validation rules:

- All optional string fields are stripped and capped to a small deterministic maximum, for example 2048 bytes.
- Empty optional strings become `None`.
- `artifact_format` must be one of the allowed values.
- `oci_layer_digests` must be a list of non-empty strings and should be capped, for example maximum 128 entries.
- Metadata validation failures raise `ImageStoreValidationError`.

Schema version:

- Keep `MANIFEST_SCHEMA_VERSION = 1` because this is additive and backward-compatible.
- Do not require migrations for existing manifests.

## Task 1: Add Failing Image-Store Metadata Tests

**Files:**

- Modify: `tldw_Server_API/tests/sandbox/test_macos_image_store.py`

**Success Criteria:** Tests fail because `TemplateRecord` and persisted manifests do not yet include artifact format or OCI metadata.

- [ ] **Step 1: Add test for bundle artifact format**

Add this test near `test_image_store_registers_bundle_with_build_provenance`:

```python
def test_image_store_registers_bundle_with_tldw_bundle_artifact_format(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "boot_mode": "linux_direct"}),
        encoding="utf-8",
    )
    store_root = tmp_path / "store"
    store = SandboxImageStore(root_path=store_root)

    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )

    manifest_path = store_root / "templates" / "vz_linux" / "debian-bookworm-arm64" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_format"] == "tldw_bundle"

    reloaded = SandboxImageStore(root_path=store_root).get_template(template_id)
    assert reloaded is not None
    assert reloaded.artifact_format == "tldw_bundle"
    assert reloaded.oci_image_ref is None
    assert reloaded.oci_layer_digests == []
```

- [ ] **Step 2: Add test for optional OCI metadata**

Add this test:

```python
def test_image_store_persists_optional_oci_metadata(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store_root = tmp_path / "store"
    store = SandboxImageStore(root_path=store_root)

    template_id = store.register_template(
        runtime="vz_linux",
        template_name="oci-backed",
        disk_paths=[str(disk)],
        artifact_format="oci_image",
        oci_image_ref="registry.example/tldw/sandbox:bookworm",
        oci_platform="linux/arm64",
        oci_manifest_digest="sha256:" + "a" * 64,
        oci_config_digest="sha256:" + "b" * 64,
        oci_layer_digests=["sha256:" + "c" * 64],
        registry="registry.example",
        imported_at="2026-05-02T00:00:00+00:00",
    )

    manifest_path = store_root / "templates" / "vz_linux" / "oci-backed" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_format"] == "oci_image"
    assert payload["oci_image_ref"] == "registry.example/tldw/sandbox:bookworm"
    assert payload["oci_platform"] == "linux/arm64"
    assert payload["oci_layer_digests"] == ["sha256:" + "c" * 64]

    reloaded = SandboxImageStore(root_path=store_root).get_template(template_id)
    assert reloaded is not None
    assert reloaded.artifact_format == "oci_image"
    assert reloaded.oci_image_ref == "registry.example/tldw/sandbox:bookworm"
    assert reloaded.oci_platform == "linux/arm64"
    assert reloaded.oci_manifest_digest == "sha256:" + "a" * 64
    assert reloaded.oci_config_digest == "sha256:" + "b" * 64
    assert reloaded.oci_layer_digests == ["sha256:" + "c" * 64]
    assert reloaded.registry == "registry.example"
    assert reloaded.imported_at == "2026-05-02T00:00:00+00:00"
```

- [ ] **Step 3: Add validation tests**

Add focused negative tests:

```python
def test_image_store_rejects_unknown_artifact_format(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")

    with pytest.raises(ImageStoreValidationError, match="artifact_format_invalid"):
        store.register_template(
            runtime="vz_linux",
            template_name="bad-format",
            disk_paths=[str(disk)],
            artifact_format="tarball",
        )


def test_image_store_rejects_invalid_oci_layer_digests(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")

    with pytest.raises(ImageStoreValidationError, match="oci_layer_digests_invalid"):
        store.register_template(
            runtime="vz_linux",
            template_name="bad-oci",
            disk_paths=[str(disk)],
            oci_layer_digests=[""],
        )
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_image_store.py -q
```

Expected: FAIL with missing `artifact_format`/OCI keyword support or missing `TemplateRecord` attributes.

## Task 2: Implement Image-Store Metadata Persistence

**Files:**

- Modify: `tldw_Server_API/app/core/Sandbox/image_store.py`

**Success Criteria:** Task 1 tests pass, old manifests still load, and current callers do not need changes.

- [ ] **Step 1: Extend `TemplateRecord`**

Add the metadata fields listed in the Metadata Contract section.

- [ ] **Step 2: Add keyword-only parameters to `register_template()`**

Extend the signature after `provenance`:

```python
artifact_format: str | None = None,
oci_image_ref: str | None = None,
oci_platform: str | None = None,
oci_manifest_digest: str | None = None,
oci_config_digest: str | None = None,
oci_layer_digests: list[str] | None = None,
registry: str | None = None,
imported_at: str | None = None,
```

Set `artifact_format` to `raw_artifacts` when omitted.

- [ ] **Step 3: Set `artifact_format="tldw_bundle"` in `register_bundle()`**

Pass `artifact_format="tldw_bundle"` into `register_template()` and do not change any existing `register_bundle()` arguments.

- [ ] **Step 4: Add validation helpers**

Add small private helpers:

```python
_ALLOWED_ARTIFACT_FORMATS = frozenset({"tldw_bundle", "raw_artifacts", "oci_image", "unknown"})
_MAX_METADATA_TEXT_BYTES = 2048
_MAX_OCI_LAYER_DIGESTS = 128
```

Use helpers shaped like:

```python
def _normalize_artifact_format(self, value: str | None, *, default: str) -> str:
    normalized = str(value if value is not None else default).strip()
    if normalized not in _ALLOWED_ARTIFACT_FORMATS:
        raise ImageStoreValidationError(f"artifact_format_invalid: {normalized}")
    return normalized

def _normalize_optional_metadata_text(self, value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ImageStoreValidationError(f"{field_name}_invalid")
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized.encode("utf-8")) > _MAX_METADATA_TEXT_BYTES:
        raise ImageStoreValidationError(f"{field_name}_too_long")
    return normalized

def _normalize_oci_layer_digests(self, values: list[str] | None) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > _MAX_OCI_LAYER_DIGESTS:
        raise ImageStoreValidationError("oci_layer_digests_invalid")
    normalized = []
    for value in values:
        item = self._normalize_optional_metadata_text(value, "oci_layer_digest")
        if item is None:
            raise ImageStoreValidationError("oci_layer_digests_invalid")
        normalized.append(item)
    return normalized
```

Keep the helpers boring. Do not implement a full OCI digest parser in this PR.

- [ ] **Step 5: Write metadata fields into manifests**

Update `_record_to_manifest()` to include the fields:

```python
"artifact_format": record.artifact_format,
"oci_image_ref": record.oci_image_ref,
"oci_platform": record.oci_platform,
"oci_manifest_digest": record.oci_manifest_digest,
"oci_config_digest": record.oci_config_digest,
"oci_layer_digests": list(record.oci_layer_digests),
"registry": record.registry,
"imported_at": record.imported_at,
```

- [ ] **Step 6: Read metadata fields from manifests with backward-compatible defaults**

Update `_read_manifest()` so missing fields do not fail:

```python
artifact_format=self._normalize_artifact_format(payload.get("artifact_format"), default="unknown"),
oci_image_ref=self._normalize_optional_metadata_text(payload.get("oci_image_ref"), "oci_image_ref"),
...
oci_layer_digests=self._normalize_oci_layer_digests(payload.get("oci_layer_digests")),
```

When reading, handle invalid field types with `ImageStoreValidationError`, not raw `TypeError`/`ValueError`.

- [ ] **Step 7: Run image-store tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_image_store.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/image_store.py tldw_Server_API/tests/sandbox/test_macos_image_store.py
git commit -m "feat(sandbox): persist oci-aware image metadata"
```

## Task 3: Surface Template Metadata In Admin Diagnostics

**Files:**

- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`

**Success Criteria:** `collect_macos_diagnostics()` includes a read-only `image_store.templates` list with template artifact format and optional OCI provenance.

- [ ] **Step 1: Add failing diagnostics assertions**

In `test_collect_macos_diagnostics_reports_image_store_correlation`, after `image_store = data["image_store"]`, add:

```python
templates = {template["template_id"]: template for template in image_store["templates"]}
assert templates[template_id]["artifact_format"] == "tldw_bundle"
assert templates[template_id]["runtime"] == "vz_linux"
assert templates[template_id]["template_name"] == "debian-bookworm-arm64"
assert templates[template_id]["artifact_count"] == 2
assert templates[template_id]["artifact_size_bytes"] == len(b"kernel") + len(b"rootfs")
```

Add a second focused unit test against `probe_image_store()` using `register_template(... artifact_format="oci_image", ...)` and assert the OCI fields are surfaced.

- [ ] **Step 2: Run diagnostics tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py -q
```

Expected: FAIL because `templates` is missing.

- [ ] **Step 3: Add Pydantic schema for template diagnostics**

In `sandbox_schemas.py`, add:

```python
class SandboxAdminMacOSImageStoreTemplate(BaseModel):
    template_id: str
    runtime: str
    template_name: str
    artifact_format: str
    source_path: str | None = None
    artifact_count: int = 0
    artifact_size_bytes: int = 0
    oci_image_ref: str | None = None
    oci_platform: str | None = None
    oci_manifest_digest: str | None = None
    oci_config_digest: str | None = None
    oci_layer_digests: list[str] = Field(default_factory=list)
    registry: str | None = None
    imported_at: str | None = None
    provenance: dict[str, object] = Field(default_factory=dict)
```

Add to `SandboxAdminMacOSImageStoreDiagnostics`:

```python
templates: list[SandboxAdminMacOSImageStoreTemplate] = Field(default_factory=list)
```

- [ ] **Step 4: Add diagnostics helper**

In `macos_diagnostics.py`, add a private helper:

```python
def _image_store_template_item(record) -> dict[str, object]:
    return {
        "template_id": record.template_id,
        "runtime": record.runtime,
        "template_name": record.template_name,
        "artifact_format": record.artifact_format,
        "source_path": record.source_path,
        "artifact_count": len(record.artifacts),
        "artifact_size_bytes": sum(int(artifact.size_bytes) for artifact in record.artifacts),
        "oci_image_ref": record.oci_image_ref,
        "oci_platform": record.oci_platform,
        "oci_manifest_digest": record.oci_manifest_digest,
        "oci_config_digest": record.oci_config_digest,
        "oci_layer_digests": list(record.oci_layer_digests),
        "registry": record.registry,
        "imported_at": record.imported_at,
        "provenance": dict(record.provenance),
    }
```

Use a concrete import type only if it does not introduce import cycles. A duck-typed helper is acceptable here because diagnostics already builds dict payloads.

- [ ] **Step 5: Include `templates` in every `probe_image_store()` return shape**

For unconfigured/missing/unavailable cases, return `"templates": []`.

For the configured healthy case, add:

```python
templates = [_image_store_template_item(record) for record in store.list_templates()]
```

and include `templates` in the returned dict.

- [ ] **Step 6: Run diagnostics tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py
git commit -m "feat(sandbox): expose image template provenance diagnostics"
```

## Task 4: Documentation Updates

**Files:**

- Modify: `tools/vz-linux-image/README.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

**Success Criteria:** Operator-facing docs say current bundles are recorded as `tldw_bundle`, OCI fields are metadata-only, and helper bootability remains helper-owned.

- [ ] **Step 1: Update `tools/vz-linux-image/README.md`**

In the Image Store Registration section, update the final paragraph to mention:

```markdown
Canonical bundle registration writes `artifact_format="tldw_bundle"`.
The manifest also has optional OCI/source provenance fields such as
`oci_image_ref`, `oci_platform`, manifest/config/layer digests, `registry`, and
`imported_at`. These fields are metadata scaffolding only; the helper still
boots the repo-owned bundle path and remains the source of truth for bootability.
```

- [ ] **Step 2: Update Sandbox README technical notes**

Extend the image-store note to say:

```markdown
Template manifests are OCI-aware metadata scaffolding: current canonical bundles
remain `artifact_format=tldw_bundle`, while optional OCI fields can describe a
future source image without changing helper boot.
```

- [ ] **Step 3: Run docs diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add tools/vz-linux-image/README.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(sandbox): document oci-aware image metadata"
```

## Task 5: Final Verification

**Files:**

- Check only.

**Success Criteria:** Focused tests and security scan pass before PR creation.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_image_store.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on touched Python code**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/image_store.py \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_oci_image_store_metadata.json
```

Expected: JSON report writes successfully with no new findings in touched code.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Inspect commit range**

Run:

```bash
git log --oneline --decorate dev..HEAD
git diff --stat dev..HEAD
```

Expected: only roadmap docs plus OCI-aware image-store metadata plan/implementation commits are present.

## Implementation Notes

- Keep diagnostics read-only. `probe_image_store()` must not create roots or write manifests.
- Keep manifest output deterministic with `json.dump(..., sort_keys=True)`.
- Do not store arbitrary nested OCI descriptors yet. The goal is provenance fields, not an OCI model.
- Do not validate that OCI digests are reachable or present in a registry.
- Do not add a new runtime enum or helper boot mode.
- If reviewers ask for a schema bump, prefer explaining that fields are optional/additive and old manifests still load; bump only if incompatible validation is introduced.

## Expected PR Summary

This PR adds OCI-aware metadata scaffolding to the sandbox image store without changing VM boot behavior. Current canonical bundles are recorded as `artifact_format=tldw_bundle`; future OCI source information can be persisted and surfaced in admin diagnostics, but helper template validation remains the bootability authority.

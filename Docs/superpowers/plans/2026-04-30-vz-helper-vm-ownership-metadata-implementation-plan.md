# VZ Helper VM Ownership Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add helper-owned VM metadata and use it to restrict orphan VM repair to VMs that can be classified as owned by the tldw `vz_linux` sandbox path.

**Architecture:** Keep the Python service as the trusted control plane and the Swift helper as the owner of live VM state. Python supplies ownership metadata on `create_vm`; the helper stores it with VM registry records and reports it through status/list responses; Python reconciliation uses that metadata to classify orphan VMs and repair only terminates owned orphans.

**Tech Stack:** Swift Package Manager, Swift Testing, Python dataclasses, pytest, existing `MacOSVirtualizationHelperClient`, existing sandbox reconciliation and repair APIs.

---

## File Structure

- Modify `tools/macos-vz-helper/Sources/Protocol/Response.swift`
  - Add `VMOwnershipMetadata`.
  - Add metadata to `VMRecord`, `HelperVMResponse`, and `HelperVMStatusResponse`.
- Modify `tools/macos-vz-helper/Sources/VM/VMRegistry.swift`
  - Store metadata with records.
  - Preserve metadata across `upsert()` state transitions unless replacement metadata is supplied.
- Modify `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
  - Accept metadata during VM creation and pass it into the registry.
- Modify `tools/macos-vz-helper/Sources/Server/HelperService.swift`
  - Accept metadata for `createVM()`.
  - Assign helper-created `created_at` when missing.
  - Include metadata in create/status/list responses.
- Modify `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
  - Forward `owner`, `runtime`, `run_id`, `session_id`, `session_mode`, `template`, `template_path`, and `workspace_path` into `HelperService.createVM()`.
- Modify Swift tests:
  - `tools/macos-vz-helper/Tests/VMRegistryTests.swift`
  - `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift`
  - `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`
- Modify Python helper models and client:
  - `tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py`
  - `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify Python runtime/reconciliation/repair:
  - `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
  - `tldw_Server_API/app/core/Sandbox/vz_reconciliation.py`
  - `tldw_Server_API/app/core/Sandbox/service.py`
- Modify Python tests:
  - `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
  - `tldw_Server_API/tests/sandbox/test_macos_helper_client.py`
  - `tldw_Server_API/tests/sandbox/test_vz_reconciliation.py`
  - `tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py`
  - `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
  - `tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py`
- Modify docs:
  - `tools/macos-vz-helper/PROTOCOL.md`
  - `tools/macos-vz-helper/README.md`
  - `Docs/Sandbox/macos-runtime-operator-notes.md`
  - `tldw_Server_API/app/core/Sandbox/README.md`

## Task 1: Add Swift VM Metadata Storage

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Protocol/Response.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VMRegistry.swift`
- Test: `tools/macos-vz-helper/Tests/VMRegistryTests.swift`

- [ ] **Step 1: Write failing registry metadata tests**

Add tests for:

```swift
@Test func vmRegistryStoresOwnershipMetadata() throws {
    let registry = VMRegistry()
    let metadata = VMOwnershipMetadata(
        owner: "tldw",
        runtime: "vz_linux",
        runID: "run-1",
        sessionID: "session-1",
        sessionMode: true,
        templatePath: "/tmp/bundle",
        workspacePath: "/tmp/workspace",
        createdAt: "2026-04-30T18:00:00Z"
    )

    registry.upsert(vmID: "vm-1", state: "booting", healthy: false, metadata: metadata)

    #expect(registry.status(vmID: "vm-1")?.metadata.owner == "tldw")
    #expect(registry.status(vmID: "vm-1")?.metadata.runID == "run-1")
}

@Test func vmRegistryPreservesMetadataAcrossStateUpdates() throws {
    let registry = VMRegistry()
    let metadata = VMOwnershipMetadata(
        owner: "tldw",
        runtime: "vz_linux",
        runID: "run-1",
        sessionID: "",
        sessionMode: false,
        templatePath: "/tmp/bundle",
        workspacePath: "/tmp/workspace",
        createdAt: "2026-04-30T18:00:00Z"
    )

    registry.upsert(vmID: "vm-1", state: "booting", healthy: false, metadata: metadata)
    registry.upsert(vmID: "vm-1", state: "running", healthy: true)

    let record = registry.status(vmID: "vm-1")
    #expect(record?.state == "running")
    #expect(record?.metadata.runID == "run-1")
}
```

- [ ] **Step 2: Run Swift tests to verify failure**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter VMRegistryTests
```

Expected: fail because `VMOwnershipMetadata` and metadata-aware `upsert()` do not exist.

- [ ] **Step 3: Implement metadata model and registry storage**

Add to `Response.swift`:

```swift
struct VMOwnershipMetadata: Codable, Equatable {
    let owner: String
    let runtime: String
    let runID: String
    let sessionID: String
    let sessionMode: Bool
    let templatePath: String
    let workspacePath: String
    let createdAt: String

    static let unknown = VMOwnershipMetadata(
        owner: "unknown",
        runtime: "",
        runID: "",
        sessionID: "",
        sessionMode: false,
        templatePath: "",
        workspacePath: "",
        createdAt: ""
    )

    private enum CodingKeys: String, CodingKey {
        case owner
        case runtime
        case runID = "run_id"
        case sessionID = "session_id"
        case sessionMode = "session_mode"
        case templatePath = "template_path"
        case workspacePath = "workspace_path"
        case createdAt = "created_at"
    }
}

struct VMRecord {
    let vmID: String
    let state: String
    let healthy: Bool
    let metadata: VMOwnershipMetadata
}
```

Update `VMRegistry.upsert()`:

```swift
func upsert(vmID: String, state: String, healthy: Bool, metadata: VMOwnershipMetadata? = nil) {
    lock.lock()
    defer { lock.unlock() }
    let existingMetadata = records[vmID]?.metadata ?? .unknown
    records[vmID] = VMRecord(
        vmID: vmID,
        state: state,
        healthy: healthy,
        metadata: metadata ?? existingMetadata
    )
}
```

- [ ] **Step 4: Run registry tests**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter VMRegistryTests
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Protocol/Response.swift tools/macos-vz-helper/Sources/VM/VMRegistry.swift tools/macos-vz-helper/Tests/VMRegistryTests.swift
git commit -m "feat(sandbox): store helper vm ownership metadata"
```

## Task 2: Surface Metadata Through Helper Service Responses

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Protocol/Response.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Test: `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift`
- Test: `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`

- [ ] **Step 1: Write failing helper service tests**

Add tests that:

- call `HelperService.createVM()` with metadata and assert create/status/list responses include it
- call `HelperService.createVM()` without metadata and assert `owner == "unknown"`
- send a `create_vm` JSON request through `UnixSocketServer.handleRequestData()` and assert nested response metadata contains `owner`, `runtime`, `run_id`, `session_id`, `session_mode`, `template_path`, and `workspace_path`

Use this assertion shape:

```swift
let response = try service.createVM(
    vmID: "vm-owned",
    templatePath: "/tmp/template.img",
    workspacePath: "/tmp/workspace",
    readinessTimeoutSeconds: 5,
    metadata: VMOwnershipMetadata(
        owner: "tldw",
        runtime: "vz_linux",
        runID: "run-owned",
        sessionID: "session-owned",
        sessionMode: true,
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        createdAt: ""
    )
)

#expect(response.metadata.owner == "tldw")
#expect(response.metadata.createdAt.isEmpty == false)
```

- [ ] **Step 2: Run Swift tests to verify failure**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter HelperServiceVMTests
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: fail because service signatures and response metadata do not exist.

- [ ] **Step 3: Implement service and response metadata**

Update response structs:

```swift
struct HelperVMResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let metadata: VMOwnershipMetadata
    let details: [String: String]
    ...
}

struct HelperVMStatusResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let healthy: Bool
    let metadata: VMOwnershipMetadata
    let details: [String: String]
    ...
}
```

Update `VZLinuxVMManager.createVM(..., metadata:)` to write metadata on the initial `booting` upsert and preserve it on the `running` upsert.

Update `HelperService.createVM(..., metadata: VMOwnershipMetadata = .unknown)` to normalize metadata with a helper-created timestamp when missing:

```swift
private func normalizeMetadata(_ metadata: VMOwnershipMetadata, templatePath: String, workspacePath: String) -> VMOwnershipMetadata {
    VMOwnershipMetadata(
        owner: metadata.owner.isEmpty ? "unknown" : metadata.owner,
        runtime: metadata.runtime.isEmpty ? "vz_linux" : metadata.runtime,
        runID: metadata.runID,
        sessionID: metadata.sessionID,
        sessionMode: metadata.sessionMode,
        templatePath: metadata.templatePath.isEmpty ? templatePath : metadata.templatePath,
        workspacePath: metadata.workspacePath.isEmpty ? workspacePath : metadata.workspacePath,
        createdAt: metadata.createdAt.isEmpty ? ISO8601DateFormatter().string(from: Date()) : metadata.createdAt
    )
}
```

Update `UnixSocketServer` to build metadata from request fields:

```swift
let templatePath = request.request["template"]?.stringValue ?? request.request["template_path"]?.stringValue ?? ""
let metadata = VMOwnershipMetadata(
    owner: request.request["owner"]?.stringValue ?? "unknown",
    runtime: request.request["runtime"]?.stringValue ?? "vz_linux",
    runID: request.request["run_id"]?.stringValue ?? "",
    sessionID: request.request["session_id"]?.stringValue ?? "",
    sessionMode: request.request["session_mode"]?.boolValue ?? false,
    templatePath: templatePath,
    workspacePath: request.request["workspace_path"]?.stringValue ?? "",
    createdAt: ""
)
```

For missing VM status, return `.unknown` metadata.

- [ ] **Step 4: Run helper Swift tests**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter HelperServiceVMTests
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Protocol/Response.swift tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift tools/macos-vz-helper/Tests/HelperServiceVMTests.swift tools/macos-vz-helper/Tests/UnixSocketServerTests.swift
git commit -m "feat(sandbox): expose helper vm metadata"
```

## Task 3: Parse VM Metadata In Python Helper Models

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_helper_client.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

- [ ] **Step 1: Write failing Python parser tests**

Add model/parser tests for:

```python
def test_parse_helper_vm_status_reads_metadata() -> None:
    reply = parse_helper_vm_status({
        "protocol_version": "1",
        "helper_version": "0.1.0",
        "vm_id": "vm-1",
        "state": "running",
        "healthy": True,
        "metadata": {
            "owner": "tldw",
            "runtime": "vz_linux",
            "run_id": "run-1",
            "session_id": "session-1",
            "session_mode": True,
            "template_path": "/tmp/bundle",
            "workspace_path": "/tmp/workspace",
            "created_at": "2026-04-30T18:00:00Z",
        },
    })

    assert reply.metadata.owner == "tldw"
    assert reply.metadata.session_mode is True
```

Add tests that missing or malformed `metadata` returns an unknown/default metadata object and does not mark ownership as trusted.

- [ ] **Step 2: Run parser tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: fail because `HelperVMStatusReply.metadata` does not exist.

- [ ] **Step 3: Implement metadata dataclass and parser**

Add to `models.py`:

```python
@dataclass(slots=True)
class HelperVMMetadata:
    owner: str = "unknown"
    runtime: str = ""
    run_id: str = ""
    session_id: str = ""
    session_mode: bool = False
    template_path: str = ""
    workspace_path: str = ""
    created_at: str = ""

    @property
    def has_tldw_owner(self) -> bool:
        return self.owner == "tldw" and self.runtime == "vz_linux"
```

Add `metadata: HelperVMMetadata = field(default_factory=HelperVMMetadata)` to `HelperVMReply` and `HelperVMStatusReply`.

Add:

```python
def _metadata_field(payload: dict[str, Any]) -> HelperVMMetadata:
    raw = payload.get("metadata")
    if not isinstance(raw, dict):
        return HelperVMMetadata()
    return HelperVMMetadata(
        owner=_str_field(raw, "owner", "unknown").strip() or "unknown",
        runtime=_str_field(raw, "runtime").strip(),
        run_id=_str_field(raw, "run_id").strip(),
        session_id=_str_field(raw, "session_id").strip(),
        session_mode=_bool_field(raw, "session_mode"),
        template_path=_str_field(raw, "template_path").strip(),
        workspace_path=_str_field(raw, "workspace_path").strip(),
        created_at=_str_field(raw, "created_at").strip(),
    )
```

Update `parse_helper_vm_status()`, `parse_helper_vm_list()`, real `create_vm()`, and TEST_MODE fake replies to set metadata. The fake `create_vm()` should echo owner/runtime/run/session values from the request so runner tests can inspect behavior without a real helper.

- [ ] **Step 4: Run parser/client tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_macos_helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "feat(sandbox): parse helper vm metadata"
```

## Task 4: Attach Ownership Metadata To Runner VM Creation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py`

- [ ] **Step 1: Write failing runner tests**

Update existing fake-helper tests to assert `create_vm()` receives:

```python
assert request["owner"] == "tldw"
assert request["runtime"] == "vz_linux"
assert request["run_id"] == run_id
assert request["session_mode"] is expected_session_mode
assert request["session_id"] == expected_session_id
assert request["template"] == template_source
```

For ephemeral runs, `session_id` should be an empty string. For session reuse creation, `session_id` should be the sandbox session id.

- [ ] **Step 2: Run runner tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py -q
```

Expected: fail because `owner` and `session_id` are not sent on create.

- [ ] **Step 3: Add metadata to `create_vm` request**

Update `VZLinuxRunner._run_real()` create request:

```python
vm = helper.create_vm(
    {
        "owner": "tldw",
        "runtime": self.runtime_type.value,
        "vm_name": run_id,
        "run_id": run_id,
        "session_id": str(spec.session_id or "").strip(),
        "session_mode": session_mode,
        "workspace_path": workspace,
        "workspace_mount": "virtiofs",
        "template": template_source,
        "network_policy": str(spec.network_policy or "deny_all").strip().lower() or "deny_all",
        "timeout_sec": int(spec.startup_timeout_sec or spec.timeout_sec or 300),
    }
)
```

- [ ] **Step 4: Run runner tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py
git commit -m "feat(sandbox): tag vz linux helper vms"
```

## Task 5: Classify Owned, Unknown, And Foreign Orphan VMs

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/vz_reconciliation.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_reconciliation.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`

- [ ] **Step 1: Write failing reconciliation tests**

Add helper VM factories that accept metadata. Test:

- owned orphan has `status == "owned_orphaned_vm"`, `termination_eligible is True`, and reason `owned_orphan`
- missing metadata is `unknown_orphaned_vm`, `termination_eligible is False`, and reason `unknown_ownership`
- tldw metadata missing `run_id`, missing `created_at`, or session-mode missing `session_id` is `unknown_orphaned_vm`
- owner/runtime mismatch is `foreign_orphaned_vm`, `termination_eligible is False`, and reason `foreign_owner`
- `orphaned_vm_ids` includes all three categories
- `owned_orphaned_vm_ids`, `unknown_orphaned_vm_ids`, and `foreign_orphaned_vm_ids` are deterministic sorted lists

- [ ] **Step 2: Run reconciliation tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_reconciliation.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py -q
```

Expected: fail because reconciliation still emits only `orphaned_vm`.

- [ ] **Step 3: Implement orphan classification**

Add constants:

```python
STATUS_OWNED_ORPHAN = "owned_orphaned_vm"
STATUS_UNKNOWN_ORPHAN = "unknown_orphaned_vm"
STATUS_FOREIGN_ORPHAN = "foreign_orphaned_vm"
REASON_OWNED_ORPHAN = "owned_orphan"
REASON_UNKNOWN_OWNERSHIP = "unknown_ownership"
REASON_FOREIGN_OWNER = "foreign_owner"
```

Add helper:

```python
def _classify_orphan_vm(vm: object) -> tuple[str, bool, str]:
    metadata = getattr(vm, "metadata", None)
    owner = str(getattr(metadata, "owner", "") or "").strip()
    runtime = str(getattr(metadata, "runtime", "") or "").strip()
    if not owner or owner == "unknown" or not runtime:
        return STATUS_UNKNOWN_ORPHAN, False, REASON_UNKNOWN_OWNERSHIP
    if owner != "tldw" or runtime != "vz_linux":
        return STATUS_FOREIGN_ORPHAN, False, REASON_FOREIGN_OWNER
    run_id = str(getattr(metadata, "run_id", "") or "").strip()
    created_at = str(getattr(metadata, "created_at", "") or "").strip()
    session_mode = bool(getattr(metadata, "session_mode", False))
    session_id = str(getattr(metadata, "session_id", "") or "").strip()
    if not run_id or not created_at or (session_mode and not session_id):
        return STATUS_UNKNOWN_ORPHAN, False, REASON_UNKNOWN_OWNERSHIP
    return STATUS_OWNED_ORPHAN, True, REASON_OWNED_ORPHAN
```

Update `_empty_report()` with the three new id lists. When appending orphan items, include:

```python
termination_eligible=eligible
```

Extend `_append_item()` to accept `termination_eligible: bool | None`.

- [ ] **Step 4: Run reconciliation and diagnostics tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_reconciliation.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/vz_reconciliation.py tldw_Server_API/tests/sandbox/test_vz_reconciliation.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py
git commit -m "feat(sandbox): classify helper vm ownership"
```

## Task 6: Gate Orphan VM Repair By Ownership

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py`

- [ ] **Step 1: Write failing repair tests**

Update existing orphan termination tests to use `owned_orphaned_vm` instead of generic `orphaned_vm` for actual termination.

Add tests:

- `unknown_orphaned_vm` with `terminate_orphaned_vms=True` returns a `skip_orphaned_vm` action and does not call helper
- `foreign_orphaned_vm` with `terminate_orphaned_vms=True` returns a `skip_orphaned_vm` action and does not call helper
- legacy `orphaned_vm` without `termination_eligible=True` is skipped, not terminated
- summary `orphaned_vms` counts owned, unknown, foreign, and legacy orphan statuses

Expected skipped action shape:

```python
{
    "type": "skip_orphaned_vm",
    "session_id": None,
    "vm_id": "vm-unknown",
    "status": "skipped",
    "reason": "unknown_ownership",
    "termination_eligible": False,
}
```

- [ ] **Step 2: Run repair tests to verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py -q
```

Expected: fail because repair still terminates generic orphan items.

- [ ] **Step 3: Implement ownership-gated repair**

Update `SandboxService.repair_macos_reconciliation()`:

```python
orphan_statuses = {
    "owned_orphaned_vm",
    "unknown_orphaned_vm",
    "foreign_orphaned_vm",
    "orphaned_vm",
}
orphaned_items = [
    item for item in report_items
    if str(item.get("status") or "").strip() in orphan_statuses
]
```

For orphan items:

```python
eligible = status == "owned_orphaned_vm" and bool(item.get("termination_eligible"))
if status == "orphaned_vm":
    eligible = bool(item.get("termination_eligible")) and (reason == "owned_orphan")
if not eligible:
    if terminate_orphaned_vms and vm_id:
        actions.append({
            "type": "skip_orphaned_vm",
            "session_id": None,
            "vm_id": vm_id,
            "status": "skipped",
            "reason": reason or "unknown_ownership",
            "termination_eligible": False,
        })
    continue
```

For eligible owned orphans, keep the existing helper-specific exception mapping and add `termination_eligible: True` to termination actions.

- [ ] **Step 4: Run repair tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py
git commit -m "fix(sandbox): gate orphan repair by vm ownership"
```

## Task 7: Update Protocol And Operator Docs

**Files:**
- Modify: `tools/macos-vz-helper/PROTOCOL.md`
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

- [ ] **Step 1: Update protocol docs**

Document `create_vm` request ownership fields and include `metadata` in `get_vm_status` and `list_vms` reply examples.

- [ ] **Step 2: Update operator docs**

Clarify:

- orphan termination is explicit and dry-run-first
- repair only terminates owned `vz_linux` helper VMs
- unknown/foreign helper VMs are reported and skipped
- metadata is local helper ownership metadata, not cryptographic proof
- legacy VMs without metadata may require manual operator cleanup

- [ ] **Step 3: Run docs diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add tools/macos-vz-helper/PROTOCOL.md tools/macos-vz-helper/README.md Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(sandbox): document helper vm ownership gate"
```

## Task 8: Run Full Focused Verification

**Files:**
- All touched files from Tasks 1-7.

- [ ] **Step 1: Run Swift helper tests**

Run:

```bash
swift test --package-path tools/macos-vz-helper
```

Expected: all Swift helper tests pass.

- [ ] **Step 2: Run focused Python sandbox tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_helper_client.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  tldw_Server_API/tests/sandbox/test_vz_reconciliation.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run Bandit on touched Python scope**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit \
  -r tldw_Server_API/app/core/Sandbox/macos_virtualization tldw_Server_API/app/core/Sandbox/vz_reconciliation.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py \
  -s B101,B106 \
  -f json \
  -o /tmp/bandit_vz_helper_vm_ownership_metadata.json
```

Expected: command exits 0 and reports no findings in touched production code. If Bandit is unavailable in the venv, record that explicitly before finishing.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Run final status check**

Run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=8
```

Expected: branch contains the design, plan, and implementation commits. Worktree is clean except for intentional uncommitted changes if a review checkpoint requires them.

- [ ] **Step 6: Commit plan updates if needed**

If the plan was adjusted during implementation, commit the final plan state:

```bash
git add Docs/superpowers/plans/2026-04-30-vz-helper-vm-ownership-metadata-implementation-plan.md
git commit -m "docs(sandbox): update vm ownership metadata plan"
```

## Review Checkpoints

- After Task 2, review Swift protocol compatibility before moving to Python parsing.
- After Task 5, review reconciliation output shape against diagnostics tests to avoid breaking admin consumers.
- After Task 6, review repair behavior carefully: unknown, foreign, and legacy orphan records must not call `terminate_vm`.
- Before PR creation, run the full focused verification in Task 8 and summarize any host-gated tests not run.

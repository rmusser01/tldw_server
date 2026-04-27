# VZ Linux Image Bundle And Boot Driver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a canonical `vz_linux` image bundle format, helper-side template resolution/validation, reference-image bundle tooling, and a real Linux VM boot driver on Apple silicon macOS hosts.

**Architecture:** Keep Python unchanged at the sandbox API layer and make the helper the source of truth for bundle validation and boot behavior. Introduce a canonical bundle format in `tools/vz-linux-image/`, resolve bundle and raw-disk inputs through one helper-side resolution interface with distinct boot-spec variants, and wire those variants into the appropriate `Virtualization.framework` boot paths while still requiring guest-agent readiness before a VM becomes healthy.

**Tech Stack:** Python, Swift Package Manager, `Virtualization.framework`, Go, systemd image assets, pytest, Swift Testing, Go test

---

### Task 1: Define The Canonical Bundle Manifest And Resolver Fixtures

**Files:**
- Create: `tools/vz-linux-image/docs/bundle-format.md`
- Create: `tools/macos-vz-helper/Sources/Templates/TemplateManifest.swift`
- Create: `tools/macos-vz-helper/Sources/Templates/TemplateBootSpec.swift`
- Create: `tools/macos-vz-helper/Tests/TemplateManifestTests.swift`
- Create: `tools/macos-vz-helper/Tests/TemplateFixtures/bundle/manifest.json`

**Step 1: Write the failing Swift tests**

Add `TemplateManifestTests.swift` with tests like:

```swift
@Test func templateManifestDecodesCanonicalBundleFields() throws {
    let data = try Data(contentsOf: fixtureURL("bundle/manifest.json"))
    let manifest = try JSONDecoder().decode(TemplateManifest.self, from: data)

    #expect(manifest.bundleVersion == "1")
    #expect(manifest.bootMode == "bundle")
    #expect(manifest.kernel == "kernel")
    #expect(manifest.rootfs == "rootfs.img")
    #expect(manifest.vsockPort == 1024)
    #expect(manifest.workspaceMountTag == "workspace")
}
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateManifestTests'
```

Expected: FAIL because the manifest types do not exist yet.

**Step 3: Add the minimal manifest and boot-spec types**

- Add `TemplateManifest.swift` with a Codable manifest model
- Add `TemplateBootSpec.swift` with:
  - an explicit bundle boot-spec variant with:
    - `bootMode`
    - `kernelPath`
    - optional `initrdPath`
    - `rootfsPath`
    - `workspaceMountTag`
    - `vsockPort`
    - `guestAgentPath`
    - `validationStrength`
  - an explicit raw-disk compatibility boot-spec variant with:
    - `bootMode`
    - `diskImagePath`
    - `workspaceMountTag`
    - `vsockPort`
    - `guestAgentPath`
    - `bootLoaderKind`
    - `validationStrength`
- Add one canonical fixture manifest under `Tests/TemplateFixtures/bundle/manifest.json`
- Document the same manifest contract in `tools/vz-linux-image/docs/bundle-format.md`

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateManifestTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/docs/bundle-format.md tools/macos-vz-helper/Sources/Templates/TemplateManifest.swift tools/macos-vz-helper/Sources/Templates/TemplateBootSpec.swift tools/macos-vz-helper/Tests/TemplateManifestTests.swift tools/macos-vz-helper/Tests/TemplateFixtures/bundle/manifest.json
git commit -m "feat(vz_linux): define canonical image bundle manifest"
```

### Task 2: Add Bundle And Raw-Disk Template Resolvers

**Files:**
- Create: `tools/macos-vz-helper/Sources/Templates/BundleTemplateResolver.swift`
- Create: `tools/macos-vz-helper/Sources/Templates/RawDiskTemplateResolver.swift`
- Modify: `tools/macos-vz-helper/Sources/Templates/TemplateValidator.swift`
- Create: `tools/macos-vz-helper/Tests/TemplateResolverTests.swift`
- Create: `tools/macos-vz-helper/Tests/TemplateFixtures/raw-disk/disk.img`

**Step 1: Write the failing resolver tests**

Add `TemplateResolverTests.swift` with tests like:

```swift
@Test func bundleResolverProducesStrongBootSpec() throws {}
@Test func rawDiskResolverProducesCompatibilityBootSpec() throws {}
@Test func validatorRejectsBundleMissingKernel() throws {}
```

The bundle test should expect:

- `bootMode == .bundle`
- `validationStrength == .strong`

The raw-disk test should expect:

- `bootMode == .rawDisk`
- `validationStrength == .compatibility`
- `diskImagePath` is populated without requiring kernel/initrd fields

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateResolverTests'
```

Expected: FAIL because the resolvers do not exist yet.

**Step 3: Write the minimal resolver implementation**

- `BundleTemplateResolver`:
  - detect `manifest.json`
  - decode the manifest
  - validate referenced files
  - return a strong `TemplateBootSpec`
- `RawDiskTemplateResolver`:
  - verify the file exists
  - return a compatibility `TemplateBootSpec`
- Update `TemplateValidator` to dispatch between resolvers and return richer details for diagnostics

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'TemplateResolverTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Templates/BundleTemplateResolver.swift tools/macos-vz-helper/Sources/Templates/RawDiskTemplateResolver.swift tools/macos-vz-helper/Sources/Templates/TemplateValidator.swift tools/macos-vz-helper/Tests/TemplateResolverTests.swift tools/macos-vz-helper/Tests/TemplateFixtures/raw-disk/disk.img
git commit -m "feat(vz_linux): resolve bundle and raw disk templates"
```

### Task 3: Make Reference Image Tooling Emit The Canonical Bundle

**Files:**
- Modify: `tools/vz-linux-image/README.md`
- Modify: `tools/vz-linux-image/Makefile`
- Create: `tools/vz-linux-image/scripts/build-bundle.sh`
- Create: `tools/vz-linux-image/scripts/write-manifest.sh`
- Create: `tools/vz-linux-image/systemd/tldw-agent-guest.service`
- Create: `tools/vz-linux-image/tests/test_bundle_layout.py`

**Step 1: Write the failing Python layout test**

Add `test_bundle_layout.py` with a test like:

```python
def test_bundle_builder_emits_manifest_and_expected_paths(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    # invoke build script here
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "rootfs.img").is_file()
```

**Step 2: Run the test to verify it fails**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_bundle_layout.py -q
```

Expected: FAIL because the scripts and bundle output do not exist yet.

**Step 3: Add the minimal bundle builder**

- Add a systemd unit for `tldw-agent-guest`
- Add a script that:
  - stages the guest binary
  - stages the systemd unit
  - writes `manifest.json`
  - creates a canonical bundle directory layout
- Update the `Makefile` and README with the new bundle-building entrypoint

The first version must not use synthetic placeholder kernel or rootfs files. It can
consume operator-supplied real inputs or checked-in minimal test assets, but the
canonical bundle builder should fail fast when required boot artifacts are absent.

**Step 4: Run the test to verify it passes**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_bundle_layout.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/README.md tools/vz-linux-image/Makefile tools/vz-linux-image/scripts/build-bundle.sh tools/vz-linux-image/scripts/write-manifest.sh tools/vz-linux-image/systemd/tldw-agent-guest.service tools/vz-linux-image/tests/test_bundle_layout.py
git commit -m "feat(vz_linux): emit canonical image bundles"
```

### Task 4: Build A Real Linux Boot Configuration In The Helper

**Files:**
- Create: `tools/macos-vz-helper/Sources/VM/VZLinuxConfigurationBuilder.swift`
- Create: `tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
- Create: `tools/macos-vz-helper/Tests/VZLinuxConfigurationBuilderTests.swift`
- Create: `tools/macos-vz-helper/Tests/VirtualizationLinuxBootDriverTests.swift`

**Step 1: Write the failing Swift tests**

Add tests that expect:

```swift
@Test func configurationBuilderCreatesLinuxBootLoaderForBundleSpec() throws {}
@Test func configurationBuilderCreatesCompatibilityBootPathForRawDiskSpec() throws {}
```

The canonical bundle test should assert that:

- `VZLinuxBootLoader` is built from the manifest kernel path
- optional initrd is included when present
- a `VZVirtioFileSystemDeviceConfiguration` is created with the manifest tag
- a `VZVirtioSocketDeviceConfiguration` is present

The raw-disk compatibility test should assert that:

- the raw-disk spec does not require kernel/initrd fields
- a compatibility boot loader is created for self-booting images, starting with
  `VZEFIBootLoader`
- storage attachment is created from the disk-image path

**Step 2: Run the tests to verify they fail**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VZLinuxConfigurationBuilderTests|VirtualizationLinuxBootDriverTests'
```

Expected: FAIL because the real builder and boot driver do not exist yet.

**Step 3: Write the minimal implementation**

- Add `VZLinuxConfigurationBuilder` that converts `TemplateBootSpec` into:
  - the canonical bundle Linux boot path
  - the raw-disk compatibility boot path
  - storage attachment
  - virtio-fs device configuration
  - virtio socket device configuration
- Add `VirtualizationLinuxBootDriver` that:
  - builds the VM configuration
  - validates it
  - starts a `VZVirtualMachine`
  - tracks enough state to stop it later
- Update `VZLinuxVMManager` to use the real driver instead of the placeholder for
  the canonical bundle path, while keeping the raw-disk lane explicitly marked as
  compatibility mode

**Step 4: Run the tests to verify they pass**

Run:

```bash
cd tools/macos-vz-helper && swift test --filter 'VZLinuxConfigurationBuilderTests|VirtualizationLinuxBootDriverTests'
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/VM/VZLinuxConfigurationBuilder.swift tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift tools/macos-vz-helper/Tests/VZLinuxConfigurationBuilderTests.swift tools/macos-vz-helper/Tests/VirtualizationLinuxBootDriverTests.swift
git commit -m "feat(vz_linux): add real helper boot configuration"
```

### Task 5: Wire Canonical Bundle Validation Into The Real Helper Protocol

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Modify: `tools/macos-vz-helper/PROTOCOL.md`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

**Step 1: Write the failing contract tests**

Extend `test_macos_virtualization_helper_client.py` with a test like:

```python
def test_helper_validate_template_preserves_validation_strength_and_boot_mode():
    ...
```

The payload should prove that bundle validation reports stronger metadata than raw disk mode.

**Step 2: Run the Python contract tests to verify they fail**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: FAIL because the helper payload/details do not yet include the richer bundle metadata.

**Step 3: Write the minimal implementation**

- Extend helper `validate_template` responses to include:
  - `boot_mode`
  - `validation_strength`
- Update `tools/macos-vz-helper/PROTOCOL.md` so the frozen host-helper contract stays
  aligned with the implementation plan
- Keep the Python helper client tolerant and backward-compatible
- Update the operator docs and sandbox README to describe the canonical bundle as the primary artifact format

**Step 4: Run the tests to verify they pass**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift tools/macos-vz-helper/PROTOCOL.md tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "feat(vz_linux): expose bundle validation through helper protocol"
```

### Task 6: Add Host-Gated Canonical Bundle Boot Smoke

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing host-gated smoke expectations**

Keep the existing helper-daemon smoke focused on daemon reachability and template
validation. Add a separate canonical-bundle boot smoke so that, when a canonical
bundle env is present, it expects:

- helper `validate_template` reports `boot_mode=bundle`
- helper `validate_template` reports `validation_strength=strong`
- canonical bundle boot smoke reaches real `create_vm` success instead of
  `boot_not_implemented`

The new env should be:

```text
TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1
TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH=/abs/path/to/bundle
```

**Step 2: Run the smoke test to verify current failure or skip**

Run:

```bash
source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH=/abs/path/to/bundle python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py -q
```

Expected: skip on an unprepared host, or fail until the real boot path is wired.

**Step 3: Write the minimal implementation**

- Keep the daemon-contract smoke narrow and add the new host-gated canonical-bundle
  boot smoke path
- Update the existing `vz_linux` E2E docstrings and comments so the canonical bundle is the preferred artifact
- Keep raw disk support available as compatibility mode

**Step 4: Run the smoke test again**

Run:

```bash
source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH=/abs/path/to/bundle python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py -q
```

Expected: PASS on a prepared Apple silicon host with a valid canonical bundle;
otherwise explicit skip reasons

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py tools/vz-linux-image/README.md
git commit -m "test(vz_linux): add canonical bundle boot smoke"
```

### Task 7: Run Verification And Security Checks

**Files:**
- Verify: `tools/macos-vz-helper/`
- Verify: `tools/tldw-agent/`
- Verify: `tools/vz-linux-image/`
- Verify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Verify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`
- Verify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Run focused native and guest tests**

Run:

```bash
cd tools/macos-vz-helper && swift test
cd ../tldw-agent && bash ./scripts/verify-local-build.sh
```

Expected: PASS

**Step 2: Run focused Python sandbox tests**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS for unit/contract tests, with host-gated tests skipping cleanly unless all real prerequisites are present

**Step 3: Run Bandit on the touched Python scope**

Run:

```bash
source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -f json -o /tmp/bandit_vz_linux_bundle_boot_driver.json
```

Expected: `results: []`

**Step 4: Commit the final verification/docs cleanup if needed**

```bash
git add Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "chore(vz_linux): finalize bundle boot driver verification"
```

# VZ Linux Debian Arm64 Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repo-owned Linux-only builder that produces a bootable Debian stable arm64 canonical bundle for `vz_linux`.

**Architecture:** Keep one real builder implementation under `tools/vz-linux-image/`. Build a Debian rootfs directory first, stage the guest agent and enabled units into it, pack it into `rootfs.img`, extract `kernel` and `initrd`, and emit the existing canonical bundle layout. Containerized Linux support should only wrap the same scripts.

**Tech Stack:** shell scripts, Debian `debootstrap`, loopback/ext filesystem tooling, Python pytest, existing `tools/vz-linux-image` bundle scripts

**Doctrine Alignment:** This plan follows `Docs/Sandbox/sandbox-architecture-doctrine.md`, especially the canonical-artifact rule, provenance requirements, and the requirement that canonical VM images include enough debug affordances to make boot/readiness failures diagnosable.

---

### Task 1: Add Package Profiles And Builder Defaults

**Files:**
- Create: `tools/vz-linux-image/profiles/minimal.packages`
- Create: `tools/vz-linux-image/profiles/debug.packages`
- Create: `tools/vz-linux-image/scripts/builder-defaults.sh`
- Create: `tools/vz-linux-image/tests/test_profiles.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_minimal_profile_contains_required_boot_packages():
    packages = load_profile("minimal.packages")
    assert "systemd" in packages
    assert "initramfs-tools" in packages

def test_debug_profile_extends_minimal_without_duplicates():
    minimal = load_profile("minimal.packages")
    debug = compose_profiles(["minimal.packages", "debug.packages"])
    assert set(minimal).issubset(set(debug))
    assert len(debug) == len(set(debug))
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_profiles.py -q
```

Expected: FAIL because the profile files and loader contract do not exist yet.

**Step 3: Write the minimal implementation**

- Add `minimal.packages` with the canonical package set
- Add `debug.packages` with additive troubleshooting packages
- Add `builder-defaults.sh` defining:
  - default suite `bookworm`
  - fixed architecture `arm64`
  - pinned kernel package name
  - helper functions for reading and composing package lists
- Document the profiles in `tools/vz-linux-image/README.md`

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_profiles.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/profiles/minimal.packages tools/vz-linux-image/profiles/debug.packages tools/vz-linux-image/scripts/builder-defaults.sh tools/vz-linux-image/tests/test_profiles.py tools/vz-linux-image/README.md
git commit -m "feat(vz_linux): add debian image builder profiles"
```

### Task 2: Build A Debian Rootfs Directory

**Files:**
- Create: `tools/vz-linux-image/scripts/build-debian-rootfs.sh`
- Modify: `tools/vz-linux-image/scripts/install-agent.sh`
- Create: `tools/vz-linux-image/tests/test_build_rootfs_args.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_build_rootfs_requires_linux_host():
    result = run_builder("--output-rootfs", "/tmp/rootfs")
    assert result.returncode != 0
    assert "Linux host required" in result.stderr

def test_build_rootfs_emits_expected_debootstrap_command():
    result = run_builder("--dry-run", "--profile", "minimal", "--output-rootfs", "/tmp/rootfs")
    assert "debootstrap" in result.stdout
    assert "--arch=arm64" in result.stdout
    assert "bookworm" in result.stdout
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_build_rootfs_args.py -q
```

Expected: FAIL because the builder script does not exist yet.

**Step 3: Write the minimal implementation**

- Create `build-debian-rootfs.sh` that:
  - validates Linux-only execution
  - accepts suite/profile/mirror/output args
  - supports `--dry-run`
  - runs `debootstrap --arch=arm64`
  - installs profile packages inside the rootfs
  - calls `install-agent.sh` on the built rootfs
- extend staging so the canonical rootfs also includes:
  - vsock module-loading configuration
  - serial console enablement
- Reuse `builder-defaults.sh` for suite/arch/kernel defaults
- Update the README with the rootfs-builder entrypoint

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_build_rootfs_args.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/build-debian-rootfs.sh tools/vz-linux-image/scripts/install-agent.sh tools/vz-linux-image/tests/test_build_rootfs_args.py tools/vz-linux-image/README.md
git commit -m "feat(vz_linux): add debian rootfs builder"
```

### Task 3: Pack The Rootfs Directory Into `rootfs.img`

**Files:**
- Create: `tools/vz-linux-image/scripts/pack-rootfs-image.sh`
- Create: `tools/vz-linux-image/tests/test_pack_rootfs_image.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_pack_rootfs_requires_existing_rootfs_dir():
    result = run_packer("--rootfs", "/tmp/missing", "--output-image", "/tmp/rootfs.img")
    assert result.returncode != 0

def test_pack_rootfs_supports_dry_run():
    result = run_packer("--dry-run", "--rootfs", "/tmp/rootfs", "--output-image", "/tmp/rootfs.img")
    assert "rootfs.img" in result.stdout
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_pack_rootfs_image.py -q
```

Expected: FAIL because the packer script does not exist yet.

**Step 3: Write the minimal implementation**

- Create `pack-rootfs-image.sh` that:
  - validates the rootfs directory
  - accepts size/output args
  - supports `--dry-run`
  - creates `rootfs.img` from the directory using a directory-to-ext4 path such
    as `mke2fs -d` when possible
  - does not mutate the source rootfs directory in place

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_pack_rootfs_image.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/pack-rootfs-image.sh tools/vz-linux-image/tests/test_pack_rootfs_image.py tools/vz-linux-image/README.md
git commit -m "feat(vz_linux): add rootfs image packer"
```

### Task 4: Extract Kernel And Initrd Artifacts

**Files:**
- Create: `tools/vz-linux-image/scripts/extract-kernel-artifacts.sh`
- Create: `tools/vz-linux-image/tests/test_extract_kernel_artifacts.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_extract_kernel_requires_boot_artifacts_in_rootfs():
    result = run_extractor("--rootfs", "/tmp/rootfs", "--output-dir", "/tmp/out")
    assert result.returncode != 0

def test_extract_kernel_supports_dry_run():
    result = run_extractor("--dry-run", "--rootfs", "/tmp/rootfs", "--output-dir", "/tmp/out")
    assert "kernel" in result.stdout
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_extract_kernel_artifacts.py -q
```

Expected: FAIL because the extractor script does not exist yet.

**Step 3: Write the minimal implementation**

- Create `extract-kernel-artifacts.sh` that:
  - locates the installed Debian kernel and initrd in the built rootfs
  - copies them to an explicit output dir as `kernel` and `initrd`
  - supports `--dry-run`
  - fails fast when boot artifacts are missing

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_extract_kernel_artifacts.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/extract-kernel-artifacts.sh tools/vz-linux-image/tests/test_extract_kernel_artifacts.py tools/vz-linux-image/README.md
git commit -m "feat(vz_linux): add kernel artifact extraction"
```

### Task 5: Add The Top-Level Debian Bundle Builder

**Files:**
- Create: `tools/vz-linux-image/scripts/build-debian-bundle.sh`
- Modify: `tools/vz-linux-image/Makefile`
- Modify: `tools/vz-linux-image/scripts/build-bundle.sh`
- Create: `tools/vz-linux-image/tests/test_build_debian_bundle.py`
- Create: `tools/vz-linux-image/tests/test_build_metadata.py`
- Modify: `tools/vz-linux-image/README.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_build_debian_bundle_dry_run_prints_all_artifact_paths():
    result = run_bundle_builder("--dry-run", "--output-dir", "/tmp/out")
    assert "rootfs/" in result.stdout
    assert "rootfs.img" in result.stdout
    assert "bundle/" in result.stdout

def test_build_metadata_includes_suite_profile_and_kernel_package():
    metadata = load_build_metadata("/tmp/out/build-info.json")
    assert metadata["suite"] == "bookworm"
    assert metadata["profile"] == "minimal"
    assert "kernel_package" in metadata
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_build_debian_bundle.py -q
```

Expected: FAIL because the orchestration script does not exist yet.

**Step 3: Write the minimal implementation**

- Create `build-debian-bundle.sh` that orchestrates:
  - rootfs build
  - image packing
  - kernel/initrd extraction
  - existing canonical bundle emission
- Emit `build-info.json` next to the output artifacts with:
  - suite
  - profile
  - architecture
  - kernel package
  - selected package list
  - source artifact paths when provided
  - validation-strength or canonical-artifact marker when appropriate
- Default to keeping intermediates in:
  - `rootfs/`
  - `rootfs.img`
  - `kernel`
  - `initrd`
  - `bundle/`
  - `build-info.json`
- Add cleanup/keep-intermediates flags
- Expose the new entrypoint via the `Makefile`

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_build_debian_bundle.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/build-debian-bundle.sh tools/vz-linux-image/Makefile tools/vz-linux-image/scripts/build-bundle.sh tools/vz-linux-image/tests/test_build_debian_bundle.py tools/vz-linux-image/tests/test_build_metadata.py tools/vz-linux-image/README.md
git commit -m "feat(vz_linux): add debian bundle builder"
```

### Task 6: Add Native Linux And Container Wrapper Smoke Coverage

**Files:**
- Create: `tools/vz-linux-image/scripts/run-linux-builder-container.sh`
- Create: `tools/vz-linux-image/tests/test_linux_builder_wrapper.py`
- Modify: `tools/vz-linux-image/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`

**Step 1: Write the failing tests**

Add tests like:

```python
def test_container_wrapper_invokes_native_builder_script():
    result = run_wrapper("--dry-run", "--output-dir", "/tmp/out")
    assert "build-debian-bundle.sh" in result.stdout
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_linux_builder_wrapper.py -q
```

Expected: FAIL because the wrapper script does not exist yet.

**Step 3: Write the minimal implementation**

- Create `run-linux-builder-container.sh` as a thin wrapper around the native builder
- Document:
  - Linux-only native path
  - container wrapper requirements
  - privilege expectations
- Update operator notes so the local prepared-host flow points at the new builder

**Step 4: Run the tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_linux_builder_wrapper.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/run-linux-builder-container.sh tools/vz-linux-image/tests/test_linux_builder_wrapper.py tools/vz-linux-image/README.md Docs/Sandbox/macos-runtime-operator-notes.md
git commit -m "docs(vz_linux): add linux builder wrapper guidance"
```

### Task 7: Run Privileged Linux Validation And macOS Follow-On Checks

**Files:**
- Modify: `tools/vz-linux-image/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Write the failing host-gated checks**

Add or tighten assertions so the existing host-gated tests clearly expect a
builder-produced canonical bundle path rather than an arbitrary manually assembled
artifact.

**Step 2: Run the tests to verify current gaps**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: PASS on unprepared hosts with explicit skips; on prepared hosts, these
should be the final E2E proof for the new builder output.

**Step 3: Run privileged Linux validation**

Run on Linux:

```bash
tools/vz-linux-image/scripts/build-debian-bundle.sh --profile minimal --output-dir /tmp/vz-linux-build
```

Expected:

- `/tmp/vz-linux-build/rootfs/`
- `/tmp/vz-linux-build/rootfs.img`
- `/tmp/vz-linux-build/kernel`
- `/tmp/vz-linux-build/initrd`
- `/tmp/vz-linux-build/bundle/manifest.json`

**Step 4: Run macOS follow-on checks with the produced bundle**

Run on prepared Apple silicon macOS host:

```bash
source .venv/bin/activate
TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1 \
TLDW_SANDBOX_MACOS_HELPER_SOCKET=/tmp/macos-vz-helper.sock \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH=/tmp/vz-linux-build/bundle \
python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py -q

TLDW_SANDBOX_VZ_LINUX_E2E=1 \
TLDW_SANDBOX_MACOS_HELPER_SOCKET=/tmp/macos-vz-helper.sock \
TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=/tmp/vz-linux-build/bundle \
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: helper smoke and `vz_linux` E2E pass on a prepared host.

**Step 5: Commit**

```bash
git add tools/vz-linux-image/README.md Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "test(vz_linux): validate debian bundle builder output"
```

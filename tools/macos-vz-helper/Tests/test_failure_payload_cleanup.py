"""Small-file regressions for automatic disposable boot-payload cleanup."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def drill() -> ModuleType:
    """Load the CLI without executing host checks or creating VMs."""
    path = Path(__file__).resolve().parents[1] / "scripts/vz-failure-drill.py"
    spec = importlib.util.spec_from_file_location("payload_cleanup_drill", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocation(drill: ModuleType, bundle: Path, names: list[str]) -> object:
    """Record fixture ownership when created, independently of cleanup's safety checks."""
    paths = [bundle, *list(bundle.parents)[:4], *[bundle / name for name in names]]
    identities = {path: (path.lstat().st_dev, path.lstat().st_ino) for path in paths if path.exists()}
    return drill.BundleAllocation(names, identities)


@pytest.fixture
def packet(drill: ModuleType, tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    """Provide one invocation-owned bundle and independently verified teardown."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "rootfs.img").write_bytes(b"canonical")
    evidence = tmp_path / "evidence"
    bundle = evidence / "image-store/runs/owned/bundle"
    bundle.mkdir(parents=True)
    for name in ("rootfs.img", "vmlinuz", "initramfs"):
        (bundle / name).write_bytes(name.encode())
    for path in (
        bundle / "manifest.json",
        bundle / "fault-agent",
        bundle.parent / "manifest.json",
        evidence / "run.log",
    ):
        path.write_bytes(b"evidence")
    receipt = {
        "errors": [],
        "source_before": {"rootfs.img": "canonical-hash"},
        "source_after": {"rootfs.img": "canonical-hash"},
        "fault_sources": {"mismatch": {"rootfs.img": "fault-hash"}},
        "fault_sources_unchanged": {"mismatch": True},
        "lifecycle": {
            "vms": {"ok": True, "remaining_vm_ids": []},
            "stop": {"ok": True},
            "absent": True,
            "runtime_removed": True,
        },
    }
    return source, evidence, receipt, {bundle: allocation(drill, bundle, ["rootfs.img", "vmlinuz", "initramfs"])}


@pytest.fixture
def second_bundle(drill: ModuleType, packet: tuple) -> Path:
    """Add an independently owned later allocation to the invocation fixture."""
    _, evidence, _, allocations = packet
    second = evidence / "image-store/runs/second/bundle"
    second.mkdir(parents=True)
    (second / "rootfs.img").write_bytes(b"second")
    allocations[second] = allocation(drill, second, ["rootfs.img"])
    return second


@pytest.mark.unit
@pytest.mark.parametrize("run_failed", [False, True])
def test_cleanup_removes_payloads_but_preserves_provenance(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, run_failed: bool
) -> None:
    """Successful teardown must not retain disks merely because a case failed."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    prior = evidence / "image-store/runs/prior/bundle"
    prior.mkdir(parents=True)
    (prior / "rootfs.img").write_bytes(b"previous")
    if run_failed:
        receipt["errors"].append("case failed")
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert all(not (bundle / name).exists() for name in allocations[bundle].names)
    assert (source / "rootfs.img").read_bytes() == b"canonical"
    assert (prior / "rootfs.img").read_bytes() == b"previous"
    assert (bundle / "manifest.json").read_bytes() == b"evidence"
    assert (bundle / "fault-agent").read_bytes() == b"evidence"
    assert (bundle.parent / "manifest.json").read_bytes() == b"evidence"
    assert (evidence / "run.log").read_bytes() == b"evidence"
    entry = receipt["bundle_cleanup"]["bundles"][0]
    assert entry["payload_sha256"] == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in ("rootfs.img", "vmlinuz", "initramfs")
    }
    assert receipt["bundle_cleanup"]["ok"] is True
    assert receipt["errors"] == (["case failed"] if run_failed else [])


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("TLDW_SANDBOX_VZ_LINUX_NATIVE_CLEANUP") != "1",
    reason="requires macOS and TLDW_SANDBOX_VZ_LINUX_NATIVE_CLEANUP=1",
)
@pytest.mark.parametrize("held_open", [True, False], ids=["open-handle", "closed-handle"])
def test_native_payload_cleanup_with_real_lsof(drill: ModuleType, packet: tuple, held_open: bool) -> None:
    """Use native lsof/deletion with tiny files and synthetic source/teardown proofs."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    names = allocations[bundle].names
    rootfs = bundle / "rootfs.img"
    with rootfs.open("rb") if held_open else nullcontext():
        drill.cleanup_payloads(source, evidence, receipt, allocations)

    cleanup = receipt["bundle_cleanup"]
    entry = cleanup["bundles"][0]
    assert "disk_handles" in entry, entry.get("retained_reason")
    probe = entry["disk_handles"]
    expected_hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    assert entry["payload_sha256"] == expected_hashes
    persisted = json.loads((evidence / "receipt.json").read_text())
    assert persisted["bundle_cleanup"]["bundles"][0]["payload_sha256"] == expected_hashes
    assert probe["ok"] is (not held_open)
    # Other named payloads have no handles, so lsof exits 1 even when rootfs is open.
    assert probe["exit_code"] == 1
    assert probe["stderr"] == ""
    assert cleanup["ok"] is (not held_open)
    if held_open:
        assert f"p{os.getpid()}" in probe["stdout"].splitlines()
        assert f"n{rootfs}" in probe["stdout"].splitlines()
        assert entry["removed"] == []
        assert "closed payload handles not confirmed" in entry["retained_reason"]
        assert receipt["errors"]
        assert all((bundle / name).read_bytes() == name.encode() for name in names)
    else:
        assert probe["stdout"] == ""
        assert entry["removed"] == names
        assert receipt["errors"] == []
        assert all(not (bundle / name).exists() for name in names)
    assert {path.name for path in bundle.iterdir()} == {"manifest.json", "fault-agent", *(names if held_open else [])}
    assert (source / "rootfs.img").read_bytes() == b"canonical"
    for path in (
        bundle / "manifest.json",
        bundle / "fault-agent",
        bundle.parent / "manifest.json",
        evidence / "run.log",
    ):
        assert path.read_bytes() == b"evidence"


@pytest.mark.unit
def test_keep_bundles_is_explicit_and_does_not_probe_or_delete(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The debug override retains all allocated payloads without claiming deletion."""
    source, evidence, receipt, allocations = packet
    monkeypatch.setattr(drill, "disk_handles", lambda paths: pytest.fail("retention must not delete"))
    drill.cleanup_payloads(source, evidence, receipt, allocations, keep_bundles=True)
    assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)
    assert receipt["bundle_cleanup"]["policy"] == "keep-bundles"
    assert receipt["bundle_cleanup"]["bundles"][0]["removed"] == []


@pytest.mark.unit
@pytest.mark.parametrize("unsafe", ["unknown_vms", "remaining_vm", "stop", "absent", "runtime", "canonical", "fault"])
def test_cleanup_requires_source_verification_and_complete_teardown(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, unsafe: str
) -> None:
    """A missing or failed safety proof must retain disks and fail acceptance."""
    source, evidence, receipt, allocations = packet
    if unsafe == "unknown_vms":
        del receipt["lifecycle"]["vms"]
    elif unsafe == "remaining_vm":
        receipt["lifecycle"]["vms"]["remaining_vm_ids"] = ["still-running"]
    elif unsafe in ("absent", "runtime"):
        receipt["lifecycle"]["absent" if unsafe == "absent" else "runtime_removed"] = False
    elif unsafe == "stop":
        receipt["lifecycle"]["stop"]["ok"] = False
    elif unsafe == "canonical":
        del receipt["source_after"]
    else:
        receipt["fault_sources_unchanged"]["mismatch"] = False
    monkeypatch.setattr(drill, "disk_handles", lambda paths: pytest.fail("unsafe teardown must retain"))
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)
    assert receipt["bundle_cleanup"]["ok"] is False
    assert receipt["errors"]


@pytest.mark.unit
@pytest.mark.parametrize("probe", ["open", "unknown", "raised"])
def test_cleanup_requires_a_fresh_closed_handle_probe(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    """Previously closed disks are not enough; final deletion needs a new probe."""
    source, evidence, receipt, allocations = packet

    def disk_handles(paths: list[Path]) -> dict:
        """Model live OS uncertainty after the helper has stopped."""
        if probe == "raised":
            raise OSError("lsof unavailable")
        return {"ok": False, "reason": probe}

    monkeypatch.setattr(drill, "disk_handles", disk_handles)
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
@pytest.mark.parametrize("unsafe", ["outside", "bundle_symlink", "ancestor_symlink", "payload_symlink", "traversal"])
def test_cleanup_never_follows_unowned_or_symlink_paths(
    drill: ModuleType, packet: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str
) -> None:
    """Path replacement or traversal must not turn cleanup into arbitrary deletion."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    if unsafe == "outside":
        allocations = {source: allocation(drill, source, ["rootfs.img"])}
    elif unsafe == "bundle_symlink":
        moved = bundle.with_name("moved")
        bundle.rename(moved)
        bundle.symlink_to(moved, target_is_directory=True)
    elif unsafe == "ancestor_symlink":
        runs = bundle.parent.parent
        moved = evidence / "moved-runs"
        runs.rename(moved)
        runs.symlink_to(moved, target_is_directory=True)
    elif unsafe == "payload_symlink":
        (bundle / "rootfs.img").unlink()
        (bundle / "rootfs.img").symlink_to(source / "rootfs.img")
    else:
        allocations[bundle] = allocation(drill, bundle, ["../manifest.json"])
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (source / "rootfs.img").read_bytes() == b"canonical"
    assert (bundle / "vmlinuz").read_bytes() == b"vmlinuz"
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["before_cleanup", "hash", "probe"])
@pytest.mark.parametrize("replaced", ["payload", "bundle", "ancestor"])
def test_cleanup_retains_objects_replaced_during_hashing_or_handle_checks(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, stage: str, replaced: str
) -> None:
    """Regular-file replacements must not inherit the original allocation's proof."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    changed = False

    def replace() -> None:
        """Keep the old objects alive so inode reuse cannot obscure the replacement."""
        nonlocal changed
        if changed:
            return
        changed = True
        if replaced == "payload":
            (bundle / "rootfs.img").rename(bundle / "saved.img")
        else:
            parent = bundle if replaced == "bundle" else bundle.parent
            parent.rename(parent.with_name("saved"))
            bundle.mkdir(parents=True)
        (bundle / "rootfs.img").write_bytes(b"unowned replacement")

    original_digest = drill.digest

    def digest(path: Path) -> str:
        """Replace immediately after reading the first original artifact."""
        value = original_digest(path)
        if stage == "hash":
            replace()
        return value

    def probe(paths: list[Path]) -> dict:
        """Return closed handles for the original object, then replace its name."""
        if stage == "probe":
            replace()
        return {"ok": True}

    monkeypatch.setattr(drill, "digest", digest)
    monkeypatch.setattr(drill, "disk_handles", probe)
    if stage == "before_cleanup":
        replace()
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (bundle / "rootfs.img").read_bytes() == b"unowned replacement"
    assert receipt["bundle_cleanup"]["ok"] is False
    assert receipt["errors"]


@pytest.mark.unit
def test_shared_directory_replacement_does_not_adopt_later_unowned_bundles(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced runs root invalidates all original allocations beneath it."""
    source, evidence, receipt, allocations = packet
    second = evidence / "image-store/runs/second/bundle"
    second.mkdir(parents=True)
    (second / "rootfs.img").write_bytes(b"original-second")
    allocations[second] = allocation(drill, second, ["rootfs.img"])
    changed = False

    def probe(paths: list[Path]) -> dict:
        """Substitute an unrelated tree between safety proof and destruction."""
        nonlocal changed
        if not changed:
            changed = True
            runs = evidence / "image-store/runs"
            runs.rename(evidence / "saved-runs")
            second.mkdir(parents=True)
            (second / "rootfs.img").write_bytes(b"unowned-second")
        return {"ok": True}

    monkeypatch.setattr(drill, "disk_handles", probe)
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (second / "rootfs.img").read_bytes() == b"unowned-second"
    assert receipt["bundle_cleanup"]["ok"] is False
    assert all(not entry["removed"] for entry in receipt["bundle_cleanup"]["bundles"])


@pytest.mark.unit
def test_in_place_payload_mutation_invalidates_recorded_hashes(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a writer before lsof must not make a stale digest trustworthy."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))

    def probe(paths: list[Path]) -> dict:
        """Change bytes without replacing the inode, then report no open handles."""
        (bundle / "rootfs.img").write_bytes(b"mutated-in-place")
        return {"ok": True}

    monkeypatch.setattr(drill, "disk_handles", probe)
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (bundle / "rootfs.img").read_bytes() == b"mutated-in-place"
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
def test_signal_after_unlink_records_deletion_before_continuing_cleanup(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal delivery must not land between unlink and deletion bookkeeping."""
    source, evidence, receipt, allocations = packet
    original_unlink = os.unlink
    raised = False

    def unlink(path: str, *args: object, **kwargs: object) -> None:
        """Deliver the actual handled signal immediately after a successful unlink."""
        nonlocal raised
        original_unlink(path, *args, **kwargs)
        if str(path) == "rootfs.img" and kwargs.get("dir_fd") is not None and not raised:
            raised = True
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert receipt["bundle_cleanup"]["bundles"][0]["removed"] == ["rootfs.img"]
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
def test_cleanup_records_unlink_failure_and_still_cleans_other_bundles(
    drill: ModuleType, packet: tuple, second_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One operational failure must not abandon later independently owned disks."""
    source, evidence, receipt, allocations = packet
    first = next(iter(allocations))
    second = second_bundle
    first_inode = (first / "rootfs.img").stat().st_ino
    unlink = os.unlink

    def fail_first(path: str, *args: object, **kwargs: object) -> None:
        """Inject a permission failure for one payload only."""
        if kwargs.get("dir_fd") is not None and os.stat(path, dir_fd=kwargs["dir_fd"]).st_ino == first_inode:
            raise PermissionError("denied")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_first)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (first / "rootfs.img").exists()
    assert not (second / "rootfs.img").exists()
    assert receipt["bundle_cleanup"]["ok"] is False
    assert receipt["errors"]


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["hash", "probe", "unlink"])
def test_cancellation_inside_cleanup_records_failure_and_continues_other_allocations(
    drill: ModuleType, packet: tuple, second_bundle: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """A signal while cleaning one disk must not orphan every later allocation."""
    source, evidence, receipt, allocations = packet
    first = next(iter(allocations))
    second = second_bundle
    raised = False

    def interrupt_once() -> None:
        """Deliver one ordinary cancellation, as the installed signal handler does."""
        nonlocal raised
        if not raised:
            raised = True
            raise KeyboardInterrupt("cancel cleanup")

    original_digest, original_unlink = drill.digest, os.unlink

    def digest(path: Path) -> str:
        """Interrupt before the first hash can establish deletion provenance."""
        if stage == "hash":
            interrupt_once()
        return original_digest(path)

    def probe(paths: list[Path]) -> dict:
        """Interrupt before confirming closed handles for the first allocation."""
        if stage == "probe":
            interrupt_once()
        return {"ok": True}

    def unlink(path: str, *args: object, **kwargs: object) -> None:
        """Interrupt exactly at an allocated payload's destructive boundary."""
        if stage == "unlink" and str(path) == "rootfs.img" and kwargs.get("dir_fd") is not None:
            interrupt_once()
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(drill, "digest", digest)
    monkeypatch.setattr(drill, "disk_handles", probe)
    monkeypatch.setattr(os, "unlink", unlink)
    try:
        drill.cleanup_payloads(source, evidence, receipt, allocations)
    except KeyboardInterrupt:
        pass  # The old implementation abandons all later allocations here.
    assert (first / "rootfs.img").is_file()
    assert not (second / "rootfs.img").exists()
    assert receipt["bundle_cleanup"]["ok"] is False
    assert receipt["errors"]


@pytest.mark.unit
def test_clone_tracks_partial_allocation_before_materialization(drill: ModuleType, tmp_path: Path) -> None:
    """A failed copy can leave a disk even though prepare_bundle never returns."""
    evidence = tmp_path / "evidence"
    bundle = evidence / "image-store/runs/partial/bundle"

    def prepare(args: object) -> Path:
        """Leave one artifact behind, as an interrupted materializer can."""
        bundle.mkdir(parents=True)
        (bundle / "rootfs.img").write_bytes(b"partial")
        raise OSError("copy interrupted")

    materializer = SimpleNamespace(
        parse_args=lambda args: args,
        resolve_run_bundle_path=lambda args: bundle,
        bundle_artifact_names=lambda source: ["vmlinuz", "rootfs.img", "initramfs"],
        prepare_bundle=prepare,
    )
    disks: list[Path] = []
    allocations: dict = {}
    with pytest.raises(OSError, match="copy interrupted"):
        drill.clone(materializer, tmp_path / "source", evidence, "partial", "healthy", disks, allocations)
    assert allocations[bundle].names == ["vmlinuz", "rootfs.img", "initramfs"]
    state = (bundle / "rootfs.img").lstat()
    assert allocations[bundle].identities[bundle / "rootfs.img"] == (state.st_dev, state.st_ino)
    assert disks == [bundle / "rootfs.img"]


@pytest.mark.unit
def test_cleanup_does_not_delete_without_durable_provenance(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed receipt persistence must leave the original payloads inspectable."""
    source, evidence, receipt, allocations = packet

    def unwritable(*args: object) -> None:
        """Model an unavailable evidence destination before destruction."""
        raise OSError("receipt unavailable")

    monkeypatch.setattr(drill, "write_json", unwritable)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: pytest.fail("must persist before deleting"))
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
def test_failed_receipt_rewrite_preserves_hashes_for_already_removed_payloads(
    drill: ModuleType, packet: tuple, second_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial later write must not truncate the only proof for deleted disks."""
    source, evidence, receipt, allocations = packet
    first = next(iter(allocations))
    second = second_bundle
    original_write = Path.write_text
    writes = 0

    def failing_write(path: Path, text: str, *args: object, **kwargs: object) -> int:
        """Model an actual partial filesystem write, not failure before truncation."""
        nonlocal writes
        writes += 1
        if writes == 2:
            original_write(path, "{", *args, **kwargs)
            raise OSError("disk full after truncation")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert not (first / "rootfs.img").exists()
    assert (second / "rootfs.img").exists()
    stored = json.loads((evidence / "receipt.json").read_text())
    assert (
        stored["bundle_cleanup"]["bundles"][0]["payload_sha256"]["rootfs.img"]
        == hashlib.sha256(b"rootfs.img").hexdigest()
    )
    assert receipt["bundle_cleanup"]["ok"] is False
    assert not list(evidence.glob(".receipt.json.*"))


@pytest.mark.unit
def test_partial_payloads_are_removed_without_requiring_uncopied_artifacts(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted allocation must not leak its disk because kernel copying failed."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    (bundle / "vmlinuz").unlink()
    (bundle / "initramfs").unlink()
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert not (bundle / "rootfs.img").exists()
    assert receipt["bundle_cleanup"]["bundles"][0]["removed"] == ["rootfs.img"]
    assert receipt["bundle_cleanup"]["ok"] is True


@pytest.mark.unit
@pytest.mark.parametrize("name", ["manifest.json", "build-info.json"])
def test_cleanup_preserves_metadata_even_if_a_manifest_selects_it_as_boot_input(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An unusual boot manifest must not make provenance eligible for deletion."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    (bundle / name).write_bytes(b"provenance")
    allocations[bundle] = allocation(drill, bundle, ["rootfs.img", name])
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert (bundle / name).read_bytes() == b"provenance"
    assert (bundle / "rootfs.img").is_file()
    assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome", ["success", "failure", "cancelled", "setup_failed", "verification_cancelled", "keep", "cleanup_failure"]
)
def test_cli_finalizes_payload_cleanup_after_source_verification(
    drill: ModuleType, packet: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """The CLI must clean on every ordinary exit, with hashes durable before unlink."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    materializer = SimpleNamespace(
        validate_bundle=lambda path: None,
        read_optional_json=lambda path: {},
        bundle_artifact_names=lambda path: ["rootfs.img"],
    )
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", "1")
    monkeypatch.setattr(drill, "create_evidence", lambda path, src: evidence)
    monkeypatch.setattr(drill, "load_module", lambda name, path: materializer)
    monkeypatch.setattr(drill.sys, "platform", "darwin")
    monkeypatch.setattr(drill.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(drill, "logged", lambda *args: 0)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": outcome != "cleanup_failure"})
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client",
        SimpleNamespace(MacOSVirtualizationHelperClient=object),
    )
    fingerprint = drill.fingerprint
    source_reads = 0

    def interrupt_final_verification(module: ModuleType, path: Path) -> dict:
        """Cancel only the final canonical read, after allocation and teardown."""
        nonlocal source_reads
        source_reads += 1
        if outcome == "verification_cancelled" and source_reads == 2:
            raise KeyboardInterrupt("cancel source verification")
        return fingerprint(module, path)

    monkeypatch.setattr(drill, "fingerprint", interrupt_final_verification)

    def exercise(*args: object) -> None:
        """Leave real files for finalization, but replace compilation and VM RPCs."""
        actual_receipt = args[5]
        actual_receipt["lifecycle"] = receipt["lifecycle"]
        if len(args) > 7:
            args[7].update(allocations)
        if outcome == "setup_failed":
            raise RuntimeError("offline installation failed before source baseline")
        actual_receipt["cases"] = {
            profile + suffix: {"ok": True} for profile in drill.TESTS for suffix in ("-positive", "-negative")
        }
        if outcome == "failure":
            raise RuntimeError("case failed")
        if outcome == "cancelled":
            raise KeyboardInterrupt("cancelled")

    monkeypatch.setattr(drill, "exercise", exercise)
    original_unlink = os.unlink

    def verify_before_unlink(path: str, *args: object, **kwargs: object) -> None:
        """Check persisted verification and payload hashes at the destructive boundary."""
        stored = json.loads((evidence / "receipt.json").read_text())
        assert stored["source_after"] == stored["source_before"]
        assert stored["bundle_cleanup"]["bundles"][0]["payload_sha256"][str(path)]
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", verify_before_unlink)
    argv = [
        "--allow-fault-injection",
        "--source-bundle",
        str(source),
        "--helper",
        str(helper),
        "--evidence-dir",
        str(evidence),
    ]
    if outcome == "keep":
        argv.append("--keep-bundles")
    assert drill.main(argv) == int(
        outcome in ("failure", "cancelled", "setup_failed", "verification_cancelled", "cleanup_failure")
    )
    assert (bundle / "rootfs.img").exists() is (outcome in ("keep", "verification_cancelled", "cleanup_failure"))
    final = json.loads((evidence / "receipt.json").read_text())
    assert final["bundle_cleanup"]["ok"] is (outcome not in ("verification_cancelled", "cleanup_failure"))
    if outcome == "setup_failed":
        assert not final.get("fault_sources")
        assert final["bundle_cleanup"]["bundles"][0]["payload_sha256"]


@pytest.mark.unit
def test_clone_rejects_names_that_image_store_would_normalize(drill: ModuleType, tmp_path: Path) -> None:
    """Reject ambiguous input before allocating a clone with different filenames."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"kernel": " kernel "}))
    for name in (" kernel ", "kernel", "rootfs.img"):
        (source / name).write_bytes(b"small fixture")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    materializer = drill.load_module(
        "cleanup_real_materializer", drill.REPO / "tools/vz-linux-image/scripts/prepare-smoke-bundle.py"
    )
    allocations = {}
    with pytest.raises(ValueError, match="whitespace"):
        drill.clone(materializer, source, evidence, "ambiguous", "healthy", [], allocations)
    assert allocations == {}
    assert not (evidence / "image-store/runs/ambiguous/bundle").exists()


@pytest.mark.unit
@pytest.mark.parametrize("replaced", ["payload", "occupied_restore", "bundle", "ancestor"])
def test_cleanup_preserves_replacement_at_deletion_boundary(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, replaced: str
) -> None:
    """Replacement after the last path check cannot redirect deletion to new files."""
    source, evidence, receipt, allocations = packet
    bundle = next(iter(allocations))
    changed = False
    unlink, rename = Path.unlink, os.rename

    def replace() -> None:
        """Substitute an object only at the final destructive namespace operation."""
        nonlocal changed
        if changed:
            return
        changed = True
        if replaced in ("payload", "occupied_restore"):
            (bundle / "rootfs.img").rename(bundle / "saved.img")
        else:
            parent = bundle if replaced == "bundle" else bundle.parent
            parent.rename(parent.with_name("saved"))
            bundle.mkdir(parents=True)
        (bundle / "rootfs.img").write_bytes(b"unowned replacement")

    def boundary_unlink(path: Path, *args: object, **kwargs: object) -> None:
        """Exercise the original direct-unlink implementation."""
        if path == bundle / "rootfs.img":
            replace()
        unlink(path, *args, **kwargs)

    def boundary_rename(src: object, dst: object, *args: object, **kwargs: object) -> None:
        """Exercise the atomic claim used instead of direct pathname deletion."""
        if kwargs.get("src_dir_fd") is not None and str(src) == "rootfs.img":
            replace()
        rename(src, dst, *args, **kwargs)
        if replaced == "occupied_restore" and kwargs.get("src_dir_fd") is not None:
            (bundle / "rootfs.img").write_bytes(b"newer occupied name")

    monkeypatch.setattr(Path, "unlink", boundary_unlink)
    monkeypatch.setattr(os, "rename", boundary_rename)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert changed
    if replaced == "occupied_restore":
        assert (bundle / "rootfs.img").read_bytes() == b"newer occupied name"
        retained = Path(receipt["bundle_cleanup"]["bundles"][0]["retained_path"])
        assert retained.read_bytes() == b"unowned replacement"
    else:
        assert (bundle / "rootfs.img").read_bytes() == b"unowned replacement"
    if replaced in ("payload", "occupied_restore"):
        assert receipt["bundle_cleanup"]["ok"] is False


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["uuid", "staging_open"])
def test_claim_setup_failure_does_not_leak_descriptors(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Cancellation or staging failure must close every acquired descriptor."""
    source, evidence, receipt, allocations = packet
    opened = set()
    original_open, original_close = os.open, os.close

    def open_descriptor(path: object, *args: object, **kwargs: object) -> int:
        """Track real descriptors and inject failure before staging is opened."""
        if stage == "staging_open" and str(path).startswith(".payload-cleanup-"):
            raise OSError("staging unavailable")
        fd = original_open(path, *args, **kwargs)
        opened.add(fd)
        return fd

    def close_descriptor(fd: int) -> None:
        """Remove only descriptors actually closed by production code."""
        original_close(fd)
        opened.discard(fd)

    def cancel_uuid() -> None:
        """Model an ordinary handled cancellation during claim setup."""
        raise KeyboardInterrupt("cancel staging name")

    monkeypatch.setattr(os, "open", open_descriptor)
    monkeypatch.setattr(os, "close", close_descriptor)
    if stage == "uuid":
        monkeypatch.setattr(drill, "uuid4", cancel_uuid)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: {"ok": True})
    try:
        drill.cleanup_payloads(source, evidence, receipt, allocations)
        assert not opened
        assert receipt["bundle_cleanup"]["ok"] is False
        assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)
    finally:
        for fd in opened:
            original_close(fd)


@pytest.mark.unit
def test_cancelled_prepared_source_verification_retains_payloads(
    drill: ModuleType, packet: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified canonical source cannot substitute for a canceled prepared source."""
    source, evidence, receipt, allocations = packet

    def fingerprint(materializer: object, path: Path) -> dict:
        """Verify the canonical source, then cancel an existing prepared baseline."""
        if path == source:
            return receipt["source_before"]
        raise KeyboardInterrupt("cancel prepared verification")

    monkeypatch.setattr(drill, "fingerprint", fingerprint)
    monkeypatch.setattr(drill, "disk_handles", lambda paths: pytest.fail("unverified source must retain"))
    drill.verify_sources(None, source, evidence, receipt)
    drill.cleanup_payloads(source, evidence, receipt, allocations)
    assert receipt["fault_sources_unchanged"]["mismatch"] is False
    assert receipt["bundle_cleanup"]["ok"] is False
    assert all((bundle / name).is_file() for bundle, record in allocations.items() for name in record.names)

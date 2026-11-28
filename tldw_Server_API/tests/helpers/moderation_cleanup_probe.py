"""Benign fault injection for actual controller cleanup ownership, stdlib only."""

import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path


def drive() -> int:
    failure, release, owner, controller, timeout, *argv = sys.argv[2:]
    namespace = runpy.run_path(controller)
    tree_type = namespace["PosixTree"]
    original = tree_type.cleanup

    def cleanup(tree, deadline):
        temporary = Path(owner).with_suffix(".new")
        temporary.write_text(json.dumps({"controller": os.getpid(), "group": tree.pgid}))
        temporary.replace(owner)
        if not Path(release).exists():
            if failure == "raise":
                raise OSError("injected benign cleanup failure")
            return False
        return original(tree, deadline)

    tree_type.cleanup = cleanup
    sys.argv = [controller, timeout, *argv]
    return namespace["main"]()


def invoke(directory: Path, failure: str):
    helper = Path(__file__).with_name("moderation_process_harness.py").resolve()
    run = runpy.run_path(str(helper))["run_isolated_probe"]
    native = subprocess.Popen
    owner, release, pids = (directory / name for name in ("owner.json", "release", "pids"))

    def launch(argv, **kwargs):
        assert Path(argv[3]).name == "moderation_probe_controller.py"
        return native(
            [*argv[:3], str(Path(__file__).resolve()), "driver", failure, str(release), str(owner), *argv[3:]], **kwargs
        )

    probe = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        (
            "import os,subprocess,sys;"
            "p=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(8)'],close_fds=True);"
            f"open({str(pids)!r},'w').write(str(os.getpid())+' '+str(p.pid));os._exit(7)"
        ),
    ]
    subprocess.Popen = launch
    try:
        try:
            run(probe, timeout_s=1)
        except RuntimeError:
            return run.__globals__
        raise AssertionError("unconfirmed containment was accepted")
    finally:
        subprocess.Popen = native


def alive(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def validate_cleanup_recovery(directory: Path, *, failure: str, disconnect: bool = False) -> None:
    """Verify a live independent owner survives refusal/exception and invoker loss."""
    owner, release, pids = (directory / name for name in ("owner.json", "release", "pids"))
    state = invoker = None
    try:
        if disconnect:
            invoker = subprocess.Popen(
                [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "invoker", failure, str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            assert invoker.wait(timeout=5) == 0, "invoker did not retain a live tree owner"
        else:
            state = invoke(directory, failure)
            controller = state["_pending_controller"]
            assert (
                controller is not None and controller.poll() is None
            ), "cleanup owner died/released before confirmation"
            try:
                state["run_isolated_probe"]([sys.executable], timeout_s=0.01)
            except RuntimeError as error:
                assert "capacity" in str(error)
            else:
                raise AssertionError("replacement admitted while cleanup was unconfirmed")
        record = json.loads(owner.read_text())
        recorded_pids = list(map(int, pids.read_text().split()))
        assert record["group"] == recorded_pids[0]
        assert alive(record["controller"]), "a dead controller reference is not an owner"
        assert alive(recorded_pids[1]), "probe did not leave a live benign descendant"
        release.touch()
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if not any(alive(pid) for pid in [record["controller"], *recorded_pids]):
                if state is None or state["_pending_controller"] is None:
                    break
            time.sleep(0.01)
        assert not any(alive(pid) for pid in [record["controller"], *recorded_pids])
        if state is not None:
            assert state["_pending_controller"] is None
            result = state["run_isolated_probe"]([sys.executable, "-I", "-S", "-c", "pass"])
            assert result.returncode == 0
    finally:
        release.touch()
        # RED-run safety net for the old controller that already abandoned this
        # bounded benign group. The fixed path must prove its LIVE owner above.
        if owner.exists():
            record = json.loads(owner.read_text())
            if not alive(record["controller"]):
                try:
                    os.killpg(record["group"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if invoker is not None and invoker.poll() is None:
            invoker.kill()
            invoker.wait(timeout=2)


if __name__ == "__main__":
    if sys.argv[1] == "driver":
        raise SystemExit(drive())
    local_state = invoke(Path(sys.argv[3]), sys.argv[2])
    active = local_state["_pending_controller"]
    if active is None or active.poll() is not None:
        os._exit(1)
    os._exit(0)

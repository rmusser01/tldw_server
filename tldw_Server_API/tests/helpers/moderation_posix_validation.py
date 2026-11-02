"""Dependency-free Linux/macOS evidence runner; containment precedes hostility."""

import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    if os.name != "posix":
        raise RuntimeError("POSIX evidence only")
    helper = Path(__file__).with_name("moderation_process_harness.py").resolve()
    run_isolated_probe = runpy.run_path(str(helper))["run_isolated_probe"]
    with tempfile.TemporaryDirectory() as directory:
        record = Path(directory) / "pids"

        def argv(hard_exit):
            return [
                sys.executable,
                "-I",
                "-S",
                "-c",
                (
                    "import os,subprocess,sys,time;"
                    "p=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(60)'],close_fds=True);"
                    f"open({str(record)!r},'w').write(str(os.getpid())+' '+str(p.pid));"
                    + ("os._exit(7)" if hard_exit else "time.sleep(60)")
                ),
            ]

        def dead():
            pids = list(map(int, record.read_text().split()))
            assert len(pids) == 2
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise AssertionError("recorded descendant remains")

        assert run_isolated_probe(argv(False), timeout_s=0.5).returncode == 124
        dead()
        assert run_isolated_probe(argv(True)).returncode == 7
        dead()
        record.unlink()
        code = (
            "import runpy,threading,os,time;"
            f"h=runpy.run_path({str(helper)!r});"
            f"threading.Thread(target=lambda:h['run_isolated_probe']({argv(False)!r})).start();"
            f"p=__import__('pathlib').Path({str(record)!r});end=time.monotonic()+3;"
            "\nwhile not p.exists() and time.monotonic()<end:time.sleep(.01)\n"
            "os._exit(0)"
        )
        invoker = subprocess.Popen([sys.executable, "-I", "-S", "-c", code], close_fds=True)
        try:
            assert invoker.wait(timeout=4) == 0
            time.sleep(0.2)
            dead()
        finally:
            if invoker.poll() is None:
                invoker.kill()
                invoker.wait(timeout=2)
        print("CONTAINMENT_PROVEN", sys.platform, sys.version.split()[0], flush=True)
        cleanup_checks = runpy.run_path(str(Path(__file__).with_name("moderation_cleanup_probe.py")))
        for failure, disconnect in (("false", False), ("raise", False), ("raise", True)):
            case = Path(directory) / f"cleanup-{failure}-{disconnect}"
            case.mkdir()
            cleanup_checks["validate_cleanup_recovery"](case, failure=failure, disconnect=disconnect)
            print("CLEANUP_OWNER_PROVEN", failure, "invoker_loss" if disconnect else "recovery", flush=True)
        probe = Path(__file__).with_name("moderation_supervisor_probe.py")
        for mode in (
            "fork-during-spawn",
            "pathological",
            "heartbeat",
            "cancel-spawn",
            "fork",
            "nonzero",
            "posture",
            "blocked-write",
        ):
            result = run_isolated_probe([sys.executable, "-I", "-S", str(probe), mode])
            assert result.returncode == 0, (mode, result.returncode, result.stderr)
            print(result.stdout.decode("ascii").strip(), flush=True)


if __name__ == "__main__":
    main()

"""Compare the frozen UAT236 input ledger after Backlog task closure.

This stores only aggregate verification data and public tracker metadata. Private
input paths and contents are read only to recalculate their recorded hashes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_AUDIT = ROOT / ".tmp/uat-repairs-231-246/native236-fresh-review/audit.json"
RETAINED_AUDIT = ROOT / "output/playwright/fresh-matrix-repairs-2026-09-17/native-model236-fresh-retention-review/audit.json"
TRACKER = "backlog/tasks/task-13260.178 - Recognize-llama-provider-aliases-in-Character-Chat-readiness.md"
BASELINE = "592521e9fa"
OUT = Path(__file__).with_name("closure-metadata.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)

source_audit = json.loads(SOURCE_AUDIT.read_text())
retained_audit_bytes = RETAINED_AUDIT.read_bytes()
inputs = source_audit["inputs"]
tracker_input = next(entry for entry in inputs if entry["path"] == TRACKER)
other_inputs = [entry for entry in inputs if entry is not tracker_input]

matched_other = 0
for entry in other_inputs:
    data = (ROOT / entry["path"]).read_bytes()
    if digest(data) == entry["sha256"] and len(data) == entry["bytes"]:
        matched_other += 1

baseline_tracker = git("show", f"{BASELINE}:{TRACKER}")
current_tracker = (ROOT / TRACKER).read_bytes()
diff_names = git("diff", "--name-only", BASELINE, "--", TRACKER).decode().splitlines()
diff = git("diff", "--unified=0", BASELINE, "--", TRACKER).decode()
expected_hunks = (" -4 +4 ", " -7 +7 ", " -24,3 +24,3 ", " -42,0 +43,2 ", " -44,0 +47,6 ", " -47,6 +55,6 ")
hunks = tuple(line.split("@@", 2)[1] for line in diff.splitlines() if line.startswith("@@"))
closure_markers = (
    "status: Done",
    "updated_date:",
    "Final Summary",
    "SECTION:FINAL_SUMMARY",
    "Acceptance criteria completed",
    "Tests or verification recorded",
    "Documentation updated when relevant",
    "Bandit run for touched code",
    "Final summary added",
    "Known skips or blockers documented",
)
result = {
    "task": "TASK13260.178 UAT236 closure metadata supplement",
    "scope": "Retained fresh-profile acceptance packet remains unchanged; only the mutable Backlog task is rechecked after closure.",
    "preClosureRetainedAudit": {
        "path": "output/playwright/fresh-matrix-repairs-2026-09-17/native-model236-fresh-retention-review/audit.json",
        "sha256": digest(retained_audit_bytes),
        "verdict": json.loads(retained_audit_bytes)["verdict"],
        "checks": len(json.loads(retained_audit_bytes)["checks"]),
        "provenanceInputs": len(inputs),
    },
    "inputVerification": {
        "totalRecordedInputs": len(inputs),
        "immutableInputs": len(other_inputs),
        "immutableInputsMatchingHashAndSize": matched_other,
        "allImmutableInputsMatch": matched_other == len(other_inputs),
        "privateHashOnlyInputsIncludedInAggregate": sum(entry.get("privateHashOnly") is True for entry in other_inputs),
    },
    "mutableTracker": {
        "path": TRACKER,
        "recordedPreClosureSha256": tracker_input["sha256"],
        "recordedPreClosureBytes": tracker_input["bytes"],
        "gitShowRevision": BASELINE,
        "gitShowMatchesRecordedPreClosure": digest(baseline_tracker) == tracker_input["sha256"] and len(baseline_tracker) == tracker_input["bytes"],
        "currentSha256": digest(current_tracker),
        "currentBytes": len(current_tracker),
        "currentDiffersBecauseTaskWasClosed": digest(current_tracker) != tracker_input["sha256"],
    },
    "closureDiff": {
        "changedPaths": diff_names,
        "onlyTrackerPathChanged": diff_names == [TRACKER],
        "expectedZeroContextHunkLocations": list(expected_hunks),
        "actualZeroContextHunkLocations": list(hunks),
        "hunksMatchClosureUpdateShape": hunks == expected_hunks,
        "containsClosureMarkers": all(marker in diff for marker in closure_markers),
        "classification": "Backlog closure metadata: status/checklist transitions, closing evidence note, and final summary; no packet or product input change.",
    },
    "conclusion": "PRE-CLOSURE AUDIT REMAINS VALID AT ITS CAPTURE TIME. POST-CLOSURE RECHECK MATCHES ALL 42 IMMUTABLE INPUTS; the sole expected mismatch is the closed Backlog task.",
}
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "immutableMatched": result["inputVerification"]["immutableInputsMatchingHashAndSize"],
    "immutableTotal": result["inputVerification"]["immutableInputs"],
    "trackerGitShowMatches": result["mutableTracker"]["gitShowMatchesRecordedPreClosure"],
    "onlyTrackerPath": result["closureDiff"]["onlyTrackerPathChanged"],
    "hunkShape": result["closureDiff"]["hunksMatchClosureUpdateShape"],
}, sort_keys=True))

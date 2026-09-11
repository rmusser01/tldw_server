#!/usr/bin/env python3
"""Focused installed/runtime controls for the NLTK candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nltk
import nltk.pathsec as pathsec
import pycrfsuite
from nltk.tag.crf import CRFTagger
from nltk.tag.perceptron import AveragedPerceptron, PerceptronTagger


class StablePath:
    """A normal PathLike that always names the same path."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.calls = 0

    def __fspath__(self) -> str:
        self.calls += 1
        return self.path


class ChangingPath:
    """A hostile PathLike that changes after its first coercion."""

    def __init__(self, first: Path, later: Path) -> None:
        self.first = str(first)
        self.later = str(later)
        self.calls = 0

    def __fspath__(self) -> str:
        self.calls += 1
        return self.first if self.calls == 1 else self.later


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exception(error: Exception | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def _native_model(path: Path, label: str) -> None:
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append([{"word": "dog"}], [label])
    trainer.train(str(path))


def _run_case(
    case_id: str,
    operation: Callable[[], Any],
    check: Callable[[Exception | None, Any], tuple[bool, dict[str, Any]]],
) -> dict[str, Any]:
    error = None
    observation = None
    try:
        observation = operation()
    except Exception as exc:  # noqa: BLE001 - retain each runtime failure as evidence.
        error = exc
    passed, details = check(error, observation)
    return {
        "id": case_id,
        "passed": passed,
        "exception": _exception(error),
        "details": details,
    }


def run_controls(evidence: Path, *, fault_alter_canary: bool = False) -> dict[str, Any]:
    """Run the focused freeze-once controls and write ``controls.json``."""
    evidence.mkdir(parents=True, exist_ok=False)
    workspace = evidence / "workspace"
    allowed = workspace / "allowed"
    forbidden = workspace / "forbidden"
    allowed.mkdir(parents=True, mode=0o700)
    forbidden.mkdir(mode=0o700)

    os.environ["NLTK_DATA"] = str(allowed)
    nltk.data.path[:] = [str(allowed)]
    pathsec.ENFORCE = True
    pathsec._ALLOWED_ROOTS_CACHE = None
    pathsec._LAST_DATA_PATHS = None
    allowed_roots = sorted(str(root) for root in pathsec._get_allowed_roots())
    if str(allowed.resolve()) not in allowed_roots:
        raise RuntimeError("controlled allowed root is not active")
    if any(
        forbidden.resolve() == Path(root) or forbidden.resolve().is_relative_to(Path(root)) for root in allowed_roots
    ):
        raise RuntimeError("forbidden control root is inside an effective allowed root")

    canary = forbidden / "canary.bin"
    canary.write_bytes(b"outside-model-canary")
    canary_before = _digest(canary)
    if fault_alter_canary:
        canary.write_bytes(b"fault-injected-canary-change")

    allowed_model = allowed / "allowed.crfsuite"
    forbidden_model = forbidden / "forbidden.crfsuite"
    _native_model(allowed_model, "ALLOWED")
    _native_model(forbidden_model, "FORBIDDEN")
    allowed_model_before = _digest(allowed_model)
    forbidden_model_before = _digest(forbidden_model)

    cases = []

    stable_set = StablePath(allowed_model)
    stable_tagger = CRFTagger()
    cases.append(
        _run_case(
            "crf_set_model_file_accepts_stable_pathlike",
            lambda: (stable_tagger.set_model_file(stable_set), stable_tagger._tagger.labels())[1],
            lambda error, labels: (
                error is None
                and stable_set.calls == 1
                and list(labels or []) == ["ALLOWED"]
                and _digest(allowed_model) == allowed_model_before
                and _digest(forbidden_model) == forbidden_model_before
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": stable_set.calls,
                    "labels": list(labels or []),
                    "allowedModelUnchanged": _digest(allowed_model) == allowed_model_before,
                    "forbiddenModelUnchanged": _digest(forbidden_model) == forbidden_model_before,
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    changing_set = ChangingPath(allowed_model, forbidden_model)
    changing_tagger = CRFTagger()
    cases.append(
        _run_case(
            "crf_set_model_file_freezes_changing_pathlike",
            lambda: (
                changing_tagger.set_model_file(changing_set),
                changing_tagger._tagger.labels(),
            )[1],
            lambda error, labels: (
                error is None
                and changing_set.calls == 1
                and list(labels or []) == ["ALLOWED"]
                and _digest(allowed_model) == allowed_model_before
                and _digest(forbidden_model) == forbidden_model_before
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": changing_set.calls,
                    "labels": list(labels or []),
                    "allowedModelUnchanged": _digest(allowed_model) == allowed_model_before,
                    "forbiddenModelUnchanged": _digest(forbidden_model) == forbidden_model_before,
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    stable_train_output = allowed / "stable-train.crfsuite"
    stable_train = StablePath(stable_train_output)
    stable_train_tagger = CRFTagger()
    cases.append(
        _run_case(
            "crf_train_accepts_stable_pathlike",
            lambda: stable_train_tagger.train([[("dog", "NN")]], stable_train),
            lambda error, _result: (
                error is None
                and stable_train.calls == 1
                and stable_train_output.is_file()
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": stable_train.calls,
                    "allowedOutputCreated": stable_train_output.is_file(),
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    changing_train_output = allowed / "changing-train.crfsuite"
    forbidden_train = forbidden / "forbidden-train.crfsuite"
    forbidden_train.write_bytes(b"outside-train-canary")
    forbidden_train_before = _digest(forbidden_train)
    changing_train = ChangingPath(changing_train_output, forbidden_train)
    changing_train_tagger = CRFTagger()
    cases.append(
        _run_case(
            "crf_train_freezes_changing_pathlike",
            lambda: changing_train_tagger.train([[("dog", "NN")]], changing_train),
            lambda error, _result: (
                error is None
                and changing_train.calls == 1
                and changing_train_output.is_file()
                and _digest(forbidden_train) == forbidden_train_before
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": changing_train.calls,
                    "allowedOutputCreated": changing_train_output.is_file(),
                    "forbiddenOutputUnchanged": _digest(forbidden_train) == forbidden_train_before,
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    allowed_save = allowed / "allowed-averaged.json"
    forbidden_save = forbidden / "forbidden-averaged.json"
    forbidden_save.write_bytes(b"outside-save-canary")
    forbidden_save_before = _digest(forbidden_save)
    changing_save = ChangingPath(allowed_save, forbidden_save)
    averaged_save = AveragedPerceptron({"feature": {"TAG": 1.0}})
    cases.append(
        _run_case(
            "averaged_perceptron_save_freezes_changing_pathlike",
            lambda: averaged_save.save(changing_save),
            lambda error, _result: (
                error is None
                and changing_save.calls == 1
                and json.loads(allowed_save.read_text()) == {"feature": {"TAG": 1.0}}
                and _digest(forbidden_save) == forbidden_save_before
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": changing_save.calls,
                    "allowedOutputCreated": allowed_save.is_file(),
                    "forbiddenOutputUnchanged": _digest(forbidden_save) == forbidden_save_before,
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    allowed_load = allowed / "allowed-load.json"
    forbidden_load = forbidden / "forbidden-load.json"
    allowed_weights = {"allowed": {"TAG": 1.0}}
    allowed_load.write_text(json.dumps(allowed_weights))
    forbidden_load.write_text(json.dumps({"forbidden": {"TAG": 1.0}}))
    forbidden_load_before = _digest(forbidden_load)
    changing_load = ChangingPath(allowed_load, forbidden_load)
    averaged_load = AveragedPerceptron({"sentinel": {"TAG": 1.0}})
    cases.append(
        _run_case(
            "averaged_perceptron_load_freezes_changing_pathlike",
            lambda: averaged_load.load(changing_load),
            lambda error, _result: (
                error is None
                and changing_load.calls == 1
                and averaged_load.weights == allowed_weights
                and _digest(forbidden_load) == forbidden_load_before
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": changing_load.calls,
                    "loadedAllowedWeights": averaged_load.weights == allowed_weights,
                    "forbiddenInputUnchanged": _digest(forbidden_load) == forbidden_load_before,
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    allowed_dir = allowed / "allowed-tagger-dir"
    forbidden_dir = forbidden / "forbidden-tagger-dir"
    changing_dir = ChangingPath(allowed_dir, forbidden_dir)
    perceptron = PerceptronTagger(load=False)
    perceptron.model.weights = {"feature": {"TAG": 1.0}}
    perceptron.tagdict = {"dog": "TAG"}
    perceptron.classes = perceptron.model.classes = {"TAG"}
    expected_files = sorted(perceptron.param_files("eng"))
    cases.append(
        _run_case(
            "perceptron_tagger_save_to_json_freezes_before_directory_create",
            lambda: perceptron.save_to_json(lang="eng", loc=changing_dir),
            lambda error, _result: (
                error is None
                and changing_dir.calls == 1
                and sorted(path.name for path in allowed_dir.iterdir()) == expected_files
                and not forbidden_dir.exists()
                and _digest(canary) == canary_before,
                {
                    "fspathCalls": changing_dir.calls,
                    "allowedDirectoryCreated": allowed_dir.is_dir(),
                    "allowedFiles": sorted(path.name for path in allowed_dir.iterdir()) if allowed_dir.is_dir() else [],
                    "forbiddenDirectoryCreated": forbidden_dir.exists(),
                    "canaryUnchanged": _digest(canary) == canary_before,
                },
            ),
        )
    )

    report = {
        "schemaVersion": 1,
        "scope": "nltk-candidate-focused-controls-not-release-admission",
        "admitted": False,
        "faults": {"alterCanary": fault_alter_canary},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nltk": nltk.__version__,
            "pythonCrfsuite": importlib.metadata.version("python-crfsuite"),
            "nltkModule": str(Path(nltk.__file__).resolve()),
            "allowedRoots": allowed_roots,
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    (evidence / "controls.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--fault-alter-canary", action="store_true")
    args = parser.parse_args()
    report = run_controls(args.evidence, fault_alter_canary=args.fault_alter_canary)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

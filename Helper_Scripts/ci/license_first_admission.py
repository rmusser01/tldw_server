from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

WORKFLOW_NAME = "Frontend License Gate Audit"
WORKFLOW_PATH = ".github/workflows/frontend-license-gate.yml"
ALLOWED_BASE_REFS = {"main", "dev"}
SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
KNOWN_FILE_STATUSES = {
    "added",
    "changed",
    "copied",
    "modified",
    "removed",
    "renamed",
    "unchanged",
}


class AdmissionError(ValueError):
    """Raised when trusted workflow or pull-request metadata is invalid."""


def _field(value: object, *path: object) -> object:
    try:
        for part in path:
            value = value[part]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        raise AdmissionError("missing or malformed admission metadata") from None
    return value


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise AdmissionError("expected a positive integer")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise AdmissionError("expected a 40-character hexadecimal SHA")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise AdmissionError("expected an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise AdmissionError("expected an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise AdmissionError("expected a timezone-aware timestamp")
    return parsed


def _require_equal(actual: object, expected: object) -> None:
    if actual != expected:
        raise AdmissionError("admission metadata does not match the trusted run")


def _class_expression(pattern: str, start: int) -> tuple[str, int] | None:
    end = pattern.find("]", start + 1)
    if end < 0:
        return None
    content = pattern[start + 1 : end]
    if not content or any(character in content for character in "/\\[]"):
        return None
    if content.startswith("!"):
        content = "^" + content[1:]
    elif content.startswith("^"):
        content = "\\" + content
    if content in {"", "^"}:
        return None
    return f"[{content}]", end + 1


def _compile_pattern(pattern: object) -> tuple[bool, re.Pattern[str]] | None:
    if not isinstance(pattern, str) or not pattern or "\n" in pattern or "\0" in pattern:
        return None
    negative = pattern.startswith("!")
    if negative:
        pattern = pattern[1:]
    if not pattern or pattern.startswith(("!", "/")):
        return None

    expression = ""
    index = 0
    can_quantify = False
    while index < len(pattern):
        character = pattern[index]
        if character in "{}()+\\":
            return None
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                if index + 2 < len(pattern) and pattern[index + 2] == "*":
                    return None
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    expression += "(?:.*/)?"
                    index += 1
                else:
                    expression += ".*"
            else:
                expression += "[^/]*"
                index += 1
            can_quantify = False
        elif character == "?":
            if not can_quantify:
                return None
            expression += "?"
            index += 1
            can_quantify = False
        elif character == "[":
            character_class = _class_expression(pattern, index)
            if character_class is None:
                return None
            part, index = character_class
            expression += part
            can_quantify = True
        else:
            expression += re.escape(character)
            index += 1
            can_quantify = True
    try:
        return negative, re.compile(f"^{expression}$")
    except re.error:
        return None


def _changed_paths(file_pages: object) -> list[str] | None:
    if not isinstance(file_pages, list):
        return None
    files: list[dict] = []
    for page in file_pages:
        if not isinstance(page, list):
            return None
        files.extend(page)
        if len(files) > 300:
            return None

    paths: list[str] = []
    for file in files:
        if not isinstance(file, dict) or file.get("status") not in KNOWN_FILE_STATUSES:
            return None
        filename = file.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or filename.startswith("/")
            or "\n" in filename
            or "\0" in filename
        ):
            return None
        paths.append(filename)

        previous = file.get("previous_filename")
        if file["status"] == "renamed" and previous is None:
            return None
        if previous is not None:
            if (
                not isinstance(previous, str)
                or not previous
                or previous.startswith("/")
                or "\n" in previous
                or "\0" in previous
            ):
                return None
            paths.append(previous)
    return paths


def _should_run(
    workflow_file: str,
    routes: dict,
    file_pages: list[list[dict]],
    files_complete: bool,
    base_drifted: bool,
) -> str:
    route = routes.get(workflow_file) if isinstance(routes, dict) else None
    if route is None:
        return "true"
    if base_drifted or files_complete is not True or not isinstance(route, dict):
        return "true"

    mode = route.get("mode")
    patterns = route.get("patterns")
    if mode not in {"paths", "paths-ignore"} or not isinstance(patterns, list) or not patterns:
        return "true"
    compiled = [_compile_pattern(pattern) for pattern in patterns]
    if any(pattern is None for pattern in compiled):
        return "true"

    paths = _changed_paths(file_pages)
    if paths is None:
        return "true"
    if not paths:
        return "false"

    for path in paths:
        selected = mode == "paths-ignore"
        for pattern in compiled:
            if pattern is None:
                return "true"
            negative, matcher = pattern
            if matcher.fullmatch(path):
                selected = negative if mode == "paths-ignore" else not negative
        if selected:
            return "true"
    return "false"


def admit(
    event: dict,
    workflow: dict,
    pull: dict,
    combined_status: dict,
    file_pages: list[list[dict]],
    workflow_file: str,
    routes: dict,
    files_complete: bool,
) -> dict[str, str]:
    """Validate a trusted license run and decide whether one workflow should run."""
    run = _field(event, "workflow_run")
    _require_equal(_field(event, "action"), "completed")
    _require_equal(_field(run, "name"), WORKFLOW_NAME)
    _require_equal(_field(run, "path"), WORKFLOW_PATH)
    _require_equal(_field(run, "event"), "pull_request_target")
    _require_equal(_field(run, "conclusion"), "success")

    event_workflow_id = _positive_int(_field(run, "workflow_id"))
    workflow_id = _positive_int(_field(workflow, "id"))
    _require_equal(event_workflow_id, workflow_id)
    _require_equal(_field(workflow, "name"), WORKFLOW_NAME)
    _require_equal(_field(workflow, "path"), WORKFLOW_PATH)

    associated_pulls = _field(run, "pull_requests")
    if not isinstance(associated_pulls, list) or len(associated_pulls) != 1:
        raise AdmissionError("expected exactly one associated pull request")
    audited_pull = associated_pulls[0]
    pr_number = _positive_int(_field(audited_pull, "number"))
    head_sha = _sha(_field(audited_pull, "head", "sha"))
    head_repo_id = _positive_int(_field(audited_pull, "head", "repo", "id"))
    base_sha = _sha(_field(audited_pull, "base", "sha"))
    base_ref = _field(audited_pull, "base", "ref")
    if not isinstance(base_ref, str) or base_ref not in ALLOWED_BASE_REFS:
        raise AdmissionError("pull request base must be main or dev")
    base_repo_id = _positive_int(_field(audited_pull, "base", "repo", "id"))

    _require_equal(_positive_int(_field(pull, "number")), pr_number)
    _require_equal(_field(pull, "state"), "open")
    _require_equal(_sha(_field(pull, "head", "sha")), head_sha)
    _require_equal(_positive_int(_field(pull, "head", "repo", "id")), head_repo_id)
    current_base_sha = _sha(_field(pull, "base", "sha"))
    _require_equal(_field(pull, "base", "ref"), base_ref)
    _require_equal(_positive_int(_field(pull, "base", "repo", "id")), base_repo_id)

    _require_equal(_sha(_field(combined_status, "sha")), head_sha)
    statuses = _field(combined_status, "statuses")
    if not isinstance(statuses, list):
        raise AdmissionError("combined status response is malformed")
    expected_context = f"frontend-license-policy/trusted/{base_ref}"
    trusted_statuses = [
        status for status in statuses if isinstance(status, dict) and status.get("context") == expected_context
    ]
    if len(trusted_statuses) != 1:
        raise AdmissionError("expected exactly one branch-qualified trusted status")
    trusted_status = trusted_statuses[0]
    _require_equal(_field(trusted_status, "state"), "success")
    if _timestamp(_field(trusted_status, "created_at")) < _timestamp(_field(run, "run_started_at")):
        raise AdmissionError("trusted status predates the audited workflow run")

    return {
        "pr_number": str(pr_number),
        "head_sha": head_sha,
        "base_sha": base_sha,
        "base_ref": base_ref,
        "should_run": _should_run(
            workflow_file,
            routes,
            file_pages,
            files_complete,
            current_base_sha != base_sha,
        ),
    }


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--pull", type=Path, required=True)
    parser.add_argument("--combined-status", type=Path, required=True)
    parser.add_argument("--file-pages", type=Path, required=True)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--files-complete", choices=("true", "false"), required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = admit(
        event=_read_json(args.event),
        workflow=_read_json(args.workflow),
        pull=_read_json(args.pull),
        combined_status=_read_json(args.combined_status),
        file_pages=_read_json(args.file_pages),
        workflow_file=args.workflow_file,
        routes=_read_json(args.routes),
        files_complete=args.files_complete == "true",
    )
    with args.github_output.open("a", encoding="utf-8") as output:
        for key in ("pr_number", "head_sha", "base_sha", "base_ref", "should_run"):
            value = outputs[key]
            if "\n" in value or "\r" in value:
                raise AdmissionError("output values must be single-line")
            output.write(f"{key}={value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

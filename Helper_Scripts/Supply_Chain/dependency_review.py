"""Project canonical package exceptions into safely scoped dependency-review allowances.

The action accepts advisory-wide allowances. Every added occurrence of an advisory
must therefore match an exact, active source approval before the advisory is passed
to it. Revalidate its complete ``dependency-changes`` output after the action runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from Helper_Scripts.Supply_Chain.exception_policy import ExceptionPolicy, PolicyError, load_policy

# Identity aliases only: none authorizes an exception without the canonical policy.
_ADVISORY_IDENTITIES = {
    "GHSA-f4j7-r4q5-qw2c": ("CVE-2026-45829", "chromadb", "1.5.9"),
    "GHSA-36p7-vc44-83pf": ("CVE-2026-45833", "chromadb", "1.5.9"),
    "GHSA-2wm9-hf6c-p5cr": ("CVE-2026-45830", "chromadb", "1.5.9"),
    "GHSA-xph7-9rjv-w5fr": ("CVE-2026-45831", "chromadb", "1.5.9"),
    "GHSA-8mgp-746c-j5xp": ("CVE-2026-81726", "nltk", "3.10.3"),
    "GHSA-qqmf-gpg7-g8gw": ("CVE-2026-58659", "lightning", "2.6.5"),
    "GHSA-2cp2-2r3c-7p7r": ("CVE-2026-68508", "hydra-core", "1.3.2"),
    "GHSA-9379-mwvr-7wxx": ("CVE-2025-33245", "nemo-toolkit", "2.0.0"),
    "GHSA-hvjw-vp7g-39h5": ("CVE-2025-33253", "nemo-toolkit", "2.0.0"),
    "GHSA-m4jw-wgmf-889x": ("CVE-2026-24157", "nemo-toolkit", "2.0.0"),
    "GHSA-v7v2-m736-cf3c": ("CVE-2026-24159", "nemo-toolkit", "2.0.0"),
}
_GHSA_PATTERN = re.compile(r"GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}")
_MANIFESTS = frozenset({"pyproject.toml", "uv.lock"})


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate JSON fields rather than choosing their last value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError("dependency review: duplicate JSON field")
        result[key] = value
    return result


def load_changes(path: Path, *, paginated: bool = False) -> list[object]:
    """Read complete action JSON or every retained GitHub comparison page."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyError("dependency review: unreadable changes JSON") from error
    if not isinstance(payload, list):
        raise PolicyError("dependency review: changes must be an array")
    if paginated:
        if not payload or any(not isinstance(page, list) for page in payload):
            raise PolicyError("dependency review: expected complete comparison pages")
        return [change for page in payload for change in page]
    return payload


def derive_allow_ghsas(changes: object, *, policy: ExceptionPolicy, today: date) -> tuple[str, ...]:
    """Allow an advisory only when every added occurrence has exactly one approval."""
    if not isinstance(changes, list):
        raise PolicyError("dependency review: changes must be an array")
    seen: set[str] = set()
    uncovered: set[str] = set()
    for change in changes:
        if not isinstance(change, dict):
            raise PolicyError("dependency review: invalid change record")
        for field in ("change_type", "manifest", "ecosystem", "name", "version", "package_url"):
            # GitHub represents unresolved manifest declarations with version="".
            if not isinstance(change.get(field), str) or (field != "version" and not change[field].strip()):
                raise PolicyError(f"dependency review: invalid {field}")
        if change["change_type"] not in {"added", "removed"}:
            raise PolicyError("dependency review: invalid change_type")
        if "scope" in change and change["scope"] not in ("runtime", "development", "unknown"):
            raise PolicyError("dependency review: invalid scope")
        vulnerabilities = change.get("vulnerabilities")
        if not isinstance(vulnerabilities, list):
            raise PolicyError("dependency review: missing vulnerabilities array")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise PolicyError("dependency review: invalid vulnerability record")
            ghsa = vulnerability.get("advisory_ghsa_id")
            severity = vulnerability.get("severity")
            if not isinstance(ghsa, str) or not _GHSA_PATTERN.fullmatch(ghsa):
                raise PolicyError("dependency review: invalid advisory_ghsa_id")
            if severity not in ("critical", "high", "moderate", "low"):
                raise PolicyError("dependency review: invalid severity")
            if change["change_type"] != "added" or ghsa not in _ADVISORY_IDENTITIES:
                continue
            cve, name, version = _ADVISORY_IDENTITIES[ghsa]
            seen.add(ghsa)
            matches = [
                item
                for item in policy.exceptions
                if change["manifest"] in _MANIFESTS
                and change["ecosystem"] == "pip"
                and change["name"] == name
                and change["package_url"] == item.purl == f"pkg:pypi/{name}@{version}"
                and change["version"] == item.installed_version == version
                and item.component == "source-python-root"
                and item.vulnerability_id == cve
                and item.severity == severity.upper()
                and item.created_on <= today <= item.expires_on
            ]
            if len(matches) != 1:
                uncovered.add(ghsa)
    return tuple(sorted(seen - uncovered))


def validate_allow_ghsas(allowed: str, changes: object, *, policy: ExceptionPolicy, today: date) -> None:
    """Reject allowances whose scope or validity changed in the action's raw output."""
    passed = allowed.split(",") if allowed else []
    if len(passed) != len(set(passed)) or any(not _GHSA_PATTERN.fullmatch(item) for item in passed):
        raise PolicyError("dependency review: invalid allow-ghsas")
    current = derive_allow_ghsas(changes, policy=policy, today=today)
    if not set(passed).issubset(current):
        raise PolicyError("dependency review: action output no longer fully covers allow-ghsas")


def main(argv: list[str] | None = None) -> int:
    """Prepare action allowances or validate them against retained complete output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--paginated", action="store_true")
    parser.add_argument("--allow-ghsas", help="Revalidate the exact allowances passed to the action")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        today = datetime.now(timezone.utc).date()
        policy = load_policy(args.policy, today=today)
        changes = load_changes(args.changes, paginated=args.paginated)
        if args.allow_ghsas is not None:
            validate_allow_ghsas(args.allow_ghsas, changes, policy=policy, today=today)
        allowed = derive_allow_ghsas(changes, policy=policy, today=today)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"allow-ghsas={','.join(allowed)}\n")
        print(json.dumps({"allow_ghsas": allowed}, sort_keys=True))
    except (OSError, UnicodeError, PolicyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

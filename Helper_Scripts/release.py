"""Pure helpers for the release automation flow."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
# Bandit B404 is expected here because the helper shells out to fixed git/gh commands.
import subprocess  # nosec B404
from typing import Any, Mapping, Protocol, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_RELEASE_BRANCH = "main"
ALLOWED_CHANGELOG_SECTIONS = ("Added", "Changed", "Fixed", "Removed")
_SEMVER_PATTERN = r"v?\d+\.\d+\.\d+"


def read_current_version(pyproject_path: str | Path) -> str:
    """Read the canonical project version from ``pyproject.toml``."""

    path = Path(pyproject_path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    version = data["project"]["version"]
    return str(version)


def bump_version(version: str, bump: str) -> str:
    """Bump a semantic version by patch or minor."""

    normalized = version.lstrip("v")
    parts = normalized.split(".")
    if len(parts) != 3:
        raise ValueError(f"Unsupported semantic version: {version!r}")

    major, minor, patch = (int(part) for part in parts)
    if bump == "patch":
        patch += 1
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        raise ValueError(f"Unsupported bump type: {bump!r}")

    return f"{major}.{minor}.{patch}"


def extract_required_check_names(doc_path: str | Path) -> list[str]:
    """Extract the stable required CI gate names from the CI gates doc."""

    path = Path(doc_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    in_section = False
    required_names: list[str] = []
    seen: set[str] = set()

    for raw_line in lines:
        line = raw_line.strip()
        if line == "## Required Check Names":
            in_section = True
            continue
        if in_section and line.startswith("#"):
            break
        if not in_section:
            continue

        match = re.match(r"^\d+\.\s+`([^`]+)`", line)
        if match is None:
            continue

        name = match.group(1)
        if name not in seen:
            seen.add(name)
            required_names.append(name)

    if not in_section:
        raise ValueError("Missing '## Required Check Names' section")
    if not required_names:
        raise ValueError("'## Required Check Names' section is empty")

    return required_names


def normalize_bullet_text(text: str) -> str:
    """Trim whitespace and collapse all internal whitespace runs."""

    return " ".join(text.split())


def find_exact_duplicate_bullets(
    subsections: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    """Find exact duplicate bullets within each changelog subsection."""

    duplicates: list[tuple[str, str]] = []

    for subsection in ALLOWED_CHANGELOG_SECTIONS:
        bullets = subsections.get(subsection, ())
        counts = Counter(normalize_bullet_text(bullet) for bullet in bullets)
        duplicates.extend(
            (subsection, bullet_text)
            for bullet_text, count in counts.items()
            if count > 1
        )

    return duplicates


def _parse_unreleased_changelog_subsections(changelog_text: str) -> dict[str, list[str]]:
    """Extract bullets from the ``Unreleased`` changelog section."""

    lines = changelog_text.splitlines()
    unreleased_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "## [Unreleased]"),
        None,
    )
    if unreleased_index is None:
        raise ValueError("Missing [Unreleased] changelog section")

    section_end = len(lines)
    for index in range(unreleased_index + 1, len(lines)):
        if lines[index].startswith("## "):
            section_end = index
            break

    subsections: dict[str, list[str]] = {section: [] for section in ALLOWED_CHANGELOG_SECTIONS}
    current_section: str | None = None
    seen_structural_content = False

    for raw_line in lines[unreleased_index + 1 : section_end]:
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("## "):
            break
        if stripped_line.startswith("### "):
            candidate = stripped_line.removeprefix("### ").strip()
            if candidate not in ALLOWED_CHANGELOG_SECTIONS:
                raise ValueError(
                    f"Unsupported changelog subsection in Unreleased: {candidate!r}"
                )
            current_section = candidate
            seen_structural_content = True
            continue
        if raw_line[:1].isspace():
            if current_section is None or not subsections[current_section]:
                raise ValueError(
                    "Indented changelog content in Unreleased must continue a bullet"
                )
            subsections[current_section][-1] = (
                f"{subsections[current_section][-1]} {stripped_line}"
            )
            continue
        if stripped_line.startswith("- "):
            if current_section is None:
                raise ValueError(
                    "Changelog bullets in Unreleased must appear under a supported subsection"
                )
            subsections[current_section].append(stripped_line[2:])
            seen_structural_content = True
            continue
        raise ValueError("Unsupported content in Unreleased changelog slice")

    if not seen_structural_content:
        raise ValueError("Unreleased changelog section is empty or malformed")
    if sum(len(bullets) for bullets in subsections.values()) == 0:
        raise ValueError("Unreleased changelog section has no bullets")

    return subsections


def _find_cross_section_near_duplicates(
    subsections: Mapping[str, Sequence[str]],
) -> list[str]:
    """Report near-duplicates across different changelog subsections."""

    warnings: list[str] = []
    subsection_names = [section for section in ALLOWED_CHANGELOG_SECTIONS if subsections.get(section)]

    for index, first_section in enumerate(subsection_names):
        first_bullets = [normalize_bullet_text(bullet) for bullet in subsections.get(first_section, ())]
        for second_section in subsection_names[index + 1 :]:
            second_bullets = [normalize_bullet_text(bullet) for bullet in subsections.get(second_section, ())]
            for first_bullet in first_bullets:
                for second_bullet in second_bullets:
                    similarity = SequenceMatcher(None, first_bullet, second_bullet).ratio()
                    if similarity >= 0.84 or first_bullet in second_bullet or second_bullet in first_bullet:
                        warnings.append(
                            "Near-duplicate changelog bullets across "
                            f"{first_section} and {second_section}: "
                            f"{first_bullet!r} ~ {second_bullet!r}"
                        )

    return warnings


def _render_changelog_section(title: str, subsections: Mapping[str, Sequence[str]], *, date: str | None = None) -> str:
    """Render a Keep-a-Changelog section with stable spacing."""

    heading = f"## [{title}]"
    if date is not None:
        heading = f"{heading} - {date}"

    lines: list[str] = [heading, ""]
    for subsection in ALLOWED_CHANGELOG_SECTIONS:
        lines.append(f"### {subsection}")
        lines.append("")
        for bullet in subsections.get(subsection, ()):
            lines.append(f"- {normalize_bullet_text(bullet)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def promote_changelog_unreleased_section(
    changelog_text: str,
    version: str,
    release_date: str,
) -> tuple[str, list[str]]:
    """Promote ``Unreleased`` into a dated changelog section."""

    subsections = _parse_unreleased_changelog_subsections(changelog_text)
    duplicate_bullets = find_exact_duplicate_bullets(subsections)
    if duplicate_bullets:
        section, bullet_text = duplicate_bullets[0]
        raise ValueError(
            f"Duplicate changelog bullet detected in {section!r}: {bullet_text!r}"
        )

    warnings = _find_cross_section_near_duplicates(subsections)

    lines = changelog_text.splitlines()
    unreleased_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "## [Unreleased]"),
        None,
    )
    if unreleased_index is None:
        raise ValueError("Missing [Unreleased] changelog section")

    section_end = len(lines)
    for index in range(unreleased_index + 1, len(lines)):
        if lines[index].startswith("## "):
            section_end = index
            break

    prefix = "\n".join(lines[:unreleased_index]).rstrip()
    if prefix:
        prefix += "\n\n"

    reset_unreleased = _render_changelog_section("Unreleased", {section: () for section in ALLOWED_CHANGELOG_SECTIONS})
    promoted_section = _render_changelog_section(version, subsections, date=release_date)
    suffix = "\n".join(lines[section_end:]).lstrip("\n")

    result = prefix + reset_unreleased + "\n" + promoted_section + suffix
    return result.rstrip() + "\n", warnings


def update_pyproject_version(pyproject_text: str, version: str) -> str:
    """Update the package version in ``pyproject.toml``."""

    updated_text, count = re.subn(
        r'''(?m)^(version\s*=\s*)(["'])(\d+\.\d+\.\d+)(\2)$''',
        rf"\g<1>\g<2>{version}\g<4>",
        pyproject_text,
        count=1,
    )
    if count == 0:
        raise ValueError("Missing pyproject version anchor")

    return updated_text


def update_readme_release_references(readme_text: str, version: str) -> str:
    """Update release-line references in the README to ``version``."""

    replacements = [
        (
            rf"(?m)^(- `){_SEMVER_PATTERN}(` Beta status\. Expect rough edges and please report issues\.)$",
            rf"\g<1>{version}\g<2>",
            "current release line",
        ),
        (
            rf"(?m)^(- The `dev` branch currently contains additional unreleased work beyond `){_SEMVER_PATTERN}(`; see \[CHANGELOG\.md\]\(CHANGELOG\.md\) for branch-level detail and \[Docs/Published/RELEASE_NOTES\.md\]\(Docs/Published/RELEASE_NOTES\.md\) for the published release entry point\.)$",
            rf"\g<1>{version}\g<2>",
            "beyond-release reference",
        ),
        (
            rf"(?m)^(Currently landing on `dev` \(post-`){_SEMVER_PATTERN}(` branch work\):)$",
            rf"\g<1>{version}\g<2>",
            "post-release reference",
        ),
    ]

    updated_text = readme_text
    for pattern, replacement, anchor_name in replacements:
        updated_text, count = re.subn(pattern, replacement, updated_text)
        if count == 0:
            raise ValueError(f"Missing README anchor for {anchor_name}")

    return updated_text


def update_mkdocs_version_metadata(mkdocs_text: str, version: str) -> str:
    """Update the MkDocs version and copyright metadata."""

    lines = mkdocs_text.splitlines()
    extra_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "extra:"),
        None,
    )
    if extra_index is None:
        raise ValueError("Missing MkDocs extra anchor")

    extra_indent = len(lines[extra_index]) - len(lines[extra_index].lstrip(" "))
    version_updated = False
    for index in range(extra_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= extra_indent:
            break
        if stripped.startswith("version:"):
            lines[index] = f"{line[:indent]}version: v{version}"
            version_updated = True
            break
    if not version_updated:
        raise ValueError("Missing MkDocs extra.version anchor")

    copyright_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "copyright: |"),
        None,
    )
    if copyright_index is None or copyright_index + 1 >= len(lines):
        raise ValueError("Missing MkDocs copyright anchor")

    copyright_indent = len(lines[copyright_index]) - len(lines[copyright_index].lstrip(" "))
    copyright_count = 0
    for index in range(copyright_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if stripped:
            indent = len(line) - len(line.lstrip(" "))
            if indent <= copyright_indent:
                break
        lines[index], count = re.subn(r"v?\d+\.\d+\.\d+", f"v{version}", line)
        copyright_count += count
    if copyright_count == 0:
        raise ValueError("Missing MkDocs copyright version anchor")

    updated_text = "\n".join(lines)
    if mkdocs_text.endswith("\n"):
        updated_text += "\n"

    return updated_text


def update_release_notes_entry_point(
    release_notes_text: str,
    release_process_doc_path: str,
) -> str:
    """Point release notes at the authoritative release-process document."""

    updated_text, count = re.subn(
        r"(?m)^For release process details, see `[^`]+`\.$",
        f"For release process details, see `{release_process_doc_path}`.",
        release_notes_text,
    )
    if count == 0:
        raise ValueError("Missing release notes process anchor")
    return updated_text


def validate_release_branch(branch: str) -> str:
    """Accept only the supported release branch."""

    normalized_branch = branch.strip()
    if normalized_branch != SUPPORTED_RELEASE_BRANCH:
        raise ValueError(
            f"Unsupported release branch: {branch!r}; expected {SUPPORTED_RELEASE_BRANCH!r}"
        )
    return normalized_branch


def classify_release_state(
    *,
    local_release_commit_exists: bool,
    local_tag_exists: bool,
    remote_tag_exists: bool,
    github_release_exists: bool,
) -> str:
    """Classify a resumable release state snapshot."""

    if github_release_exists and not remote_tag_exists:
        raise ValueError("inconsistent release state: GitHub Release exists without remote tag")

    if github_release_exists:
        return "existing_github_release"
    if remote_tag_exists:
        return "remote_tag_without_github_release"
    if local_tag_exists:
        return "local_tag_only"
    if local_release_commit_exists:
        return "local_release_commit_only"
    raise ValueError("No resumable release state could be determined")


def release_tag_name(version: str) -> str:
    """Return the canonical tag for ``version``."""

    return f"v{version.lstrip('v')}"


def release_commit_message(version: str) -> str:
    """Return the canonical release commit subject."""

    return f"chore(release): {release_tag_name(version)}"


def extract_release_notes_for_version(changelog_text: str, version: str) -> str:
    """Extract the rendered changelog section for ``version``."""

    tag = version.lstrip("v")
    lines = changelog_text.splitlines()
    heading = f"## [{tag}]"
    start_index = next(
        (index for index, line in enumerate(lines) if line.startswith(heading)),
        None,
    )
    if start_index is None:
        raise ValueError(f"Missing release section for {tag}")

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("## "):
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip() + "\n"


class ReleaseRunner(Protocol):
    """Runtime contract for release orchestration."""

    def fetch_origin_main(self) -> str: ...

    def get_current_branch(self) -> str: ...

    def is_worktree_clean(self) -> bool: ...

    def get_origin_main_sha(self) -> str: ...

    def get_release_state(self, version: str) -> str: ...

    def ensure_required_checks_green(self, sha: str, required_checks: list[str]) -> None: ...

    def prepare_metadata(self, current_version: str, next_version: str) -> None: ...

    def create_release_commit(self, next_version: str) -> str: ...

    def create_or_update_tag(self, version: str) -> None: ...

    def push_main(self) -> None: ...

    def push_tag(self, version: str) -> None: ...

    def prepare_release_notes_from_tag(self, version: str) -> None: ...

    def get_metadata_warnings(self) -> list[str]: ...

    def create_github_release(self, version: str) -> bool: ...


def _call_runner_if_present(runner: object, method_name: str, *args: object) -> object | None:
    method = getattr(runner, method_name, None)
    if method is None:
        return None
    return method(*args)


def _runner_dry_run(runner: object) -> bool:
    return bool(getattr(runner, "dry_run", False))


def _runner_metadata_warnings(runner: object) -> list[str]:
    warnings = _call_runner_if_present(runner, "get_metadata_warnings")
    if isinstance(warnings, list):
        return [str(warning) for warning in warnings]
    return []


def _create_github_release(runner: ReleaseRunner, version: str) -> bool:
    created = runner.create_github_release(version)
    if isinstance(created, bool):
        return created
    return not _runner_dry_run(runner)


def _release_result(
    *,
    branch: str,
    current_version: str,
    next_version: str,
    required_checks: list[str],
    state: str,
    validated_sha: str,
    release_commit_sha: str | None,
    github_release_created: bool,
    dry_run: bool,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "branch": branch,
        "current_version": current_version,
        "next_version": next_version,
        "required_checks": required_checks,
        "state": state,
        "validated_sha": validated_sha,
        "release_commit_sha": release_commit_sha,
        "github_release_created": github_release_created,
        "dry_run": dry_run,
        "warnings": warnings,
    }


def orchestrate_release(
    *,
    bump: str,
    runner: ReleaseRunner,
    pyproject_path: str | Path,
    ci_gates_doc_path: str | Path,
) -> dict[str, Any]:
    """Run the release flow against an injected runner."""

    validated_sha = runner.fetch_origin_main()
    branch = validate_release_branch(runner.get_current_branch())
    if not runner.is_worktree_clean():
        raise RuntimeError("Release automation requires a clean worktree")

    current_version = read_current_version(pyproject_path)
    next_version = bump_version(current_version, bump)
    state = runner.get_release_state(next_version)
    required_checks = extract_required_check_names(ci_gates_doc_path)

    _call_runner_if_present(runner, "ensure_github_auth")

    dry_run = _runner_dry_run(runner)
    warnings: list[str] = []
    github_release_created = False
    release_commit_sha: str | None = None

    if state == "existing_github_release":
        return _release_result(
            branch=branch,
            current_version=current_version,
            next_version=next_version,
            required_checks=required_checks,
            state=state,
            validated_sha=validated_sha,
            release_commit_sha=None,
            github_release_created=False,
            dry_run=dry_run,
            warnings=warnings,
        )

    if state == "remote_tag_without_github_release":
        _call_runner_if_present(runner, "prepare_release_notes_from_tag", next_version)
        github_release_created = _create_github_release(runner, next_version)
        return _release_result(
            branch=branch,
            current_version=current_version,
            next_version=next_version,
            required_checks=required_checks,
            state=state,
            validated_sha=validated_sha,
            release_commit_sha=None,
            github_release_created=github_release_created,
            dry_run=dry_run,
            warnings=warnings,
        )

    if state == "fresh":
        head_sha = _call_runner_if_present(runner, "get_head_sha")
        if isinstance(head_sha, str) and head_sha != runner.get_origin_main_sha():
            raise RuntimeError("Release automation requires local main to match origin/main")

        runner.ensure_required_checks_green(validated_sha, required_checks)
        runner.prepare_metadata(current_version, next_version)
        warnings = _runner_metadata_warnings(runner)
        release_commit_sha = runner.create_release_commit(next_version)
        runner.create_or_update_tag(next_version)
    elif state == "local_release_commit_only":
        head_sha = _call_runner_if_present(runner, "get_head_sha")
        release_commit_sha = head_sha if isinstance(head_sha, str) else None
        runner.create_or_update_tag(next_version)
    elif state == "local_tag_only":
        head_sha = _call_runner_if_present(runner, "get_head_sha")
        tag_target_sha = _call_runner_if_present(runner, "get_local_tag_target_sha", next_version)
        release_commit_sha = head_sha if isinstance(head_sha, str) else None
        if (
            isinstance(head_sha, str)
            and isinstance(tag_target_sha, str)
            and tag_target_sha != head_sha
        ):
            raise RuntimeError(
                "Release aborted: local release tag does not point at current HEAD"
            )
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported release state: {state}")

    try:
        runner.push_main()
    except RuntimeError as exc:
        if "non-fast-forward" in str(exc).lower():
            raise RuntimeError(
                "Release aborted: non-fast-forward push to origin/main; main moved and the release must be rerun from a fresh fetch"
            ) from exc
        raise

    runner.push_tag(next_version)
    github_release_created = _create_github_release(runner, next_version)

    return _release_result(
        branch=branch,
        current_version=current_version,
        next_version=next_version,
        required_checks=required_checks,
        state=state,
        validated_sha=validated_sha,
        release_commit_sha=release_commit_sha,
        github_release_created=github_release_created,
        dry_run=dry_run,
        warnings=warnings,
    )


class ShellReleaseRunner:
    """Real release runner backed by ``git`` and ``gh``."""

    def __init__(self, repo_root: str | Path, *, dry_run: bool = False):
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        self._prepared_paths: list[Path] = []
        self._release_notes_body: str | None = None
        self._metadata_warnings: list[str] = []
        self._release_date = datetime.now(timezone.utc).date().isoformat()
        self._repo_slug: str | None = None

    def _run_command(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if not args:
            raise ValueError("Command arguments must not be empty")
        executable = args[0]
        resolved_executable = executable if Path(executable).is_absolute() else shutil.which(executable)
        if resolved_executable is None:
            raise FileNotFoundError(f"Executable not found: {executable}")
        resolved_path = Path(resolved_executable)
        if not resolved_path.is_absolute():
            raise FileNotFoundError(
                f"Executable did not resolve to an absolute path: {executable}"
            )
        command = [str(resolved_path), *args[1:]]
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            capture_output=capture_output,
            check=False,
        )  # nosec B603
        if check and completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"Command failed: {' '.join(args)}: {detail}")
        return completed

    def _run_text(self, args: Sequence[str]) -> str:
        return self._run_command(args).stdout.strip()

    def ensure_github_auth(self) -> None:
        self._run_command(["gh", "auth", "status"])

    def fetch_origin_main(self) -> str:
        self._run_command(["git", "fetch", "origin", "main"])
        return self.get_origin_main_sha()

    def get_current_branch(self) -> str:
        return self._run_text(["git", "branch", "--show-current"])

    def is_worktree_clean(self) -> bool:
        return self._run_text(["git", "status", "--porcelain"]) == ""

    def get_head_sha(self) -> str:
        return self._run_text(["git", "rev-parse", "HEAD"])

    def get_origin_main_sha(self) -> str:
        return self._run_text(["git", "rev-parse", "origin/main"])

    def _get_repo_slug(self) -> str:
        if self._repo_slug is None:
            self._repo_slug = self._run_text(
                ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
            )
        return self._repo_slug

    def _local_tag_exists(self, tag_name: str) -> bool:
        result = self._run_command(
            ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
            check=False,
        )
        return result.returncode == 0

    def _remote_tag_exists(self, tag_name: str) -> bool:
        output = self._run_text(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag_name}"])
        return output != ""

    def get_local_tag_target_sha(self, version: str) -> str | None:
        tag_name = release_tag_name(version)
        result = self._run_command(
            ["git", "rev-list", "-n", "1", tag_name],
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _github_release_exists(self, tag_name: str) -> bool:
        result = self._run_command(["gh", "release", "view", tag_name], check=False)
        if result.returncode == 0:
            return True

        error_text = f"{result.stderr}\n{result.stdout}".lower()
        if "not found" in error_text or "404" in error_text:
            return False

        raise RuntimeError(f"GitHub release lookup failed for {tag_name}: {error_text.strip()}")

    def get_release_state(self, version: str) -> str:
        tag_name = release_tag_name(version)
        local_release_commit_exists = self._run_text(["git", "log", "-1", "--format=%s"]) == release_commit_message(version)
        local_tag_exists = self._local_tag_exists(tag_name)
        remote_tag_exists = self._remote_tag_exists(tag_name)
        github_release_exists = self._github_release_exists(tag_name)

        if not local_release_commit_exists and not local_tag_exists and not remote_tag_exists and not github_release_exists:
            return "fresh"

        return classify_release_state(
            local_release_commit_exists=local_release_commit_exists,
            local_tag_exists=local_tag_exists,
            remote_tag_exists=remote_tag_exists,
            github_release_exists=github_release_exists,
        )

    def ensure_required_checks_green(self, sha: str, required_checks: list[str]) -> None:
        repo_slug = self._get_repo_slug()
        payload = self._run_text(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo_slug}/commits/{sha}/check-runs?per_page=100",
            ]
        )
        data = json.loads(payload)
        if isinstance(data, list):
            check_runs = [
                check_run
                for page in data
                if isinstance(page, dict)
                for check_run in page.get("check_runs", [])
            ]
        elif isinstance(data, dict):
            check_runs = data.get("check_runs", [])
        else:
            raise RuntimeError("GitHub check-runs response had an unexpected shape")
        status_by_name = {
            str(check_run.get("name")): (
                str(check_run.get("status")),
                str(check_run.get("conclusion")),
            )
            for check_run in check_runs
        }

        missing = [name for name in required_checks if name not in status_by_name]
        failing = [
            name
            for name in required_checks
            if name in status_by_name and status_by_name[name] != ("completed", "success")
        ]
        if missing or failing:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if failing:
                details.append(f"non_green={failing}")
            raise RuntimeError(
                f"Required CI gates are not green on {sha}: " + ", ".join(details)
            )

    def prepare_metadata(self, current_version: str, next_version: str) -> None:
        pyproject_path = self.repo_root / "pyproject.toml"
        changelog_path = self.repo_root / "CHANGELOG.md"
        readme_path = self.repo_root / "README.md"
        mkdocs_path = self.repo_root / "Docs" / "mkdocs.yml"
        release_notes_path = self.repo_root / "Docs" / "Published" / "RELEASE_NOTES.md"
        release_process_doc = "Docs/Development/Release_Process.md"

        pyproject_text = pyproject_path.read_text(encoding="utf-8")
        changelog_text = changelog_path.read_text(encoding="utf-8")
        readme_text = readme_path.read_text(encoding="utf-8")
        mkdocs_text = mkdocs_path.read_text(encoding="utf-8")
        release_notes_text = release_notes_path.read_text(encoding="utf-8")

        updated_pyproject = update_pyproject_version(pyproject_text, next_version)
        updated_changelog, warnings = promote_changelog_unreleased_section(
            changelog_text=changelog_text,
            version=next_version,
            release_date=self._release_date,
        )
        self._metadata_warnings = list(warnings)

        updated_readme = update_readme_release_references(readme_text, next_version)
        updated_mkdocs = update_mkdocs_version_metadata(mkdocs_text, next_version)
        updated_release_notes = update_release_notes_entry_point(
            release_notes_text,
            release_process_doc,
        )

        self._release_notes_body = extract_release_notes_for_version(updated_changelog, next_version)
        self._prepared_paths = [
            pyproject_path,
            changelog_path,
            readme_path,
            mkdocs_path,
            release_notes_path,
        ]

        if self.dry_run:
            return

        pyproject_path.write_text(updated_pyproject, encoding="utf-8")
        changelog_path.write_text(updated_changelog, encoding="utf-8")
        readme_path.write_text(updated_readme, encoding="utf-8")
        mkdocs_path.write_text(updated_mkdocs, encoding="utf-8")
        release_notes_path.write_text(updated_release_notes, encoding="utf-8")

    def get_metadata_warnings(self) -> list[str]:
        return list(self._metadata_warnings)

    def prepare_release_notes_from_tag(self, version: str) -> None:
        tag_name = release_tag_name(version)
        self._run_command(["git", "fetch", "origin", "--tags"])
        changelog_text = self._run_text(["git", "show", f"{tag_name}:CHANGELOG.md"])
        self._release_notes_body = extract_release_notes_for_version(changelog_text, version)

    def create_release_commit(self, next_version: str) -> str:
        if not self._prepared_paths:
            raise RuntimeError("Release metadata has not been prepared")

        rel_paths = [str(path.relative_to(self.repo_root)) for path in self._prepared_paths]
        if self.dry_run:
            return f"dry-run-release-{next_version}"

        self._run_command(["git", "add", *rel_paths])
        self._run_command(["git", "commit", "-m", release_commit_message(next_version)])
        return self.get_head_sha()

    def create_or_update_tag(self, version: str) -> None:
        tag_name = release_tag_name(version)
        if self.dry_run:
            return
        self._run_command(["git", "tag", "-a", tag_name, "-m", tag_name])

    def push_main(self) -> None:
        if self.dry_run:
            return
        self._run_command(["git", "push", "origin", "main"])

    def push_tag(self, version: str) -> None:
        if self.dry_run:
            return
        self._run_command(["git", "push", "origin", release_tag_name(version)])

    def create_github_release(self, version: str) -> bool:
        if self._release_notes_body is None:
            changelog_path = self.repo_root / "CHANGELOG.md"
            self._release_notes_body = extract_release_notes_for_version(
                changelog_path.read_text(encoding="utf-8"),
                version,
            )

        if self.dry_run:
            return False

        tag_name = release_tag_name(version)
        self._run_command(
            [
                "gh",
                "release",
                "create",
                tag_name,
                "--verify-tag",
                "--title",
                tag_name,
                "--notes",
                self._release_notes_body,
            ]
        )
        return True


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cut a local release from main.")
    parser.add_argument("--bump", choices=("patch", "minor"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    runner = ShellReleaseRunner(repo_root=repo_root, dry_run=args.dry_run)
    result = orchestrate_release(
        bump=args.bump,
        runner=runner,
        pyproject_path=repo_root / "pyproject.toml",
        ci_gates_doc_path=repo_root / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

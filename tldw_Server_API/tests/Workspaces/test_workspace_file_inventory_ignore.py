from __future__ import annotations

from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import (
    MAX_GITIGNORE_BYTES,
    build_inventory_ignore_policy,
    should_ignore_inventory_path,
)


def test_builtin_generated_directories_are_ignored() -> None:
    policy = build_inventory_ignore_policy()

    node_modules = should_ignore_inventory_path("src/node_modules/pkg/index.js", is_dir=False, policy=policy)
    git_dir = should_ignore_inventory_path(".git", is_dir=True, policy=policy)
    normal_file = should_ignore_inventory_path("src/app.py", is_dir=False, policy=policy)

    assert node_modules.ignored is True
    assert node_modules.reason == "builtin:generated_dir:node_modules"
    assert git_dir.ignored is True
    assert git_dir.reason == "builtin:generated_dir:.git"
    assert normal_file.ignored is False


def test_builtin_secret_like_files_are_ignored_by_basename() -> None:
    policy = build_inventory_ignore_policy()

    assert should_ignore_inventory_path(".env.local", is_dir=False, policy=policy).reason == "builtin:secret_file:.env.*"
    assert should_ignore_inventory_path("keys/id_ed25519", is_dir=False, policy=policy).reason == (
        "builtin:secret_file:id_ed25519"
    )
    assert should_ignore_inventory_path("certs/server.pem", is_dir=False, policy=policy).reason == (
        "builtin:secret_file:*.pem"
    )


def test_simple_gitignore_subset_matches_anchored_unanchored_and_directory_rules() -> None:
    policy = build_inventory_ignore_policy(
        gitignore_texts=[
            (
                ".gitignore",
                """
                # comments and blanks are ignored
                logs/
                *.tmp
                /site-build
                """,
            )
        ]
    )

    assert should_ignore_inventory_path("logs/app.log", is_dir=False, policy=policy).reason == "gitignore:logs/"
    assert should_ignore_inventory_path("src/cache.tmp", is_dir=False, policy=policy).reason == "gitignore:*.tmp"
    assert should_ignore_inventory_path("site-build/index.js", is_dir=False, policy=policy).reason == (
        "gitignore:/site-build"
    )
    assert should_ignore_inventory_path("src/site-build/index.js", is_dir=False, policy=policy).ignored is False


def test_directory_only_patterns_do_not_ignore_files_with_same_name() -> None:
    policy = build_inventory_ignore_policy(
        gitignore_texts=[
            (
                ".gitignore",
                """
                logs/
                /build/
                """,
            )
        ]
    )

    assert should_ignore_inventory_path("logs", is_dir=False, policy=policy).ignored is False
    assert should_ignore_inventory_path("logs", is_dir=True, policy=policy).reason == "gitignore:logs/"
    assert should_ignore_inventory_path("logs/app.log", is_dir=False, policy=policy).reason == "gitignore:logs/"
    assert should_ignore_inventory_path("build", is_dir=False, policy=policy).ignored is False
    assert should_ignore_inventory_path("build", is_dir=True, policy=policy).reason == "gitignore:/build/"
    assert should_ignore_inventory_path("build/index.js", is_dir=False, policy=policy).reason == "gitignore:/build/"


def test_workspace_patterns_are_included_in_policy() -> None:
    policy = build_inventory_ignore_policy(workspace_patterns=["private/"])

    decision = should_ignore_inventory_path("notes/private/source.md", is_dir=False, policy=policy)

    assert decision.ignored is True
    assert decision.reason == "workspace:private/"


def test_unsupported_gitignore_constructs_produce_diagnostics_and_fail_closed() -> None:
    policy = build_inventory_ignore_policy(
        gitignore_texts=[
            (
                ".gitignore",
                """
                !keep.txt
                **/cache
                """,
            )
        ]
    )

    diagnostic_codes = [diagnostic["code"] for diagnostic in policy.diagnostics]

    assert diagnostic_codes == ["unsupported_gitignore_pattern", "unsupported_gitignore_pattern"]
    assert should_ignore_inventory_path("keep.txt", is_dir=False, policy=policy).reason == "gitignore:!keep.txt"
    assert should_ignore_inventory_path("src/cache", is_dir=True, policy=policy).reason == "gitignore:**/cache"


def test_malformed_and_oversized_gitignore_inputs_produce_diagnostics_without_crashing() -> None:
    policy = build_inventory_ignore_policy(
        gitignore_texts=[
            (".gitignore", "valid\nbad\0pattern\n"),
            ("large/.gitignore", "a" * (MAX_GITIGNORE_BYTES + 1)),
        ]
    )

    diagnostic_codes = [diagnostic["code"] for diagnostic in policy.diagnostics]

    assert "malformed_gitignore_pattern" in diagnostic_codes
    assert "ignore_file_too_large" in diagnostic_codes
    assert should_ignore_inventory_path("valid", is_dir=False, policy=policy).reason == "gitignore:valid"


def test_policy_fingerprint_is_stable_for_equivalent_rules_and_changes_for_new_rules() -> None:
    first = build_inventory_ignore_policy(gitignore_texts=[(".gitignore", "*.tmp\n.cache\n")])
    reordered = build_inventory_ignore_policy(gitignore_texts=[("other.gitignore", ".cache\n*.tmp\n")])
    changed = build_inventory_ignore_policy(gitignore_texts=[(".gitignore", "*.tmp\n.cache\n*.log\n")])

    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint != changed.fingerprint

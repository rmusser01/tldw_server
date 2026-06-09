from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_diff import (
    FilesystemPatchError,
    apply_patch_to_text,
    parse_unified_diff,
)


def test_parse_unified_diff_modify_headers_and_hunk_ranges() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETTA
 gamma
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    assert len(parsed) == 1  # nosec B101
    patch_file = parsed[0]
    assert patch_file.old_path == "docs/story.txt"  # nosec B101
    assert patch_file.new_path == "docs/story.txt"  # nosec B101
    assert patch_file.action == "modify"  # nosec B101
    assert len(patch_file.hunks) == 1  # nosec B101
    hunk = patch_file.hunks[0]
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 3, 1, 3)  # nosec B101
    assert [line.kind for line in hunk.lines] == ["context", "remove", "add", "context"]  # nosec B101
    assert [line.text for line in hunk.lines] == ["alpha", "beta", "BETTA", "gamma"]  # nosec B101


def test_parse_unified_diff_preserves_safe_paths_with_spaces() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/my note.txt
+++ b/docs/my note.txt
@@ -1 +1 @@
-alpha
+beta
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    assert parsed[0].old_path == "docs/my note.txt"  # nosec B101
    assert parsed[0].new_path == "docs/my note.txt"  # nosec B101


def test_parse_unified_diff_create_and_reject_delete() -> None:
    created = parse_unified_diff(
        """--- /dev/null
+++ b/docs/new.txt
@@ -0,0 +1,2 @@
+alpha
+beta
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    assert created[0].old_path is None  # nosec B101
    assert created[0].new_path == "docs/new.txt"  # nosec B101
    assert created[0].action == "create"  # nosec B101

    with pytest.raises(FilesystemPatchError) as exc_info:
        parse_unified_diff(
            """--- a/docs/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-alpha
""",
            max_files=10,
            max_hunks=10,
            max_bytes=10_000,
        )
    assert exc_info.value.reason_code == "delete_not_supported"  # nosec B101


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:/Users/alice/secret.txt",
        "../secret.txt",
        "docs/../secret.txt",
        "",
    ],
)
def test_parse_unified_diff_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(FilesystemPatchError) as exc_info:
        parse_unified_diff(
            f"""--- a/docs/story.txt
+++ b/{path}
@@ -1 +1 @@
-alpha
+beta
""",
            max_files=10,
            max_hunks=10,
            max_bytes=10_000,
        )
    assert exc_info.value.reason_code == "invalid_patch_path"  # nosec B101


def test_parse_unified_diff_enforces_limits() -> None:
    two_file_diff = """--- a/one.txt
+++ b/one.txt
@@ -1 +1 @@
-one
+ONE
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-two
+TWO
"""

    with pytest.raises(FilesystemPatchError) as size_exc:
        parse_unified_diff(two_file_diff, max_files=10, max_hunks=10, max_bytes=3)
    assert size_exc.value.reason_code == "diff_too_large"  # nosec B101

    with pytest.raises(FilesystemPatchError) as file_exc:
        parse_unified_diff(two_file_diff, max_files=1, max_hunks=10, max_bytes=10_000)
    assert file_exc.value.reason_code == "diff_file_limit_exceeded"  # nosec B101

    with pytest.raises(FilesystemPatchError) as hunk_exc:
        parse_unified_diff(two_file_diff, max_files=10, max_hunks=1, max_bytes=10_000)
    assert hunk_exc.value.reason_code == "diff_hunk_limit_exceeded"  # nosec B101


def test_apply_patch_rejects_add_only_hunk_beyond_end_of_file() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -50,0 +50,1 @@
+late
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    with pytest.raises(FilesystemPatchError) as exc_info:
        apply_patch_to_text("alpha\n", parsed[0])

    assert exc_info.value.reason_code == "patch_context_mismatch"  # nosec B101


def test_apply_patch_to_text_preserves_missing_final_newline() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1,2 +1,2 @@
 alpha
-beta
\\ No newline at end of file
+BETTA
\\ No newline at end of file
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    result = apply_patch_to_text("alpha\nbeta", parsed[0])

    assert result == "alpha\nBETTA"  # nosec B101


def test_parse_unified_diff_rejects_orphan_no_newline_marker() -> None:
    with pytest.raises(FilesystemPatchError) as exc_info:
        parse_unified_diff(
            """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1 +1 @@
\\ No newline at end of file
-alpha
+beta
""",
            max_files=10,
            max_hunks=10,
            max_bytes=10_000,
        )

    assert exc_info.value.reason_code == "invalid_no_newline_marker"  # nosec B101


def test_apply_patch_to_text_modifies_content_in_memory() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETTA
 gamma
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    result = apply_patch_to_text("alpha\nbeta\ngamma\n", parsed[0])

    assert result == "alpha\nBETTA\ngamma\n"  # nosec B101


def test_apply_patch_to_text_detects_context_mismatch() -> None:
    parsed = parse_unified_diff(
        """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETTA
 gamma
""",
        max_files=10,
        max_hunks=10,
        max_bytes=10_000,
    )

    with pytest.raises(FilesystemPatchError) as exc_info:
        apply_patch_to_text("alpha\nchanged\ngamma\n", parsed[0])
    assert exc_info.value.reason_code == "patch_context_mismatch"  # nosec B101

"""TASK-13288: one `extract_text_from_input`, with the file read gone on purpose.

`Summarization_General_Lib` defined `extract_text_from_input` twice at module scope. The
second binding won at runtime, marked `# noqa: F811`:

* the shadowed copy did `os.path.isfile(input_data)` -> `open(input_data).read()`, so any
  caller-supplied string naming a readable file returned that file's contents;
* the winning copy did not, which is why TASK-2425 (Done, 2026-06-24) concluded the
  summarization arbitrary-file-read was "not active through `analyze()`".

So the shadowing was load-bearing as an accidental security control: a property of
definition order plus an inline suppression, which any tidy-up that deleted the *second*
definition would have silently removed. These tests make the intended behaviour explicit
so it stops resting on that.

They also cover two defects the winning copy carried, both reachable from `analyze()`,
which 17+ core modules import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as _module
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    extract_text_from_input,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The security property, pinned rather than inferred from definition order.
# ---------------------------------------------------------------------------


def test_a_path_to_a_readable_file_is_not_read(tmp_path: Path) -> None:
    """A string naming a readable file is text, never a file to open.

    The shadowed copy returned the file's contents here. `analyze()` reaches this with
    caller-supplied `input_data` for every caller that does not pass
    `input_is_literal_text=True` -- only two sites in the whole app do.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("SENTINEL-DO-NOT-LEAK", encoding="utf-8")

    result = extract_text_from_input(str(secret))

    assert "SENTINEL-DO-NOT-LEAK" not in result, (
        "extract_text_from_input read a file named by its argument; that is the "
        "arbitrary file read TASK-2425 believed was inactive because of the shadowing"
    )
    assert result == str(secret), "a path should come back as the plain string it is"


def test_a_json_file_path_is_not_read_and_reparsed(tmp_path: Path) -> None:
    """The shadowed copy recursed into parsed file content; that path must stay gone."""
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"title": "LEAKED-TITLE"}), encoding="utf-8")

    result = extract_text_from_input(str(payload))

    assert "LEAKED-TITLE" not in result
    assert result == str(payload)


def test_only_one_definition_survives() -> None:
    """A ratchet: the duplicate must not come back, with or without a noqa.

    F811 is not sanctioned policy here -- it appears nowhere in pyproject.toml, so the
    inline `# noqa: F811` was a per-line suppression hiding a real shadow.
    """
    import ast
    import pathlib

    source = pathlib.Path(_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_text_from_input"
    ]

    assert len(definitions) == 1, (
        f"extract_text_from_input is defined {len(definitions)} times at module scope "
        f"(lines {definitions}). The later binding wins, so a second definition makes the "
        "module's behaviour depend on definition order -- and the copy that lost carried "
        "an arbitrary file read. See TASK-13288."
    )


def test_no_noqa_f811_remains() -> None:
    """The suppression that hid the shadow must not return on this function.

    Scoped to extract_text_from_input's own definition line: an unrelated F811
    elsewhere in the module is not this defect, and a whole-file text scan would fail
    on it. The AST test above already catches a second definition however it is
    suppressed.
    """
    import ast
    import pathlib

    source = pathlib.Path(_module.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "extract_text_from_input":
            assert "noqa: F811" not in lines[node.lineno - 1], (
                "extract_text_from_input carries a # noqa: F811, so a module-scope "
                "redefinition of it is being suppressed again. See TASK-13288."
            )


# ---------------------------------------------------------------------------
# Defect 1: dicts carrying text under 'text' or 'content' extracted to nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["text", "content"])
def test_dict_text_and_content_keys_are_extracted(key: str) -> None:
    """`analyze(api, {"text": ...})` returned "Error: Could not extract text content."

    The winning copy looked only at title/description/transcription/segments, so a dict
    carrying its text under either of these keys produced "" and `analyze()` at
    Summarization_General_Lib.py:699 turned that into an error string. The shadowed copy
    handled both keys; that half is worth keeping, unlike the file read.
    """
    result = extract_text_from_input({key: "  the actual body  "})

    assert "the actual body" in result, (
        f"a dict with only a {key!r} key extracted to {result!r}, so analyze() reports "
        "'Could not extract text content' for input that plainly has text"
    )


def test_known_structures_still_win_over_text() -> None:
    """Control: adding the new keys must not displace the existing precedence."""
    result = extract_text_from_input(
        {"title": "T", "transcription": "spoken words", "text": "ignored-if-lower"}
    )

    assert "Title: T" in result
    assert "spoken words" in result


# ---------------------------------------------------------------------------
# Defect 2: a JSON scalar raised TypeError, swallowed into a generic error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["123", "true", "null", "false", "1.5", '"quoted"'])
def test_json_scalar_strings_do_not_raise(raw: str) -> None:
    """`json.loads` on these yields a scalar, then `'title' in data` raised TypeError.

    TypeError is in `_SUMMARIZATION_NONCRITICAL_EXCEPTIONS`, so `analyze()` swallowed it
    into a generic error string and the caller never learned the input was usable text.
    """
    result = extract_text_from_input(raw)

    assert isinstance(result, str)
    assert result != "", f"{raw!r} extracted to empty, so analyze() reports an error"


@pytest.mark.parametrize("raw", ["123", "true"])
def test_json_scalar_keeps_the_original_spelling(raw: str) -> None:
    """A scalar is text the caller wrote; return it, do not re-render it."""
    assert extract_text_from_input(raw) == raw


# ---------------------------------------------------------------------------
# Controls: the shapes that already worked must keep working.
# ---------------------------------------------------------------------------


def test_plain_text_is_unchanged() -> None:
    """Control."""
    assert extract_text_from_input("just some prose") == "just some prose"


def test_json_object_string_is_parsed() -> None:
    """Control."""
    result = extract_text_from_input(json.dumps({"title": "T", "description": "D"}))

    assert "Title: T" in result
    assert "Description: D" in result


def test_segments_list_in_a_dict_is_extracted() -> None:
    """Control."""
    result = extract_text_from_input(
        {"segments": [{"Text": "first"}, {"Text": "second"}]}
    )

    assert "first" in result
    assert "second" in result


def test_transcription_list_is_joined() -> None:
    """Control."""
    result = extract_text_from_input(
        {"transcription": [{"Text": "alpha"}, {"Text": "beta"}]}
    )

    assert "alpha" in result
    assert "beta" in result


# ---------------------------------------------------------------------------
# The dead pair that carried a second copy of the same file read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["extract_metadata_and_content", "format_input_with_metadata"]
)
def test_the_dead_metadata_helpers_are_gone(name: str) -> None:
    """Neither helper had a caller anywhere in the repo, and one was a live hazard.

    `extract_metadata_and_content` did `os.path.exists(input_data)` ->
    `open(input_data)` -> `json.load(file)`: the same caller-controlled file read this
    task removed from `extract_text_from_input`, sitting unreferenced in the same module
    waiting for someone to wire it up. `format_input_with_metadata` only formatted its
    output, so the pair goes together.

    Deleting them is the point rather than tidiness -- AC#6 offered "removed, or retained
    with a documented reason", and "it reads any file the caller names" is not a reason
    to retain dead code.
    """
    assert not hasattr(_module, name), (
        f"{name} is back. It had no callers and carried a caller-controlled file read; "
        "if it is genuinely needed, it must not resolve strings as filesystem paths. "
        "See TASK-13288."
    )


def test_no_caller_controlled_path_probe_remains_in_the_module() -> None:
    """Ratchet: nothing here may test a caller-supplied value for being a path.

    Both removed hazards had the same shape -- an `os.path` existence check on a plain
    argument, followed by `open()` on it.

    Deliberately an AST walk rather than a substring scan. A scan cannot tell code from
    prose, so it trips on the docstrings that explain the removal and could equally be
    dodged by reformatting the call across two lines.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(_module.__file__).read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"isfile", "exists", "isdir"}:
            continue
        # Only flag a probe on a bare argument name, which is what "caller-supplied"
        # looks like here; a probe on a locally-constructed path is not this defect.
        if node.args and isinstance(node.args[0], ast.Name):
            offenders.append(f"line {node.lineno}: os.path.{node.func.attr}({node.args[0].id})")

    assert not offenders, (
        "a caller-supplied value is being resolved as a filesystem path in "
        f"Summarization_General_Lib: {offenders}. analyze() passes caller-supplied "
        "input_data straight through, so this is the arbitrary file read removed in "
        "TASK-13288 -- the one TASK-2425 believed was inactive because of shadowing."
    )

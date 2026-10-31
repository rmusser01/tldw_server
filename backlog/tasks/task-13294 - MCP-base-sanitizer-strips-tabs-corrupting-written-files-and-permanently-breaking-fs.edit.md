---
id: TASK-13294
title: >-
  MCP base sanitizer strips tabs, corrupting written files and permanently
  breaking fs.edit
status: Done
assignee: []
created_date: '2026-09-22 04:44'
updated_date: '2026-09-28 19:37'
labels:
  - bug
  - mcp
  - security
dependencies: []
references:
  - 'tldw_Server_API/app/core/MCP_unified/modules/base.py:766'
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py:1480
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/MCP_unified/modules/base.py:sanitize_input` filters control characters with:

```python
return "".join(ch for ch in s if ch >= " " or ch == "\n")
```

It keeps `\n` and **drops `\t` and `\r`**. It is applied to every tool call via `tool_execution/security.py:harden_and_sanitize_tool_arguments`, and 22 modules inherit it unchanged.

Executed proof:
```
input : 'all:\n\tgcc -o x x.c\n'
output: 'all:\ngcc -o x x.c\n'   # the required tab is gone
```

Two consequences:
1. **`fs.write` silently corrupts tab-significant files and reports success.** A Makefile loses its required tabs. The `expected_sha256` receipt hashes the on-disk pre-image, so the integrity guard structurally cannot catch it.
2. **`fs.edit` is permanently unusable on tab-indented files.** It does exact string replacement, so an `old_string` containing a tab can never match content whose tabs were stripped on the way in.

The correct whitespace class already exists in the same package — `filesystem_module.py:_sanitize_patch_diff` preserves `\n`, `\r` **and** `\t`, with a docstring saying so. Its per-tool exemption table is dead code because the blanket sanitizer already ran upstream.

**Second defect in the same function (same fix window).** The denylist at `:779-788` rejects `["';", '";', "--", "/*", "*/", "xp_", "sp_"]` on pure data bound to parameterised queries, so it buys no injection protection while rejecting ordinary content: Markdown `---` rules, `src/*.py` pathspecs, `SELECT 1 -- note`, git `-- path`, and filenames like `exp_data.csv`. `web_tool_base.py:51-70` already fixed this for web tools only; its docstring diagnoses the problem.

The base sanitizer test uses `os.urandom(4).hex()` as its "safe" fixture — a hex string can never contain a denied substring, so nothing asserts that legitimate content survives.

Found by the comprehensive core-module review; the tab-stripping behaviour independently reproduced by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test writes tab-indented content through fs.write and asserts the tab survives round-trip
- [x] #2 A failing test proves fs.edit can match an old_string containing a tab
- [x] #3 sanitize_input preserves \\t and \\r, matching filesystem_module._sanitize_patch_diff
- [x] #4 The denylist no longer rejects ordinary content -- Markdown ---, src/*.py, git -- path, exp_ filenames -- or is removed as ineffective for parameterised queries
- [x] #5 The base sanitizer test uses a fixture that could actually contain denied substrings and asserts legitimate content survives
- [x] #6 filesystem_module's now-dead exemption table is removed or made reachable
- [x] #7 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 475bdfb929 and 6a16f76e48.

Root cause was two layers, not one. BaseModule.sanitize_input stripped every character below U+0020 except "\n", but fixing that alone changed nothing for the reported path: filesystem_module carried its own sanitize_input override with the same defect (`ch >= " " or ch == "\n"`), which shadowed the base. run_command, sandbox and web_tool_base had near-identical overrides too, each drifted to a different whitespace class, all existing only to escape the SQL denylist. All four deleted; the base is now the one implementation and strips CONTROL_CHARS_RE = [\x00-\x08\x0b\x0c\x0e-\x1f\x7f], preserving tab, newline and carriage return.

AC6: the "exemption table" is execute_tool's per-tool key bypass. It is not dead -- it keeps file content byte-exact -- but it was incomplete: fs.edit's old_string/new_string and notebook.edit_cell's source were exempt while fs.write/fs.write_text's content was not, so fs.write silently stripped form feeds (the conventional page separator in Python, Lisp and C sources) and reported success. Replaced with one _VERBATIM_ARGS table covering all four tools; fs.patch keeps its own path through _sanitize_patch_diff.

CORRECTION to this task's consequence 2: fs.edit was NOT permanently unusable on tab-indented files. old_string and new_string have been exempt from sanitization since f57da0ef3a, so the tab always survived. The AC2 test passes against the pre-fix source; it is kept as a regression guard on an exemption nothing else covered. Consequence 1 (fs.write corrupting Makefiles) was correct, and the AC1 test reproduces it -- red pre-fix, green after.

Tests:
- test_filesystem_module.py::test_fs_write_preserves_tabs_and_form_feeds_round_trip (AC1, red pre-fix)
- test_filesystem_module.py::test_fs_edit_matches_a_tab_indented_old_string (AC2, via the real fs.read -> read_receipt -> fs.edit workflow)
- test_base_sanitizer_preserves_content.py, 18 cases (AC3/4/5), including test_no_module_shadows_the_base_sanitizer, which fails if any module re-declares sanitize_input

Verification:
- MCP_unified in-app: 13 failed / 3301 passed vs baseline 13 / 3281, identical failure set
- tests/MCP + MCP_Hub + MCP_unified: 4 failed, unchanged
- tests/sandbox + Services: 24 failed, identical with and without the change (stash-isolated)
- AC7 bandit: clean over the five touched files, no issues at any severity. Run via `uvx bandit`; bandit is CI-only (security-required.yml), not a declared local dependency, so the venv was left untouched.

Notes recorded on dev by the parallel core-review work (merged 2026-09-23):
Whitespace half SHIPPED in PR #2980 (merge 8045fa2956): BaseModule.sanitize_input and the FilesystemModule override now preserve \t and \r, so fs.write no longer corrupts tab-significant files and fs.edit can match tab-indented content. Qodo correctly caught that fixing only the base class was ineffective, since the override shadows it on the production path.
STILL OPEN - the dangerous_patterns denylist. It rejects '--', '/*', 'xp_', so ordinary Markdown rules, src/*.py pathspecs, git '-- path' and exp_ filenames are refused on data that is bound to parameterised queries. Removing it requires confirming all 22 inheriting modules actually parameterise, which is an owner decision rather than a drive-by edit.
Also still divergent and out of scope for that change: sandbox_module.py strips \t, and run_command_module.py keeps \t but strips \r. Three predicates remain across the four overrides.
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented

Notes recorded on dev by the parallel work (merged 2026-09-27):
Denylist half implemented on fix/mcp-denylist (commit 09671d65b7).
AUDIT (the blocker named when this was deferred in #2980): MCP_unified contains zero
f-string, %-formatted, .format()ed or concatenated SQL. SQL-keyword-anchored grep over
non-test code returns 0 hits, so the denylist had nothing to protect -- the substrings it
rejected had no concatenation to escape into. Its own NUL entry was the four-character
literal "\x00", not the byte, so even that was handled by the control-char strip beside it.
PROBED the denylist against ordinary input before removing it: 7 of 9 routine strings were
rejected -- exp_data.csv (via "xp_"), src/*.py, "--- a markdown rule", "git log -- path",
sp_reports.txt, a/*glob*/b, "SELECT 1 -- note".
THIRD DEFECT FOUND AND FIXED (not previously recorded): all four subclass overrides were
strictly worse than the base they shadowed, and two carried live bugs.
  - sandbox_module kept only "\n" -> stripped every tab AND carriage return. An inline
    Makefile or TSV passed to sandbox.exec lost its tabs and the tool reported success --
    the same defect class this task opened on, in a second module.
  - run_command_module stripped carriage returns, corrupting CRLF payloads.
  - all four let DEL (\x7f) through, because `ch >= " "` admits it.
  - web_tool_base's had become byte-identical in behaviour to the base.
All four deleted rather than kept in sync; that drift is exactly what let #2980 fix the base
while fs.write stayed broken behind the filesystem override. CONTROL_CHARS_RE now has one
definition in base.py, and filesystem_module._sanitize_patch_diff derives from it instead of
hand-copying the predicate its own comment said had to match.
AC#6 NOT DONE, stays open. filesystem_module's exemption table (fs.edit old_string/
new_string, notebook.edit_cell source, fs.patch diff) cannot be made reachable from the
module: tool_execution/security.py:harden_and_sanitize_tool_arguments calls
module.sanitize_input(arguments) unconditionally before execute_tool, passing the whole dict
with no tool name, so the upstream pass cannot know which keys a given tool exempts -- and
has already stripped them by the time the table runs. Honouring the intent means threading
tool_name through sanitize_input across all 22 inheriting modules, which is a boundary
redesign, not part of a denylist removal. Annotated in place with that finding.
VERIFICATION
- test_base_sanitizer_preserves_whitespace.py: 34 passed. Adds the ordinary-content cases
  the old test could not express (AC#5 -- its "safe" fixture was os.urandom(4).hex(), which
  cannot contain a denied substring), a ratchet that fails if any of the four modules
  reintroduces an override, and per-module tab/CR/DEL assertions.
- test_validation_and_sanitization.py: 2 passed. test_deep_argument_sanitization_blocks_
  nested_patterns asserted the removed behaviour, so it is rewritten as
  ..._reaches_nested_values, keeping its real subject (recursion + depth guard) while
  asserting legitimate content survives.
- PROBE-THE-FIX: restoring the denylist turns 7 tests red; restoring either the sandbox or
  the run_command override turns 3 red. The tests are load-bearing, not decorative.
- Bandit: clean, 0 issues across 5333 LOC of touched scope (run via uvx; not in the venv).
- Ruff on touched files: 2 findings, both pre-existing in sandbox_module at lines 22 and 180,
  outside this change's only hunk (256-268).
BASELINE: full MCP suite (app/core/MCP_unified/tests + tests/MCP_unified) is 53 failures/
errors on clean dev at 91e8bbf84c, and the identical 53 with this change -- byte-identical
FAILED/ERROR lists. Zero regressions. The pre-existing set is concentrated in
test_slides_module_standalone_html.py (29) and test_runtime_package_boundary.py (10); the one
hit in a module this change touches, test_filesystem_glob_marks_file_size_unavailable,
monkeypatches Path.stat and is unrelated to sanitization.

Notes from the other branch (merged 2026-09-27):
Re-checked on dev 2026-09-27: ACs 1-5 are in place (#2980, #2996). Stays open for AC #6, the unreachable exemption table in filesystem_module, which needs tool_name threaded through sanitize_input as recorded above.

AC6 done 2026-09-28 by making the exemption table reachable, without threading tool_name through all 22 modules. BaseModule.verbatim_argument_keys(tool_name) returns frozenset() by default; FilesystemModule returns its _VERBATIM_ARGS entry. ToolExecutionSecurity.harden_and_sanitize_tool_arguments now takes tool_name (both call sites pass it) and leaves str values under the declared keys byte-exact. Non-str values under those keys are still fully sanitised (depth guard included), and the ownership-override strip still runs first. Duck-typed modules without the hook default to nothing verbatim. Tests: MCP_unified/tests/test_verbatim_argument_keys.py (9) covers form-feed content surviving for fs.write/fs.edit/notebook.edit_cell, per-key scope, non-str fallback, other tools and modules unchanged, and the override strip. MCP trees: the 5 remaining failures are identical on dev; one parallel-run failure passes alone and with its file.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Base sanitizer now preserves tab/newline/CR and the four shadowing overrides are gone, so the fix actually reaches fs.write. The per-tool verbatim-argument exemption was extended to fs.write/fs.write_text content, which had been silently losing form feeds. This task's fs.edit claim was disproved: those arguments were already exempt.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

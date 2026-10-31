# Stage 2 — `BaseModule` contract and the `sanitize_input` override cluster (C10)

## Scope

The shared module contract in `modules/base.py`, the five `sanitize_input` overrides, where
sanitization is actually applied in the execution pipeline, and the filesystem module's
per-key exemptions. Plus one correctness defect found while running the filesystem suite.

## Code Paths Reviewed

Sanitizer definitions — the C10 cluster, all re-confirmed at these exact ranges:

- `modules/base.py:BaseModule.sanitize_input (766-807)` — depth guard `_depth > 20`;
  denylist `["';", '";', "--", "/*", "*/", "xp_", "sp_", "\\x00"]` matched
  case-insensitively against the whole string; then `"".join(ch for ch in s if ch >= " " or ch == "\n")`.
- `modules/implementations/web_tool_base.py:WebToolBase.sanitize_input (51-70)` — regex
  predicate `CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")` (`web_tool_base.py:23`),
  `_MAX_SANITIZE_DEPTH = 20` (`:24`) — the only copy that names the constant.
  **Preserves `\t`, `\n`, `\r`.** Drops the denylist entirely, with a 10-line docstring
  explaining why (`:52-61`).
- `modules/implementations/run_command_module.py:RunCommandModule.sanitize_input (309-328)`
  — explicit character loop `if ch == "\n" or ch == "\t" or ch >= " "`. **Preserves `\t`,
  drops `\r`.** Hardcodes `_depth > 20`.
- `modules/implementations/sandbox_module.py:SandboxModule.sanitize_input (259-279)` —
  genexp `ch >= " " or ch == "\n"`. **Drops `\t` and `\r`.** Hardcodes `_depth > 20`.
- `modules/implementations/filesystem_module.py:FilesystemModule.sanitize_input (1465-1477)`
  — identical genexp to sandbox. **Drops `\t` and `\r`.** Hardcodes `_depth > 20`.
- `modules/implementations/filesystem_module.py:FilesystemModule._sanitize_patch_diff (1480-1485)`
  — the fifth variant: `ch >= " " or ch in {"\n", "\r", "\t"}`. **Preserves `\t` and `\r`.**
  No depth guard (single-string helper).

Application points:

- `tool_execution/security.py:ToolExecutionSecurity.harden_and_sanitize_tool_arguments (614-626)`
  — strips ownership-override keys then calls `module.sanitize_input(hardened_args)` on the
  **whole** argument dict; any raised exception becomes `InvalidParamsException`.
- `tool_execution/security.py:2004` — the main `tools/call` execution path invokes it
  **before** `build_canonical_snapshot` and before the module's own `execute_tool`.
- `protocol.py:MCPProtocol._harden_and_sanitize_tool_arguments (887-892)` and its call at
  `protocol.py:1764` — the permission-check path.
- `modules/implementations/filesystem_module.py:FilesystemModule.execute_tool (607-931)`,
  specifically the per-tool exemption table at **`:609-626`**: `fs.edit` passes
  `old_string`/`new_string` through raw, `notebook.edit_cell` passes `source` through raw,
  `fs.patch` routes `diff` to `_sanitize_patch_diff`, everything else goes through the
  tab-stripping `sanitize_input`.
- `modules/implementations/filesystem_module.py:409-437` (`fs.write`) and `:441-470`
  (`fs.write_text`) — both take a required `content: string` argument with **no** exemption
  at either layer.

Correctness defect found while executing the suite:

- `modules/implementations/filesystem_module.py:FilesystemModule._glob_paths (1705-1841)`,
  specifically `is_symlink = candidate.is_symlink()` at **`:1778`** (unguarded) versus
  `candidate.stat(follow_symlinks=False)` at **`:1797-1803`** (wrapped in `try/except OSError`
  that sets `size: None, size_unavailable: True`).

Subclass census — 26 module classes inherit `BaseModule` (or `WebToolBase`) under
`modules/implementations/`; **4 files override `sanitize_input`**
(`filesystem_module.py`, `run_command_module.py`, `sandbox_module.py`, `web_tool_base.py`).
The remaining 22 inherit the base denylist: `browser_cdp_module.py`, `characters_module.py`,
`chats_module.py`, `codegraph_module.py`, `cooking_module.py`, `docs_module.py`,
`external_federation_module.py`, `flashcards_module.py`, `git_module.py`,
`governance_module.py`, `kanban_module.py`, `knowledge_module.py`,
`mcp_discovery_module.py`, `media_module.py`, `notes_module.py`,
`persona_visuals_module.py`, `prompts_module.py`, `quizzes_module.py`, `rag_module.py`,
`rpg_module.py`, `skills_module.py`, `slides_module.py`, `template_module.py`
(`web_fetch_module.py`, `web_search_module.py`, `web_research_module.py` inherit
`WebToolBase`'s override).

## Tests Reviewed

Found by import-grep on `sanitize_input`, not by path.

- `app/core/MCP_unified/tests/test_validation_and_sanitization.py` — the only test of the
  **base** sanitizer. `test_deep_argument_sanitization_blocks_nested_patterns (63-…)` uses
  `os.urandom(4).hex()` as its "safe case" (`:66`). A hex string can never contain `--`,
  `/*` or `xp_`, so the test only ever asserts the denylist **fires**; nothing asserts that
  legitimate content survives. **Does not downgrade the risk in finding 1 — it is the reason
  the risk survives.**
- `app/core/MCP_unified/tests/test_web_tool_base.py` — 4 tests
  (`test_sanitize_input_allows_sql_like_substrings :42`, `…preserves_punycode_in_lists :47`,
  `…strips_control_characters :52`, `…depth_guard :57`). These are exactly the assertions
  the other 22 modules lack. **Downgrades the risk for web tools only.**
- `app/core/MCP_unified/tests/test_web_fetch_module.py:374-392`,
  `test_web_search_module.py:217-235`, `test_web_research_module.py:232-236` — the same
  three assertions repeated per web tool.
- `app/core/MCP_unified/tests/test_scope_and_fallbacks.py:215` — exercises
  `mod.sanitize_input(payload)` on a scope payload; does not assert content preservation.
- `app/core/MCP_unified/tests/test_filesystem_module.py` — 104 tests. None asserts that
  `fs.write` round-trips tab-containing content or that `fs.edit` matches a tab-indented
  preimage. `test_filesystem_glob_marks_file_size_unavailable (2640-2683)` **is currently
  red** (see Validation Commands) and is the test that names the intended `_glob_paths`
  behaviour.
- No test in either tree covers `run_command_module.sanitize_input` or
  `sandbox_module.sanitize_input` content preservation (import-grep reachability, not
  measured coverage).

## Validation Commands

Behaviour of the five sanitizer variants, run against the real classes:

```
$ python -  <<'PY'   # (loguru startup lines elided)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from ...filesystem_module import FilesystemModule
from ...sandbox_module import SandboxModule
from ...run_command_module import RunCommandModule
...
PY
fs.write  content  -> 'all:\ngcc -o x x.c\n'
fs.edit   old_str  -> 'def f():\nreturn 1'
fs.patch  diff     -> '--- a\n+++ b\n@@\n-\tx\n+\ty\n'
sandbox   command  -> ['python', '-c', 'def f():\nprint(1)\nf()']
run_cmd   command  -> 'def f():\n\treturn 1'
crlf  fs  ->  'a\nb'
```

Inputs were, respectively: a Makefile body `"all:\n\tgcc -o x x.c\n"`; `"def f():\n\treturn 1"`;
a unified diff with tab-prefixed lines; `["python","-c","def f():\n\tprint(1)\nf()"]`;
`"def f():\n\tprint(1)"`; `"a\r\nb"`. Every tab survives only in `fs.patch`'s `diff` and in
`run_command`.

Base denylist behaviour, via `NotesModule` (which inherits it):

```
markdown hr          REJECT  -> ValueError: Potentially dangerous input detected: --
glob in prose        REJECT  -> ValueError: Potentially dangerous input detected: /*
c comment snippet    REJECT  -> ValueError: Potentially dangerous input detected: /*
sql comment          REJECT  -> ValueError: Potentially dangerous input detected: --
em dash typed        REJECT  -> ValueError: Potentially dangerous input detected: --
plain                OK      -> 'hello world'
tab indented code    OK      -> 'def f():\nreturn 1'
```

Inputs: `"Heading\n---\nbody"`, `"run tests in src/*.py"`, `"/* keep this */"`,
`"SELECT 1 -- note"`, `"wait--what"`, `"hello world"`, `"def f():\n\treturn 1"`.

Same, via `GitModule`:

```
git pathspec sep   REJECT -> Potentially dangerous input detected: --
exp_ filename      REJECT -> Potentially dangerous input detected: xp_
branch name        REJECT -> Potentially dangerous input detected: --
normal path        OK     -> 'src/app.py'
GitModule overrides sanitize_input: False
```

Test runs:

```
$ python -m pytest -q -p no:randomly \
    tldw_Server_API/app/core/MCP_unified/tests/test_validation_and_sanitization.py \
    tldw_Server_API/app/core/MCP_unified/tests/test_web_tool_base.py \
    tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py
======================= 22 passed, 22 warnings in 0.12s ========================

$ python -m pytest -q -p no:randomly tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
================== 1 failed, 103 passed, 19 warnings in 8.28s ==================
FAILED tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_glob_marks_file_size_unavailable
  filesystem_module.py:1778: in _glob_paths
      is_symlink = candidate.is_symlink()
  pathlib.py:914: in is_symlink
      return S_ISLNK(self.lstat().st_mode)
  E   OSError: metadata unavailable
```

```
$ git log -1 --format='%h %ad %s' --date=short -S "test_filesystem_glob_marks_file_size_unavailable" \
    -- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
5009fc8b95 2026-06-03 fix: harden filesystem helper grep budgets
```

## Findings

```
FINDING mcp-unified-1
  axis:        correctness
  class:       divergent-copies
  severity:    High
  sites:       modules/base.py:BaseModule.sanitize_input (766-807) — the denylist
               ["';", '";', "--", "/*", "*/", "xp_", "sp_", "\x00"] at :779-788;
               inherited unchanged by 22 module classes:
               modules/implementations/{browser_cdp_module.py, characters_module.py,
               chats_module.py, codegraph_module.py, cooking_module.py, docs_module.py,
               external_federation_module.py, flashcards_module.py, git_module.py,
               governance_module.py, kanban_module.py, knowledge_module.py,
               mcp_discovery_module.py, media_module.py, notes_module.py,
               persona_visuals_module.py, prompts_module.py, quizzes_module.py,
               rag_module.py, rpg_module.py, skills_module.py, slides_module.py,
               template_module.py};
               applied on every tool call at tool_execution/security.py:
               harden_and_sanitize_tool_arguments (614-626), reached from :2004.
               The one correct copy: modules/implementations/web_tool_base.py:
               WebToolBase.sanitize_input (51-70), whose docstring (:52-61) already
               diagnoses the problem for web tools only.
  canonical:   modules/implementations/web_tool_base.py:51 is the least-wrong existing
               implementation (control-char stripping, no SQL denylist, named depth constant).
  destination: n/a — promote web_tool_base's predicate into BaseModule.sanitize_input and
               parameterise the allowed-control-character set; no new module needed.
  knowledge:   "what characters may appear in an MCP tool argument". It is currently
               answered five times with five different predicates, and the one answer that
               is wrong is the default every new module inherits.
  scenario:    Confirmed by execution (see Validation Commands). `notes.create` with body
               "Heading\n---\nbody" (an ordinary Markdown horizontal rule) raises
               ValueError("Potentially dangerous input detected: --"), which
               harden_and_sanitize_tool_arguments converts to InvalidParamsException, so the
               MCP client sees JSON-RPC -32602. Likewise: any note or chat message containing
               "src/*.py" (matches "/*"); any code snippet with a C-style comment; and
               `git.log` / `git.diff` with the standard pathspec separator "-- path".
               The "xp_" entry additionally rejects the innocuous substring inside
               "exp_data.csv" — confirmed above. These are all pure-data arguments handed to
               parameterised DB_Management queries; the denylist buys no injection protection
               and costs a hard 400 on ordinary content.
  impact:      High: it is a user-visible functional block on the primary write tools
               (notes, chats, characters, prompts, kanban, quizzes, flashcards, slides, git)
               that fails with a security-flavoured message, and the workaround has already
               been written once (web_tool_base) instead of applied at the base.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/test_validation_and_sanitization.py (asserts only
               that the denylist fires; its "safe" fixture is os.urandom(4).hex(), which
               cannot contain any denied substring);
               app/core/MCP_unified/tests/test_web_tool_base.py:42-62 holds the assertions
               the other 22 modules lack. Import-grep reachability, not measured coverage.
  effort:      cheap — delete the denylist from base.py, keep the control-char strip, and
               lift the three web_tool_base assertions into a BaseModule-level test. The
               22 affected modules need no edit.
  owner-only:  no
  confidence:  confirmed (the rejection behaviour, executed above);
               confirmed (that 22 classes inherit it, by class census)
```

```
FINDING mcp-unified-2
  axis:        correctness
  class:       divergent-copies
  severity:    High
  sites:       modules/implementations/filesystem_module.py:FilesystemModule.sanitize_input
               (1465-1477) — strips every char < U+0020 except "\n", so drops "\t" and "\r";
               modules/implementations/sandbox_module.py:SandboxModule.sanitize_input
               (259-279) — same predicate, same effect;
               modules/base.py:BaseModule.sanitize_input (766-807) — same predicate at :796;
               versus modules/implementations/run_command_module.py:RunCommandModule.sanitize_input
               (309-328), which explicitly keeps "\t";
               versus modules/implementations/filesystem_module.py:_sanitize_patch_diff
               (1480-1485), which keeps "\t" and "\r";
               versus modules/implementations/web_tool_base.py:CONTROL_CHARS_RE (:23), which
               keeps "\t", "\n", "\r".
               Defeated exemption table: modules/implementations/filesystem_module.py:
               FilesystemModule.execute_tool (607-931) at :609-626.
               Blanket application point: tool_execution/security.py:
               harden_and_sanitize_tool_arguments (614-626), called at :2004 before
               execute_tool ever runs.
               Unexempted write surface: filesystem_module.py:409-437 (`fs.write`,
               required `content: string`) and :441-470 (`fs.write_text`, same).
  canonical:   modules/implementations/filesystem_module.py:_sanitize_patch_diff (1480-1485)
               — the only variant in this file that gets the whitespace class right, written
               by the same authors for the same reason.
  destination: n/a — one BaseModule.sanitize_input whose preserved-whitespace set is a
               parameter, defaulting to {"\t", "\n", "\r"}.
  knowledge:   "tab and carriage return are data, not control characters". Known in three
               places (run_command, _sanitize_patch_diff, web_tool_base), forgotten in three
               (base, sandbox, filesystem's general path).
  scenario:    Confirmed by execution. (a) `fs.write {"path": "Makefile", "content":
               "all:\n\tgcc -o x x.c\n"}` writes "all:\ngcc -o x x.c\n" — a Makefile whose
               recipe line has lost its required leading tab — and reports success. The
               `expected_sha256` / `read_receipt` integrity guards hash the *pre-image* on
               disk, not the submitted content, so nothing catches it. Same for any
               tab-indented Go/Python/C source or any TSV.
               (b) `fs.edit` is worse: execute_tool:609-612 deliberately exempts
               `old_string`/`new_string` from sanitization, but security.py:2004 has already
               run the blanket sanitizer over the whole dict, so the exemption never sees
               raw input. `_edit_file (2823-…)` does an exact string replacement against the
               file's real bytes, so `old_string="def f():\n\treturn 1"` arrives as
               "def f():\nreturn 1" and can never match a tab-indented file. fs.edit is
               unusable on any tab-indented file, permanently.
               (c) `sandbox.run {"command": ["python","-c","def f():\n\tprint(1)\nf()"]}`
               executes "def f():\nprint(1)\nf()" -> IndentationError, attributed to the
               user's code rather than to the transport.
               (d) CRLF content silently becomes LF ("a\r\nb" -> "a\nb").
  impact:      High: silent data corruption on a write tool, with the tool reporting success
               and the hash guard structurally unable to detect it; plus a permanently broken
               edit path. The `fs.edit` exemption being dead is the sharpest evidence that the
               two-layer sanitize design is not understood by its own callers.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/test_filesystem_module.py (104 tests, none asserts
               tab round-trip through fs.write or a tab-containing fs.edit preimage);
               test_web_tool_base.py:52 is the only control-char assertion anywhere and it
               covers the variant that is already correct. No test covers sandbox_module or
               run_command_module content preservation. Import-grep reachability, not coverage.
  effort:      moderate — the fix is small, but it changes observable behaviour of fs.write /
               fs.edit / sandbox.run and needs round-trip tests added per tool first. Gate on
               writing those tests; the sites themselves are well covered structurally.
  owner-only:  no
  confidence:  confirmed (the stripping, executed above; the ordering, read at security.py:2004
               vs filesystem_module.py:609)
```

```
FINDING mcp-unified-5
  axis:        correctness
  class:       n/a
  severity:    Medium
  sites:       modules/implementations/filesystem_module.py:FilesystemModule._glob_paths
               (1705-1841) — unguarded `candidate.is_symlink()` at :1778, immediately
               followed by the guarded `candidate.stat(follow_symlinks=False)` at :1797-1803.
  canonical:   NONE
  destination: n/a
  knowledge:   "a per-entry filesystem metadata error degrades that entry, it does not fail
               the walk". Encoded correctly 15 lines below the site that violates it.
  scenario:    `Path.is_symlink()` swallows only the errno set `_ignore_error` treats as
               "does not exist" (ENOENT/ENOTDIR/…); it re-raises EACCES, EIO and ESTALE.
               `fs.glob` walks the whole workspace root, so a single unreadable entry — an
               NFS/SMB stale handle, a directory the walker can descend but whose child it
               cannot lstat, a file unlinked between os.walk and the probe — raises OSError
               out of `asyncio.to_thread` at :845 and aborts the entire glob with a
               transport-level error. The surrounding code at :1797 is written to handle
               exactly this and emit `{"size": null, "size_unavailable": true}`.
               `test_filesystem_glob_marks_file_size_unavailable` (test_filesystem_module.py
               :2640-2683) asserts that degraded shape and currently FAILS on HEAD with
               `OSError: metadata unavailable` — see Validation Commands.
  impact:      Medium: availability, not correctness of data; scoped to workspaces on
               network or permission-restricted filesystems. Raised above Low because the
               intended behaviour is already specified by a test, and that test is red.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/test_filesystem_module.py::
               test_filesystem_glob_marks_file_size_unavailable (2640-2683) — RED on HEAD.
               Added 2026-06-03 in 5009fc8b95. It is in `pyproject.toml` testpaths but in no
               ci.yml shard (see stage 4, finding mcp-unified-4), which is why it has stayed red.
  effort:      cheap — wrap :1778 in the same `try/except OSError` shape as :1797 and decide
               the degraded `type` value; the test already encodes the expected result.
  owner-only:  no
  confidence:  confirmed (the red test and its traceback, reproduced above);
               probable-risk (that EACCES/ESTALE reach it in production — inferred from
               CPython's `_ignore_error` errno set, not observed on a live deployment)
```

## Suggested Refactor/Actions

Ordered cheapest-first; none of these touches an owner-only path.

1. **Collapse the five sanitizer predicates into one parameterised base implementation.**
   `BaseModule.sanitize_input(input_data, _depth=0)` keeps the depth guard (lifting
   `_MAX_SANITIZE_DEPTH` out of `web_tool_base.py:24` into `base.py` so it is defined once
   instead of four times), drops the SQL denylist entirely, and strips control characters
   using the `web_tool_base.CONTROL_CHARS_RE` predicate — which preserves `\t`, `\n`, `\r`.
   The four overrides then delete their bodies; `_sanitize_patch_diff` can go too. That is a
   net deletion of roughly 70 lines and removes findings 1 and 2 together.
   Small enough to land without a design doc, **but** it changes the observable behaviour of
   `fs.write`, `fs.edit`, `sandbox.run` and every inheriting module, so it needs: a
   round-trip test per write tool (tab, CRLF, `--`, `/*`) added *before* the change, and a
   Backlog task noting the behaviour change for release notes.

2. **Delete the dead exemption table** at `filesystem_module.py:609-626` as part of (1). Once
   the base sanitizer preserves tabs, the `fs.edit` / `notebook.edit_cell` / `fs.patch`
   special-casing has no purpose — and today it actively misleads, because
   `security.py:2004` has already sanitized the arguments before it runs. If the exemptions
   are genuinely needed for something else, they must move to `harden_and_sanitize_tool_arguments`
   where they can take effect; that would be a design decision (per-tool sanitizer policy at
   the protocol layer) and would warrant `Docs/Design/2026-MM-DD-mcp-tool-argument-sanitization-design.md`
   plus an ADR, since it changes a security-boundary contract.

3. **Fix `_glob_paths:1778`** independently of (1) and (2) — it is a two-line change that
   turns a currently-red test green.

4. Propose (do not create) three Backlog tasks: one per finding, each linking this stage file.
   Task 1 and 3 are self-contained; task 2 should link the design doc named in action (2)
   if the exemption table is kept rather than deleted.

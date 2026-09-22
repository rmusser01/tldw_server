# Stage 1 — Architecture survey and inventory

## Scope

Establish the real shape of `tldw_Server_API/app/core/MCP_unified/` before reading any
code linearly: how the 141,588 LOC split between production code, in-app tests, and tests
for a *different* package; which files are hot by size × churn; which ADRs bind; and which
seed clusters actually land here.

## Code Paths Reviewed

- `tldw_Server_API/app/core/MCP_unified/` — 253 `.py` files, 141,588 LOC total.
  Split: **51,546 LOC production** (106 files) / **90,042 LOC in-app tests** (147 files).
- `tldw_Server_API/app/core/MCP_unified/server.py:MCPServer._register_default_modules (1180-1265)`
  — the module registration path and the only construction site of `ModuleConfig`.
- `tldw_Server_API/app/core/MCP_unified/server.py:_resolve_env_placeholders (166-179)` —
  resolves `${ENV:default}` in module `settings`; returns `os.getenv(...)`, i.e. **always a
  `str`**. Load-bearing for finding `mcp-unified-6`.
- `tldw_Server_API/app/core/MCP_unified/modules/base.py:ModuleConfig (132-152)` — the
  `settings: dict[str, Any]` bag every module accessor reads.
- `tldw_Server_API/app/core/MCP_unified/protocol_types.py:RequestContext (58-95)` — the
  neutral per-request context. Note for stage 3: it has **no `is_admin` attribute**.
- `apps/mcp-unified/src/mcp_unified/` — 125 files, 46,521 LOC, own `pyproject.toml`,
  `LICENSE`, `README.md`, `USER_GUIDE.md`, and **no `tests/` directory of its own**.
- `tldw_Server_API/app/core/MCP_unified/README.md:3` — "the standalone package/gateway is
  planned but not shipped in this tree yet", which the 46,521-LOC `apps/mcp-unified/`
  package and the 11-file `mcp-unified-rc` CI workflow contradict.

### Hot set (size × churn), from the sidecars

| File | LOC | Commits/12mo | Read? |
| --- | ---: | ---: | --- |
| `protocol.py` | 2,532 | 91 | yes (skim + targeted) |
| `server.py` | 2,348 | 74 | yes (auth/env/registration paths) |
| `tests/test_gateway_fastapi_package.py` | 6,575 | 57 | classified only |
| `tests/test_runtime_package_boundary.py` | 1,669 | 50 | header + trigger analysis |
| `tests/test_extraction_contracts.py` | 4,429 | 43 | header + contract list |
| `modules/implementations/media_module.py` | 2,408 | 32 | yes (cache, admin, DB) |
| `modules/implementations/filesystem_module.py` | 3,227 | 32 | yes (full sanitize/glob/settings) |
| `modules/base.py` | 1,003 | 32 | yes (full) |
| `modules/implementations/notes_module.py` | 2,258 | 27 | yes (admin, DB) |
| `tool_execution/security.py` | 2,290 | 12 | yes (sanitize/exec pipeline) |

The churn list is dominated by **test** files (5 of the top 6). That is itself the stage-4
signal: this module's maintenance cost is concentrated in a test tree that CI does not run.

## Tests Reviewed

Located by import-grep, never by path, per the briefing.

- **63 files** under `tldw_Server_API/tests/` import `core.MCP_unified`, concentrated in
  `tests/MCP_unified/` (33) and `tests/MCP/` (2), with singletons in `Workflows` (5),
  `DB_Management` (5), `unit` (3), `Services` (2), `AuthNZ` (2) and 11 others.
- **147 files / 90,042 LOC** live *inside* the module at `app/core/MCP_unified/tests/`.
  Classified by import target: 105 import `core.MCP_unified`, **35 import the standalone
  `mcp_unified` package** (39,533 LOC), 6 import both, 13 import neither.
- `app/core/MCP_unified/tests/conftest.py:mcp_ws_client (12-33)` — the in-app fixture that
  disables WS auth and IP allowlists. It exists only in the in-app tree; `tests/MCP_unified/`
  has its own separate `conftest.py`.
- Only one basename collides across the two trees (`test_run_command_module.py`). I checked
  whether that produces a pytest import-mismatch error — **it does not** (see Validation
  Commands); `tldw_Server_API/tests` is an importable package while the in-app dir collects
  as a plain `Dir`. Not a finding; recorded so the next reviewer does not re-derive it.

## Validation Commands

```
$ find tldw_Server_API/app/core/MCP_unified -name '*.py' | wc -l
     253
$ find tldw_Server_API/app/core/MCP_unified -name '*.py' -exec wc -l {} + | tail -1
  141588 total
$ find tldw_Server_API/app/core/MCP_unified/tests -name '*.py' -exec wc -l {} + | tail -1
   90042 total
$ find apps/mcp-unified/src/mcp_unified -name '*.py' -not -path '*__pycache__*' -exec wc -l {} + | tail -1
   46521 total
$ find apps/mcp-unified -type d -name tests -not -path '*__pycache__*'
(no output)

$ cd tldw_Server_API/app/core/MCP_unified/tests
$ grep -rlE "^(import|from) mcp_unified" *.py | wc -l
      35
$ grep -rl "core\.MCP_unified" *.py | wc -l
     105
$ grep -rlE "^(import|from) mcp_unified" *.py | xargs wc -l | tail -1
   39533 total

$ grep -rl "core\.MCP_unified" tldw_Server_API/tests | wc -l
      63

$ grep -rn --include='*.py' "from tldw_Server_API.app.api" tldw_Server_API/app/core/MCP_unified | grep -v "/tests/" | wc -l
       1
  (the single hit: modules/implementations/rag_module.py -> api.v1.schemas.rag_schemas_unified)

$ grep -rln --include='*.py' -E "cursor\.execute\(|\.execute\(\"SELECT|execute\(f\"" \
    tldw_Server_API/app/core/MCP_unified | grep -v "/tests/"
(no output — no raw SQL outside DB_Management)

$ grep -rn --include='*.py' -E "def .*=\s*(\[\]|\{\})\s*[,)]" tldw_Server_API/app/core/MCP_unified | grep -v "/tests/"
(no output — no mutable default arguments)

$ python -m pytest --collect-only -q \
    tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py \
    tldw_Server_API/tests/MCP_unified/test_run_command_module.py 2>&1 | tail -1
========================= 98 tests collected in 0.61s ==========================
  (no "import file mismatch" — the same-basename pair collects cleanly)
```

## Findings

No standalone findings are filed at this stage. Stage 1 establishes three facts the later
stages depend on, all of which invert the naive reading of "141k-LOC god module":

1. **Production code is 51.5k LOC, not 141k.** The module is large but not the monolith the
   raw count suggests; there is no single-file god module (largest production file is
   `filesystem_module.py` at 3,227 LOC). The `Media_DB_v2.py -> media_db/` decomposition
   precedent cited in the briefing does **not** apply here — nothing needs splitting on size.
2. **`_resolve_env_placeholders` stringifies every env-sourced setting.** Every per-module
   numeric/boolean settings accessor therefore receives `str`, not `int`/`bool`. This is the
   mechanism that turns the stage-3 accessor divergence into a live config bug.
3. **The layering axis is clean here.** One schema-only `core -> api` import; no raw SQL
   outside `DB_Management/`; no mutable defaults; no `time.sleep`/`requests`/`subprocess.run`
   on async paths. The repo-wide layering ratchet recommendation does not need a seat in this
   module's backlog.

Two clusters from the briefing were checked and **dropped here**:

- **C1 (base64 cursor padding)** — exactly one site in this module
  (`modules/implementations/prompts_catalog.py:decode_prompt_cursor (94-160)`), carried
  forward to stage 4 as a participant in the repo-wide cluster, not as local duplication.
- **C3–C9** — no sites mapped to this module, confirmed by the module map, and not searched
  beyond incidental greps.

## Suggested Refactor/Actions

- None at this stage. Stage 1 is a reading-order artifact; actions are filed against the
  stage that produced the finding.
- One documentation correction worth folding into whichever change lands first:
  `tldw_Server_API/app/core/MCP_unified/README.md:3` states the standalone gateway is not
  shipped in this tree. It is (`apps/mcp-unified/`, 46,521 LOC, with a dedicated CI
  workflow). See finding `mcp-unified-12` in stage 4.

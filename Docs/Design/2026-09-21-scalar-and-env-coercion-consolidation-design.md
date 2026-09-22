# Scalar and environment coercion consolidation

Parent: TASK-13293 (comprehensive core-module code review). Scope: define one coercion contract for
string→bool, string→int and environment-variable reads, adopt it by ratchet rather than by mass
refactor, and close the two confirmed fail-open defects the current sprawl produced. This document
defines a bounded correction to existing behaviour, not a new capability.

## Problem

Measured across `tldw_Server_API/app` on 2026-09-21:

| Cluster | Definition sites | Modules touched |
|---|---|---|
| `_truthy` / `_is_truthy` / `_coerce_bool` / `_as_bool` / `_to_bool` | 103 | 37 |
| `_coerce_int` / `_safe_int` / `_as_int` / `_to_int` | 102 | 36 |
| `_env_bool` / `_env_int` / `_env_flag` / `_env_str` | 51 | 18 |

Roughly 256 private re-implementations of five-line functions, spanning most of the codebase.

**This is not a tidiness problem.** Three of the copies are wrong in ways that reached security- and
correctness-relevant switches, and two of them fail *open*:

1. `core/TTS/adapters/audio_cpp_config.py:_as_bool` ends with `return bool(value)`, so any
   unrecognised non-empty string is `True`. `_as_bool("n")`, `("none")`, `("disabled")`, `("nope")`
   all return `True`. It gates `allow_remote_base_url` (`:133`), whose `False` value is what enables
   the loopback check at `validate_base_url:54`. An operator writing `allow_remote_base_url = "n"`
   to mean "no" **disables the egress guard** and the adapter POSTs synthesis requests — including
   uploaded voice-reference audio — to an arbitrary remote host. The same parser gates `managed`
   (spawns a sidecar subprocess) and `retain_request_artifacts` (retains request audio on disk).
   The canonical `parse_bool` returns `False` for every one of those inputs: the private copy is not
   merely laxer, it is **inverted** on the tokens an operator is most likely to type.
2. `core/LLM_Calls/providers/google_adapter.py:_env_flag` has the same shape:
   `return lowered not in {"0","false","no","off",""}`. `_env_flag("disabled")` is `True`. It gates
   `LLM_ADAPTERS_GEMINI_{IMAGE,AUDIO,VIDEO}_URLS_BETA`, which control whether a caller-supplied URL
   is forwarded into `{"fileData": {"fileUri": ...}}` for Google to fetch server-side.
3. `core/RAG/rag_service/request_resolution.py:_is_truthy_value` accepts
   `{"1","true","yes","on"}` — **no `"y"`** — and is applied to all ten public Search-Agent boolean
   flags. `SEARCH_QUERY_CLASSIFICATION=y` silently resolves the feature *off*, with no warning and
   no validation error, while `RAG_GUARDRAILS_STRICT=y` in the same request resolves *on* because
   that path routes through `core/testing.py:is_truthy`. Two dialects, one request.

Three designated canonicals already exist and are bypassed:

| Helper | Contract | Truthy set |
|---|---|---|
| `core/testing.py:is_truthy` | two-way; anything not truthy is `False` | `{"1","true","yes","y","on"}` |
| `core/MCP_unified/environment.py:is_truthy` | byte-identical to the above | `{"1","true","yes","y","on"}` |
| `core/TTS/utils.py:parse_bool` | **three-way**; separate truthy and falsy token sets, unrecognised returns an explicit `default` | truthy + falsy sets |

## Decisions

**1. `parse_bool`'s three-way contract is the correct one, and becomes canonical.** The two-way
contract (`in set → True, else False`) cannot distinguish "the operator said no" from "the operator
typed something we do not understand". Both of the fail-open defects above are the result of a copy
inventing a third answer for that case. A three-way contract makes the ambiguous case explicit at
the call site and is the only one that can be made to fail closed on a security switch.

**2. Unrecognised input never silently becomes `True`.** The default is supplied by the caller. For
any flag that gates egress, subprocess execution, credential handling or authorization, the default
is `False` and an unrecognised value is logged at WARNING with the key name (never the value).

**3. The new home is `core/Utils/coercion.py`, not `core/Utils/Utils.py`.** `Utils.py` is 1,110 LOC
of unrelated concerns of which 282 LOC across 18 public symbols currently have zero importers; adding
to it would deepen the problem the review identified. `coercion.py` has one job: turn an untyped
scalar or environment value into a Python scalar, with an explicit contract for the unparseable case.
`core/http_client.py` (6,600 LOC) is likewise excluded as a destination.

**4. Adoption is by ratchet, not by mass refactor.** 256 sites across 37 modules cannot be changed in
one reviewable commit, and a sweeping automated rewrite across security-relevant switches is exactly
the change most likely to introduce the next fail-open. A ratchet test seeded at today's counts makes
the number monotonically decreasing while each module migrates on its own schedule, under its own
tests. Precedent in-repo: `tests/lint/test_endpoint_auth_deps_import_boundary.py` (AST ban list),
`tests/lint/test_no_dict_usage.py` (explicit file list), and
`tests/lint/test_noncritical_exception_tuples.py`.

**5. The two fail-open defects are fixed immediately and do not wait for the migration.** They are
three-line changes with named failure scenarios; coupling them to a 37-module programme would leave
live egress and subprocess switches misparsing for the duration.

## The contract

```
parse_bool(value, *, default: bool) -> bool
    bool        -> returned as-is
    int | float -> 0 is False, non-zero True
    str         -> case-insensitive, stripped, matched against TRUTHY then FALSY
                   unmatched -> `default`, logged at WARNING with the key, not the value
    None        -> `default`

parse_int(value, *, default: int | None, minimum=None, maximum=None) -> int | None
    unparseable or out of range -> `default`, logged at WARNING

env_bool(key, *, default: bool) -> bool      # os.environ read + parse_bool
env_int(key, *, default, minimum, maximum)   # os.environ read + parse_int
```

`TRUTHY = {"1","true","yes","y","on","enabled"}` and
`FALSY = {"0","false","no","n","off","disabled","none","null",""}`.
The union of every set currently in use, so no existing accepted spelling stops working;
`"none"` and `"null"` come from `core/TTS/utils.py:FALSY_STRINGS`, which already treats them as false. `"y"` and
`"n"` are both included — their asymmetric treatment today is defect 3 above. `"enabled"` is included
because `core/LLM_Calls/cache_intents.py:88` already accepts it and `core/Setup/readiness_service.py:69`
does not.

`is_truthy` in `core/testing.py` and `core/MCP_unified/environment.py` stay where they are as
two-way wrappers delegating to `parse_bool(value, default=False)`. Neither is deleted:
`MCP_unified/environment.py` exists specifically so the standalone package can move without
depending on host-level modules, and that constraint is still real.

## Implementation boundaries

Excluded from this design, deliberately:

- **Domain-specific parsers that only look like coercers.** `core/Chunking`'s option parsing,
  per-provider payload coercion in `core/LLM_Calls/providers/`, and the ~40 per-table row mappers in
  `core/DB_Management/Sync_DB.py` answer different questions over different input domains. They are
  `justified-divergence` and are not ratchet targets.
- **Row-value coercion from database drivers.** `core/AuthNZ/repos/` has five private
  `_to_bool`/`_load_json_dict` copies, but those normalize *driver output* where the same logical
  column is `TEXT` on SQLite and `JSONB` on PostgreSQL. That is an AuthNZ schema fact, not a general
  one; it gets its own AuthNZ-local module and must not be routed through an env helper.
- **Any change to what a currently-accepted value means.** The union sets above are additive. If a
  migration would change an existing deployment's resolved configuration, it stops and is recorded.
- **`api/v1/**`** — owner-only per `CONTRIBUTING.md`. Endpoint-layer call sites are listed in the
  ratchet seed but migrated only by the owner.

## Migration stages

| Stage | Content | Gate |
|---|---|---|
| 1 | Fix the two fail-open defects (`audio_cpp_config._as_bool`, `google_adapter._env_flag`) and the missing `"y"` in `request_resolution._is_truthy_value`. Each with a failing test first. | Tests red→green; no behaviour change for currently-recognised values |
| 2 | Add `core/Utils/coercion.py` with the contract above and its own unit tests, including a table test over every token in both sets. | New module only; no call sites changed |
| 3 | Re-point `core/testing.py:is_truthy` and `MCP_unified/environment.py:is_truthy` at it as wrappers. | Full suite; these two have the widest blast radius |
| 4 | Add `tests/lint/test_private_coercion_ratchet.py`, seeded at the current per-module counts. | The ratchet fails if any count rises |
| 5 | Migrate opportunistically, module by module, whenever a module is touched for other reasons. Each migration lowers its seed. | Per-module tests |

Stage 5 has no completion date by design. The ratchet is the deliverable; reaching zero is not.

## Verification and release gate

- Stage 1 is not complete until a test proves `_as_bool("n")` resolves the loopback guard **on**, and
  that a remote `base_url` is rejected when `allow_remote_base_url` is an unrecognised token.
- Stage 2's table test must assert every token in `TRUTHY` and `FALSY`, and must assert that a token
  in neither returns the supplied default rather than `True`.
- Stage 4's ratchet must fail when a new private coercer is added. Prove it by adding one in a
  scratch commit and observing red, then reverting — the same way the existing lint ratchets were
  validated.
- No stage may change the resolved value of any configuration key that parses today. A migration that
  would is a finding, not a merge.
- `make lint-changed` and Bandit on touched scope per `Docs/ADR/005`.

## Alternatives rejected

- **Mass automated rewrite across all 256 sites.** Not reviewable in one change, and the sites
  include egress, subprocess and authorization switches where a wrong rewrite is a security defect
  rather than a style regression. The ratchet reaches the same end state without a single high-risk
  commit.
- **Promote `core/testing.py:is_truthy` as canonical.** It is the most-adopted helper, but its
  two-way contract is what makes the fail-open class expressible. Adopting it repo-wide would make
  unrecognised values silently `False` on switches that currently default `True`, changing resolved
  configuration — which boundary 3 forbids.
- **Put the helpers in `core/Utils/Utils.py`.** Rejected under decision 3.
- **Delete the two `is_truthy` copies and import `coercion` directly.** Rejected: the MCP copy exists
  to keep the standalone package boundary intact, and that reason still holds.

## Follow-ups not in scope

`core/Utils/iso_datetime.py` (9 endpoint parsers plus ~40 module copies) and
`core/Infrastructure/retry.py` (28 copies in `PromptStudioDatabase.py` plus 8 module-level
implementations) are the same shape as this cluster and should reuse this document's ratchet pattern.
They need their own design docs; the decisions above about destination, contract explicitness and
ratchet-over-refactor are intended to be reusable.

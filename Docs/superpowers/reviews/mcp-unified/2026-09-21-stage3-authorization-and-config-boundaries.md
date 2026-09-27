# Stage 3 — Authorization, per-user data attribution, and configuration boundaries

## Scope

Every place a sibling MCP module re-answers a question the module system already answers
once: "is this caller an admin", "which database and whose identity", "what does this
setting mean", "is this flag on". This stage absorbs the standard arc's API/schema-boundary
and data-source-boundary stages, which have no independent findings here (stage 1,
Validation Commands).

## Code Paths Reviewed

Admin predicates — five definitions:

- `protocol_types.py:_metadata_has_admin_claims (134-146)` — MCP's own normalizer:
  `roles` and `permissions` both run through `_metadata_claim_values (111-117)`
  (accepts `str | list | tuple | set | frozenset`), `.strip().lower()`, then
  `"admin" in roles or "*" in permissions`. Consumed only by
  `_has_trusted_compat_claims (148-169)`; re-exported through `protocol.py:79`.
- `modules/implementations/media_module.py:MediaModule._is_admin (2058-2063)` — `roles`
  must be a `list`; no `.strip()`; `permissions` ignored entirely.
- `modules/implementations/notes_module.py:NotesModule._is_admin (2200-2205)` — identical
  body to media's, differing only in the caught-exception tuple name.
- `modules/implementations/kanban_module.py:KanbanModule._is_admin (1585-1592)` — probes
  `getattr(context, "is_admin", False)` first, then the same list-only roles check.
- `modules/implementations/sandbox_module.py:SandboxModule._is_admin (210-229)` — probes
  `context.is_admin`, accepts `roles` as `str` or `list` with `.strip()`, and additionally
  grants on `permissions` containing `"*"` or `"system.configure"`.
- `modules/implementations/mcp_discovery_module.py:McpDiscoveryModule._is_admin (283-302)` —
  a sixth shape: roles `str`-or-list, then a **raw SQL** fallback
  `SELECT 1 FROM user_roles ur JOIN roles r ON r.id = ur.role_id WHERE ur.user_id = ? AND r.name = 'admin'`
  issued directly against `pool` from inside `core/MCP_unified/` (`:296-299`).

The host's definition, for comparison:

- `core/AuthNZ/auth_principal_resolver.py:_claims_mark_admin (131-140)` with
  `_PLATFORM_ADMIN_ROLES = {"admin", "owner", "super_admin"}` (`:70`) and
  `_ADMIN_CLAIM_PERMISSIONS = {"*", "system.configure", "admin"}` (`:71`), normalizing via
  `_normalized_claim_values (121-128)`.

Where the answers are consumed — all four are irreversible or cross-user gates:

- `modules/implementations/sandbox_module.py:142` — `if not self._is_admin(context) and str(owner) != user_id: raise PermissionError` — **cross-user sandbox session access**.
- `modules/implementations/media_module.py:1847` — permanent (hard) media delete.
- `modules/implementations/notes_module.py:2008` — permanent note delete.
- `modules/implementations/kanban_module.py:1595` (`_require_admin`) — kanban workflow-policy admin ops.

Context plumbing:

- `protocol_types.py:RequestContext.__init__ (66-95)` — the attribute list is
  `request_id, user_id, client_id, session_id, metadata, start_time, db_paths,
  server_auth_scope, logger`. **There is no `is_admin`.**
- `server.py:1483-1487` — translates the `AuthPrincipal` into `RequestContext.metadata`,
  copying `roles` and `permissions` and **dropping `principal.is_admin`**.
- `server.py:1592`, `:1615`, `:1636-1638`, `:1658-1669` — the other four metadata-population
  sites (AuthNZ identity, MCP JWT, single-user compat key, API key + scopes).

Per-user database opening — seven definitions, six of them over the same ChaChaNotes DB:

- `modules/implementations/notes_module.py:_open_db (533-541)` — `client_id = user_id or "mcp_notes_<name>"`.
- `modules/implementations/quizzes_module.py:_open_db (533-539)` + `_get_client_id (541-551)` — `user_id`, else `client_id`, else **raises**.
- `modules/implementations/chats_module.py:_open_db (130-136)` — constant `f"mcp_chats_{self.config.name}"`.
- `modules/implementations/characters_module.py:_open_db (111-117)` — constant `f"mcp_characters_{…}"`.
- `modules/implementations/flashcards_module.py:_open_db (442-448)` — constant `f"mcp_flashcards_{…}"`.
- `modules/implementations/persona_visuals_module.py:_open_db (486-495)` — constant `f"mcp_persona_visuals_{…}"`.
- `modules/implementations/kanban_module.py:_open_db (1003-1012)` — different DB (`KanbanDB`), and the only one that **requires** `user_id` and passes it as `user_id=`, not `client_id=`.
- `core/DB_Management/ChaChaNotes_DB.py:35-36` — "The library requires a `client_id` upon
  initialization, which is used to attribute changes in the `sync_log` and in individual
  records." (`sync_log` inserts at `:1467, 1487, 1500, 1514, 1526, 1545, 1557, 1567, 1578, 1609, 1623, 1633, 1646, 1660, 1670, 1680, 1689, 1702, 1712` and further.)

Workspace-root resolution — three copies of the same six-key metadata contract:

- `modules/implementations/filesystem_module.py:_resolve_workspace_root (1487-1520)` — trust guard at `:1502-1503`, fails closed with `PermissionError`.
- `modules/implementations/git_module.py:_resolve_workspace_root (1220-1265)` — same guard at `:1235-1239`, fails closed with `_GitToolError`.
- `modules/implementations/run_command_module.py:_resolve_workspace_root (591-629)` — **no trust guard**; wraps the resolver in `except Exception: return None` (`:621-623`) and returns `None` on empty (`:626-627`).
- Supporting triplicate: `_first_nonempty` at `filesystem_module.py:57-62`, `git_module.py:63-68`, `run_command_module.py:794-799`.

Settings accessors — six definitions over the same `ModuleConfig.settings` dict:

- `modules/implementations/filesystem_module.py:_setting_positive_int (949-955)` — `int()`-coerces, clamps `max(1, …)`, falls back to `default` only on `TypeError/ValueError`.
- `modules/implementations/browser_cdp_module.py:_setting_positive_int (452-459)` — `int()`-coerces, returns `default` when `<= 0`, and guards a non-dict `settings`.
- `modules/implementations/git_module.py:_setting_positive_int (2071-2075)` — **strict**: `if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value <= 0: return default`.
- `modules/implementations/run_command_module.py:_setting_int (514-519)` — `max(1, int(raw))`, keyword-only `default`.
- `modules/implementations/filesystem_module.py:_setting_bool (957-963)` — truthy set `{"1","true","yes","on","y"}`; an unrecognised string yields `False` regardless of `default`.
- `modules/implementations/browser_cdp_module.py:_setting_bool (461-473)` — separate truthy `{"1","true","t","yes","y","on"}` and falsy `{"0","false","f","no","n","off"}` sets; an unrecognised string yields `default`.

Truthy coercion (cluster C2) inside the module — one shared helper, seven bypasses:

- `environment.py:is_truthy (17-22)` with `_TRUTHY = {"1","true","yes","y","on"}` (`:13`) — the designated helper. Exactly **one** importer in the module: `modules/implementations/mcp_discovery_module.py:17`.
- `protocol.py:_is_truthy (147-152)` with its own `_TRUTHY_VALUES` at `:144` — same set, re-declared.
- `tool_execution/security.py:_TRUTHY_VALUES (58)` — same set, re-declared a third time.
- `server.py:_fallback_truthy (523-529)` — adds `"t"`.
- `modules/implementations/browser_cdp_module.py:469` — adds `"t"` (inside `_setting_bool`).
- `modules/implementations/kanban_module.py:_parse_bool_like (34-45)` — base set, no `"t"`.
- `modules/implementations/filesystem_module.py:962` — base set minus nothing, order differs.
- `modules/implementations/{characters_module.py:58, chats_module.py:64, notes_module.py:157, prompts_module.py:86}` — `str(os.getenv("MCP_HEALTHCHECK_DB_WRITE_TEST","")).lower() in {"1","true","yes"}`, four byte-identical copies that accept neither `"on"`/`"y"` nor a leading/trailing space.
- `server.py:_env_flag_explicitly_disabled (532-534)` — the inverse set `{"0","false","off","no","n"}`, with no shared counterpart.

Caching:

- `modules/implementations/media_module.py:110-111` — `self._media_cache = {}` (plain dict), `self._cache_ttl = settings.get("cache_ttl", 300)`.
- `modules/implementations/media_module.py:_clean_cache (1968-1978)` — full O(n) scan, TTL eviction only, awaited at `:1219` after **every** cache write.
- `modules/implementations/media_module.py:_evict_user_db_cache_locked (500-521)` and `_get_or_create_user_db (523-558)` — the contrast case in the same class: `OrderedDict`, TTL **and** LRU `max_size` eviction, lock-held, explicit resource close.

Verbatim helper:

- `_safe_exception_family` — five byte-identical bodies differing only in docstring:
  `server.py (81-96)`, `tool_execution/security.py (90-107)`, `tool_execution/runtime.py (30-…)`,
  `tool_execution/reporting.py (28-…)`, `modules/base.py (58-75)`.

## Tests Reviewed

By import-grep, not path.

- `app/core/MCP_unified/tests/support.py:build_mcp_admin_auth_override (85-109)` — builds an
  `AuthPrincipal(..., roles=["admin"], permissions=["system.logs"], is_admin=True)`. Note it
  sets `is_admin` on the **principal**, which is correct; nothing constructs a
  `RequestContext` with `is_admin`.
- `app/core/MCP_unified/tests/test_http_mapping.py:61` — the only other `is_admin=True`, also
  on an `AuthPrincipal`.
- `app/core/MCP_unified/tests/test_kanban_module.py`, `test_notes_crud_tags.py`,
  `test_mcp_discovery_module.py`, `test_knowledge_search_defaults.py`,
  `test_protocol_scope_enforcement.py`, `test_mounted_jsonrpc_transport_contract.py` —
  reference roles/admin. They exercise the `roles=["admin"]` list shape, which every copy
  handles identically, so **none of them discriminates between the five predicates**. They
  do not downgrade finding 3.
- `tldw_Server_API/tests/MCP_unified/{test_mcp_hub_audit_findings.py, test_mcp_http_auth_paths.py, test_tool_catalogs_api.py, test_tool_catalogs_pg.py, test_mcp_hub_management_api.py}` — hub/API-level admin paths; also list-shaped roles only.
- `grep -rln "_setting_positive_int|_setting_bool|_setting_int"` across **both** test trees
  returns nothing. The four accessor variants have **no direct test at all** (import-grep
  reachability, not measured coverage) — they are only exercised transitively via default
  values, i.e. the branch that reads a configured value is never taken in tests.
- `tldw_Server_API/tests/MCP_unified/test_mcp_server_test_mode_truthiness.py` — the only test
  aimed at truthy parsing; it covers `is_test_mode`, not the seven bypass sites.

## Validation Commands

```
$ grep -rn --include='*.py' "def _is_admin" tldw_Server_API/app/core/MCP_unified | grep -v tests
modules/implementations/media_module.py:2058:    def _is_admin(self, context: Any | None) -> bool:
modules/implementations/kanban_module.py:1585:    def _is_admin(self, context: Any | None) -> bool:
modules/implementations/mcp_discovery_module.py:283:    async def _is_admin(self, context: RequestContext, pool: Any) -> bool:
modules/implementations/sandbox_module.py:210:    def _is_admin(self, context: Any | None) -> bool:
modules/implementations/notes_module.py:2200:    def _is_admin(self, context: Any | None) -> bool:

$ grep -rn --include='*.py' "\.is_admin = \|is_admin=" tldw_Server_API/app/core/MCP_unified
tldw_Server_API/app/core/MCP_unified/tests/support.py:105:            is_admin=True,
tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py:61:            is_admin=True,
  (both on AuthPrincipal; zero production assignments, and RequestContext has no such field)

$ grep -rn --include='*.py' "def _safe_exception_family" tldw_Server_API/app/core/MCP_unified | grep -v tests
server.py:81 / tool_execution/security.py:90 / tool_execution/runtime.py:30
tool_execution/reporting.py:28 / modules/base.py:58            -> 5 definitions

$ grep -rn --include='*.py' "def _open_db" tldw_Server_API/app/core/MCP_unified | grep -v tests | wc -l
      10

$ grep -rn --include='*.py' "is_truthy" tldw_Server_API/app/core/MCP_unified | grep -v tests | grep import
config.py:25:from .environment import env_flag_enabled, is_explicit_pytest_runtime, is_test_mode
security/request_guards.py:21:from ..environment import is_test_mode
security/ip_filter.py:17:from ..environment import is_explicit_pytest_runtime, is_test_mode
modules/implementations/mcp_discovery_module.py:17:from ...environment import is_truthy
  (is_truthy itself: 1 importer)

$ grep -rln "_setting_positive_int\|_setting_bool\|_setting_int" \
    tldw_Server_API/app/core/MCP_unified/tests tldw_Server_API/tests
(no output)

$ grep -n "_media_cache\|maxlen\|OrderedDict" tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py | head -4
110:            self._media_cache = {}
1146:        cached = self._media_cache.get(cache_key)
1215:            self._media_cache[cache_key] = {
  (no maxlen, no OrderedDict on _media_cache; _user_db_cache uses OrderedDict at :538)

$ sed -n '166,179p' tldw_Server_API/app/core/MCP_unified/server.py
def _resolve_env_placeholders(value: Any) -> Any:
    ...
    return os.getenv(env_name, default if default is not None else "")
  (every ${VAR:default} setting resolves to a str)
```

## Findings

```
FINDING mcp-unified-3
  axis:        correctness
  class:       adoption-gap
  severity:    High
  sites:       modules/implementations/media_module.py:MediaModule._is_admin (2058-2063),
               consumed at :1847 (permanent media delete);
               modules/implementations/notes_module.py:NotesModule._is_admin (2200-2205),
               consumed at :2008 (permanent note delete);
               modules/implementations/kanban_module.py:KanbanModule._is_admin (1585-1592),
               consumed at :1595 (_require_admin);
               modules/implementations/sandbox_module.py:SandboxModule._is_admin (210-229),
               consumed at :142 (cross-user sandbox session access);
               modules/implementations/mcp_discovery_module.py:McpDiscoveryModule._is_admin
               (283-302), consumed at :157 — plus raw SQL at :296-299 inside core/.
               Dead attribute probes: kanban_module.py:1587 and sandbox_module.py:212 read
               getattr(context, "is_admin", False) on protocol_types.py:RequestContext
               (58-95), which defines no such attribute; server.py:1483-1487 builds the
               context from an AuthPrincipal and drops principal.is_admin.
  canonical:   protocol_types.py:_metadata_has_admin_claims (134-146), built on
               _metadata_claim_values (111-117) — MCP's own normalizer, currently used only
               by _has_trusted_compat_claims (148-169).
               Upstream of that: core/AuthNZ/auth_principal_resolver.py:_claims_mark_admin
               (131-140) with _PLATFORM_ADMIN_ROLES (:70) and _ADMIN_CLAIM_PERMISSIONS (:71).
  destination: n/a — a single `BaseModule.caller_is_admin(context)` delegating to
               protocol_types._metadata_has_admin_claims (promoted from private), whose
               claim sets are aligned with AuthNZ's. No new module.
  knowledge:   "what claims make a caller an administrator". Answered six times: AuthNZ
               {admin,owner,super_admin} + {*,system.configure,admin}; MCP's own normalizer
               {admin} + {*}; sandbox {admin} + {*,system.configure}; kanban/media/notes
               {admin} only, list-shaped only, unstripped; discovery {admin} + a DB lookup.
  scenario:    Two reachable, opposite-direction failures.
               (a) UNDER-GRANT. A principal whose AuthNZ role is "owner" — a member of
               _PLATFORM_ADMIN_ROLES, so AuthPrincipal.is_admin is True — authenticates over
               the MCP WebSocket. server.py:1486 writes metadata["roles"] = ["owner"].
               Every MCP module predicate tests only for the literal "admin", so the platform
               owner is refused `media.delete permanent=true`, `notes.delete permanent=true`,
               and every kanban workflow-policy operation, while being an administrator
               everywhere else in the product. The same holds for "super_admin".
               (b) OVER-GRANT relative to MCP's own gate. An API key normalized at
               server.py:1662-1669 into metadata["permissions"] containing "system.configure"
               (no admin role) is treated as admin by sandbox_module._is_admin:219-226 and
               therefore passes the cross-user check at sandbox_module.py:142, gaining access
               to another user's sandbox session. protocol_types._metadata_has_admin_claims
               — the predicate MCP uses for its own trusted-claims gate — returns False for
               exactly that input, because its permission set is {"*"} only. So the module
               grants a cross-user capability its own protocol layer would refuse.
               (c) Dead branch: the getattr(context, "is_admin", …) probes in kanban and
               sandbox can never fire in production because RequestContext has no such
               attribute, so those two modules are quietly running the same roles-only logic
               as media and notes despite appearing to be the stricter implementations.
  impact:      High. The consumers are a cross-user resource gate and two irreversible
               deletes. Even the under-grant direction is high-severity for an audit: it means
               the product has no single answer to "is this caller an admin", and the next
               module author copies whichever neighbour they open first.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/{test_kanban_module.py, test_notes_crud_tags.py,
               test_mcp_discovery_module.py, test_protocol_scope_enforcement.py,
               test_mounted_jsonrpc_transport_contract.py, test_http_mapping.py, support.py};
               tldw_Server_API/tests/MCP_unified/{test_mcp_hub_audit_findings.py,
               test_mcp_http_auth_paths.py, test_mcp_hub_management_api.py,
               test_tool_catalogs_api.py, test_tool_catalogs_pg.py}.
               All of them use roles=["admin"], the one shape every copy agrees on, so none
               discriminates between the predicates. Import-grep reachability, not coverage.
  effort:      moderate — the code change is small, but aligning MCP's claim sets with
               AuthNZ's is a security-semantics decision that widens who counts as admin
               (owner/super_admin/"admin" permission), so it needs a design record and a
               table-driven test over the claim matrix before landing.
  owner-only:  no
  confidence:  confirmed (the six divergent definitions; RequestContext having no is_admin;
               the dead getattr probes; server.py dropping principal.is_admin);
               probable-risk (that a "system.configure"-only API key is issued in a given
               deployment — the code path exists and is reachable, but I did not observe such
               a key in a live configuration)
```

```
FINDING mcp-unified-6
  axis:        correctness
  class:       divergent-copies
  severity:    Medium
  sites:       modules/implementations/git_module.py:_setting_positive_int (2071-2075) —
               the strict copy, consumed at :2078, :2081, :2084, :2087, :2090, :2093, :2096,
               :2099, :2102, :2105 (ten limits);
               modules/implementations/filesystem_module.py:_setting_positive_int (949-955) —
               coercing copy, 24 call sites between :119 and :2184;
               modules/implementations/browser_cdp_module.py:_setting_positive_int (452-459) —
               coercing copy with a non-dict guard, 8 call sites :173-:326;
               modules/implementations/run_command_module.py:_setting_int (514-519) —
               coercing copy, 3 call sites :176-:178;
               modules/implementations/filesystem_module.py:_setting_bool (957-963) and
               modules/implementations/browser_cdp_module.py:_setting_bool (461-473) — two
               truthy vocabularies and two different unrecognised-value behaviours;
               shared upstream: server.py:_resolve_env_placeholders (166-179), whose output
               feeds ModuleConfig.settings at server.py:1255.
  canonical:   NONE inside the module. browser_cdp_module.py:452-459 is the least-wrong
               integer copy (coerces, guards a non-dict settings bag, rejects <= 0 to the
               declared default rather than clamping to 1).
  destination: n/a — these belong on BaseModule (modules/base.py), which already owns
               ModuleConfig; four protected accessors (`setting_int`, `setting_positive_int`,
               `setting_bool`, `setting_str`) with one documented coercion policy.
  knowledge:   "how a value in ModuleConfig.settings is coerced to a number or a boolean".
               Four integer answers and two boolean answers for one settings bag populated by
               one loader.
  scenario:    server.py:_resolve_env_placeholders (166-179) resolves every `${VAR:default}`
               setting via os.getenv, so an env-sourced setting is **always a str**. An
               operator who configures `max_log_entries: "${MCP_GIT_MAX_LOG:500}"` gets the
               string "500". git_module._setting_positive_int rejects any non-int
               (`not isinstance(raw_value, int)`) and silently returns the hardcoded default
               100 — the override is discarded with no log line. The same spelling under
               filesystem, browser_cdp or run_command is honoured, because those three
               int()-coerce. So the identical config idiom works for three module families
               and silently does nothing for the fourth, across all ten git limits.
               Second divergence: filesystem_module._setting_bool("require_lock_for_mutation",
               default) with a mistyped value such as "enabled" returns False even when the
               default is True, because an unrecognised string falls through the truthy set;
               browser_cdp's copy returns the default instead. A typo therefore silently
               disables a mutation-lock requirement under one module and is ignored under
               the other.
  impact:      Medium: silent misconfiguration rather than data loss — the operator sees the
               default behaviour and no error. Raised above Low because one of the affected
               booleans (`require_lock_for_mutation`, filesystem_module.py:2316) gates a
               write-safety control, and because the accessors have literally zero direct
               test coverage.
  cost-driver: n/a
  tests:       none. `grep -rln "_setting_positive_int\|_setting_bool\|_setting_int"` across
               app/core/MCP_unified/tests and tldw_Server_API/tests returns no files. The
               accessors are exercised only through their default branch.
  effort:      cheap to implement (one accessor set on BaseModule, six deletions), moderate to
               land safely — behaviour changes for git_module's ten limits, so the accessor
               needs its own unit test table (int, "5", "0", "-1", "", None, True, "abc")
               before the six copies are deleted.
  owner-only:  no
  confidence:  confirmed (the four divergent bodies; that _resolve_env_placeholders returns
               str); probable-risk (that a deployment actually uses an ${ENV} placeholder for
               a git limit today — the mechanism is wired, I did not find a config that uses it)
```

```
FINDING mcp-unified-7
  axis:        duplication
  class:       divergent-copies
  severity:    Medium
  sites:       modules/implementations/filesystem_module.py:_resolve_workspace_root (1487-1520),
               trust guard at :1502-1503;
               modules/implementations/git_module.py:_resolve_workspace_root (1220-1265),
               trust guard at :1235-1239;
               modules/implementations/run_command_module.py:_resolve_workspace_root (591-629),
               **no trust guard**, blanket `except Exception: return None` at :621-623;
               supporting triplicate `_first_nonempty` at filesystem_module.py:57-62,
               git_module.py:63-68, run_command_module.py:794-799;
               shared dependency app/services/mcp_hub_workspace_root_resolver.py:
               McpHubWorkspaceRootResolver.resolve_for_context, wired at
               filesystem_module.py:116, git_module.py:311, run_command_module.py:597-603.
  canonical:   modules/implementations/filesystem_module.py:1487-1520 is the correct copy:
               it enforces the guard and fails closed with a generic PermissionError rather
               than a module-specific error type.
  destination: n/a — one `resolve_module_workspace_root(context)` helper next to
               McpHubWorkspaceRootResolver (app/services/mcp_hub_workspace_root_resolver.py),
               owning "translate a RequestContext into the resolver's scope arguments",
               returning the resolved Path or raising a single PermissionError. The three
               modules keep their own error translation.
  knowledge:   the six-key metadata contract of resolve_for_context — session_id, user_id,
               workspace_id, workspace_trust_source (or selected_workspace_trust_source),
               owner_scope_type (or selected_workspace_scope_type), owner_scope_id (or
               selected_workspace_scope_id) — plus the rule "a session_id without a user_id
               is only trusted when workspace_trust_source == 'shared_registry'".
  scenario:    n/a (duplication axis). The change-amplification cost is concrete: adding a
               seventh scope dimension, or renaming one of the six `selected_*` metadata
               aliases, means editing three files; missing one silently degrades workspace
               scope resolution for that tool family rather than failing. The divergence is
               already present — run_command's copy omits the trust guard entirely and
               converts every resolver failure into `None`. In run_command that feeds only
               the relative-spill-dir path (`_resolve_spill_dir`, :521-533), so it is not
               itself a privilege bug; the hazard is that it is the copy a fourth module
               would most plausibly clone, and the guard it drops is the trust boundary.
  impact:      Medium: no confirmed live privilege defect, but this is the workspace trust
               boundary for the three filesystem-touching tool families, and one of the three
               copies already fails open where the other two fail closed.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/{test_filesystem_module.py, test_git_module.py,
               test_run_command_module.py, test_workspace_root_resolver.py};
               tldw_Server_API/tests/MCP_unified/{test_mcp_hub_workspace_root_resolver.py,
               test_mcp_hub_multi_root_path_execution.py, test_mcp_ws_workspace_context.py}.
               The first two cover the guarded copies; nothing asserts run_command's
               fail-open behaviour is intended. Import-grep reachability, not coverage.
  effort:      cheap — the two guarded copies are well covered, so extracting the shared
               resolver-call shape is low-risk. Deciding whether run_command should adopt the
               guard is the only judgement call.
  owner-only:  no
  confidence:  confirmed (the three copies and the missing guard)
```

```
FINDING mcp-unified-8
  axis:        duplication
  class:       divergent-copies
  severity:    Medium
  sites:       modules/implementations/notes_module.py:_open_db (533-541) — client_id = user_id
               or "mcp_notes_<name>";
               modules/implementations/quizzes_module.py:_open_db (533-539) with
               _get_client_id (541-551) — user_id, else client_id, else raise;
               modules/implementations/chats_module.py:_open_db (130-136) — constant;
               modules/implementations/characters_module.py:_open_db (111-117) — constant;
               modules/implementations/flashcards_module.py:_open_db (442-448) — constant;
               modules/implementations/persona_visuals_module.py:_open_db (486-495) — constant.
               Contrast: modules/implementations/kanban_module.py:_open_db (1003-1012), the
               only copy that requires user_id and passes it as the identity argument.
               Contract: core/DB_Management/ChaChaNotes_DB.py:35-36.
  canonical:   modules/implementations/quizzes_module.py:_get_client_id (541-551) is the
               correct copy — it resolves the real caller identity and refuses to open the DB
               without one, rather than silently substituting a module label.
  destination: n/a — `BaseModule.open_chacha_db(context, *, purpose: str)` on
               modules/base.py, owning "resolve the per-user ChaChaNotes path and the caller
               identity that will be written into sync_log". Six deletions.
  knowledge:   "who is the attributed author of a ChaChaNotes mutation". ChaChaNotes_DB's own
               docstring (:35-36) states client_id "is used to attribute changes in the
               sync_log and in individual records"; four of six MCP writers discard it.
  scenario:    n/a (duplication axis). Consequence, stated as risk rather than defect:
               notes.* and quizzes.* write sync_log rows with client_id = the real user id,
               while chats.*, characters.*, flashcards.* and persona_visuals.* write
               "mcp_chats_Chats" / "mcp_characters_Characters" / etc. for every caller. In a
               shared-registry workspace (the trust source named at filesystem_module.py:1502)
               two users mutating the same ChaChaNotes DB are indistinguishable in the sync
               log for four of the six entity families, so conflict resolution and the audit
               trail cannot attribute the change. This is the same "the split can silently
               diverge" shape the briefing flags as known-real.
  impact:      Medium: it degrades sync attribution and audit rather than granting access.
               Elevated from Low because the divergence is 4-vs-2 within one physical DB and
               one entity set, which makes the sync_log internally inconsistent rather than
               uniformly coarse.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/{test_notes_crud_tags.py, test_chats_module.py,
               test_characters_module.py, test_flashcards_module.py, test_quizzes_module.py,
               test_persona_visuals_module.py} exercise the CRUD paths; no test in either
               tree asserts the client_id written to sync_log. Import-grep reachability.
  effort:      cheap to consolidate; the attribution change itself is behavioural and should
               be paired with a check of what reads sync_log.client_id before landing.
  owner-only:  no
  confidence:  confirmed (the six divergent client_id values; the ChaChaNotes contract at
               :35-36); probable-risk (the sync/audit consequence — I read the contract and
               the insert sites, I did not exercise a two-user shared-workspace sync)
```

```
FINDING mcp-unified-9
  axis:        duplication
  class:       adoption-gap
  severity:    Medium
  sites:       environment.py:is_truthy (17-22) with _TRUTHY (:13) — the designated helper,
               with exactly one importer in the whole module
               (modules/implementations/mcp_discovery_module.py:17).
               Bypasses, each with its own literal set:
               protocol.py:_TRUTHY_VALUES (144) + _is_truthy (147-152) — same set, re-declared;
               tool_execution/security.py:_TRUTHY_VALUES (58) — same set, re-declared;
               server.py:_fallback_truthy (523-529) — adds "t";
               modules/implementations/browser_cdp_module.py:469 — adds "t";
               modules/implementations/kanban_module.py:_parse_bool_like (34-45);
               modules/implementations/filesystem_module.py:962;
               modules/implementations/characters_module.py:58,
               modules/implementations/chats_module.py:64,
               modules/implementations/notes_module.py:157,
               modules/implementations/prompts_module.py:86 — four byte-identical
               `str(os.getenv("MCP_HEALTHCHECK_DB_WRITE_TEST","")).lower() in {"1","true","yes"}`;
               inverse with no shared counterpart: server.py:_env_flag_explicitly_disabled
               (532-534), set {"0","false","off","no","n"}.
  canonical:   environment.py:is_truthy (17-22). Note it is itself a deliberate fork of
               core/testing.py:is_truthy (30-32) — the module docstring (environment.py:1-6)
               justifies the fork by the standalone-package boundary, so the fork is
               justified-divergence and is NOT the finding. The finding is that the fork is
               then ignored seven times inside its own package.
  destination: n/a — the destination already exists. Add `is_falsy`/`env_flag_disabled` to
               environment.py (it owns exactly this responsibility, 50 lines, one concern)
               to absorb server.py:532-534, and import is_truthy at the seven bypass sites.
  knowledge:   "which strings mean true in MCP configuration". Currently three vocabularies:
               {1,true,yes,y,on} (environment, protocol, security, kanban, filesystem),
               {1,true,t,yes,y,on} (server fallback, browser_cdp),
               {1,true,yes} (the four health-check copies).
  scenario:    n/a (duplication axis). Concretely divergent today: `MCP_HEALTHCHECK_DB_WRITE_TEST=on`
               enables nothing in characters/chats/notes/prompts but is truthy everywhere
               else in the module; `MCP_TRUSTED_..._FLAG=t` is truthy for server.py's fallback
               and browser_cdp but not for protocol.py or tool_execution/security.py. Change
               amplification: adding "enabled" or "y" to the accepted spellings — which the
               repo-wide C2 cluster shows has already happened elsewhere
               (core/LLM_Calls/cache_intents.py accepts "enabled", core/Setup/readiness_service.py
               does not) — requires finding and editing eight literal sets in this module alone.
  impact:      Medium: no single site is dangerous, but the module ships a purpose-built
               helper with one importer while eight sites re-implement it, which is the
               clearest adoption-gap instance in the module and the cheapest to close.
  cost-driver: n/a
  tests:       tldw_Server_API/tests/MCP_unified/test_mcp_server_test_mode_truthiness.py
               covers is_test_mode only; app/core/MCP_unified/tests/test_profile_presets.py
               and test_mcp_config_sanitization.py touch config flags transitively. No test
               covers any of the eight bypass sites' vocabulary. Import-grep reachability.
  effort:      cheap — mechanical, and environment.py is dependency-free so importing it
               cannot create a cycle. The four health-check copies are a one-line change each.
  owner-only:  no
  confidence:  confirmed
```

```
FINDING mcp-unified-10
  axis:        efficiency
  class:       divergent-copies
  severity:    Medium
  sites:       modules/implementations/media_module.py:110-111 (`self._media_cache = {}`,
               `self._cache_ttl = settings.get("cache_ttl", 300)`);
               write at :1215-1218; read at :1146-1149;
               modules/implementations/media_module.py:_clean_cache (1968-1978), awaited at
               :1219 after every write;
               invalidation at :1795, :1861 via _clear_media_cache (1980-1982), which clears
               everything.
               Correct sibling in the same class:
               modules/implementations/media_module.py:_evict_user_db_cache_locked (500-521)
               and _get_or_create_user_db (523-558).
  canonical:   modules/implementations/media_module.py:_evict_user_db_cache_locked (500-521)
               — same file, same class, same author: OrderedDict, TTL **and** LRU max_size,
               under a lock, with resource close on eviction.
  destination: n/a — apply the sibling's shape to _media_cache; no new module.
  knowledge:   "how this module bounds an in-process cache". Answered twice in one class,
               once correctly.
  scenario:    n/a (efficiency axis).
  cost-driver: Two costs, both on the `media.search` hot path.
               (1) Memory: `_media_cache` is an unbounded plain dict keyed by
               _make_cache_key("search_media", payload) where the payload (:1114-1143)
               includes the query text, every filter, offset, limit, user_id, and — when
               present — the full `query_vector`. Each entry stores a whole formatted result
               page including `rows`. Nothing bounds entry count; the only eviction is by
               TTL (default 300 s) and a blanket clear on media mutation. Scales with
               *distinct search argument tuples across all users within the TTL window*,
               times the page size, with no ceiling.
               (2) CPU: `_clean_cache` (1968-1978) is awaited at :1219 on **every** cache
               write and iterates the entire dict to build `expired_keys`. So each cache
               miss costs O(current cache size). Scales quadratically with the number of
               distinct searches per TTL window.
               The fix is already written 1,450 lines up: `_user_db_cache` bounds itself with
               `_user_db_cache_max_size` (default 100) and evicts LRU via `popitem(last=False)`.
  tests:       app/core/MCP_unified/tests/{test_media_module.py, test_media_search_semantic.py}
               and tldw_Server_API/tests files importing core.MCP_unified.modules.implementations
               .media_module exercise search results; none asserts a cache bound or eviction
               behaviour. Import-grep reachability, not coverage.
  effort:      cheap — swap the dict for OrderedDict, add a `cache_max_entries` setting
               alongside the existing `cache_ttl`, and move the TTL sweep out of the write
               path, mirroring :500-521 exactly.
  owner-only:  no
  confidence:  confirmed (the unbounded dict and the per-write full scan);
               probable-risk (the magnitude — I did not profile a loaded server)
```

```
FINDING mcp-unified-11
  axis:        duplication
  class:       true-duplication
  severity:    Low
  sites:       server.py:_safe_exception_family (81-96);
               tool_execution/security.py:_safe_exception_family (90-107);
               tool_execution/runtime.py:_safe_exception_family (30-…);
               tool_execution/reporting.py:_safe_exception_family (28-…);
               modules/base.py:_safe_exception_family (58-75).
               Five byte-identical bodies (bounded-length, ASCII, identifier-shaped
               exception-type-name allowlist); only the docstrings differ.
  canonical:   NONE — no copy is designated; all five are private to their file.
  destination: environment.py, which already exists as this package's dependency-free
               host-neutral helper module (50 lines, one stated responsibility: "helpers MCP
               needs without depending on host-level modules"). `_safe_exception_family` is
               exactly that shape — stdlib-only, no MCP imports — and adding it keeps
               environment.py cohesive rather than turning it into a junk drawer.
  knowledge:   "what of an exception is safe to put in a log line". A single security policy
               (never the message, only a validated type name) with five sources of truth.
  scenario:    n/a (duplication axis). Change amplification: tightening the policy — say,
               excluding third-party exception type names, or bounding to 32 chars — requires
               finding all five. A miss leaves one log path emitting under the old policy,
               and because the sites are spread across server / protocol-security / runtime /
               reporting / module base, the one that is missed is the one nobody greps for.
  impact:      Low: the five copies are currently identical, so there is no live defect. It
               is filed because this is a security-logging policy, the class of thing that
               must have one owner, and consolidating it is a pure deletion.
  tests:       app/core/MCP_unified/tests/test_extraction_contracts.py names
               `_safe_exception_family` in SAFE_EXCEPTION_LOG_HELPERS (:43-47) as part of its
               AST ratchet over safe logging — so an extraction contract already knows this
               helper exists by name, in five places. Import-grep reachability.
  effort:      cheap — one move, five deletions, and one line changed in
               test_extraction_contracts.py's helper set if it is path-scoped.
  owner-only:  no
  confidence:  confirmed
```

## Suggested Refactor/Actions

1. **`BaseModule` should own the four things its subclasses keep re-deriving.** Findings 3,
   6, 8 and (partly) 7 are the same structural gap: `modules/base.py` defines the lifecycle
   and the `ModuleConfig`, but not the accessors every subclass needs. Add, on `BaseModule`:
   `caller_is_admin(context)`, `setting_int/setting_positive_int/setting_bool/setting_str`,
   and `open_chacha_db(context, *, purpose)`. That is one file touched and roughly sixteen
   private helpers deleted across nine module files.
   **Needs design-first treatment**, because `caller_is_admin` changes who counts as an
   administrator: `Docs/Design/2026-MM-DD-mcp-module-base-accessors-design.md`, an ADR entry
   for the admin-claims alignment specifically (it is a durable security decision that must
   name AuthNZ's `_PLATFORM_ADMIN_ROLES` / `_ADMIN_CLAIM_PERMISSIONS` as the source of
   truth), a Backlog task linking both, and `IMPLEMENTATION_PLAN_mcp-module-base-accessors.md`
   staged as: (1) settings accessors + their test table, (2) `open_chacha_db`,
   (3) `caller_is_admin` + the claim-matrix test, (4) delete the private copies.
   Bandit runs on touched scope (ADR-005); the admin-claims change must clear HIGH/CRITICAL.
2. **Promote `protocol_types._metadata_has_admin_claims` to a public name** as the first step
   of that plan — it is already correct in shape and already re-exported through
   `protocol.py:79`; only the claim sets need to match AuthNZ.
3. **Close the truthy adoption gap (finding 9) independently and first.** It needs no design
   doc: import `environment.is_truthy` at the eight bypass sites and add `env_flag_disabled`
   to `environment.py` for `server.py:532-534`. Pure deletion, no behaviour change except
   that `"t"` and `"on"` become uniformly accepted — call that out in the PR body.
4. **Move `_safe_exception_family` into `environment.py` (finding 11)** in the same change as
   (3); both are the "host-neutral helper" concern and both are pure deletions.
5. **Bound `_media_cache` (finding 10)** as a standalone change, copying the shape of
   `_evict_user_db_cache_locked` verbatim. No design doc; it is a local fix with an in-file
   template.
6. **`mcp_discovery_module.py:296-299` issues raw SQL from `core/MCP_unified/`.** Reportable
   on taste grounds per the briefing's bounded list ("no raw SQL in endpoints" /
   "keep storage access centralized via core/DB_Management/", `Docs/Architecture.md`). It is
   a single two-table lookup and is subsumed by the `caller_is_admin` work in (1) — the
   DB-backed role check should move behind an AuthNZ repo call rather than being re-homed.
   Not filed as a separate finding; recorded here so it is not lost.

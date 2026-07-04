# First-Run MCP Tool Packs Setup Design

Date: 2026-07-04
Status: Approved for implementation planning
Backlog: TASK-12131

## Summary

Add an optional first-run setup step named `mcp_tools` that introduces users to MCP tools through outcome-oriented packs, saves a visible `First-run default` MCP Hub permission profile, and validates tool availability with a direct safe sample tool run. The step should not require a chat request or successful LLM tool call.

The first version uses five core packs:

- Research
- Learning
- Writing
- Media Library
- Personal Knowledge

Low-risk, server-native, read-only tools are selected by default. Local file reading is not selected by default because the server may not be running on the user's local machine. External network, local file reads, writes, destructive actions, and process/run-command capabilities are visible as disabled add-ons inside packs and require explicit opt-in.

No manual external MCP server builder is added to first-run setup. If external servers are already configured, first-run setup can refresh discovery and validate one safe read-only external tool when available. If no external servers exist, setup links to MCP Hub for adding them later.

## Goals

- Give first-time single-user installs a clear MCP tools starting point without exposing raw module internals.
- Seed useful low-risk MCP permissions through existing MCP Hub profiles and assignments.
- Keep packs as setup/catalog metadata, not a second permission system.
- Let users validate tool execution directly from setup without relying on chat or model tool-call behavior.
- Show enabled packs and available tools at the end of setup for both built-in-only and external-server configurations.
- Keep risky capabilities visible but disabled by default.

## Non-Goals

- Do not build a new MCP pack database or standalone permission model.
- Do not add a manual external server builder to first-run setup.
- Do not enable local file reads, external network, writes, destructive actions, or process execution by default.
- Do not require MCP validation to complete first-run setup.
- Do not redesign MCP Hub.

## User Flow

The `MCP Tool Packs` step appears near the end of `UnifiedSetupWizard`, after provider/default setup and before first chat.

1. **Pack selection**
   Users see five outcome packs. Low-risk server-native read-only packs are selected by default. Each pack can expand to show add-ons, but the default view stays compact.

2. **Risk add-ons**
   Add-ons are disabled by default. Selecting an add-on changes generated policy, not only UI display. External network and local file read can use normal opt-in. Writes, destructive actions, and process/run-command require stronger confirmation and remain off even when a user selects all packs.

3. **Save packs**
   The user saves the selected packs before validation. Saving upserts the generated MCP Hub profile and assignment, then returns the effective profile id and effective tool summary.

4. **Run sample tool**
   Built-in validation uses the saved effective permissions. External discovery and external sample validation are setup-only readiness checks against servers that were already configured before first-run validation starts. They do not require the `external_network_read` add-on, do not add external tools to the generated profile, and do not authorize ongoing app use. Ongoing external tool use still requires the generated policy to include the relevant external add-on.

5. **Summary**
   The user sees enabled packs, available tool count, available tools for the current configuration, disabled add-ons, external server status, and an `Open MCP Hub` link. If no external servers are configured, the summary explains that external tools can be added later in MCP Hub.

6. **Skip**
   `Skip MCP tools` remains available. Skipping records `skipped` and does not block setup completion. MCP Hub should later show a recovery action for validation not run during setup.

## Validation States

Persist one explicit validation state in first-run step data:

- `not_run`
- `built_in_passed`
- `external_discovered`
- `external_tool_passed`
- `no_safe_external_tool`
- `external_discovery_incomplete`
- `failed`
- `skipped`

The optional setup step must not become a hidden completion gate. Setup completion should accept every validation state listed above. Only malformed `mcp_tools` state should block.

`not_run` covers the path where the user saves packs, sees the current available tools, and chooses `Continue` without running validation. MCP Hub should later show this as `not validated during setup`, not as a setup failure.

`no_safe_external_tool` is not an error. It means discovery worked or an external server exists, but no discovered tool could be confidently classified as safe read-only for setup validation.

## Backend Contract

Add `mcp_tools` to the first-run step allowlist in `tldw_Server_API/app/api/v1/endpoints/setup.py` with explicit allowed fields. Do not accept arbitrary nested secrets or raw external server config through this step.

Allowed persisted fields for the `mcp_tools` step:

- `acknowledged`: boolean
- `selected_pack_ids`: list of strings
- `selected_addon_ids`: list of strings
- `confirmed_addon_ids`: list of strings
- `confirmation_version`: string or null
- `validation_state`: one of the states listed above
- `profile_id`: integer or null
- `assignment_id`: integer or null
- `catalog_version`: string
- `effective_tool_count`: integer
- `validated_at`: ISO timestamp string or null
- `validation_message`: redacted string or null
- `last_validation_run_id`: string or null

The setup step must reject raw external server definitions, credentials, environment variables, headers, filesystem paths, command strings, and arbitrary nested tool config.

Add a small setup MCP tools service that delegates to existing MCP Hub/profile/catalog/readiness services.

Minimal endpoints:

- `GET /api/v1/setup/first-run/mcp-tools/catalog`
- `POST /api/v1/setup/first-run/mcp-tools/apply`
- `POST /api/v1/setup/first-run/mcp-tools/validate`

The catalog response may also include current saved state so the frontend can resume accurately. If this grows too large, add `GET /api/v1/setup/first-run/mcp-tools/state`.

### Pack Catalog

The server-side catalog is static in v1 and includes:

- `pack_id`
- label
- purpose
- default selected flag
- included module/tool patterns
- risky add-ons
- sample validation candidates
- required module/server conditions
- catalog version

Unknown saved pack ids from older catalog versions should display as older setup choices instead of being silently dropped.

Initial v1 pack ids and defaults:

| Pack id | Label | Default | Server-native module targets | Initial read-only tool patterns |
| --- | --- | --- | --- | --- |
| `research` | Research | selected | `knowledge`, `media`, `prompts`, `mcp_discovery` | `knowledge.search*`, `knowledge.get*`, `media.search*`, `media.get*`, `prompts.list*`, `prompts.get*`, `mcp.discovery*` |
| `learning` | Learning | selected | `quizzes`, `flashcards`, `media` | `quizzes.list*`, `quizzes.get*`, `flashcards.list*`, `flashcards.get*`, `media.search*`, `media.get*` |
| `writing` | Writing | selected | `prompts`, `notes` | `prompts.list*`, `prompts.get*`, `notes.list*`, `notes.get*` |
| `media_library` | Media Library | selected | `media` | `media.list*`, `media.search*`, `media.get*` |
| `personal_knowledge` | Personal Knowledge | selected | `notes`, `prompts`, `knowledge` | `notes.list*`, `notes.get*`, `prompts.list*`, `prompts.get*`, `knowledge.search*`, `knowledge.get*` |

These patterns define the intended v1 behavior, not a permission bypass. Implementation planning must bind them to the actual registered tool names in the current registry and keep the registry risk metadata authoritative. If a target module lacks a matching read-only tool, the catalog should show the pack with fewer available tools instead of widening the policy.

Initial disabled add-ons:

| Add-on id | Label | Default | Requirement |
| --- | --- | --- | --- |
| `external_network_read` | External network reads | off | Explicit opt-in; only read-only tools may be added. |
| `local_file_read` | Local file reads | off | Explicit opt-in; explain server-local path semantics. |
| `workspace_write` | Writes and updates | off | Strong confirmation; generated policy must enumerate writable tools. |
| `destructive_actions` | Delete/destructive actions | off | Strong confirmation; never implied by pack selection. |
| `process_run_command` | Process or command execution | off | Strong confirmation; never implied by pack selection. |

`external_network_read` controls ongoing external tool availability in the generated profile. It does not control the setup-only external validation check described above.

For add-ons with strong confirmation, the apply endpoint must require the add-on id in both `selected_addon_ids` and `confirmed_addon_ids` for the current `confirmation_version`. If the confirmation is missing or stale, the server must reject the apply request and leave the existing profile unchanged.

### First-Run Default Profile

Saving packs upserts one MCP Hub permission profile:

- Display name: `First-run default`
- Stable marker: `metadata.setup_origin = "first_run_mcp_tools"`
- Stable setup id: `metadata.setup_instance_id`
- Generated policy provenance: selected pack ids, selected add-on ids, catalog version, generated policy hash, and last generated hash

The profile is assigned visibly through MCP Hub so users can inspect or replace it later.

Do not locate the profile by display name alone. Use the stable metadata marker and setup id.

### Manual Edit Protection

The apply endpoint must not blindly overwrite a profile that was manually edited in MCP Hub.

Store `last_generated_hash` for the generated policy. On re-apply:

- If the current generated section still matches `last_generated_hash`, update it.
- If it differs, return a conflict requiring the frontend to show `Keep existing` or `Replace generated profile`.
- `Keep existing` records the current profile and continues without overwriting.
- `Replace generated profile` explicitly overwrites the generated profile and stores a new hash.

The hash covers only the generated policy section plus selected pack ids, selected add-on ids, and catalog version. It must not cover user-owned profile metadata outside `metadata.first_run_mcp_tools`, display name changes, or MCP Hub annotations.

### Authorization And Setup Boundary

During unfinished first-run setup, MCP tools endpoints use the same local setup trust boundary as other first-run setup writes. After setup completion, the same operations must require normal MCP Hub admin/config permissions.

External validation does not imply authorization. Discovery and a safe sample run only prove that the server can respond. Enabled app/tool use still depends on the generated `First-run default` profile and selected add-ons.

## Sample Tool Contract

Built-in validation must use a deterministic read-only tool that works with empty user data. Media, notes, knowledge, or search tools should not be the default validation candidate because empty databases can look like failure.

If no existing registered tool is guaranteed to work, add a tiny diagnostic MCP tool such as `mcp.health` or `mcp.tools.list_sample`. It must:

- be read-only
- require no user content
- avoid external network calls
- avoid local filesystem access
- return stable success data

Built-in validation safety must come from internal registry metadata or the static setup catalog. Do not infer safety from a tool name alone.

External validation may run only a discovered tool that is clearly safe read-only. A tool is eligible only when all of these are true:

- The external server is already configured and enabled before first-run validation starts.
- Discovery has refreshed successfully for that server during the validation attempt.
- The tool has explicit trusted metadata from MCP tool annotations or the server registry, not from a name heuristic.
- The tool is marked read-only or `mutates_state = false`.
- The tool has a low-risk classification.
- The tool does not require filesystem, process, write, delete, or destructive privileges.
- The validation call can use a static safe example input or no input.

Missing, conflicting, or heuristic-only safety metadata makes the tool ineligible. If no eligible external tool exists, return `no_safe_external_tool`.

## Frontend UX

Add `McpToolsStep` under `apps/packages/ui/src/components/Option/Onboarding/steps/` and wire it into `UnifiedSetupWizard`.

The primary action sequence should be:

1. `Save packs`
2. `Run sample tool`
3. `Continue`

`Skip MCP tools` is always available.

The default screen should show the pack picker and validation panel. Add-ons stay collapsed inside each pack to avoid a dense settings wall.

Validation states need visible progress:

- saving
- discovering external tools
- running built-in sample
- running external sample
- passed
- failed
- skipped
- no safe external tool
- external discovery incomplete

`Open MCP Hub` should include `source=first-run` and preserve setup state. If possible, MCP Hub should provide a return affordance to setup.

## MCP Hub Follow-Up State

MCP Hub should surface first-run MCP setup status when relevant:

- `validated during setup`
- `not validated during setup`
- `validation failed`
- `external discovery incomplete`
- `profile manually changed`

For skipped or failed validation, MCP Hub should offer a recovery action to run validation or review the `First-run default` profile.

Existing completed installs should not unexpectedly see the first-run wizard again. They should use MCP Hub recovery and management entry points.

## Error Handling

- Pack catalog load failure: show a setup-recoverable error and keep `Skip MCP tools`.
- Apply failure: do not advance; show safe server detail and no stack traces, secrets, raw headers, raw env values, or local paths.
- Validation failure: preserve saved pack state, mark `failed`, allow retry or skip.
- External discovery timeout: mark `external_discovery_incomplete`, explain that discovery is not complete, allow retry or continue, and link to MCP Hub.
- Manual edit conflict: show `Keep existing` and `Replace generated profile`.
- Concurrent save/validation calls: use idempotent upsert behavior and avoid duplicate profiles or assignments.

All errors returned through setup endpoints must be redacted. Tests should cover secret-like tokens, headers, env names, and local path leakage.

## Testing

Backend unit tests:

- pack catalog shape and versioning
- default low-risk read-only selection
- local file read excluded by default
- risky add-ons excluded until opt-in
- generated policy from selected packs and add-ons
- stable profile upsert by metadata marker
- duplicate save and retry idempotency
- manual edit hash conflict and replace behavior
- validation state transitions
- redaction for validation/config errors

Backend integration tests:

- `GET /api/v1/setup/first-run/mcp-tools/catalog`
- `POST /api/v1/setup/first-run/mcp-tools/apply`
- `POST /api/v1/setup/first-run/mcp-tools/validate`
- setup completion accepts `not_run`, skipped, failed, no-safe-tool, and discovery-incomplete validation states
- existing external server discovery passing with no safe validation tool returns `no_safe_external_tool`

Frontend tests:

- default pack selection
- add-on opt-in and high-risk confirmation
- `Save packs` before `Run sample tool`
- validation state rendering
- skip path
- no external servers summary
- external server discovered but no safe tool state
- manual edit conflict actions
- MCP Hub link with `source=first-run`

MCP Hub tests:

- skipped or failed first-run validation shows recovery guidance
- `First-run default` profile is visible and inspectable

Verification:

- Run focused backend tests for setup and MCP Hub touched code.
- Run focused frontend tests for onboarding and MCP Hub touched code.
- Run Bandit on touched backend setup/MCP code.

## Acceptance Criteria

- A fresh single-user setup can skip MCP tools and still complete setup.
- A fresh single-user setup can save default read-only packs and run built-in validation without user data.
- Local file read is not selected by default.
- External servers, when already configured, can be discovered during setup.
- Discovery passing with no safe external validation tool is degraded but not failed.
- The user sees enabled packs and available tools at the end of the step.
- The generated profile and assignment are visible in MCP Hub.
- Re-running setup does not create duplicate profiles or assignments.
- Manual MCP Hub edits are not overwritten without explicit confirmation.
- Endpoint responses do not leak secrets, raw auth material, stack traces, or local paths.

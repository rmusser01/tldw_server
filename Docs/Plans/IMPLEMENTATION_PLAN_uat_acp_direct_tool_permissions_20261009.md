# C05 direct ACP runner tool permission gates

Associated task: TASK-13260.278.18.83.45 (root-owned; direct-tool note recorded and read back before edits).

## Scope and ADR assessment

Scope is `tools/tldw-agent/internal/acp/runner.go`, one new direct-tool wire/handler test module, and this plan. Reuse the existing permission request contract, bound runner session, workspace guards and terminal allowlist. Gate direct `fs/write_text_file` and `terminal/create` according to existing configured approval flags before mutation/process creation. Only an explicit nested `outcome.outcome == "approved"` permits the operation; unavailable, denied, cancelled, error or malformed responses refuse. Approval-disabled behavior remains unchanged.

ADR required: no. ADR-016 governs unchanged shared ACP permission/session persistence; ADR-020 governs unchanged owner-bound access. This repair enforces the existing permission authority contract and introduces no new policy, persistence or public route. Existing backend runtime/governance decisions remain authoritative. No Conn or Python dispatch edits are authorized here; normal prompt permission/Stop responsiveness is a separately coordinated unit and remains unqualified.

## Stage 1: Source contract and causal controls
**Goal**: Reproduce direct tool bypass with real runner handlers and JSON-RPC wire permission responses.
**Success Criteria**: Refused direct writes preserve existing file bytes; refused terminal creation starts no terminal; exact bound session/method/arguments reach existing permission consumer; invalid requests and approval-disabled behavior remain fenced.
**Tests**: New `runner_direct_permissions_test.go` using real Conn/net.Pipe, temp workspace/FSTools/TerminalManager and a harmless configured echo command. No downstream agent/model/provider.
**Status**: Complete

## Stage 2: Minimal direct gates
**Goal**: Reuse `handlePermissionRequest` in one private strict approval helper and both handlers.
**Success Criteria**: Session/absolute/write-workspace guards precede permission. Existing terminal enabled/allowlist/meta/cwd checks remain in unchanged Create. Strict response classification precedes effects; existing no-approval configuration works.
**Tests**: Causal refusal/approval, unavailable/error/malformed response matrix and existing handler/terminal/workspace controls.
**Status**: Complete

## Stage 3: Qualification and freeze
**Goal**: Run bounded affected Go checks, format/vet/security delta and independent source review.
**Success Criteria**: At most three distinct attempts; outer600/default-case cap300; actual installed Go/dependencies with no installation; touched source/test format clean, no new vet findings, exact frozen hashes. Root owns build and live workflow acceptance.
**Tests**: New direct-tool controls plus genuinely affected existing ACP and workspace/FSTools tests. Bandit N/A for Go; report installed Go security-tool availability truthfully.
**Status**: In Progress

## Evidence ledger

- Source read: direct write/terminal handlers never consulted the configured approval flags. Existing permission helper forwards to the current upstream and returns cancelled for unavailable/error. Existing Python consumer requires `sessionId`, `tool.name`, `tool.input` and returns nested `outcome.outcome`. Exact pre-unit runner bytes retained in functions memory only.
- Actual workspace/path mutation and terminal process guards remain in existing implementations; direct wire tests do not qualify normal nested session/prompt reader progress or live permission UI.
- Baseline normal Go vet passed, natural0/1.489 s. Initial sandbox default-cache access and gofmt temporary-file denials occurred before compilation/formatting; exact normal checks were rerun with authorized scoped escalation, no cache/environment policy or dependency change.
- Attempt1 selected two denial controls: terminal is a valid causal red (real terminal created without any permission request). Write fixture targeted a missing parent, which existing ResolvePath refuses; only permission-wire absence failed, so that write result is excluded from mutation causal evidence. Before production edits: correct the fixture to an existing owned file with retained original bytes, keep wire/decision assertions, and repeat the two causal IDs. No production change yet. Maximum three attempts remains.
- Attempt2: corrected write fixture now causally overwrote the original owned file without approval, natural1/0.521 s outer; terminal red was also repeated in the already-started command before root's corrected-write-only direction arrived. Retain original terminal red, exclude malformed write result, and do not replay either unchanged red again. Production repair starts only after this valid overwrite evidence; attempt3 is full affected green.
- Root separately recorded/authorized connected responsive prompt/control scope before edits: runner-only tracked asynchronous session/prompt and current-operation permission cancellation; scoped Python permission-request tasks, one active prompt with operation-owned futures, authenticated same-WebSocket prompt task. Generic Conn remains untouched. Record connected causal evidence separately before that production work; direct gate results alone do not qualify normal prompt approval/Stop.
- Attempt3 direct affected qualification passed naturally0/0.680 s outer (0.371 s ACP):56 leaf cases including29 new direct controls and27 existing ACP controls. FSTools/workspace packages have no test files; they are not configured test skips. Normal vet0 and format clean. Gosec is not installed; no installation or substituted qualification; Bandit is N/A for Go.
- Connected final runner source supersedes the direct-only freeze through captured operation/context fences; direct gate/classifier/wire bodies remain unchanged. Final affected Go scope58 ACP leaf passes/vet0 retained, including2 connected wire cases; no later unchanged Go replay. Complete connected plan and split Python/timeout evidence: `IMPLEMENTATION_PLAN_uat_acp_prompt_permission_control_20261009.md`.
- Current direct source/test byte freeze sent for independent review; root owns combined build and actual controlled live workflow acceptance. No real provider/model/browser permission or already-created terminal Stop acceptance is claimed.
- Independent prompt_cookie_sync_fix full connected/direct source and new body review COMPLETE/CLEAR; final narrow lifetime/instant cancellation corrections reviewed separately without replay. Root combined build/live acceptance remains open; direct test and gate bytes stay unchanged in the connected freeze.

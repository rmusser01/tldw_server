# Persona Tool Administration PRD

Status: Draft

Owner: Persona module / MCP Hub / AuthNZ

Tracking: #1922, split from #1902

Backlog: TASK-472

## Summary

Define the future Persona Tool Administration product contract: how Persona users and administrators discover, configure, grant, restrict, audit, revoke, and troubleshoot tools available to Personas without creating a second permission system.

The current Persona module already supports Persona-local scope and policy editing, minimal MCP/tool discovery, runtime policy evaluation, confirmation requirements, and live tool-plan denial messaging. MCP Hub already owns broader tool governance concepts: external server registry, tool catalogs, permission profiles, effective policy previews, approval policies, credential bindings, governance packs, tool registry metadata, and audit events. Persona Tool Administration should compose those systems into a Persona-aware administration layer rather than duplicating MCP Hub or weakening runtime enforcement.

This PRD does not make full tool administration a current Persona module completion blocker. It documents the future slice moved out of `Docs/Product/Persona_Agent_Design.md`.

## Problem

Today, Persona Garden intentionally stops at minimal Persona-local discovery and policy/scopes editing. That is enough for configuring already-authorized tools, but it is not enough for full lifecycle administration:

- installing or registering external MCP servers,
- mapping broad capabilities to concrete tools,
- assigning reusable permission profiles to Personas or contexts,
- managing credentials and credential-slot bindings,
- requiring runtime approvals for risky tools,
- auditing tool availability and blocked states,
- revoking access cleanly across Persona sessions.

If this work is added directly to Persona Garden, it risks creating split-brain governance where Persona policy rules and MCP Hub effective policy both claim to authorize tool execution. The future contract needs a clear boundary: MCP Hub remains the canonical control plane; Persona Tool Administration provides Persona-specific assignment, preview, and runtime explanation surfaces.

## Goals

- Define Persona-facing tool administration as a composition layer over MCP Hub and MCP Unified.
- Keep MCP Hub as the canonical owner of global/admin tool lifecycle, credentials, governance packs, and effective policy.
- Preserve Persona scope/policy rules as Persona-local restrictions, not broad grant authority.
- Let administrators assign tool catalogs, permission profiles, approval policies, and credential bindings to Persona contexts.
- Let Persona owners preview effective tool access before live use.
- Keep runtime execution gated by AuthNZ, MCP Unified, deployment policy, MCP Hub effective policy, Persona scopes/policies, and session confirmation.
- Provide trace-safe blocked/unavailable explanations without leaking hidden/admin-only tools or secrets.
- Define audit, revocation, and health-check behavior for Persona tool access.
- Keep first implementation backend/contract-oriented.

## Non-goals

- No Buddy animation, Buddy runtime, or visual-pack work.
- No design-system backlog work.
- No implementation in this PRD slice.
- No standalone marketplace outside MCP Hub.
- No client-side or Persona-only credential storage.
- No Persona policy rule that grants capabilities the authenticated user, server, deployment, or MCP Hub policy does not grant.
- No silent installation or activation of external MCP servers from Persona chat.
- No multi-agent coordination workflow; that belongs to the future Persona Collaboration PRD.
- No scheduled autonomous work beyond policy and approval integration points already covered by `Persona_Scheduled_Work_PRD.md`.

## Current Contract Evidence

- `Docs/Product/Persona_Agent_Design.md` keeps Scopes/Policies editing and minimal MCP/tool capability discovery in current scope, while moving marketplace-style tool administration, global configuration, and admin lifecycle to this future PRD.
- `tldw_Server_API/app/core/Persona/policy_evaluator.py` implements deny-by-default Persona policy semantics, bounded wildcard matching, explicit deny precedence, scope requirements, export/delete gating, session rules, and skill rules.
- `tldw_Server_API/app/api/v1/endpoints/persona.py` exposes Persona scope and policy rule list/replace endpoints and applies policy evaluation during live websocket tool planning and execution.
- `tldw_Server_API/app/api/v1/schemas/persona.py` defines `PersonaScopeRule` and `PersonaPolicyRule` as Persona-local rule shapes.
- `tldw_Server_API/app/core/Persona/README.md` documents Persona policy/scope as authoritative over tools and capabilities inside Persona live sessions.
- `tldw_Server_API/app/core/MCP_unified/README.md` documents MCP Unified as the secure MCP server with HTTP/WebSocket transport, AuthNZ/MCP JWT, RBAC, rate limits, module registry, status, metrics, and tool execution.
- `Docs/MCP/mcp_hub_management.md` defines MCP Hub as the shared management surface for external MCP servers, secrets, and tool catalog management, with mutation endpoints requiring admin role, `system.configure`, or wildcard permission.
- `Docs/MCP/mcp_tool_catalogs.md` defines named MCP tool catalogs that shape discovery while RBAC still gates visibility and execution.
- `Docs/Plans/2026-03-09-mcp-hub-tool-permissions-design.md` defines MCP Hub as the canonical control plane for permission profiles, assignments, overrides, runtime approval rules, credential bindings, and effective policy resolution.
- `Docs/Plans/2026-03-10-mcp-hub-credential-slot-bindings-design.md` and `Docs/Plans/2026-03-10-mcp-hub-external-slot-runtime-approval-design.md` define credential-slot granularity and approval as confirmation, not temporary broadening.
- `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py` exposes tool registry, capability mappings, permission profiles, policy assignments, approval policies/decisions, credential bindings, governance packs, effective policy, and audit-oriented helpers.

## Product Shape

Persona Tool Administration should introduce a Persona-facing effective tool access contract:

```json
{
  "persona_id": "persona-id",
  "subject": {
    "scope_type": "persona",
    "scope_id": "persona-id"
  },
  "assigned_profiles": ["research-readonly"],
  "assigned_catalogs": ["research-kit"],
  "effective_tools": [
    {
      "tool_name": "knowledge.search",
      "module_id": "knowledge",
      "status": "available",
      "capabilities": ["tool.invoke"],
      "risk_class": "read",
      "approval": { "required": false },
      "credential_status": "not_required",
      "persona_policy": { "allowed": true, "requires_confirmation": false }
    }
  ],
  "blocked_tools": [
    {
      "tool_name": "media.delete",
      "status": "blocked",
      "reason_code": "PERSONA_POLICY_EXPLICIT_DENY",
      "safe_reason": "Blocked by this Persona policy."
    }
  ]
}
```

The exact schema can change during implementation, but V1 must preserve these concepts: Persona subject, MCP Hub assignments, effective tools, blocked tools, reason codes, approval status, credential status, and Persona-local policy contribution.

## Authority Model

Tool execution authority should be layered:

1. Deployment configuration and feature flags decide what MCP/HUB features exist.
2. AuthNZ decides who the current user is and which global permissions they hold.
3. MCP Unified RBAC, rate limits, module registry, and server security decide whether a tool can be listed or executed.
4. MCP Hub effective policy decides which catalogs, profiles, credentials, approvals, and governance rules apply to the current Persona context.
5. Persona scopes decide what resource contexts the Persona can use.
6. Persona policies can further restrict tools, require confirmation, or deny actions.
7. Session policy and runtime approvals can add narrower, time-bounded confirmation.

No lower layer may broaden access beyond an upper layer. Persona policy is allowed to narrow or require confirmation. It is not allowed to install tools, grant credentials, or bypass MCP Hub effective policy.

## Persona And MCP Hub Boundary

MCP Hub owns:

- external MCP server registration and lifecycle,
- secret and credential-slot storage,
- tool registry metadata,
- tool catalogs,
- capability mappings,
- permission profiles,
- policy assignments and overrides,
- approval policies and decisions,
- governance packs and governance audit findings,
- global/org/team/user administration permissions.

Persona owns:

- Persona identity and profile references,
- Persona-local scope rules,
- Persona-local policy restrictions,
- live-session policy evaluation and tool-plan messaging,
- Persona-specific effective-access preview,
- mapping Persona context into MCP Hub policy assignment subjects,
- trace-safe Persona runtime explanations.

The Persona UI may deep-link into MCP Hub for admin tasks, but it should not reimplement MCP Hub configuration forms unless a future implementation explicitly shares the same backend contracts.

## Assignment Model

Preferred V1 assignment subjects:

- `persona:<persona_id>` for one Persona.
- `workspace:<workspace_id>:persona_default` for Workspace Persona defaults after that PRD is implemented.
- `schedule:<schedule_id>` for scheduled Persona work after that PRD is implemented.
- `user:<user_id>` as inherited MCP Hub policy, not Persona-specific storage.

Persona-specific assignments should be reference-backed. They should store IDs of MCP Hub permission profiles, tool catalogs, approval policies, or policy assignments, not snapshots of tool names, Persona names, credentials, or profile documents.

## Tool Lifecycle

V1 should model lifecycle states rather than treating tools as simply present or absent:

- `not_installed`: no MCP server/module provides the tool.
- `installed_disabled`: server or deployment has disabled the provider/module.
- `discoverable`: visible in registry/catalog but not executable for this Persona.
- `available`: executable subject to policy and approvals.
- `requires_configuration`: credential slots, path scopes, or server config are incomplete.
- `requires_approval`: runtime approval is required before execution.
- `blocked_by_policy`: denied by MCP Hub, Persona policy, session policy, or deployment rule.
- `revoked`: access previously existed but is no longer valid.
- `unhealthy`: server/module is configured but health checks fail.

Blocked and unavailable messages must be truth-preserving and safe. If a user lacks permission to know a hidden/admin-only tool exists, Persona surfaces should not leak its name.

## Runtime Approval

Runtime approval is confirmation, not temporary widening:

- Approval can only apply to tools already allowed by effective policy.
- Missing credential binding or missing secret material is a hard deny, not an approval prompt.
- Approval identity should include tool name, server ID, credential slot set when relevant, Persona/session context, and approval duration.
- Persona runtime approval cards should display reason, scope, risk, and expiry without exposing secrets.
- Persona policy `require_confirmation` and MCP Hub approval policy should combine conservatively: if either requires approval, approval is required.

## Credentials And Secrets

Persona configuration must never store raw credentials. Credential management remains in MCP Hub:

- External server credentials are write-only and brokered at execution time.
- Persona assignments may reference credential bindings through MCP Hub policy.
- Persona effective-access previews may show credential status such as `not_required`, `configured`, `missing_binding`, or `missing_secret`.
- Secret plaintext, headers, environment variables, and credential refs must not appear in Persona state docs, memory, transcripts, or diagnostics.

## Audit And Revocation

Persona Tool Administration should emit or reference audit events for:

- Persona tool assignment creation, update, and deletion.
- Persona policy changes that affect tool access.
- Effective-access preview requests when they include admin-only context.
- Runtime approvals and denials.
- Credential-binding availability changes that affect a Persona.
- Revocation and session invalidation when access changes.

Revocation requirements:

- Removing a profile/catalog/credential binding should affect new tool calls immediately.
- Active sessions should re-check effective policy before every tool execution.
- Long-running tool calls should record the policy decision used at start and fail closed before follow-up mutations if policy changed.
- Persona live UI should receive a safe notice when access is revoked mid-session.

## API Direction

Preferred V1 contract additions:

- `GET /api/v1/persona/profiles/{persona_id}/tool-access`
  - Effective Persona tool access preview.
  - Includes available, blocked, unavailable, approval, credential, and policy contribution summaries.
- `PUT /api/v1/persona/profiles/{persona_id}/tool-assignments`
  - Reference-backed assignment to MCP Hub profiles/catalogs/approval policies where the acting user has grant authority.
- `POST /api/v1/persona/profiles/{persona_id}/tool-access/preview`
  - Dry-run proposed assignment or policy changes without persisting.
- `GET /api/v1/persona/profiles/{persona_id}/tool-audit`
  - Persona-filtered audit feed or linkable audit query into MCP Hub.

Implementation can also choose MCP Hub-native endpoints with `subject_type=persona` and keep Persona endpoints read-only wrappers. The key requirement is one backend resolver, not two different effective-policy calculators.

## Data Model Direction

Prefer reference-backed Persona assignments in existing MCP Hub policy tables:

- subject type: `persona`
- subject ID: Persona ID
- references to permission profile IDs, catalog IDs, approval policy IDs, credential bindings, and governance packs
- no snapshots of Persona profile content, tool display names, or secrets
- enough audit metadata to explain who assigned what and when

If a small Persona-side table is needed for UI preferences, it should only store Persona UI defaults such as preferred catalog view or pinned setup warnings. It must not become an authorization source.

## UI Direction

This PRD is backend/contract-first. Future UI can add:

- Persona-local "Tool access" summary in Persona Garden.
- Effective access preview with available, approval-required, blocked, unhealthy, and not-configured groups.
- Safe deep links to MCP Hub setup for server install, credentials, catalogs, and governance findings.
- Dry-run before saving assignments.
- Clear warning when a Persona policy narrows MCP Hub access.
- No admin-only tool names for users without visibility.

Persona Garden should remain a Persona configuration surface. MCP Hub remains the advanced governance/admin surface.

## Staged Delivery

### Stage 1: Contract Audit And Resolver Design

Goal: define the Persona subject model and effective tool access resolver.

Deliverables:

- Persona subject mapping for MCP Hub policy assignments.
- Effective access response schema.
- Permission layering and precedence rules.
- Safe blocked/unavailable reason taxonomy.
- Decision on Persona endpoint wrapper versus MCP Hub-native subject endpoints.

### Stage 2: Backend Preview API

Goal: let Persona owners preview effective tool access without changing assignments.

Deliverables:

- Read-only effective access endpoint.
- Integration with MCP Hub tool registry, catalogs, profiles, approval policies, and credential status.
- Tests for hidden/admin-only tool redaction.
- Tests for Persona policy narrowing.

### Stage 3: Reference-Backed Assignments

Goal: assign MCP Hub policies/catalogs to Persona contexts safely.

Deliverables:

- Assignment create/update/delete flow with grant-authority checks.
- Dry-run change preview.
- Audit events.
- Tests for unauthorized broadening, stale references, deleted Personas, and deleted profiles/catalogs.

### Stage 4: Runtime Enforcement Integration

Goal: ensure Persona live tool execution uses MCP Hub effective policy and Persona policy together.

Deliverables:

- Shared runtime effective-policy resolver.
- Re-check before every tool execution.
- Runtime approval identity including Persona/session context.
- Safe notices for revoked or newly blocked access.
- Tests for missing credential binding, missing secret, approval required, and revocation.

### Stage 5: Admin Health And Troubleshooting

Goal: make tool access failures explainable and recoverable.

Deliverables:

- Persona-filtered audit feed or MCP Hub audit deep links.
- Health/status summaries for configured Persona tool sources.
- Governance finding surfacing.
- Revocation and stale-assignment diagnostics.

## Validation Plan

- Unit tests for effective access precedence and blocked reason taxonomy.
- API tests for preview, assignment dry-run, grant-authority checks, and reference-backed persistence.
- Integration tests with MCP Hub tool catalogs, permission profiles, approval policies, credential bindings, and tool registry.
- Persona runtime tests proving MCP Hub denial, Persona policy denial, session policy denial, and approval requirements all fail closed.
- Redaction tests proving hidden/admin-only tools and secrets do not appear in user-visible blocked reasons.
- Audit tests for assignment changes, approval decisions, denial, and revocation.
- Bandit on touched backend implementation paths when code is added.

## Risks And Mitigations

- Risk: Persona becomes a second tool governance plane.
  Mitigation: MCP Hub owns global/admin lifecycle and effective policy; Persona references and previews it.

- Risk: Persona policy accidentally grants broader access.
  Mitigation: Persona policy can only narrow or require confirmation; grant authority stays in MCP Hub/AuthNZ.

- Risk: blocked reasons leak hidden tools.
  Mitigation: reason taxonomy separates visible blocked tools from hidden tools and returns generic counts where needed.

- Risk: approval prompts become temporary elevation.
  Mitigation: approvals only confirm already-authorized, already-configured access.

- Risk: active sessions keep stale access after revocation.
  Mitigation: re-check effective policy before each tool execution and emit safe runtime notices.

- Risk: credentials leak into Persona state, memory, or logs.
  Mitigation: brokered credentials remain execution-time only, and Persona stores only status/provenance metadata.

## Acceptance Criteria

- Persona Tool Administration uses MCP Hub as the canonical governance source.
- Persona-specific assignments are reference-backed and do not snapshot tool/admin/credential content.
- Persona policy remains a narrowing layer and cannot broaden access.
- Effective access preview reports available, blocked, unavailable, approval, credential, and health states safely.
- Runtime tool execution re-checks MCP Hub effective policy and Persona policy before every tool call.
- Approval is confirmation only and cannot fix missing credentials or missing grants.
- Audit and revocation behavior are defined and tested.
- Hidden/admin-only tools and secrets are not leaked through Persona surfaces.

## Open Questions

- Should V1 expose Persona assignment writes through Persona endpoints, or should Persona UI call MCP Hub subject-assignment endpoints directly?
- Should Persona Tool Administration support only profile/catalog assignment in V1, deferring credential binding selection to MCP Hub?
- What default tool catalog should a newly created Persona inherit, if any?
- Should Persona-specific effective access previews include hidden-tool counts for admins only?
- How should ordinary Persona-backed chat display tool-access drift if the underlying Persona assignment changes while a chat session is active?

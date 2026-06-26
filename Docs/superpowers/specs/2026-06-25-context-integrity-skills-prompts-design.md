# Context Integrity For Skills And Prompts Design

**Date:** 2026-06-25
**Status:** Draft for review
**Backlog Task:** TASK-12015
**Scope:** Skills, bundled/plugin skills, prompt-bearing files, DB-backed prompt versions, MCP prompt exposure, prompt loaders, startup warnings, admin review, and trust manifest operations.

## Summary

This design adds a shared Context Integrity subsystem for prompt-bearing assets that can influence model context or tool-capable behavior. The first protected assets are:

- user-managed skills under per-user `skills/` directories
- bundled, plugin, and repo skill files
- config prompt YAML/Markdown files
- repo prompt-skill files such as `Docs/Prompts/Skills/*`
- DB-backed prompt records and specific prompt versions that can be injected into model context
- persona, character, voice-command, profile-default, and other prompt sources when they can enter model context or tool instructions

The subsystem compares current asset digests against an approved signed manifest. The server continues booting when unexpected changes are detected, but affected assets are quarantined and cannot be injected, exposed through MCP, or executed until an operator approves a new manifest version.

The default policy is: **server boots, context use fails closed**.

## Threat Model

The default design assumes an attacker may be able to modify skill files, prompt files, and app databases while the server is offline. The attacker cannot access an external/admin-held trust manifest or an OS/hardware-backed signing key.

This design does not claim to detect full filesystem compromise when the only trust anchor is stored on the same compromised filesystem. Hardened deployments require an external/admin-held manifest, an external key, or OS/hardware-backed key storage that survives app-data tampering.

## Goals

1. Detect offline and live tampering of prompt-bearing assets.
2. Prevent changed or unapproved assets from influencing model context or tools.
3. Cover both user-managed and bundled/plugin/repo skill and prompt assets.
4. Support explicit operator approval for new baselines.
5. Reuse the existing startup warning framework for boot-time findings.
6. Provide a hardened mode for deployments that require an external/admin-held trust anchor.

## Non-Goals

1. Proving system integrity after full host compromise without an external trust anchor.
2. Blocking server startup for every changed optional skill or prompt.
3. Replacing package manager or plugin signature verification.
4. Building a general-purpose file integrity monitoring system for all repository files.
5. Letting model-assisted workflows approve their own prompt-bearing assets.

## Architecture

The protected unit is a prompt-bearing asset: any file or DB version that can be injected into model context, alter prompt assembly, appear through MCP prompt discovery, or drive skill execution.

The inclusion rule is capability-based: if an asset can affect system/user/developer instructions, prompt assembly, tool instructions, MCP prompt discovery, or skill execution, it must be treated as protected once that source is wired into Context Integrity. The first implementation can stage adapters, but new prompt-bearing surfaces should default to protected rather than relying on per-feature opt-in.

The subsystem stores approved asset state in a signed trust manifest. The manifest records canonical hashes, asset IDs, source type, owner scope, trust tier, signer/key ID, approval event, manifest schema version, and manifest sequence number. Runtime code compares current assets against the latest approved manifest. It never treats the live filesystem or app DB as automatically trusted.

Manifest signatures protect manifest contents, but they do not by themselves prevent rollback to an older valid manifest. The latest accepted manifest sequence number and digest must be anchored outside mutable app DB state. Acceptable anchors include OS/hardware-backed key storage metadata, an external/admin-held manifest ledger, or hardened-mode external verification. If no anti-rollback anchor is available, the system must report degraded integrity and avoid claiming rollback protection.

The default trust anchor is an OS/hardware-backed signing key. If that is unavailable, the server must either use an admin-held manifest or run in clearly labeled degraded integrity mode. Hardened deployments require an external/admin-held manifest or external key. Full filesystem compromise is only detectable with that external trust anchor.

Startup integrity verification runs before any protected asset can be injected, executed, exposed through MCP, or included in skill discovery. Unexpected additions, deletions, or modifications create startup warning records and quarantine affected assets. The server continues booting, but quarantined assets are excluded from all context-injection and execution paths until reviewed through a static, prompt-independent admin workflow.

## Components

### Asset Inventory Adapters

Adapters emit normalized `ContextAssetDescriptor` records with stable IDs, source metadata, and canonical bytes for hashing.

- `SkillsFilesystemAdapter`: user skill directories plus bundled/plugin/repo skill directories.
- `PromptFilesAdapter`: config prompt YAML/Markdown and repo prompt-skill files.
- `PromptDatabaseAdapter`: DB-backed prompt records and immutable prompt versions.

Adapters should be narrow and read-only. They should parse only enough metadata to identify and hash an asset. Trust decisions belong to the verifier and resolver.

### Canonicalization And Hashing

The shared hasher produces `sha256` over deterministic canonical payloads.

Filesystem assets include:

- source type
- normalized asset ID
- normalized relative path
- file bytes
- support-file bytes when a skill is directory-backed
- metadata that affects model context or execution behavior

Filesystem hashing defaults to raw file bytes, sorted by normalized POSIX-style relative path for directory-backed assets. This intentionally detects comment-only and formatting-only edits in protected files. Any source-specific semantic hash may be added for review display, but enforcement must use the raw canonical asset digest unless a source explicitly defines a safer canonical form.

DB prompt assets use stable JSON over fields that affect context, such as prompt UUID, prompt version, name, system content, user content, structured fields, model/tool-affecting metadata, and deleted state when it affects availability. DB canonical JSON must use deterministic field ordering, Unicode NFC normalization for text fields, and LF line endings for text values before hashing. The hash approves a specific version/content digest, not all future edits to a prompt UUID.

YAML/Markdown prompt-file review may show parsed semantic diffs, but enforcement still hashes the canonical file payload. This avoids a parser bug or comment channel becoming a place to hide unapproved instructions.

### Trust Manifest Store

The manifest store owns approved manifest versions, signatures, key IDs, approval metadata, and import/export.

Required behavior:

- verify manifest signatures before using manifest contents
- select the newest valid manifest
- fall back to the newest prior valid manifest if the latest manifest signature is invalid
- reject rollback to an older valid manifest when the external anti-rollback anchor says a newer accepted manifest exists
- export admin-held manifests for hardened deployments
- import external/admin-held manifests
- enter degraded integrity mode when no strong trust anchor is available

### Integrity Verifier And Runtime Resolver

The verifier compares current inventory to approved manifest entries and emits asset states:

- `trusted`
- `changed_approved_executable`
- `changed_approved_non_executable`
- `new_unapproved`
- `missing_required`
- `missing_optional`
- `signature_invalid`
- `verification_error`
- `degraded_integrity`
- `quarantined`

The runtime resolver is the centralized enforcement point. Skills, prompt loaders, MCP prompt catalog code, DB prompt access, and chat slash skill invocation must ask the resolver before returning or using protected content.

The resolver derives enforcement from verified in-memory boot state, a boot-verified immutable snapshot, or signed state. It must not trust mutable app DB quarantine flags as the source of enforcement truth; DB flags are operational metadata only.

The resolver either verifies the current digest at use time or serves content from a boot-verified immutable snapshot. This avoids trusting a startup-only scan after files change while the server is running. At-use verification must hash the exact bytes or DB row version that will be injected or executed. Filesystem at-use mode therefore needs single-read semantics, such as reading bytes once and passing those bytes onward after digest verification. DB at-use mode needs transaction or version discipline so the verified row version is the row version used. Implementations that cannot guarantee this should use immutable snapshots.

### Admin And Audit Surfaces

Boot-time findings reuse `StartupWarningRecord` with component names such as:

- `context_integrity.skills`
- `context_integrity.prompt_files`
- `context_integrity.prompt_db`
- `context_integrity.manifest`

Durable audit records cover verification findings, quarantine events, approval, rejection, manifest import/export, degraded-mode entry, and break-glass configuration. Material audit events should be signed or hash-chained. Local audit rows are operational evidence, not proof against an attacker who can modify app DBs; hardened deployments should export audit events to an external sink.

Review UI and API responses must not render untrusted prompt content as rich HTML or feed it into model-assisted review. The review path uses static server/UI strings and escaped canonical diffs.

## Data Flow

### Startup

1. Build current inventory from filesystem skills, prompt files, and DB prompt versions.
2. Load and verify the latest approved signed manifest before trusting any manifest content.
3. Validate the manifest sequence and digest against the external anti-rollback anchor when one is configured.
4. Fall back to the newest prior valid manifest if the newest manifest is invalid and rollback policy permits that fallback.
5. Compare current assets to approved digests.
6. Record current-boot verification state and signed/hash-chained audit events for material findings.
7. Register startup warnings for changed, missing, unapproved, signature-invalid, rollback, or degraded assets.
8. Quarantine unsafe assets.
9. Continue booting with unsafe assets unavailable.

### Runtime Use

Every context-injection or execution path asks the integrity resolver before using an asset.

- `trusted` assets proceed.
- quarantined, signature-invalid, changed, missing, and unapproved assets are hidden from discovery and blocked from injection/execution.
- degraded mode defaults to blocking executable and context-injection use.
- unsafe read-only visibility requires an explicit break-glass config and must never allow injection or execution.
- at-use verification must pass through the verified content bytes or verified DB row version, not re-open or re-fetch content after the check.

### Edit And Import

Saving creates a draft or pending asset version. The new version is visible as pending, but not eligible for model context, MCP prompt exposure, or skill execution.

Operator review uses escaped canonical diffs. Approval re-checks the current digest and manifest version, signs a new manifest version, and records a signed audit event. If the asset digest changed between review and approval, approval fails with `asset changed during review`.

In single-user mode, the local operator is the admin, but approval still requires explicit non-model confirmation.

### Upgrade

Bundled and plugin assets are matched to package identity and update metadata. Verified trusted updates can use a lower-friction update flow only when:

- the previous package identity was already trusted
- the update source signature validates
- the updated asset digest matches signed package metadata

Unverified changed bundled assets are treated as possible tampering and quarantined.

### Recovery

Operators can:

- restore from the last approved manifest
- approve a new baseline after reviewing diffs
- import an external/admin-held manifest
- rotate the OS/hardware-backed key
- keep assets quarantined

Full compromise recovery guidance must direct operators to external manifest verification, key rotation, and restoring from known-good backups.

## Policy

Default policy is **server boots, context use fails closed**. Suspicious assets cannot influence model context, MCP prompt exposure, or tool-capable skill execution.

Actions:

- `signature_invalid`: try the newest prior valid manifest. If none exists, quarantine the protected scope and enter degraded integrity. Requires key or manifest recovery.
- `manifest_rollback_detected`: high severity, quarantine the protected scope unless an operator imports a valid external manifest or resolves the anchor mismatch. A locally valid older signature is not sufficient.
- `changed_approved_executable`: high severity, quarantine immediately.
- `changed_approved_non_executable`: medium or high severity based on source tier, quarantine for injection.
- `new_unapproved`: pending review, not usable for injection or execution.
- `missing_required`: high severity only for required assets.
- `missing_optional`: warn and keep unavailable.
- `degraded_integrity`: no injection or execution. Admin review may show escaped metadata and canonical diffs only.
- `verification_error`: quarantine the narrowest source scope that cannot be verified. If the scope cannot be determined, quarantine all protected assets.

Runtime errors are stable and content-free:

- `Asset is quarantined pending admin review.`
- `Integrity manifest signature is invalid.`
- `Integrity manifest rollback detected.`
- `Asset version is not approved for execution.`
- `Asset changed during review; reload before approving.`

Break-glass configuration must be intentionally named and noisy, for example `CONTEXT_INTEGRITY_UNSAFE_ALLOW_DEGRADED_READONLY=true`. It must log and audit on startup. It must never allow injection or execution in degraded mode.

## Initial Enrollment And Migration

Initial enrollment is the riskiest moment. The server must not silently bless all current files and DB prompt rows.

First run produces a source-grouped baseline report:

- package-signed bundled/plugin assets may be pre-trusted when package identity and signature metadata validate
- user assets require explicit local operator approval or imported manifest
- unknown external files remain pending
- DB prompt versions require explicit approval unless imported from a trusted manifest

Migration can run in `audit_only` mode, but the UI/API must label it as non-enforcing. `audit_only` is for diagnostics and rollout only, not the recommended steady state.

## Rollout Modes

### audit_only

Verification runs and findings are audited, but assets are not quarantined by default. UI and admin API status must say that integrity findings are not enforced.

### enforce

Default mode. Changed or unapproved context-injection assets are quarantined while the server continues booting.

### hardened

Requires an external/admin-held manifest or external key. Local-only fallback is not allowed. If no valid external trust anchor is available, protected assets are unavailable for injection and execution.

## Testing

Backend unit tests should cover:

- deterministic canonical hashing for filesystem and DB prompt-version assets
- manifest signing and signature verification
- manifest anti-rollback anchor verification
- invalid-latest-manifest fallback to prior valid manifest
- rollback detection when a valid older manifest conflicts with the anti-rollback anchor
- changed, missing, new, restored, degraded, and verification-error outcomes
- approval race locking on manifest version and current digest
- runtime resolver enforcement from verified in-memory or signed state, not mutable DB flags
- at-use verification using the exact bytes or row version consumed by the caller
- degraded mode blocking injection and execution
- tamper-evident audit chain verification

Integration tests should cover:

- startup detection after modifying an approved `SKILL.md`
- live runtime detection after modifying a skill while the server is running
- config prompt file modification blocked through `prompt_loader`
- DB prompt-version modification blocked through prompt APIs and MCP prompt catalog
- chat slash skill invocation blocked for quarantined skills
- MCP prompt catalog hiding quarantined prompts
- external manifest import trusting matching assets
- invalid latest manifest falling back to prior valid manifest
- older valid manifest replay being rejected when the anti-rollback anchor records a newer accepted manifest

Performance tests should cover:

- large skill and prompt inventories
- boot-verified immutable snapshot resolver mode
- at-use digest recheck resolver mode
- worst-case latency for skill context payload and prompt catalog listing

Security validation should include Bandit on the new integrity subsystem and touched Skills/Prompt paths.

## Open Design Decisions For Implementation Planning

1. Which OS key stores are supported in the first implementation slice.
2. Whether filesystem assets use immutable boot snapshots, at-use rehashing, or a hybrid.
3. The exact DB schema split between trust manifests, verification state, and audit events.
4. How package/plugin signature metadata is discovered for bundled and plugin skills.
5. Which prompt DB fields are included in the first canonical prompt-version payload.

These are implementation planning decisions, not blockers for the design direction.

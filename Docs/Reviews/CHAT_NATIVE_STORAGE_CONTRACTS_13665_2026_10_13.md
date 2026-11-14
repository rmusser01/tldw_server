# Native storage contract review — TASK-13665

PR #3229, isolated chat remediation worktree, base `a01287c8c2`. No new protocol, serialization format, storage operation or ownership rule. Existing ADR-049/068 govern.

Five real native source diagnostics exposed mismatched SDK contracts. The existing tolerant serde runtime accepts encoded strings and structured browser-storage values; its public type now exposes unknown input while retaining SDK generic callers. Runtime decoder/serializer bodies are unchanged. The settings presentation watcher forwards the SDK callback context alongside its stamped value and retains the same mapped callback for unsubscription. Guarded direct-runtime set/remove return the platform outcome after the unchanged post-I/O ownership check; no conversion or assertion bypass is introduced.

Watcher context regression: RED missing the second argument, then GREEN. Eleven connected settings, actual native storage hydration, transport and local-recovery suites pass227 checks, zero skips, in each native/WebUI profile. Scoped production dependency compilers pass in both contexts, including all three TASK13664 feedback callers; this is not test-file or whole-project typing qualification. Early private checker configurations that selected an implicit broken cytoscape type stub or nonexistent chrome ambient entry are excluded; actual native SDK diagnostics were reproduced and corrected.

All four owned files are genuinely linted from the repository root with the frontend rules and correct Next project-root setting. Zero errors, five unchanged warnings, no added messages against identical HEAD/stdin comparison. Earlier frontend-CWD invocations ignored shared files outside their base path and are excluded from qualification. No rule/threshold is suppressed.

Independent bounded backend source review is clear: callback identity, original serialization and pre/post ownership guards remain. Bandit is not applicable to TypeScript-only scope. TASK13664 actual Phase7/updated production preview, full critical/cockpit/master qualification, fresh whole-PR independent review, remote-dev integration and required CI remain pending. This unit is not merge approval.

Private evidence: `/private/tmp/chat-pr3229-storage-13665-*`; owned lint config `/private/tmp/chat-pr3229-owned-eslint-config.mjs`. Original nine conversations/28 rows and separate four-row fixture are checked by the ongoing parent native qualification; this source/unit increment itself performs no live conversation writes.

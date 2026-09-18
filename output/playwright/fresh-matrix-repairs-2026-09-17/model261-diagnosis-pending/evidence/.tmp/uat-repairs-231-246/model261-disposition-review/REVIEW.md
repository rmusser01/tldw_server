# UAT261 / TASK13260.203 disposition review

## Verdict

**No application defect is established. Keep TASK13260.203 open pending a causal request-boundary record.** The retained UAT261 failures remain valid observations, but they do not justify a product change or provider tuning.

The audit completed **13 checks with zero failures** across the maintained journey, current prompt assembly source, task record, and safe canonical projections. It writes only hashes, lengths, booleans, public identifiers, and final-answer checks; no raw provider reasoning, prompt body, stream body, or credential is copied.

## Fixture and workflow scope

The maintained [character journey](../../../../apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts) creates a new `E2E-TestBot-<timestamp>` card, supplies a description, selects it from the library, and sends the expected question through `complete-v2` with character context and streaming enabled.

The UAT261 helper instead opens the existing decorated TestBot card and confirms that the action starts a fresh conversation. These differ in a prompt-relevant character name and in creation fields, so they cannot be treated as byte-identical executions. The maintained journey also requires an HTTP success status or a defined credential-recovery UI; it does **not** assert the canonical final answer. It therefore does not independently qualify the UAT261 exact-final requirement.

## Causal boundary

The default prompt builder includes the character name before the card fields, and `complete-v2` appends the generic emote directive after character context. Thus the decorated name and expression directive are valid hypotheses to inspect. They are not a demonstrated cause:

- the retained UAT246 review records exact fresh final answers on the existing TestBot card in both PostgreSQL modes;
- the UAT261 Retry child for the tagged card has a canonical exact final answer, while the stale parent remains a separate no-final-answer record; and
- the retained UAT261 materials do not include the outbound provider message sequence or an assembled-prompt fingerprint for the failed turn.

The same tag can therefore coexist with both the anomaly and exact output. Name conflict alone is insufficient attribution. The fresh UAT236 static `E2E-TestBot` result is also explicitly a separate card and does not erase UAT261.

## Justified next evidence

Do not run unbounded retries or change model/provider settings. If this finding advances, collect one paired same-tagged-card, same-provider request-boundary record that hashes:

1. the ordered, normalized assembled message sequence and effective generation settings immediately before provider dispatch; and
2. terminal/persistence metadata with only final-answer status, lengths, and hashes after completion.

Compare a retained anomalous result with an exact result using those fingerprints. Keep raw prompts, provider reasoning, bodies, and credentials out of the packet. A production repair is warranted only if that comparison shows an application-controlled difference or a deterministic source-level instruction conflict.

## Evidence

- Audit: `node .tmp/uat-repairs-231-246/model261-disposition-review/audit.mjs` → **13 checks, 0 failed, 11 inputs**.
- Audit script SHA-256: `5b0344ef53a70ff60fbe8e6fb588f315a9bfd5cf7914f494c8779884004f9dfc`.
- Audit JSON SHA-256: `2f01600cb7e0d43b9dabd80fd38a1148911b6977314a23ee0c5827fbc1050003`.

No source, test, browser, runtime, provider, database, Git, Backlog, or tracker changes were made. Full-matrix execution was not started.

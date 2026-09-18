# UAT261 — bounded live diagnostic

## Outcome

**The diagnostic is complete; UAT261 remains open.** Two fresh chats from the same original Character3 produced **one nonexact final and one exact `BEEP BOOP.` final**. Their captured current outbound message and generation-setting fingerprints match. Both provider streams reported `stop`, completed without a visible error, persisted, and survived normal reload.

The safe evidence audit passes **25 checks across56 hashed inputs**. This is an **instrumented diagnosis**, not ordinary production acceptance or a repair. No Retry, model tuning, replacement card or third provider submission occurred.

| Attempt | Fresh conversation | Canonical final after think-block removal | Result |
| --- | --- | --- | --- |
| A, sent04:58:46 UTC | `044945f5-55ee-4d68-8120-e5c2cc9bd4d8` | 55 characters; SHA-256 `101b02cd71721840b701b0324d9e6fd934e49c78d007008ebfbac86132651160` | **Fails exact criterion**. No visible interruption, missing-final error or Retry. Canonical reload05:00:25 retained the answer. |
| B, sent05:01:48 UTC | `9e0bf036-c06b-4edd-8ca9-367365481ee0` | Exactly `BEEP BOOP.`;10 characters; SHA-256 `e7b60945f214701508feede75e9847c3ff6096a8d097cf6143d07c1ad07b09b0` | Exact visible final and canonical reload05:03:09 agree. |

Each canonical conversation has exactly the public question and one assistant row. Assistant IDs are `pa_cb67-cb5f-677-188e` and `pa_99d6-0e73-e09-b4d7`. Raw message hashes/lengths are retained in the safe audit; provider reasoning is not copied or printed.

## What the comparison establishes

For both calls, the diagnostic seam captured two messages: one system and one user. Input projection and stream observation states are complete; terminal metadata was observed for both.

- Message fingerprint: `prompt-v1:sha256:3e28242da9a7e37f8bd49467b08d74dda9b9701040253054b420740c615f7ada`.
- Allowlisted generation-setting fingerprint: `sha256:3a7d5dc86d63744310207c0514a9e7e003afd3514c1683102ff141fa5a86fb5f`.
- Provider system-fingerprint hash: `sha256:10ae9764ba464af8c86a82a2a47d56c40c903e05613b207c4e420a2e8e0c17b7`.
- Both finish reasons are `stop`; usage is `null`. The wrapper deliberately records `final_answer: null`; final-answer assessment belongs to the independent native canonical projection.

This shows **different current final outputs with matching captured input projections**. It does not identify the precise source of variation, fingerprint every provider/server/environment input, or prove deterministic sampling. No sampling value, token count or effective token budget is inferred from absent fields.

Historical outbound fingerprints and terminal metadata remain unavailable. These new records do not explain the earlier reasoning-only attempt, establish token exhaustion, prove a UAT246 recurrence, or retroactively turn prior failures into passes. Earlier exact successes and instruction-following failures remain separately retained.

## Native and source controls

The diagnostic used the original `repairs231-250-targeted-20260917` PGsingle browser/profile, API18702 and Web18782, on unchanged source `2ff90d14ae2d8cc1686cb403b8c3a147cc53427c`. The existing tagged Character3 remained version1 with identical card-response hash and saved-instruction hash across both library entries. Both fresh chat URLs lacked a saved conversation ID before submission; both drafts were initially empty. The exact public question was sent once in each.

Native completion requests selected the same saved llama.cpp model and identical projected control fields. Neither supplied an explicit `max_tokens`. Both streams returned200 SSE with `Cache-Control: no-cache, no-transform`, no content encoding, followed by completion-persist200. These observations establish delivery and persistence for these calls, not every per-token timing criterion.

The passive observer captured388 API requests and388 responses with no HTTP≥400 or safe-projection failure. Eight expected conversation POSTs occurred. Four additional automatic `/api/v1/rag/feedback/implicit` POSTs accompanied final-message dwell/reload behavior; they are disclosed and are not extra provider submissions. No direct API write or manual feedback action was performed.

The shared model was idle before the first call, between calls, and after the second. The wrapper captured exactly2 calls, reported persistence available, and restored its provider seam binding. Its reviewed source hash is `3ff6ea2ae18c572e6cba57626a230a9c156f38a3ac524b320a296b6fb1c3cc4e`. The released module/launcher/protocol hashes match the diagnostic binding; the retained synthetic18-test preparation review is separate from these live outcomes.

Original profile, initialization, official restricted PostgreSQL holder and config hashes are unchanged. Prepared product source hashes match the retained manifest. Diagnostic API94249 and existing frontend78635 were observed still started at05:03:58 with matching receipts after both calls. Only the reviewer’s passive browser listeners were removed; browser and model leases were explicitly returned.

## Ordinary backend restored

Root restored the ordinary API launcher after handback. `restored-ordinary-safe.json` records API97528 and the unchanged frontend healthy200 at **05:07:08 UTC**, original source/profile/init/holder/config unchanged. The independent audit verifies the current receipt hashes/binding and ordinary `uvicorn tldw_Server_API.app.main:app` entrypoint. The old diagnostic process receipt’s later exit is expected and does not invalidate its recorded acceptance interval.

## Retained limits and helper corrections

- The first passive request-body filter covered `/characters` but missed the actual `/chats` paths. It was corrected **before any provider submission**, retaining the original/corrected helpers and captures.
- One read-only Node projection mixed `require` with top-level await and failed before executing, leaving an empty08 file. The explicit-ESM v2 receipt is authoritative; no provider submission resulted.
- The initial offline write-count guard omitted automatic implicit feedback. Inspection showed the four successful application-generated feedback POSTs; the audit now records them explicitly. No native outcome was changed.
- Canonical final parsing removes closed or unterminated `<think>` blocks and trims the remaining content; both retained assistants have a closing think tag. Raw reasoning is never published in the projection.
- This is one original PostgreSQL single-user configuration with reused isolated dependencies. It is not a fresh full matrix, all-mode acceptance, or a provider reliability estimate.
- Private config/credential/process data are read only for hashes or allowlisted identity checks. Known credential values are used in memory for the safe-output scan and never emitted. No runtime, source, database provisioning, Git or Backlog mutation was performed by this reviewer.

# Native acceptance audit — UAT207 and UAT208

## Disposition

**Both remaining native gates pass.** Recommend closure of TASK13260.145 (UAT207) and TASK13260.146 (UAT208), combined with their already recorded independent source/test reviews. The authenticated profile catalogue loads correctly for Alice and Bob; ordinary PostgreSQL Character chat receives an explicit unsupported-metadata placeholder instead of resolver 500 responses. Full reload preserves both results.

This is an audit of retained native evidence, not a new source review or test run. The auditor authored UAT207’s earlier repair; the task separately records root’s independent source/test review. No browser, runtime, source, task or git changes were performed.

## UAT207: authenticated profile catalogue

All times are UTC on 2026-09-17. Safe identity events establish Alice/user 2 before the switch and Bob/user 3 at **07:22:53.195** and repeatedly afterward.

| Account and boundary | Actual evidence |
| --- | --- |
| Alice initial Character chat | Three `/persona/profiles` responses at **06:45:51.861, 06:45:52.250, 06:45:52.397**, all 200, each containing only legacy profile `research_assistant`, version 1. |
| Alice full reload | Four responses at **07:18:09.086 through 07:18:09.853**, all 200, same sole profile, same version 1 and unchanged timestamps. |
| Bob ordinary Chat entry | The retained command clicks **Chat as Helpful AI Assistant**, opening `characterId=3`. Three catalogue responses at **07:25:12.434, .685, .797**, all 200, contain only `research_assistant:3`, version 1. |
| Bob full reload | The retained command executes `page.reload()`. Four catalogue responses at **07:26:10.317, .400, .772 and 07:26:11.004**, all 200, contain the same sole Bob profile and version 1. |

Alice’s legacy profile preserves `created_at` and `last_modified` **2026-09-16 22:23:13.441000+00:00**. Bob’s new owner-specific profile has both timestamps **2026-09-17 07:25:12.418000+00:00**, unchanged across every observed response after reload. This is a real populated catalogue, not an empty-success substitution. No duplicate default or repeated creation is visible in the returned lists.

The native clause of AC2 is satisfied. Official PostgreSQL/SQLite, tombstone/race/caller-transaction controls and scoped static checks remain supported by the prior review recorded in the task; this audit does not recertify them or infer raw database ownership from a response field that is not exposed.

## UAT208: optional visual resolver

Across the allowlisted, de-duplicated event captures, all **six** actual `/visual-identities/bindings/resolve` responses are **200**:

| Actor | Time | Requested expression |
| --- | --- | --- |
| Alice’s character 4 | 06:45:52.396 | neutral |
| Alice’s character 4, after actual successful Retry | 06:48:03.543 | happy |
| Alice’s character 4, full reload | 07:18:10.220 | neutral |
| Alice’s character 4, restored answer | 07:18:10.528 | happy |
| Bob’s character 3, initial Chat | 07:25:12.745 | neutral |
| Bob’s character 3, full reload | 07:26:10.855 | neutral |

Each response explicitly has `fallback_reason: metadata_backend_unsupported`, `resolution_source: placeholder`, and null pack, asset and asset-URL fields. No unsupported visual data or asset success is fabricated. Bob’s settled initial and reload snapshots retain the Character chat and greeting; the expression area uses the existing placeholder prompt.

The retained Bob windows extend through **07:25:53.972** before reload and **07:26:37.345** after reload. Each load has one neutral resolver response, with no repeated failing optional-resolver requests in those observed windows. Alice’s distinct neutral/happy requests accompany load, answer and reload; they are successful responses, not an error-polling loop. This supports AC2 and the native load/reload portion of AC3. It is not a claim about indefinitely sustained polling behavior.

### Unsupported authoring remains a limit

This fallback does **not** add PostgreSQL visual metadata authoring, pack storage, asset lookup or explicit override support. The task’s reviewed behavior preserves explicit override/authoring 501 responses and meaningful auth, rate and database errors; those are prior automated controls, not new native exercises here. SQLite visual resolution, deletion/ownership negative cases, native `thinking` expression, and explicit-override workflows were not rerun. No whole-chat clean-console or complete matrix claim is made.

## Provenance and reproducibility

`input-manifest.json` hashes the exact **15** allowlisted native files and both official CLI task snapshots. Native files were neither copied nor normalized. `audit_inputs.py` de-duplicates identical events across cumulative captures and verified **14 populated catalogue responses / 6 resolver responses**, identities, stable default versions/timestamps, actual Chat click and actual reload; all assertions passed. `verified-facts.json` retains only the relevant safe fields.

Parent handoff attributes the current backend to `47e23`, PID 56113. The supplied browser receipts establish native behavior and account identity, not an independent process-start/source attestation. No older a130 startup manifest is used. Source attribution remains linked to the runtime owner’s receipt chain. No credential, session store, login-private or private helper files were read.

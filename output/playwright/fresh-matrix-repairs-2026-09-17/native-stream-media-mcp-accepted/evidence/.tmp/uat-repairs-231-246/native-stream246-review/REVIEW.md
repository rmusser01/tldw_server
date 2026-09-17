# Independent native review — UAT246 Character stream delivery

**CLEAR for repaired byte delivery and the exact fresh TestBot first-turn scenario in both PostgreSQL modes.** This is bounded acceptance on corrected source `a7d3155a567afb25982eb360ea24b973cc3249c9`, using preserved profiles. It is not fresh initialization, the full 48-cell matrix, or proof of the original 45-second failure's cause.

Associated task: TASK-13260.188. This reviewer changed only `REVIEW.md`, `audit.mjs`, and `audit.json`. No browser, database, runtime, source, Git, Backlog, or tracker mutation was performed.

## Native acceptance

Both native requests send exactly `Hello, who are you?` in a newly created, empty conversation on the existing TestBot card. Card IDs, instruction text, version and update timestamp match their original creation receipts. Complete-v2 request settings, including provider and model, match the original native request within each mode. The visible final answer is exactly `BEEP BOOP.`; the full canonical assistant fields also contain retained model reasoning and are not represented as nine-character-only records.

| Evidence | PG single | PG multi, Alice |
| --- | --- | --- |
| Character | 3, original version 1 | 4, original version 1 |
| Fresh conversation | `5ceb72a9-61f1-42af-bc7a-434cb8c7052b` | `335f57df-78c2-4831-b7de-5275f376bd69` |
| CDP request | `55687.2573` | `55884.1891` |
| Request UTC | 21:52:05.274 | 21:54:18.655 |
| Headers UTC | 21:52:05.562 | 21:54:18.900 |
| First body after headers, monotonic CDP time | **13.382 ms** | **14.104 ms** |
| Nonempty body events | 2,691 | 1,083 |
| Canonical assistant timestamp UTC | 21:52:39.434 | 21:54:32.532 |
| Settled UI capture UTC | 21:52:45.518 | 21:55:14.807 |
| Canonical reload capture UTC | 21:53:29.334 | 21:56:07.940 |

All times are 2026-09-17. Both selected responses are HTTP 200 with `Content-Type: text/event-stream`, `Cache-Control: no-cache, no-transform`, and no `Content-Encoding`. The header observer omits request IDs, so the audit correlates the unique header timestamp to the selected CDP request. Accumulated observer arrays contain older requests; their timings are excluded from fresh-run measurements.

Each fresh turn has one user POST, one complete-v2 request, and one acknowledged completion-persistence request. The terminal UI is settled, displays the expected answer, and has no timeout/no-final-answer indication. Subsequent fresh canonical GETs each contain the same two distinct messages with the original acknowledged IDs, parent link, timestamps and content hashes; neither next pages nor duplicated rows appear. Reload shows two articles and the expected answer. The single run completed and reloaded before multi started.

These are separate new conversations. The observer records no mutation to either previous failed conversation during the new turn. Old failures and histories were retained.

## Network terminal marker

Both successful selected CDP requests end with `net::ERR_ABORTED`. The inspected parser returns on `[DONE]` and calls `reader.cancel()` in `finally`, so that marker is compatible with normal stream consumption. It alone does not prove `[DONE]` delivery or a failure. Visible terminal content, successful persistence acknowledgement, and canonical reload establish the success verdict here.

CDP body events prove browser delivery, not when the provider generated individual tokens. These runs establish prompt delivery of body bytes through the actual repaired runtime; they do not guarantee upstream output latency.

## Earlier attempts retained and corrected

1. **Wrong exact scenario:** the first upgraded single send submitted a 1,971-character collapsed Media handoff despite the harness filling the textbox with the TestBot question. It produced a reply and early bytes, but is excluded from exact-prompt acceptance.
2. **Single existing-history no-answer outcome:** the second single send used the exact question in a long existing history. An early UI capture at 21:47:46 was still generating. Complete later CDP evidence shows that request `55687.1677` ended at **21:47:59.038**, and canonical assistant storage occurred at **21:47:59.054**. Reload's final article says “No final answer.” This is a retained unsuccessful exact attempt, not a successful completion.
3. **Multi existing-history no-answer outcome:** request `55884.1082` ran from 21:48:13.675 to 21:49:17.903, about 64 seconds, delivering body bytes and ending with retained reasoning but no final answer. The final UI and canonical readback preserve that failure.

The initial controller claim that the earlier single attempt overlapped multi or was canceled by reload was incorrect: its recorded end precedes both the multi request and the single reload. The current `UPGRADED_NATIVE_NOTES.md` and a superseding task note correctly acknowledge this. No model-length-limit or reload-causation diagnosis is established. Successful fresh first turns do not erase either existing-history no-answer outcome.

## Source, causal evidence and limits

The separate UAT254 native review establishes the corrected runtime source, original profile identities and actual startup provenance. This audit hashes that review and independently confirms that all three previously reviewed UAT246 repair files match the workspace and both corrected source copies. The production header map remains confined to the three Character SSE response branches.

The retained independent implementation review reports **35 backend plus 8 installed-Next tests passing, zero skipped**. Its gated compressor test demonstrates that old headers buffer the first frame while the no-transform response delivers it before terminal release. This is causal evidence for the bounded compression repair. Those tests were not rerun by this native reviewer.

The native result supports repaired byte delivery and exact fresh-scenario completion/persistence/reload. **It does not retrospectively establish why the original 45-second native request failed.** It also does not resolve the separate earlier existing-history reasoning-only outcomes. Those limits remain explicit rather than converting headers or body activity into answer acceptance.

## Reproduction and safe evidence

Run the offline audit:

```sh
node /Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-repairs-231-246/native-stream246-review/audit.mjs
```

`audit.json` contains selected IDs, timestamps, byte counts, reply checks, canonical content hashes, source comparisons and SHA256 hashes of every reviewed input. Full model reasoning remains in original safe evidence; private credentials and raw runtime logs were not opened or printed. Passive CDP/header observers use existing responses without intercepting, replaying or fulfilling them.

Verification: **42/42 offline checks passed**, and a separate readback confirmed all **53 reviewed file hashes** unchanged. `node --check` passed. Scoped Bandit ran from the project virtual environment on the auditor MJS and reported one unsupported JavaScript AST parse error; it supplies no JavaScript security certification.

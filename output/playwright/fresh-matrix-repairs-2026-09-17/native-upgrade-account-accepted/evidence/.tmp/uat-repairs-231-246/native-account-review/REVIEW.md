# Independent native account review — UAT243 / UAT248

**Verdict: CLEAR for both bounded PostgreSQL multi-user acceptance cases.** No remaining gap in the submitted first-saved ordinary completion transition or native logout cancellation case. This does not certify the fresh full 48-cell matrix.

Associated existing tasks: TASK-13260.185 and TASK-13260.190. The reviewer changed only this report, `audit.mjs`, and `audit.json`; no source, task, tracker, Git, runtime, browser, or database changes. Prior implementation review is supporting context, not a newly executed test result.

## UAT248: account invalidation retires the active Character turn

The successful second attempt is distinguishable by conversation `d817d54a-cc28-456b-b773-8340db5495ad`, CDP request `19816.5555`, and canonical user message `cbd509c8-39e8-4227-97cf-346306a80644`:

| Event | UTC, 2026-09-17 |
| --- | --- |
| User canonical timestamp | 21:13:50.555 |
| Observer receives user POST 201 | 21:13:50.564 |
| Actual `/chats/<id>/complete-v2` streaming headers, 200 | 21:13:51.064 |
| Normal second-tab Logout response, 200 | 21:13:51.119 |
| Original request canceled, `net::ERR_ABORTED` | 21:13:51.161 |
| Normal Bob Login response, 200 | 21:14:42.582 |
| Canonical Alice history readback captured | 21:45:55.855 |

The observer reports zero body bytes for this specific second request, an abort 42 ms after the logout response, and “Signed out” in the original page. Bob's later `/auth/me` identifies id 3. The retained observer interval through 21:15:15 records no later write to Alice's conversation.

The later normal Bob logout/Alice login identifies Alice id 2 and reads the original conversation through the UI. Every captured canonical message response has exactly five rows and no next page: the initial TestBot user/assistant pair, the first unsuccessful harness attempt's normally completed pair, and only the second attempt's user message. The conversation tail is that canceled user; there is no assistant successor or assistant row after its timestamp. The request prompt and canonical canceled user content match, without reproducing model reasoning in this report.

This establishes native retirement before response-body delivery plus absence of a late persisted assistant at a roughly 32-minute readback. It does not prove provider-side cancellation, roll back previously acknowledged writes, or exercise native late-byte delivery after cancellation. Existing independently reviewed actual-lease regressions cover controlled late success/persistence suppression; that unit evidence remains distinct.

## UAT243: ordinary completion resolves the saved-chat mode

Bob's empty fresh chat initially retains the allowed Character preference. On native send, the actual ordinary endpoint `/api/v1/chat/completions` is requested at 21:17:45.577 and returns 200; the newly created conversation `e3755001-1cd4-40a4-aaaf-edd4b67b3418` has `character_id: null`. The settled capture at 21:19:22.946, still on `/chat` before reload, displays “Standard chat” and `BOB ORDINARY CHAT OK.` with neither “Character Chat” nor “Choose a character.” This directly satisfies the first-saved-completion transition that UAT243 targets.

The later normal saved-chat navigation/readback at 21:41–21:42 identifies Bob id 3, displays “Standard chat,” and reads the same ordinary `webui-chat` conversation. All canonical responses contain the same three distinct rows: system, user, and one exact assistant reply. The assistant row retains its original 21:17:47.590 timestamp. Genuine Character presentation is also still visible when Alice returns to her actual Character conversation.

An early reload capture at 21:20:27 **does show “Character Chat” and “Choose a character.”** Its harness waited only for answer text; its included canonical response events are stale 21:17:45–46 events containing one/two rows, not fresh post-reload readback. It is therefore excluded from settled/canonical acceptance. The evidence does not prove why that snapshot was misaligned or establish its duration. The direct pre-reload success and later genuinely settled readback support the bounded UAT243 verdict; no claim that every interim render is correct is made.

## Failed attempts and continuity limits

- Preparation failed while reading a strict `main` locator matching two elements, after creating the normal second Settings tab. It did not undo that setup.
- The first logout harness waited for the wrong `/chat/complete-v2` path. It timed out before clicking logout; the actual `/chats/<id>/complete-v2` turn completed and persisted normally. It is not a successful account-boundary test.
- The Bob-login helper failed when reading an undefined observer array after submission. Subsequent normal login 200 and Bob identity receipts establish that login succeeded; there was no need to count a repeat submission.
- Automation continuation initially showed `about:blank`, and the first upgraded readback wait timed out. Later evidence follows normal navigation with preserved authentication. It is not uninterrupted original-page continuity. A subsequent normal Bob logout and Alice login precede the Alice canonical readback.

## Source and reproducibility

The controller identifies initial native source as `86458ab88ce3fa62e6518c9d813c3860254ddb2c` and upgraded source as `a7d3155a567afb25982eb360ea24b973cc3249c9`. Without Git access, this reviewer independently compared the three relevant production files (combined chat hook, Character domain adapter, and scope-error route policy) against the prior UAT248 review and all retained targeted/upgrade/upgrade2 copies. Their bytes match in every location. The hook retains both the ordinary conversation-ID publication guard and captured account-lease signal. Commit identities themselves are supplied provenance, not independently rederived here.

`audit.mjs` reads only the named proof files and source copies, checks the correlations above, and writes concise safe fields to `audit.json`. The JSON hashes every reviewed input and this report/script. Full model content remains only in the original evidence files; this report and audit project IDs, roles, timestamps, content lengths/hashes, and the short acceptance marker. Private credential scripts and raw runtime logs were not opened. The audit file's own hash is printed by the command, avoiding a circular self-hash.

Run from any directory:

```sh
node /Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-repairs-231-246/native-account-review/audit.mjs
```

Verification: the offline audit passed **26/26 checks**, and all **50 reviewed file hashes** matched in a separate readback. `node --check` passed for the audit script. Scoped Bandit was run from the project virtual environment on `audit.mjs`; it returned one JavaScript AST parse error and no usable JavaScript security coverage.

No additional product tests were required for this offline evidence review. Prior implementation reviews disclose differential compiler/lint baselines and Bandit's TypeScript parse limitations; this report does not convert those into clean-build or security-certification claims.

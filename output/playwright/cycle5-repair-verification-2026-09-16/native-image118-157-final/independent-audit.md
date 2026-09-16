# UAT118 / UAT157 final native acceptance — independent audit

Reviewer: source013_diagnosis. Date: 2026-09-16. Tasks: TASK-13260.58 (UAT118) and TASK-13260.95 (UAT157).

## Decision

**Bounded acceptance is supported; both repair units can close with the limits below recorded.** The final-source SQLite single-user run demonstrates successful image persistence without the UAT157 fallback duplicates, then a real HTTP502 failed image turn that survives reload and retries with the same identity and bytes. The Retry completes with answer 3 and remains a single canonical user/assistant pair after another reload. Existing automated image-only, conflict, ownership, and partial-acknowledgement controls complement this targeted native path. This is not a full UAT matrix or a native PostgreSQL image acceptance claim.

This audit read retained evidence and source only. It performed no browser interaction, model request, runtime operation, database change, test execution, production edit, tracker update, or git mutation. Only this independent-audit.md artifact was written.

## Source and fixture freeze

- Git HEAD independently read as `d76746ac9a69d2ead927e0f2b15434d6740556ce`, matching summary.json. An independent `git diff --name-only HEAD -- apps tldw_Server_API packages` returned empty. The root summary also records an empty tracked source diff. This establishes the retained checkout source state; it is not an independent process-memory attestation.
- The two explicit UAT157 source/test hashes in summary.json match the current files and previously reviewed hashes:

| File | SHA-256 |
| --- | --- |
| `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx` | `b80564988c79cbac35dcc167409fc2307463699112ac1d77e5c587e35089b309` |
| `apps/packages/ui/src/hooks/chat/useChatActions.ts` | `b7d882669470b8c1243b11b8d96beeed1acf5fec8fb4aa732b029e5c9dda7394` |

- Public fixture: `apps/tldw-frontend/public/icon.png`, **2962 bytes**, SHA-256 `1792198785947731fc31e4c6184adeb00b988e703fc548893e3c1049a9b45453`. All 5 image parts across the three completion bodies and both final canonical image arrays decode to those exact bytes. No image URL or payload was substituted for this audit.
- Captured requested model: `llama.cpp/gemma-4-26B-A4B-it`; save_to_db is true for all three completion requests. All use conversation `525e225e-8557-40f9-a489-133ec0f9bbd2`.

## Native request and persistence evidence

The installed observer matches both /api/v1/chat/ and /api/v1/chats/ paths. final-events.txt contains 70 requests and 280 events. The only four mutations are one conversation creation and the three completion POSTs below. There are **zero captured fallback /chats/{id}/messages writes** and no captured update/delete mutations. This conclusion is scoped to the monitored page and interval.

| Request | UTC start | HTTP | Client message identity | Result |
| --- | --- | --- | --- | --- |
| 5, first Send | 20:56:57.551 | 200 | `pa_5527-3d40-178-8273` | Correct image description |
| 32, second Send | 20:58:26.911 | 502 | `pa_16b0-4c73-953-348b` | Failed turn retained |
| 52, Retry after reload | 20:59:23.691 | 200 | `pa_16b0-4c73-953-348b` | Answer `3` |

The second Send and Retry have deeply equal messages arrays, including prior successful context, current text, image ordering, MIME, and PNG bytes. Retry adds tldw_retry_failed_turn:true while preserving the same client ID and conversation. The first turn uses a distinct client ID. No second Send is used to explain an earlier duplicate.

Canonical complete-page observations (has_more:false on the final page):

1. First successful reload: **3 rows**, system plus one image user and one assistant. The assistant says “A white speech bubble icon containing three black dots.”
2. Failed-turn reload before Retry: **4 rows**, the original three plus failed user `1966fdf9-511d-4a16-8b10-89fd6f13f47f`. The saved PNG and client metadata remain present. The UI snapshot includes the attachment, local error, and Retry action.
3. Final reload: **5 rows**, system plus exactly two image users and two assistants. The failed user keeps its original UUID and version1; the new assistant is answer `3`. Both user images remain present and byte-identical. Final ordering is system, first user, first assistant, second user, second assistant.

| Role | Canonical row ID | Client ID | Attachment |
| --- | --- | --- | --- |
| system | `fe14f58c-5344-470b-9f9d-a055bbb23cde` | `—` | none |
| user | `fcbab1d5-a597-47a2-b3d3-ce45d31ae0ac` | `pa_5527-3d40-178-8273` | one PNG |
| assistant | `183d67f9-69b2-40e2-a890-a3e637890c00` | `—` | none |
| user | `1966fdf9-511d-4a16-8b10-89fd6f13f47f` | `pa_16b0-4c73-953-348b` | one PNG |
| assistant | `eb731564-7685-4567-8a25-1b16d030fae8` | `—` | none |

The final accessibility snapshot contains both image users and answer 3. The reviewer also inspected final-reloaded.png: the failed question's PNG remains visible, answer 3 is displayed, and the status reads Chat 5 messages. Screenshot viewport limits mean the canonical complete-page response, rather than the cropped screenshot alone, proves full row count.

## Direct Retry SSE evidence

Unlike the first successful completion, **Retry request52 has a captured readable SSE body** in final-events.txt. Parsing it yields:

- tldw_user_message_id: `1966fdf9-511d-4a16-8b10-89fd6f13f47f`.
- Assistant receipt field tldw_message_id: `eb731564-7685-4567-8a25-1b16d030fae8`.
- tldw_conversation_id: `525e225e-8557-40f9-a489-133ec0f9bbd2`.
- Visible content deltas concatenate to `3`; finish_reason is stop, a terminal event reports success:true, and [DONE] is present.

Both acknowledgement IDs match the final canonical rows exactly. Request5's response body is unavailable through the capture API, so this audit makes **no claim about observed first-completion SSE acknowledgements**. First-completion persistence and absence of fallback writes are instead established by the captured requests and canonical reload. summary.json was corrected before this audit to distinguish these cases and to summarize image strings as MIME/bytes/digest objects.

## PostgreSQL and existing controls

The retained command runs the explicit `test_postgres_strict_image_snapshot` plus the PostgreSQL ChaCha transaction, media, and backend suites against the owned PostgreSQL service, with postgresRequired:true. The redacted log independently shows **32 passed, 17 deselected, zero skipped**, in 11.73s. The image integration file has an executed passing item; it was not skipped. Generic teardown logs mentioning SQLite do not change the explicitly selected PostgreSQL test's backend fixture.

Read-only inspection of test_postgres_strict_image_snapshot confirms it creates CharactersRAGDB with the PostgreSQL backend from pg_database_config and invokes the common strict-image assertions: one page snapshot, exact ordered multi-image bytes, legacy image bytes, and over-budget reads returning no blobs. This establishes PostgreSQL snapshot behavior, not browser/model execution against PostgreSQL.

Existing image integration controls cover image-only and text+image failed turns, streaming/non-streaming retry, unchanged-user reuse, prior-success overlap, literal template text, wrong text/image/order/detail, ambiguous or edited placeholders, missing/corrupt/truncated attachments, pagination, and foreign account/workspace denial. UAT157's previously reviewed seven ACK/capability controls preserve genuinely unpersisted rows when only one role is acknowledged and preserve ordinary no-ACK fallback. Those existing controls and retained earlier verification support the cases not exercised natively here; this reviewer did not rerun them.

## Acceptance limits

- Native coverage is targeted **SQLite, single-user, text+PNG** with the configured real local vision model and reused dependencies. Image-only and account/history ownership remain automated boundary coverage. No native multi-user or PostgreSQL image claim is made.
- The controlled provider outage provenance is supplied by root; this audit directly observes the real502 response and later successful Retry, not the process stop/start action itself.
- First-completion SSE acknowledgement contents are unavailable. Retry acknowledgements are directly captured. There is no inference that an unavailable body means missing server acknowledgements.
- No full UAT restart, global account matrix, generic network-dispatch proof, or fix for the separate source-summary wrong-answer issue is claimed.

## Evidence hashes

All paths in the table are relative to this private native evidence folder.

| Artifact | SHA-256 |
| --- | --- |
| `install-monitor.js` | `4748c9212248d183a359f33139df53f5da96eab6583142f3056e192421f3d7a7` |
| `first-reloaded-events.txt` | `bf7fc782c08ab4d7c79ea0cd6ba9dd7a30373fff773d94b33f6009bdc1364f22` |
| `failed-events.txt` | `54a733dee5449d5261c2129eaf8020699fd3913a36f5649a8b7f93cc2cf1e2b5` |
| `final-events.txt` | `f570ed037aefc0a3d2f8262ad089c0b9cb19c4e377684d6e2bcac6460742e5e3` |
| `first-reloaded.txt` | `a11e739dd81b77fdc7c075e3dfcc36a15f01fbfe713cfb32ed8ff55479bd8bfa` |
| `failed-reloaded.txt` | `530284a60d265427a39174a58bc83962ff9a2184a0d42d468adea7eb4e043d05` |
| `final-reloaded.txt` | `814644b38e09626ac92e7d8c8f97e4db635f9d1e0867a49d992b3813f99a987a` |
| `first-reloaded.png` | `2bf123b29f92d476f77c9d0c8db0426fe534dfc6503f54e49bbac74843b65b86` |
| `final-reloaded.png` | `01d876d48fcc5948ab788d644c5867a64be2c53a4a639afec6829e70bdad5ee7` |
| `summary.json` | `fe43fca17c9384e480332be5e10ffcab4b329940ff2f7d3c7e00ab56a3cf0593` |

PostgreSQL log: `.tmp/fresh-uat-recovery-20260916/pg-final-image-acceptance.redacted.log`, SHA-256 `6c2c7b822be771945293e2e1956e3df2221abece6b05faae35bb10459fec70e3`.

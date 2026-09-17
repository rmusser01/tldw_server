# Independent targeted native audit — UAT209 / UAT210

## Verdict

**The retained native evidence satisfies the outstanding native portions of TASK13260.147 AC3 and TASK13260.148 AC3.** With the already reviewed required-PG regressions/static checks, both scoped findings support closure once this evidence package is retained. This review does not close UAT221 or qualify the native service role as a restricted PostgreSQL login.

Task criteria were read through the official Backlog CLI.209 requires original native two-owner reads plus own save/reload;210 requires authenticated native graph acceptance. Prior implementation/regression/security review is retained in `followup209-210-217-reviewed-supplemental` and `followup210-graph-compatibility-reviewed`; this audit adds native acceptance and does not rerun or replace that review.

## UAT209 — Notes ownership and save/reload: PASS

`notes209-native-api-result.json` contains23 actual requests at08:15:46–08:15:47UTC, with fresh authenticated sessions confirmed as Alice2 and Bob3. The supplied receipt contains no login response tokens or session headers. Both catalogues contain only their respective owner's rows. Attached keyword ownership matches each note owner. Original Alice Citrine note `b83dca90-…` returns200 to Alice and404 to Bob; the final Alice read is byte-for-value equal to the initial body at version1.

The controlled new Alice note `87c1d02b-…` and Bob note `66bc499c-…` return404 on the other actor’s GET and PATCH in both directions. Each actor’s own PATCH returns200/version2, and the following GET is exactly equal. The attempted foreign writes target only the disposable synthetic notes; the original Citrine note is read-only throughout this receipt.

Native browser evidence then confirms:

- Alice’s list contains4 notes and her linked tag catalogue contains6 owner2 entries. The browser sends the edited synthetic body at08:30:45.977, receives200/version3, then manual Save sends at08:30:57.343 and receives200/version4. The UI reports All changes saved and Version4.
- Actual `page.reload()` is recorded. The subsequent08:32:03.587 catalogue200 contains the identical saved body at version4, plus the other3 owner2 rows. The settled snapshot displays that body in the four-item catalogue. The editor is **New note** after reload; this is catalogue persistence evidence, not a claim that the editor reopened the saved note.
- Bob’s UI opens his synthetic note and displays its saved body/version2. His actual full reload is followed by08:37:39.959 catalogue200 with exactly3 owner3 rows, including the saved version2 body. The settled snapshot has3 entries, no Alice note, and the tag response has7 owner3 entries. Bob’s editor also resets to New note.

## UAT210 — PostgreSQL graph compatibility: PASS

An actual administrator session is confirmed by `/auth/me` as id1. The initial script creates one synthetic admin-owned note `11c86b62-…` and a linked `uat210-admin+tag`. Its focused graph GET already returns200 with the correct UUID note node, tag node and membership edge. The script incorrectly expects a frontend-style `note:` prefix on the raw API UUID and marks the receipt `passed:false`; this original failed harness receipt is preserved.

The separate read-only continuation uses the existing note, a fresh admin session, and the actual raw API ID contract. Both focused and all-notes graph GETs return200 at08:32:02.745/08:32:02.774. Each result contains exactly the admin note, its tag and their edge, active_note_count1, all_notes_eligible true and no truncation. Neither contains Alice/Bob note IDs. Login/logout and read operations are recorded; no permission/account changes or second note creation occurred in the continuation. The inherited field name `aliceId` in these graph receipts is a harness variable: `/auth/me` and note client_id establish that the actor is administrator1.

Alice’s earlier graph403 explicitly reports missing `notes.graph.read`; it remains a valid authorization result and the separate UAT221 UX finding. No successful ordinary-Alice graph UI acceptance is claimed.210 acceptance here is authenticated live HTTP graph behavior under the existing authorized administrator.

## Provenance and verification

The restart’s before/after source manifests both identify commit `6d06aae9bd03364f1ce68940c41b14f12fb5febe`; all3668 recorded source hashes are equal. The health receipt records replacement API76778 healthy at08:12:45.278 before these captures. `runtime-source-provenance.json` preserves both original manifest hashes, relevant Notes/ChaCha/NoteStore/KeywordStore hashes, times and the exact health receipt. This reviewer did not inspect or control the running process; runtime attribution uses the retained source/start receipts and parent capture context.

`audit-evidence.py` reads only existing files and asserts the identities, status codes, owner/version/body invariants, reload wire/UI receipts and graph shape. Its fresh result is `audit-verification.json`. Four scoped event extracts preserve unmodified event objects at/after08:15UTC whose event is identity or URL contains `/api/v1/notes`; each records the exact source-file SHA and extraction rule. The cumulative originals remain unchanged in the private native packet. Prior unrelated events are not silently counted as this acceptance.

## Limits and retention

The running PostgreSQL application service role is privileged/BYPASSRLS. These native checks establish the application ownership predicates despite that privilege; they do not prove restricted-role bootstrap or database RLS enforcement. SQLite, transaction, linked edge/metadata and restricted-role controls remain the separate previously reviewed fixture evidence. Workspace sharing and every linked operation were not re-exercised natively here.

This audit used no browser, HTTP, credential extraction, native database, product, runtime, task/tracker or git actions. Evidence-only artifacts and the allowlisted durable package are the only writes. No native tokens/headers, raw private logs, executable credential helpers, runtime configs or databases are retained. The copier scans both original and normalized artifacts, plus the full cumulative source event files, against known runtime credentials and JWT/PEM patterns before creating the durable destination. Text-only trailing whitespace normalization is recorded with original/retained hashes; original evidence stays unchanged.

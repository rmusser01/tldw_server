# ADR-070: Native chat retained-byte quota authority

**Status:** Proposed
**Date:** 2026-10-11
**Backfilled from:** Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md §8
**Decision owner:** Human requester / chat UX remediation session
**Related task:** TASK-13589
**Related spec/plan:** Docs/Plans/IMPLEMENTATION_PLAN_chat_native_quota_13589.md; Docs/Plans/IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md Task 2.2

## Decision

Use the existing AuthNZ storage quota authority to reserve immutable owner/candidate-keyed native retained bytes, and combine legacy-only counters L with fresh active native reservations N once in admission and public usage.

## Context

ADR-049/050 already require owned native storage, durable receipts and conservative cleanup. ADR-064 owns configured per-user limits and existing shared pools. The approved H2 design requires allocation before native byte writes and safe recovery across separate AuthNZ and chat transactions. Filesystem scans, legacy counters and native reservations cannot each charge the same managed bytes.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Charge only the filesystem scan | A pending writer can spend the same available quota; uncertain writes have no durable same-key authority. |
| Update legacy counters and retain a reservation | Recalculation and retry can double-charge or erase live reservations. |
| Introduce a separate native allowance | Bypasses the existing budget and conflicts with the accepted H2 scope. |
| Hold AuthNZ work inside chat commit | Couples database locks and failures across stores; breaks the existing outermost byte-I/O and cleanup boundary. |

## Consequences

Stored and cached legacy counters exclude the managed chat_assets tree. Native admission locks stable user, organization and team identities before optional pool rows, reads the current uncached allowance and fresh active ledger, and inserts once. The unique user/candidate key freezes account UUID, server namespace, exact global/workspace scope, pool attribution and size; conflicting identities cannot allocate a second charge. Exact released reservation keys remain terminal. Adoption and recoverable deletion retain their charge. Durable chat intents precede external reservation; original-key resolution precedes writes after an unknown response. Release follows guarded physical removal and a durable release intent; missing authority remains unknown. User/team/org identity comes from authenticated destination scope and current membership, not client quota selectors. Counting continues with enforcement disabled. Required native reads and ordinary generated-file metadata/counter changes share the existing owning AuthNZ transaction; physical voice removal follows its commit. These lifecycles lock selected quota identities in user/org/team order, reread the same authorized IDs after waits and account exact changed IDs. Durable native reserve/resolve/release facades refuse caller-bound uncommitted authority views, so only committed receipts can confirm chat intent. Required admin statistics failures return unavailable instead of a fabricated zero-storage snapshot. Billing preserves its existing exact active-member policy and caches only legacy usage.

Existing non-native check-then-write races and temporary peak disk space remain outside this guarantee. Public H2 capability stays disabled until its remaining typed projection, lifecycle, client and runtime gates qualify. This proposed record does not claim PostgreSQL, installed extension, assistive technology or participant-study acceptance.

## Accounting isolation boundary

The AuthNZ native_chat_quota_reservations ledger is a trusted accounting authority, separate from ADR050's tenant-owned ChaCha native_chat_quota_intents. AuthNZ connections do not establish app.current_user_id sessions. Organization/team admission, active-member Billing and system administration must count several users' live reservations; an ordinary forced owner policy would hide those rows or deny all normal AuthNZ calls. The ledger therefore has an explicit reviewed coverage exemption, rather than an uncovered-table baseline entry. Reservation/release writes use existing quota-owner locks; exact-key resolution is a committed read that checks frozen authenticated identity. Pool totals preserve original org/team attribution; Billing applies active membership and user/admin totals match account UUID. Authenticated service/route authorization governs aggregate access. This clarification does not promote this proposed ADR or exempt chat intents.

## Follow-up

TASK-13589 records actual qualification and review. Later H2 projection/read/send/adoption work must preserve this accounting boundary.

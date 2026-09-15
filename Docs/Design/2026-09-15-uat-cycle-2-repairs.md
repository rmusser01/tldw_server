# UAT cycle 2 repair design

Tracking: TASK-13260 and repair children13260.5–13260.11. Starting revision: `5791427481` (product tested at `68863b90b7`). Evidence and acceptance requirements are in [the running tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md), UAT-020–045. The user authorizes repeating UAT → review → fix until a full fresh single-user/multi-user pass encounters no issues. Fixing this set alone does not complete that goal.

## Repair contracts

These are bounded repairs to existing workflows. Keep current server ownership/permissions, remote-site restrictions, the real model path and source-grounding verification. Use the existing stores, settings registry, API dependencies and UI components. Add tests that fail for the observed behavior before implementation, then focused regressions and touched-scope security checks. New findings stay in the tracker and receive the same review/repair loop.

### Account state and provenance — TASK-13260.5

Recent Notes currently reads/writes the global `tldw:notesRecentOpened` setting; its hydration effect runs only on mount. Reuse the verified server/principal authority scope already supplied to `useNotesEditorState`, including immediate masking during unresolved identity and stale async-read guards. Persist sensitive note metadata under that authority, without assigning unowned legacy history to the next logged-in account. Review sibling persisted Notes data for the same boundary. Chat-derived note origin must follow its source metadata rather than default to manually typed. Clear stale document titles on route/auth changes and expose the existing logout/credential-clear behavior for manual API-key connections. Test account A→B, A→unresolved→B, delayed A hydration, reload, and successful disconnect/re-entry; backend ACLs remain unchanged.

### Flashcards and derived content — TASK-13260.6

Verification currently receives fragment-only card backs, losing the question needed to interpret the answer. Supply question/answer context to verification and require source-only concise generation. A matching word alone must never prove an answer: swapped answers and unsupported additions must fail. Preserve source evidence and verification of notes/extra fields.

Use the existing reasoning parser's visible text segments for derived content, including incomplete reasoning blocks. Saving a Chat flashcard opens the standard question/answer editing flow, prefilled with the visible answer; require both sides before persistence and validate server-side before creating any supporting note. Explicit editing avoids guessing a question from a potentially different branch. Surface generation failure with concise actionable copy and optional structured details. The initial study dashboard must not simultaneously show queue-start guidance and session-complete copy.

### Chat state and local-provider dispatch — TASK-13260.7

Trace multi-user `complete-v2` rejection before upstream dispatch against working ordinary Chat using the same provider/model. Apply the same authorized local-provider configuration contract, without broadening identity or secret access. A retry must preserve or replace conversation and message identifiers together. Character selection must update the active workspace store and actual request context, not merely a visible character label. Saved/temporary transport must agree with its displayed persistence state from first use, without an off/on toggle. Test actual request payloads and server ownership, plus delayed state transitions and retry actions.

### Media ingestion and source lifecycle — TASK-13260.8

HTTP scraping must reject terminal upstream denial status before content extraction and persistence; do not turn remote denial into article content or bypass that denial. Reuse successful ingestion error-sanitization patterns for plaintext analysis and ensure requested analysis has a configured model. Report partial/failure outcomes explicitly.

Deep-link navigation to a newly stored item must work while the library cache is empty. Deletion clears URL/restoration state before selected-item reset; Trash receives the actual deletion timestamp. Derive delete affordances from an authoritative self-capability result, preserving the existing permission gate, and describe recoverable deletion accurately. Media→Chat uses the existing Chat route. Source-original actions only open supported absolute HTTP(S) links; uploaded filenames resolve to the owned Media view. Test initial empty library, stale selected URL, denied delete, restore metadata, unsafe/relative source links, upstream403/429 and valid200 extraction.

### Retrieval, setup and prompt editor — TASK-13260.9–13260.11

Trace ordinary-user stored-media ownership/index/search boundaries against the successful single-user control before changing retrieval. Preserve security exclusions and verify both positive owned/public retrieval and negative foreign/confidential controls. Trace readiness warning payloads and local model discovery validation; a configured Chat provider must give QA a valid default or a concrete setup requirement before the user sends a query. A successful prompt save must establish a clean saved-editor state and remove the new-editor route trigger. These remaining investigations retain the exact tracker acceptance criteria; no workaround is promoted to a passing original flow.

## Verification and completion

Each unit receives independent review, focused regression/lint/Bandit where applicable, and a live targeted control. Changes are committed in reviewable units. Then create new empty config/data/browser profiles for both modes and execute every named journey/shared workflow, including provenance, actual prompt payloads, Notes/card reuse, Media analysis/recovery and cross-account privacy. Keep exact external-fixture failures visible; do not swap fixtures merely to manufacture a green run. Passing isolated tests or resolving26 findings is not sufficient: only a complete issue-free UAT meets the active goal.

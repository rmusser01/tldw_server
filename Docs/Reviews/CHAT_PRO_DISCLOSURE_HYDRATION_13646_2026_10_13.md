# Pro disclosure hydration — TASK-13646

Pro Advanced controls now use the guarded presentation preference logic already used by Casual. Its established key and storage area remain. The toggle waits for its original preference read and exposes busy state; read failure enables visit-only interaction without overwriting unknown saved data. Newer watched choices beat late hydration, and a failed write retains the visible session controls with a truthful recovery notice.

A held real native read reproduced an incorrectly enabled early toggle. The source repair adds no storage backend or readiness shortcut. Four actual controls cover pending read, unread preference failure, newer watch versus late read, and failed publication. The watch fault holds only its initial read; later real WebUI reads remain available for pre-write/watch delivery. Existing direct Pro selectors, expert controls, Casual behavior, focus and refusal assertions remain.

Native and WebUI matrices each pass986tests across10suites. Final affected native7controls and WebUI5controls pass; both normal client source compilers exit0, with no dedicated test-file type claim. Independent bounded review is clear. Diff whitespace checks pass; Bandit N/A TypeScript-only source. Exact-file hooks and the shared commit with TASK-13645 are recorded in the tasks.

ADR required:no; UI preference race correction under existing semantics. Other whole-PR findings, final independent review, latest remote-dev rebase and merge remain open; installed-extension/physical AT acceptance is not claimed.

PR: https://github.com/rmusser01/tldw_server/pull/3229

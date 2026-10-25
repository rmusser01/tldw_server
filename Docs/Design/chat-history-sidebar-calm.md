# Calm chat history sidebar

Task: TASK-13934. Mode: Operate. The requester selected the calm conversation navigator and explicitly asked that folders remain. Start from dev 1deeea56a6ec1a1c81a872986c2135baa8ba41c0, after the composer, rails, header, message views and Research Workspace fixes merged.

## Direction contract

THESIS: Finding and resuming a conversation leads. Search and New chat remain immediately available, followed by compact source controls and a readable list rather than stacked filters and individually framed cards.

OWN-WORLD: Extend the existing chat system: semantic neutral surfaces, current typography, tonal selection with existing semantic focus and selection accents, visible keyboard focus, and existing Ant Design controls. No global design-system replacement.

STORY: Choose All, Server, On this device or Folders, search the appropriate history, then resume a chat. Folders remain a visible destination with their existing management tools. Preserve ownership verification, temporary-mode read-only permissions, cached-copy distinctions, search matches, pagination and saved expansion choices.

FIRST VIEWPORT: Chats, a labeled New chat action and collapse control lead the sidebar. Search and a compact two-column source selector stay above the independently scrolling list. Pinned and recent rows use flat backgrounds and concise metadata; secondary server-chat actions share one menu. Existing app shortcuts remain separately collapsible.

FORM: A code-led local extension of the approved calm chat direction; no new visual world or concept seed. Existing state and callbacks remain authoritative. Selecting a source explicitly opens its history; ordinary collapse/expand preferences remain saved.

FINISH: The local extension ends with code and visual review, recorded verification, and this surface brief. Preserve the incumbent DESIGN.md and .impeccable/design.json; screenshots are evidence, with no new shipping raster assets.

## Validation and architecture

ADR required: no; presentation changes do not alter interfaces, persistence or ownership. ADR-049 governs history selection and local copies. Test the new source-navigation and single-menu behavior first, then existing ownership/search/folder tests, TypeScript and scoped lint. Inspect desktop/mobile in light/dark themes with populated synthetic history, keyboard navigation, folder access, overflow and scoped accessibility. Preserve all previous worktrees and evidence; do not install or stage dependencies. Bandit has no Python scope and provides no TypeScript security clearance.


## Completed implementation and verification

Implemented on `codex/chat-history-sidebar-calm` from the recorded dev base. Search, the labeled New chat action and all four source destinations stay above the scrolling history. Folders and its existing management tools remain visible. Flat rows retain ownership, source/fork identity and metadata; pin/unpin joins the existing secondary menu. Both shared and web mobile drawers use one heading and the available frame, while footer controls reflow at narrow widths.

- 476 focused tests across 37 suites passed: 409 shared sidebar/layout cases and 67 web-layout cases. Source reopening, menu permissions and the mobile frame include failing-first regressions.
- Normal frontend TypeScript passed. Installed scoped lint checked 11 files with zero errors and seven messages identical to dev. The advanced webpack build wrapper passed; temporary local React alignment was removed and the original Next configuration restored.
- Production-served synthetic-data browser checks passed at desktop 1440px and mobile 390px in both themes: folder access, source keyboard navigation, search restoration, pin/unpin, draft retention, fixed search while scrolling, mobile footer fit and no horizontal overflow. Controls also fit at the supported 150px sidebar width. Scoped axe reported zero violations; no page errors occurred.
- Fresh independent code review found no remaining material code issues. The finish reviewer returned `ship` for the five requested fixes, scoring source layout, mobile frame, narrow footer, contrast and healthy folder fixture resolved. This verdict covers those fixes.

Evidence is preserved in `.impeccable/review/sidebar/`; browser results are in `/private/tmp/chat-sidebar-browser-final-20261025.log`. The ordinary-extension documentation comparison preserves the incumbent global design files. ADR-049 remains authoritative; no new architecture rule is introduced.

Limits: browser data was synthetic. Live backend/providers, microphone hardware and full repository tests were not checked locally. The webpack wrapper does not certify the CI Turbopack profile; hosted qualification is tracked below. Bandit scanned no Python code in this TypeScript scope and provides no TypeScript security clearance. No dependency or workflow change is included.


## Publication and CI qualification

The requester approved the final preview and requested continuation. [PR #3316](https://github.com/rmusser01/tldw_server/pull/3316) records their verbatim rationale: improving the sidebar UX and lowering users' active cognitive load.

The first hosted frontend shard exposed a timing assumption in the existing local-history selection test: a completed preview read precedes model, prompt, file and H1 preparation. A held model-preparation promise reproduced the same idle-versus-ready assertion failure. The repaired test waits for the original ready condition and additionally verifies no premature ready state or callback. All current/failed/unmounted, history and prompt assertions remain; production loading and ownership code is unchanged. The focused 26 cases and deterministic eight-suite context (108 cases) pass, with independent review complete. Local context replay needed temporary React alignment; it was removed before committing. Normal TypeScript and the touched test's lint passed.

Hosted qualification must use the new PR head. The first head's separate advisory Notes creation case failed before issuing a Notes POST (45 passed, one failed); the unsaved editor screenshot/trace is preserved, but its cause and relationship to the sidebar are unproven. Notes implementation and E2E files are unchanged. No CI retry, test weakening or workflow policy change was made.

# Calm Chat composer

Task: TASK-13275. Mode: Operate. User-selected direction: a calmer writing space, with tools revealed when needed.

The existing Chat page identity, semantic theme tokens, typography, and configured composer-style preference remain authoritative. This is a local composer redesign, with no new visual identity or dependencies.

The message field and essential footer form one bordered editing surface. The footer retains attachments, a labelled Tools disclosure, the model picker, and the existing delivery control. Mobile keeps these controls reachable instead of suppressing the toolbar. Prompt assistance moves into Tools. Command hints share the attachment/Tools row, and history/temporary mode shares the model/delivery row; neither gets a dedicated footer line.

Tools uses one persisted disclosure (the existing playgroundComposerOptionsExpanded key), closed for new users and preserving explicit saved choices. Its groups use headings and visible rules: context setup, prompts and people, and response/tools. Related actions have explicit rows: source search/web/research; Modes/MCP; Prompt/System prompts/assistance; character selection; Buddy/role-play; response settings/generation/compare; and formatting/OpenUI/other tools. The bounded scrolling tray keeps a growing tool list from consuming the viewport. Escape closes the tray and returns focus to Tools. Existing feature pickers and settings remain responsible for their own interaction. The current dev storage hook retains watch-before-read ordering, pending-read protection, visit-only recovery and owned focus handling.

Selected context and warnings remain actionable when Tools is closed; idle counters are omitted. Dictation and voice share the model/delivery controls row, with a compact microphone button on small screens; active sessions remain available to stop when Tools is closed. Selectors remain mounted, and external starter events reveal Tools before opening them. Buddy setup retains its opening control for return focus.

Preserve text ownership, drafts, IME input, slash commands, mentions, privacy locks, capability checks, queue/stop behavior, errors, and notices. The existing optional V1/V3/V5 shells continue to wrap the same shared input/footer; V5 delegates delivery to the shared footer.

ADR required: no. This changes presentation and disclosure within existing shared UI boundaries; storage, server APIs, provider integration, and security ownership do not change.

Validation: focused component and integration tests, responsive browser checks at 1440×1000 and 390×844 in light and dark themes, composer-scoped axe checks, optional-shell delivery checks, TypeScript, scoped ESLint, Bandit applicability check, and a bounded Impeccable review.

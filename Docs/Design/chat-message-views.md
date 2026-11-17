# Chat message views

Task: TASK-13926. Mode: Operate, with Read typography for answer content. This bounded presentation change inherits the approved calm chat world and the existing PRODUCT.md / DESIGN.md. It is separate from header PR #3311, composer PR #3309 and rails PR #3310.

The user chose a readable transcript as the default and requested Conversation for roleplay and Research for source-heavy work as saved alternatives. Use one explicit Message view choice in the existing conversation-view row and Chat appearance settings. Keep it a local UI preference through the existing setting registry; switching it must not replace the chat, reset drafts, remount messages or change model behavior.

- **Transcript:** compact prompts, a clear speaker/model/time line, borderless assistant prose with a comfortable reading measure, restrained source/tool disclosures, and quiet rectangular actions. The latest answer keeps its useful actions visible.
- **Conversation:** aligned user/assistant bubbles, stronger speaker and character identity, existing portraits/mood preferences and conversational spacing. Preserve all existing actions through their current menus.
- **Research:** denser aligned turns, clear model and available generation metadata, source details easier to inspect, and persistent message actions. Never invent usage or source evidence.

Reuse the shared message renderer, user bubble, action bar, existing source/reasoning/tool renderers, icon system and semantic tokens. Preserve custom fonts/colors, wide mode, opacity where a bubble is present, character render preferences, editing, branches/variants, feedback, error recovery, streaming, generated images, attachments, citations and privacy/capability gates. Other consumers of the shared renderer keep their existing presentation unless /chat passes a view. Keep virtualization keys and history ownership unchanged.

ADR required: no. This uses the existing settings registry and shared UI conventions; ADR-049 history ownership, public APIs, persistence ownership, providers and dependencies do not change.

Validate setting fallback and selection, populated message behavior and retained edits, existing user/action/source/character/recovery regressions, responsive light/dark layouts, keyboard/touch controls and viewport overflow. Use the installed project TypeScript, scoped lint comparison and Bandit applicability check. The offline preview must label synthetic conversation content and cannot prove live provider behavior. Fresh code/visual review and the Impeccable documenter finish the work without changing the global design system.

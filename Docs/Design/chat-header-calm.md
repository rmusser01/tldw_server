# Calm chat header

Task: TASK-13925. Mode: Operate. User approved redesigning both /chat header rows together as a separate piece, following the accepted composer and rails.

The conversation is the anchor: title and acknowledged save/privacy state lead the top bar. Search, new saved chat and sharing stay close at hand. An explicit More menu groups temporary/character entry, page navigation, settings, appearance and keyboard help. On narrow screens, secondary actions move into that menu rather than wrapping into several header rows. Notification lifecycle and unread counts remain visible and retain their status dialog and recovery actions.

The view toolbar uses the same flat vocabulary as the rails: workflow context, Focus, chat shortcuts and Artifacts. Remove the duplicate theme toggle and static timeline pill. Preserve character workflow identity, artifact state/counts, focus mode exit and chat-help keyboard bindings. The non-chat header keeps its existing behavior. No provider, model, source, persistence or sharing permissions change.

Use the incumbent typography, neutral surfaces, semantic state tokens, visible keyboard focus and touch targets. Reuse existing Ant Design menu behavior, callbacks and route policy. ADR required: no; this is shared UI presentation, with existing ADR-049 title/history ownership and sharing restrictions retained.

Validation: failing-first header/menu contracts, existing header/title/share/notification and toolbar checks, TypeScript and scoped lint, then a bounded desktop/mobile light/dark browser pass with scoped axe, menu focus, save/privacy states and draft preservation. Fresh code and visual reviews, documenter, and Bandit applicability check complete the change. Live provider execution and unrelated global design drift are outside scope.

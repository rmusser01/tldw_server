# Native sidepanel reopen freshness — TASK-13659

A restored native tab can have an old cursor and saved latest-seen tip, but a newly admitted manifest containing later turns. Version-only background checks see the newly admitted version as current; skipping the shown-tab comparison then leaves the restored view silently behind. Compare the existing current manifest with the saved latest-seen value. Preserve intentional older positions whose saved latest-seen tip still matches, and explicit pending branches.

When the original native view is unexpectedly behind on show, reuse H1 holdStaleSend(view, original-currentness, freshness). Its existing UI preserves draft/shown messages and offers Load latest or Branch here. Keep native background version checking and legacy controller behavior. Reuse original tab/owner lease and H1 epoch fence; no new queue, persistence, version API or automatic selection.

ADR required: no; ADR049 ownership/operation lifetime governs. Original CI XP08 remains unqualified until connected source and actual reopen evidence pass.

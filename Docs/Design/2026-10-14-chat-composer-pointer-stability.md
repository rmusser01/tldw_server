# Composer pointer stability — TASK-13669

A Pro Tools click loses its target: the native trace shows pointerdown on Tools at top756.89, focusin on Tools, then pointerup/click on a DIV after Tools moves down16px. Empty textarea expansion uses any composer focus, so the action being focused moves before its click completes. Clean Casual Tools opens, and actual onClick/ref forwarding is intact.

Keep idle collapse. Entering the textarea expands it; retain that height while focus moves to actions within the composer. Entering an action from outside leaves the idle textarea compact. Leaving the composer resets the expansion latch. Existing focus-within state still pins the action bar; input geometry has its own short-lived latch in that same hook. No timer, pointer-event workaround, added dependency or persistent state.

Scope: shared action-bar hook, full Form compact predicate, existing caller fixture contracts, actual hook plus textarea focus controls, installed browser pointer/keyboard opening. Existing selection, draft, source, model and owner guards remain authoritative. ADR assessment: transient focus/layout policy only; ADR-049 ownership remains, no new durable rule.

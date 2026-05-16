# Shortcut Keys

This document lists the current in-app keyboard shortcuts and known conflicts.
Shortcuts are shown in the in-app Keyboard Shortcuts modal and stored in local
storage under `keyboardShortcuts`.

## Global
- Cmd/Ctrl+K: Open command palette
- Cmd/Ctrl+Shift+?: Open keyboard shortcuts modal
- Shift+?: Toggle header shortcuts tray (Options header)
- Ctrl+Shift+Y: Open Sidebar (browser extension shortcut; configurable in browser)

## Chat
- Shift+Esc: Focus message input
- Ctrl+Shift+U: Start a new chat
- Ctrl+E: Toggle chat mode (chat with current page)
- Alt+W: Toggle web search
- Ctrl+Shift+H: Toggle Quick Chat Helper
- Ctrl+B: Toggle sidebar (in-app)

## Navigation (Options)
- Alt+1: Playground
- Alt+3: Media
- Alt+4: Knowledge
- Alt+5: Notes
- Alt+6: Prompts
- Alt+7: Flashcards
- Alt+8: World Books
- Alt+9: Dictionaries
- Alt+0: Characters

## Knowledge source selector

- /: Focus the specific-source filter while the selector dialog is focused
- [: Switch the selector to media/docs while the selector dialog is focused
- ]: Switch the selector to notes while the selector dialog is focused

These selector shortcuts are local to the `/knowledge` specific-source dialog.
They are ignored while focus is inside an input, textarea, or select so typing
queries and pasted text are not intercepted.

## Flashcards review
- Space: Flip card
- 1: Rate Again
- 2: Rate Hard
- 3: Rate Good
- 4: Rate Easy

## Known conflicts
- Cmd/Ctrl+K: Often focuses the browser address bar search in Chrome/Edge.
- Ctrl+B: Toggles the bookmarks bar in Chromium browsers.
- Alt-based shortcuts can be reserved by the OS or browser menu accelerators.
- /, [, and ] in `/knowledge`: Scoped to the specific-source selector and ignored
  inside form fields.

If a shortcut does not fire, adjust it via the in-app settings when available
or change the browser-level extension shortcut for the sidebar.

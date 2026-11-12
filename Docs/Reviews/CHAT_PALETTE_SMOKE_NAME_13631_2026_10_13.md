# Palette smoke accessible name — TASK-13631

The smoke test now finds the exact current dialog name, Search commands. Its former command-palette regular expression did not match the production accessible label. This changes one locator only; first keyboard opening, input focus, command filtering, selected option, Enter execution and settings URL assertions remain intact. No production behavior or timeout changed.

On the existing task-owned headless page19769, first Control+K on Home opened the dialog and focused its input. Filtering Go to Settings selected the actual command, and Enter navigated to/settings with the palette closed. No browser window/tab/context was opened. Original9conversations28rows compare deeply equal before and after; the immutable baseline was not rewritten. Focused Host, focus-ownership, shortcut and Layout suites pass63tests. Independent bounded review is clear; normal hooks are recorded in the task.

ADR required: no; current accessible-copy test contract. Bandit N/A TypeScript test-only scope. This live single-page flow and focused tests do not claim the entire CI UX smoke suite passed. Installed-extension/physical AT, other whole-PR issues, final independent review, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229

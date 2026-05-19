# FTUE Sprint 3: Extension Clarity

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the browser extension approachable for non-technical users by grouping context menu items into submenus, renaming jargon labels, and adding a post-setup onboarding hint.

**Architecture:** Three changes: (1) restructure context menu registration to use Chrome's `parentId` submenu feature for grouping, (2) rename i18n message strings, (3) add a post-onboarding toast/notification pointing users to key extension features. No new components needed — leverages existing i18n and notification infrastructure.

**Tech Stack:** WXT extension framework, Chrome contextMenus API, browser.i18n

---

### Task 1: Group context menu items into submenus (EXT-001)

**Files:**
- Modify: `apps/packages/ui/src/entries/shared/background-init.ts:241-320`

**Step 1: Read the current registration code**

Read `background-init.ts` lines 230-325. Currently 14 items are registered flat with no `parentId`.

**Step 2: Restructure into 3 parent groups + ungrouped top-level items**

Replace the flat registration with grouped submenus. Chrome MV3 supports one level of nesting via `parentId`.

The new structure:

**Top-level (always visible):**
- Open Sidebar / Open Web UI (dynamic, existing)

**"AI Actions" submenu** (selection context):
- Summarize
- Explain
- Rephrase
- Translate
- Custom
- AI Popup (renamed from "Contextual action")

**"Save" submenu** (selection context):
- Save to Notes
- Save to Companion
- Read Aloud (renamed from "Narrate selection")

**"Process" submenu** (page/link context, conditional):
- Save to Library (renamed from "Send to tldw_server")
- Analyze without Saving (renamed from "Process with tldw_server (don't store)")
- Transcribe Video/Audio (conditional)
- Transcribe + Summarize (conditional)

Implementation: Create parent menu items first, then child items with `parentId`:

```typescript
// Parent: AI Actions
browser.contextMenus.create({
  id: "tldw-ai-actions",
  title: browser.i18n.getMessage("contextMenuAIActions") || "AI Actions",
  contexts: ["selection"]
})

// Children of AI Actions
browser.contextMenus.create({
  id: "summarize-pa",
  parentId: "tldw-ai-actions",
  title: browser.i18n.getMessage("contextSummarize") || "Summarize",
  contexts: ["selection"]
})
// ... etc for explain, rephrase, translate, custom, contextual-popup

// Parent: Save
browser.contextMenus.create({
  id: "tldw-save",
  title: browser.i18n.getMessage("contextMenuSave") || "Save",
  contexts: ["selection"]
})

// Children of Save
browser.contextMenus.create({
  id: "save-to-notes-pa",
  parentId: "tldw-save",
  title: browser.i18n.getMessage("contextSaveToNotes") || "Save to Notes",
  contexts: ["selection"]
})
// ... etc for companion, narrate

// Parent: Process (only if any capability enabled)
if (capabilities.sendToTldw || capabilities.processLocal || capabilities.transcribe) {
  browser.contextMenus.create({
    id: "tldw-process",
    title: browser.i18n.getMessage("contextMenuProcess") || "Process",
    contexts: ["page", "link"]
  })
  
  if (capabilities.sendToTldw) {
    browser.contextMenus.create({
      id: "send-to-tldw",
      parentId: "tldw-process",
      title: browser.i18n.getMessage("contextSendToTldw") || "Save to Library",
      contexts: ["page", "link"]
    })
  }
  // ... etc
}
```

IMPORTANT: The `browser.contextMenus.onClicked` handler in `background.ts` does NOT need changes — it matches on menu item IDs which are unchanged. Only the registration structure and titles change.

**Step 3: Commit**

```
feat(extension): group context menu items into AI Actions, Save, Process submenus

14 flat context menu items are now organized into 3 submenus plus a
top-level Open action (EXT-001). Reduces visual overwhelm when
right-clicking. Menu item IDs are unchanged so click handlers work
without modification.
```

---

### Task 2: Rename jargon labels in i18n messages (EXT-002, EXT-003, EXT-004)

**Files:**
- Modify: `apps/packages/ui/src/public/_locales/en/messages.json`

**Step 1: Update message strings**

Change these entries:

| Key | Old Value | New Value |
|-----|-----------|-----------|
| `contextCopilotPopup` | "Contextual action" | "AI Popup" |
| `contextSendToTldw` | "Send to tldw_server" | "Save to Library" |
| `contextProcessLocalTldw` | "Process with tldw_server (don't store)" | "Analyze without Saving" |
| `contextNarrateSelection` | "Narrate selection" | "Read Aloud" |

Also ADD new parent menu keys:

| Key | Value |
|-----|-------|
| `contextMenuAIActions` | "AI Actions" |
| `contextMenuSave` | "Save" |
| `contextMenuProcess` | "Process" |

Update related error/status messages to match:
- `contextCopilotPopupNoSelection` → "Select text first to use AI Popup."
- `contextNarrateSelectionNoSelection` → "Select text first to read aloud."

**Step 2: Commit**

```
fix(extension): rename jargon labels in context menu

"Contextual action" → "AI Popup" (EXT-002)
"Send to tldw_server" → "Save to Library" (EXT-003)
"Process with tldw_server (don't store)" → "Analyze without Saving" (EXT-003)
"Narrate selection" → "Read Aloud" (EXT-004)
```

---

### Task 3: Add post-onboarding feature hint (EXT-006)

**Files:**
- Modify: `apps/tldw-frontend/extension/routes/option-index.tsx`

**Step 1: Read the current first-run flow**

Read `option-index.tsx` lines 57-120. After `markFirstRunComplete()` is called, the user sees the normal Playground UI. There's no mention of context menus, sidepanel, or key features.

**Step 2: Add a notification/toast after first-run completes**

After `markFirstRunComplete()` succeeds (in the `onFinish` callback of OnboardingWizard, around line 103-108), show a brief notification explaining key features:

```typescript
onFinish={async () => {
  try {
    await markFirstRunComplete()
  } catch { }
  void checkOnce().catch(() => undefined)
  
  // Show feature hint notification
  if (typeof browser?.notifications?.create === "function") {
    browser.notifications.create("ftue-features", {
      type: "basic",
      iconUrl: browser.runtime.getURL("icon/128.png"),
      title: "You're all set!",
      message: "Right-click on any page to use AI actions. Click the extension icon to open the sidebar for chat."
    })
  }
}}
```

Also add the same hint for demo mode (in the `onDemo` callback, around line 89-93).

**Step 3: Commit**

```
feat(extension): add post-onboarding feature discovery notification

After completing extension setup (or entering demo mode), a browser
notification explains how to use the extension: right-click for AI
actions, click icon for sidebar chat (EXT-006).
```

---

### Task 4: Run regression and verify

**Step 1: Check for extension-related tests**

```bash
cd apps/packages/ui && npx vitest run --reporter=verbose 2>&1 | grep -i "extension\|background\|context.menu" || echo "No extension-specific tests found"
```

The extension uses WXT build which is separate from vitest. Context menu registration is integration-level code that runs in the service worker — no unit tests exist for it. The changes are safe because:
- Menu item IDs are unchanged (handlers still match)
- Only i18n strings and menu structure changed
- Notification API has graceful fallback

**Step 2: Manual verification**

1. Load extension in Chrome dev mode
2. Right-click on page → verify 3 submenus appear: "AI Actions", "Save", "Process"
3. Select text → right-click → verify "AI Actions" submenu has 6 items with new labels
4. Verify "Save" submenu has 3 items
5. Complete fresh onboarding → verify notification appears

**Step 3: Commit any fixups**

---

## Summary

| File | Change | Issue |
|------|--------|-------|
| `background-init.ts` | Restructure into 3 submenu groups with `parentId` | EXT-001 |
| `messages.json` | Rename 4 labels + add 3 parent menu keys | EXT-002, EXT-003, EXT-004 |
| `option-index.tsx` | Post-onboarding notification | EXT-006 |

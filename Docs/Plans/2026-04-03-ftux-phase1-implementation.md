# FTUX Phase 1: Targeted Friction Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the worst first-time user experience friction for parent/family and researcher personas across the webui's chat, moderation, and navigation.

**Architecture:** Eight independent changes to existing components — no new architecture, stores, or services. Navigation items get descriptions and regrouping. Moderation gets a proper onboarding card and Joyride tutorial. Chat gets a Settings button. Onboarding success screen gets an intent selector. All changes are in `apps/packages/ui/src/`.

**Tech Stack:** React, TypeScript, Vitest, Ant Design, react-i18next, Joyride (via tutorial registry), Zustand stores

**Design doc:** `docs/plans/2026-04-03-ftux-audit-design.md`

---

## Task 1: Add guard tests for OnboardingConnectForm

Guard tests must exist before we touch the onboarding success screen (Task 8). These are source-scanning tests following the existing pattern.

**Files:**
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`
- Reference: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.ingest-cta.guard.test.ts`
- Reference: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`

**Step 1: Write guard tests**

```typescript
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readOnboardingSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "OnboardingConnectForm.tsx"),
    "utf8"
  )

describe("OnboardingConnectForm success screen guards", () => {
  it("renders a success screen container with data-testid", () => {
    const source = readOnboardingSource()
    expect(source).toContain('data-testid="onboarding-success-screen"')
  })

  it("has ingest, media, chat, and settings action handlers", () => {
    const source = readOnboardingSource()
    expect(source).toContain("handleOpenIngestFlow")
    expect(source).toContain("handleOpenMediaFlow")
    expect(source).toContain("handleOpenChatFlow")
    expect(source).toContain("handleOpenSettingsFlow")
  })

  it("includes provider and model selector on success screen", () => {
    const source = readOnboardingSource()
    // Provider/model defaults section exists
    expect(source).toContain("Set your defaults")
  })

  it("shows showSuccess state to gate the success screen", () => {
    const source = readOnboardingSource()
    expect(source).toContain("showSuccess")
  })
})
```

**Step 2: Run tests to verify they pass against current source**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`

Expected: 4 PASS (these are asserting existing code patterns)

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts
git commit -m "test: add guard tests for OnboardingConnectForm success screen

Ensures key success screen elements (testid, action handlers, provider
selector, showSuccess gate) are present before FTUX refactoring begins."
```

---

## Task 2: Fix disconnected state on Chat page

Add a direct "Open Settings" button to the disconnected description in PlaygroundEmpty.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx` (lines 148-152)

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

```typescript
import { render, screen, fireEvent } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import React from "react"

const mockNavigate = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, opts?: string | { defaultValue?: string }) =>
      typeof opts === "string" ? opts : opts?.defaultValue ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useIsConnected: () => false
}))

vi.mock("@/store/tutorials", () => ({
  useHelpModal: () => ({ open: vi.fn() })
}))

vi.mock("@/routes/route-paths", () => ({
  buildResearchLaunchPath: () => "/research"
}))

vi.mock("@/utils/quick-ingest-open", () => ({
  requestQuickIngestOpen: vi.fn()
}))

describe("PlaygroundEmpty disconnected state", () => {
  it("renders an Open Settings button when disconnected", async () => {
    const { PlaygroundEmpty } = await import("../PlaygroundEmpty")
    render(<PlaygroundEmpty />)
    const btn = screen.getByRole("button", { name: /open settings/i })
    expect(btn).toBeInTheDocument()
  })

  it("navigates to /settings/tldw when Open Settings is clicked", async () => {
    const { PlaygroundEmpty } = await import("../PlaygroundEmpty")
    render(<PlaygroundEmpty />)
    const btn = screen.getByRole("button", { name: /open settings/i })
    fireEvent.click(btn)
    expect(mockNavigate).toHaveBeenCalledWith("/settings/tldw")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

Expected: FAIL — no "Open Settings" button exists yet

**Step 3: Implement the change**

In `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`, replace the disconnected description (around lines 148-152):

Find:
```typescript
              ? t("playground:empty.disconnectedDescription", {
                  defaultValue:
                    "Connect to a tldw server to start chatting. Go to Settings to configure your connection."
                })
```

Replace with:
```typescript
              ? (
                  <>
                    {t("playground:empty.disconnectedDescription", {
                      defaultValue:
                        "Connect to a tldw server to start chatting."
                    })}
                    <button
                      type="button"
                      onClick={() => navigate("/settings/tldw")}
                      className="mt-2 block text-sm font-medium text-primary hover:underline"
                    >
                      {t("playground:empty.openSettings", {
                        defaultValue: "Open Settings"
                      })}
                    </button>
                  </>
                )
```

Note: `navigate` is already imported via `useNavigate` at line 3/26.

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

Expected: 2 PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx
git commit -m "fix: add Open Settings button to Chat disconnected state

Previously the disconnected state just said 'Go to Settings' as text.
Now there's a clickable button that navigates to /settings/tldw."
```

---

## Task 3: Replace moderation "Loading..." with skeleton loaders

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx` (lines 136-172)

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.skeleton.test.tsx`

```typescript
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell skeleton loaders", () => {
  it("uses Skeleton component instead of plain Loading text in Suspense fallbacks", () => {
    const source = readShellSource()
    // Should import Skeleton from antd
    expect(source).toContain("Skeleton")
    // Should NOT have bare "Loading..." text in fallbacks
    const fallbackMatches = source.match(/fallback=\{<div[^>]*>Loading\.\.\.<\/div>\}/g)
    expect(fallbackMatches).toBeNull()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.skeleton.test.tsx`

Expected: FAIL — source currently contains `Loading...` text and no `Skeleton` import

**Step 3: Implement the change**

In `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`:

Add `Skeleton` to the antd import at line 3:
```typescript
import { message, Skeleton } from "antd"
```

Replace all 5 Suspense fallbacks (lines ~140, 147, 152, 158, 164) from:
```typescript
fallback={<div className="py-8 text-center text-text-muted">Loading...</div>}
```
to:
```typescript
fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}
```

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.skeleton.test.tsx`

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.skeleton.test.tsx
git commit -m "fix: replace bare Loading text with Skeleton loaders in moderation panels

Improves perceived performance by showing animated skeleton placeholders
instead of plain 'Loading...' text while lazy panels load."
```

---

## Task 4: Add plain-language descriptions to navigation items

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` (type + ~15 items)
- Modify: `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx` (renderer)

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts`

```typescript
import { describe, expect, it } from "vitest"
import { getHeaderShortcutGroups } from "../header-shortcut-items"

describe("header shortcut descriptions", () => {
  const groups = getHeaderShortcutGroups()
  const allItems = groups.flatMap((g) => g.items)

  const JARGON_IDS = [
    "stt-playground",
    "tts-playground",
    "knowledge-qa",
    "chunking-playground",
    "moderation-playground",
    "acp-playground",
    "chatbooks-playground",
    "world-books",
    "deep-research",
    "workspace-playground",
    "prompt-studio",
    "model-playground",
    "chat-dictionaries",
    "evaluations",
    "repo2txt"
  ]

  for (const id of JARGON_IDS) {
    it(`item "${id}" has a descriptionDefault`, () => {
      const item = allItems.find((i) => i.id === id)
      expect(item).toBeDefined()
      expect(item!.descriptionDefault).toBeTruthy()
      expect(item!.descriptionDefault!.length).toBeGreaterThan(5)
    })
  }
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts`

Expected: FAIL — `descriptionDefault` does not exist on type

**Step 3: Add description fields to the type**

In `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`, update the type at lines 41-49:

```typescript
export type HeaderShortcutItem = {
  id: HeaderShortcutId
  to: string
  icon: LucideIcon
  labelKey: string
  labelDefault: string
  /** Optional 1-9 index for Cmd+number shortcut when the launcher is open */
  shortcutIndex?: number
  /** Optional plain-language description for non-technical users */
  descriptionKey?: string
  descriptionDefault?: string
}
```

**Step 4: Add descriptions to jargon-heavy items**

In the same file, add `descriptionDefault` (and `descriptionKey`) to the 15 items identified. Examples:

For `stt-playground`:
```typescript
{
  id: "stt-playground",
  to: "/stt",
  icon: Mic,
  labelKey: "option:header.modeStt",
  labelDefault: "STT Playground",
  descriptionKey: "option:header.modeSttDesc",
  descriptionDefault: "Speech to Text — transcribe audio and video"
},
```

For `tts-playground`:
```typescript
descriptionKey: "option:header.modeTtsDesc",
descriptionDefault: "Text to Speech — generate spoken audio from text"
```

For `knowledge-qa`:
```typescript
descriptionKey: "option:header.modeKnowledgeDesc",
descriptionDefault: "Search your ingested documents and get cited answers"
```

For `chunking-playground`:
```typescript
descriptionKey: "settings:chunkingPlayground.desc",
descriptionDefault: "Split documents into searchable segments"
```

For `moderation-playground`:
```typescript
descriptionKey: "option:moderationPlayground.desc",
descriptionDefault: "Content safety rules, blocklists, and testing"
```

For `acp-playground`:
```typescript
descriptionKey: "option:header.acpPlaygroundDesc",
descriptionDefault: "Agent Client Protocol — run and manage AI agents"
```

For `chatbooks-playground`:
```typescript
descriptionKey: "option:header.chatbooksPlaygroundDesc",
descriptionDefault: "Export and import chat sessions as portable bundles"
```

For `world-books`:
```typescript
descriptionKey: "option:header.modeWorldBooksDesc",
descriptionDefault: "Shared lore and context injected into character chats"
```

For `deep-research`:
```typescript
descriptionKey: "option:header.deepResearchDesc",
descriptionDefault: "Long-running research with citations and checkpoints"
```

For `workspace-playground`:
```typescript
descriptionKey: "settings:researchStudioDesc",
descriptionDefault: "Three-pane workspace: sources, chat, and generated outputs"
```

For `prompt-studio`:
```typescript
descriptionKey: "option:header.modePromptStudioDesc",
descriptionDefault: "Design, test, and optimize prompts across models"
```

For `model-playground`:
```typescript
descriptionKey: "settings:modelPlaygroundDesc",
descriptionDefault: "Compare model outputs side by side"
```

For `chat-dictionaries`:
```typescript
descriptionKey: "option:header.modeDictionariesDesc",
descriptionDefault: "Custom word lists for pronunciation and spelling"
```

For `evaluations`:
```typescript
descriptionKey: "option:header.evaluationsDesc",
descriptionDefault: "Score and benchmark model quality with automated tests"
```

For `repo2txt`:
```typescript
descriptionKey: "option:repo2txt.desc",
descriptionDefault: "Convert code repositories into text for ingestion"
```

**Step 5: Run test to verify descriptions pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts`

Expected: 15 PASS

**Step 6: Update the renderer to display descriptions**

In `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx`, in the two-panel item renderer (around line 453-501), add a description line after the label span. Find the label span:

```typescript
<span className="min-w-0 flex-1 truncate">{ri.label}</span>
```

Wrap it and add the description below:

```typescript
<span className="min-w-0 flex-1">
  <span className="truncate block">{ri.label}</span>
  {ri.item.descriptionDefault && (
    <span className="block truncate text-xs text-text-subtle/70">
      {t(ri.item.descriptionKey ?? "", ri.item.descriptionDefault)}
    </span>
  )}
</span>
```

Ensure `useTranslation` is already imported (it should be — check and add `const { t } = useTranslation()` if not present in the component).

**Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx apps/packages/ui/src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts
git commit -m "feat: add plain-language descriptions to jargon-heavy nav items

Adds descriptionDefault field to HeaderShortcutItem type and populates
it for 15 items (STT, TTS, RAG, ACP, etc). Descriptions render as
subtitles in the header shortcut launcher."
```

---

## Task 5: Elevate moderation + family guardrails in navigation

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` (regroup)
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts` (remove beta, distinct icons)

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Layouts/__tests__/header-shortcut-safety-group.test.ts`

```typescript
import { describe, expect, it } from "vitest"
import { getHeaderShortcutGroups } from "../header-shortcut-items"

describe("header shortcut safety group", () => {
  const groups = getHeaderShortcutGroups()

  it("has a 'safety' group", () => {
    const safety = groups.find((g) => g.id === "safety")
    expect(safety).toBeDefined()
    expect(safety!.titleDefault).toBe("Safety")
  })

  it("safety group contains family-guardrails, content-controls, and guardian", () => {
    const safety = groups.find((g) => g.id === "safety")!
    const ids = safety.items.map((i) => i.id)
    expect(ids).toContain("family-guardrails")
    expect(ids).toContain("content-controls")
    expect(ids).toContain("guardian")
  })

  it("moderation-playground is no longer in the tools group", () => {
    const tools = groups.find((g) => g.id === "tools")
    if (tools) {
      const ids = tools.items.map((i) => i.id)
      expect(ids).not.toContain("moderation-playground")
    }
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/header-shortcut-safety-group.test.ts`

Expected: FAIL — no "safety" group exists

**Step 3: Implement navigation changes**

In `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`:

a) Add new imports at the top (alongside existing lucide imports):
```typescript
import { Eye, Users } from "lucide-react"
```

b) Remove the `moderation-playground` item from the "tools" group (lines 352-359).

c) Add a new "safety" group. Insert it after the "library" group (after line 213) and before "creation" (line 216):

```typescript
{
  id: "safety",
  titleKey: "option:header.groupSafety",
  titleDefault: "Safety",
  items: [
    {
      id: "family-guardrails",
      to: "/settings/family-guardrails",
      icon: Users,
      labelKey: "settings:familyGuardrailsWizardNav",
      labelDefault: "Family Guardrails",
      descriptionKey: "settings:familyGuardrailsWizardDesc",
      descriptionDefault: "Set up family profiles, safety templates, and invite guardians"
    },
    {
      id: "content-controls",
      to: "/moderation-playground",
      icon: ShieldCheck,
      labelKey: "option:moderationPlayground.nav",
      labelDefault: "Content Controls",
      descriptionKey: "option:moderationPlayground.desc",
      descriptionDefault: "Content safety rules, blocklists, and testing"
    },
    {
      id: "guardian",
      to: "/settings/guardian",
      icon: Eye,
      labelKey: "settings:guardianNav",
      labelDefault: "Guardian",
      descriptionKey: "settings:guardianDesc",
      descriptionDefault: "Monitor and manage dependent account activity"
    }
  ]
},
```

d) In `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`:

Remove `beta: true` from the family-guardrails entry (line 213) and the guardian entry (line 221).

Change the family-guardrails icon from `ShieldCheck` to `Users` (add `Users` to the lucide import).

Change the guardian icon from `ShieldCheck` to `Eye` (add `Eye` to the lucide import).

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/header-shortcut-safety-group.test.ts`

Expected: 3 PASS

**Step 5: Run existing nav tests to check for regressions**

Run: `cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/`

Expected: All existing tests PASS (some may need updating if they assert moderation is in "tools")

**Step 6: Fix any broken existing tests**

Check `settings-nav.moderation.test.ts` and `settings-nav.guardian.test.ts` — update assertions if they check for specific group membership or beta flags.

**Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/packages/ui/src/components/Layouts/__tests__/
git commit -m "feat: create Safety nav group with Family Guardrails, Content Controls, Guardian

Moves moderation out of Tools into a dedicated Safety group. Gives each
safety item a distinct icon (Users, ShieldCheck, Eye). Removes beta tag
from Family Guardrails."
```

---

## Task 6: Add Joyride tutorial for moderation

**Files:**
- Create: `apps/packages/ui/src/tutorials/definitions/moderation.ts`
- Modify: `apps/packages/ui/src/tutorials/registry.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/tutorials.json`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.tutorial.test.tsx`

```typescript
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell tutorial integration", () => {
  it("calls startTutorial with moderation-basics", () => {
    const source = readShellSource()
    expect(source).toContain('startTutorial("moderation-basics")')
  })

  it("checks MODERATION_ONBOARDING_KEY before starting tutorial", () => {
    const source = readShellSource()
    expect(source).toContain("ONBOARDING_KEY")
    expect(source).toContain("startTutorial")
  })
})

describe("moderation tutorial definition", () => {
  it("exists in the tutorial registry", async () => {
    const { TUTORIAL_REGISTRY } = await import("@/tutorials/registry")
    const modTutorial = TUTORIAL_REGISTRY.find((t) => t.id === "moderation-basics")
    expect(modTutorial).toBeDefined()
    expect(modTutorial!.routePattern).toBe("/moderation-playground")
    expect(modTutorial!.steps.length).toBeGreaterThanOrEqual(4)
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.tutorial.test.tsx`

Expected: FAIL — no `startTutorial` call in shell, no tutorial definition

**Step 3: Create the tutorial definition**

Create `apps/packages/ui/src/tutorials/definitions/moderation.ts`:

```typescript
import { ShieldCheck } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const moderationBasics: TutorialDefinition = {
  id: "moderation-basics",
  routePattern: "/moderation-playground",
  labelKey: "tutorials:moderation.basics.label",
  labelFallback: "Content Controls Basics",
  descriptionKey: "tutorials:moderation.basics.description",
  descriptionFallback:
    "Learn how to set up content safety rules, blocklists, and test moderation.",
  icon: ShieldCheck,
  priority: 1,
  steps: [
    {
      target: '[data-testid="moderation-hero"]',
      titleKey: "tutorials:moderation.basics.heroTitle",
      titleFallback: "Moderation Dashboard",
      contentKey: "tutorials:moderation.basics.heroContent",
      contentFallback:
        "This is your content safety hub. The status badge shows whether your server is connected.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="moderation-tab-policy"]',
      titleKey: "tutorials:moderation.basics.policyTitle",
      titleFallback: "Policy & Settings",
      contentKey: "tutorials:moderation.basics.policyContent",
      contentFallback:
        "Start here. Set your base content safety policy — what categories to filter and how strictly.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="moderation-tab-blocklist"]',
      titleKey: "tutorials:moderation.basics.blocklistTitle",
      titleFallback: "Blocklist Studio",
      contentKey: "tutorials:moderation.basics.blocklistContent",
      contentFallback:
        "Add specific words, phrases, or patterns you want to always block or flag.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="moderation-tab-test"]',
      titleKey: "tutorials:moderation.basics.testTitle",
      titleFallback: "Test Sandbox",
      contentKey: "tutorials:moderation.basics.testContent",
      contentFallback:
        "Try your rules in real time. Type a message and see whether it would be allowed or blocked.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="moderation-family-guardrails-link"]',
      titleKey: "tutorials:moderation.basics.guardrailsTitle",
      titleFallback: "Family Guardrails",
      contentKey: "tutorials:moderation.basics.guardrailsContent",
      contentFallback:
        "Setting up for a family? The Family Guardrails wizard walks you through creating profiles for each family member.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const moderationTutorials: TutorialDefinition[] = [moderationBasics]
```

**Step 4: Register the tutorial**

In `apps/packages/ui/src/tutorials/registry.ts`, add the import and spread:

Add import:
```typescript
import { moderationTutorials } from "./definitions/moderation"
```

Add to `TUTORIAL_REGISTRY` array:
```typescript
export const TUTORIAL_REGISTRY: TutorialDefinition[] = [
  ...gettingStartedTutorials,
  ...playgroundTutorials,
  ...workspacePlaygroundTutorials,
  ...moderationTutorials,
  ...mediaTutorials,
  // ... rest unchanged
]
```

**Step 5: Add i18n strings**

In `apps/packages/ui/src/assets/locale/en/tutorials.json`, add a `moderation` section alongside existing features:

```json
"moderation": {
  "basics": {
    "label": "Content Controls Basics",
    "description": "Learn how to set up content safety rules, blocklists, and test moderation.",
    "heroTitle": "Moderation Dashboard",
    "heroContent": "This is your content safety hub. The status badge shows whether your server is connected.",
    "policyTitle": "Policy & Settings",
    "policyContent": "Start here. Set your base content safety policy — what categories to filter and how strictly.",
    "blocklistTitle": "Blocklist Studio",
    "blocklistContent": "Add specific words, phrases, or patterns you want to always block or flag.",
    "testTitle": "Test Sandbox",
    "testContent": "Try your rules in real time. Type a message and see whether it would be allowed or blocked.",
    "guardrailsTitle": "Family Guardrails",
    "guardrailsContent": "Setting up for a family? The Family Guardrails wizard walks you through creating profiles for each family member."
  }
}
```

**Step 6: Add data-testid targets and tutorial trigger to ModerationPlaygroundShell**

In `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`:

a) Add imports:
```typescript
import { useTutorialStore } from "@/store/tutorials"
```

b) In the component body (around line 52), add:
```typescript
const startTutorial = useTutorialStore((s) => s.startTutorial)
const tutorialInitializedRef = React.useRef(false)
```

c) Add a useEffect for auto-starting the tutorial (after the existing `showOnboarding` state, around line 76):
```typescript
React.useEffect(() => {
  if (tutorialInitializedRef.current) return
  if (hasPermissionError) return
  tutorialInitializedRef.current = true
  if (typeof window === "undefined") return
  try {
    const dismissed = window.localStorage.getItem(ONBOARDING_KEY)
    if (!dismissed) {
      startTutorial("moderation-basics")
    }
  } catch {
    // On storage error, skip tutorial
  }
}, [hasPermissionError, startTutorial])
```

d) Add `data-testid` to the hero section (line ~267):
```typescript
<div
  data-testid="moderation-hero"
  className="relative overflow-hidden rounded-[28px] ..."
```

e) Add `data-testid` to tab buttons in the tab bar. In the TABS map (around line ~305), add testid per tab:
```typescript
{TABS.map((tab) => (
  <button
    key={tab.key}
    data-testid={`moderation-tab-${tab.key}`}
    type="button"
    ...
```

**Step 7: Run tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.tutorial.test.tsx`

Expected: All PASS

**Step 8: Run existing moderation tests for regressions**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/`

Expected: All PASS

**Step 9: Commit**

```bash
git add apps/packages/ui/src/tutorials/definitions/moderation.ts apps/packages/ui/src/tutorials/registry.ts apps/packages/ui/src/assets/locale/en/tutorials.json apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.tutorial.test.tsx
git commit -m "feat: add Joyride tutorial for moderation page

Creates a 5-step guided tour covering the hero, policy tab, blocklist,
test sandbox, and family guardrails link. Auto-triggers on first visit
using the existing tutorial registry infrastructure."
```

---

## Task 7: Improve moderation onboarding card

Replace the dismissible banner with a structured onboarding card that links to Family Guardrails Wizard.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx` (lines 248-263)

**Step 1: Write the failing test**

Create: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.onboarding-card.test.ts`

```typescript
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell onboarding card", () => {
  it("links to the Family Guardrails Wizard", () => {
    const source = readShellSource()
    expect(source).toContain("/settings/family-guardrails")
  })

  it("has a data-testid for the family guardrails link", () => {
    const source = readShellSource()
    expect(source).toContain('data-testid="moderation-family-guardrails-link"')
  })

  it("shows recommended tab order guidance", () => {
    const source = readShellSource()
    // Should indicate which tab to start with
    expect(source).toContain("Start here")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.onboarding-card.test.ts`

Expected: FAIL — no link to family-guardrails, no testid, no "Start here"

**Step 3: Implement the onboarding card**

In `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`, replace the onboarding section (lines 248-263):

Replace:
```typescript
{showOnboarding && (
  <div className="mx-4 sm:mx-6 lg:mx-8 mb-4 p-4 border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
    <p className="text-sm font-medium">Welcome to Moderation Playground</p>
    <p className="text-sm text-text-muted mt-1">
      Configure content safety rules, test them live, and manage per-user overrides.
    </p>
    <button
      type="button"
      onClick={dismissOnboarding}
      className="text-sm text-blue-600 hover:underline mt-2"
    >
      Got it, let&apos;s start
    </button>
  </div>
)}
```

With:
```typescript
{showOnboarding && (
  <div className="mx-4 sm:mx-6 lg:mx-8 mb-4 p-5 border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
    <p className="text-base font-semibold">
      {t("option:moderationPlayground.onboarding.title", "Welcome to Content Controls")}
    </p>
    <p className="text-sm text-text-muted mt-1">
      {t(
        "option:moderationPlayground.onboarding.description",
        "Set up content safety rules to protect your family or enforce server guardrails."
      )}
    </p>

    <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
      <button
        type="button"
        data-testid="moderation-family-guardrails-link"
        onClick={() => navigate("/settings/family-guardrails")}
        className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90"
      >
        {t("option:moderationPlayground.onboarding.guardrailsCta", "Set up Family Guardrails")}
      </button>
      <button
        type="button"
        onClick={dismissOnboarding}
        className="text-sm text-text-muted hover:text-text transition"
      >
        {t("option:moderationPlayground.onboarding.dismiss", "Skip — I'll explore on my own")}
      </button>
    </div>

    <p className="mt-3 text-xs text-text-muted">
      {t(
        "option:moderationPlayground.onboarding.tabHint",
        "Start here: Policy & Settings tab sets your base rules. Then test them in the Test Sandbox."
      )}
    </p>
  </div>
)}
```

Also add a "Start here" badge to the Policy tab in the tab bar. In the TABS map, add a conditional indicator:

```typescript
{TABS.map((tab) => (
  <button
    key={tab.key}
    data-testid={`moderation-tab-${tab.key}`}
    type="button"
    role="tab"
    aria-selected={activeTab === tab.key}
    onClick={() => setActiveTab(tab.key)}
    className={cn(
      "whitespace-nowrap border-b-2 px-4 py-3 text-sm font-medium transition-colors",
      activeTab === tab.key
        ? "border-primary text-primary"
        : "border-transparent text-text-muted hover:text-text hover:border-border"
    )}
  >
    {tab.label}
    {tab.key === "policy" && showOnboarding && (
      <span className="ml-1.5 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
        Start here
      </span>
    )}
  </button>
))}
```

**Step 4: Run tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.onboarding-card.test.ts`

Expected: 3 PASS

**Step 5: Run all moderation tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/ModerationPlayground/__tests__/`

Expected: All PASS

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.onboarding-card.test.ts
git commit -m "feat: replace moderation onboarding banner with structured card

New onboarding card links directly to Family Guardrails wizard as the
primary CTA. Adds 'Start here' badge to Policy tab. Replaces generic
'Got it' dismiss with 'Skip — I'll explore on my own'."
```

---

## Task 8: Add intent selector on onboarding success screen

This is the highest-leverage change — routing users to their persona-appropriate next step.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx` (success screen ~lines 899-1226)

**Step 1: Update the guard test to expect the intent selector**

In `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`, add:

```typescript
it("includes intent selector cards on the success screen", () => {
  const source = readOnboardingSource()
  expect(source).toContain('data-testid="intent-selector"')
  expect(source).toContain("/settings/family-guardrails")
  expect(source).toContain("handleOpenChatFlow")
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`

Expected: FAIL — no `intent-selector` testid yet

**Step 3: Implement the intent selector**

In `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`, find the success screen container (around line 899 where `showSuccess` is checked).

Add these imports at the top of the file (alongside existing lucide imports):
```typescript
import { MessageSquare, Shield, BookOpen } from "lucide-react"
```

Inside the success screen container, before the "Set your defaults" section (around line 934), insert the intent selector:

```typescript
{/* Intent selector — route users to persona-appropriate next step */}
<div data-testid="intent-selector" className="mt-6 mb-4">
  <p className="text-sm font-medium text-text-muted mb-3">
    {t("onboarding:success.intentTitle", {
      defaultValue: "What would you like to do first?"
    })}
  </p>
  <div className="grid gap-3 sm:grid-cols-3">
    <button
      type="button"
      onClick={handleOpenChatFlow}
      className="flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors hover:border-primary/50 hover:bg-surface2"
    >
      <MessageSquare className="h-5 w-5 text-primary" />
      <span className="text-sm font-medium text-text">
        {t("onboarding:success.intentChat", {
          defaultValue: "Chat with AI"
        })}
      </span>
      <span className="text-xs text-text-muted">
        {t("onboarding:success.intentChatDesc", {
          defaultValue: "Start a conversation with your configured models."
        })}
      </span>
    </button>

    <button
      type="button"
      onClick={() => finishAndNavigate("/settings/family-guardrails")}
      className="flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors hover:border-primary/50 hover:bg-surface2"
    >
      <Shield className="h-5 w-5 text-primary" />
      <span className="text-sm font-medium text-text">
        {t("onboarding:success.intentFamily", {
          defaultValue: "Set up family safety"
        })}
      </span>
      <span className="text-xs text-text-muted">
        {t("onboarding:success.intentFamilyDesc", {
          defaultValue: "Create family profiles and content safety rules."
        })}
      </span>
    </button>

    <button
      type="button"
      onClick={() => {
        handleOpenIngestFlow()
      }}
      className="flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors hover:border-primary/50 hover:bg-surface2"
    >
      <BookOpen className="h-5 w-5 text-primary" />
      <span className="text-sm font-medium text-text">
        {t("onboarding:success.intentResearch", {
          defaultValue: "Research my documents"
        })}
      </span>
      <span className="text-xs text-text-muted">
        {t("onboarding:success.intentResearchDesc", {
          defaultValue: "Import documents and ask questions about them."
        })}
      </span>
    </button>
  </div>
</div>
```

Note: `finishAndNavigate` and `handleOpenIngestFlow` and `handleOpenChatFlow` are already defined in the component. Verify that `finishAndNavigate` can accept the guardrails path — it should be a generic navigate function. If it doesn't exist or only accepts specific routes, use `navigate("/settings/family-guardrails")` directly after calling the success completion logic.

**Step 4: Run guard tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`

Expected: All PASS including the new intent-selector test

**Step 5: Run existing onboarding tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/Onboarding/__tests__/`

Expected: All PASS

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts
git commit -m "feat: add intent selector to onboarding success screen

After successful connection, shows three cards: 'Chat with AI', 'Set up
family safety' (links to Family Guardrails wizard), and 'Research my
documents' (opens Quick Ingest). Bridges the gap between 'connected' and
'productive' for all personas."
```

---

## Final Verification

After all 8 tasks are complete:

**Step 1: Run the full test suite**

```bash
cd apps/packages/ui && npx vitest run
```

Expected: All tests PASS with no regressions.

**Step 2: Manual smoke test**

1. Clear localStorage in browser
2. Navigate to `/` — verify onboarding wizard appears
3. Complete connection — verify intent selector appears with 3 cards
4. Click "Set up family safety" — verify it navigates to Family Guardrails
5. Navigate to `/moderation-playground` — verify new onboarding card with "Set up Family Guardrails" CTA
6. Verify Joyride tutorial auto-starts on first moderation visit
7. Open header shortcuts (Cmd+K) — verify "Safety" group exists with distinct icons
8. Verify jargon items (STT, TTS, etc.) show descriptions
9. Navigate to `/chat` without connection — verify "Open Settings" button
10. Verify skeleton loaders appear briefly when switching moderation tabs

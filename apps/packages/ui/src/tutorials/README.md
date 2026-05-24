# Tutorial System

Per-page guided tours using React Joyride. For comprehensive documentation, see [Tutorial System Developer Guide](../../../../../Docs/Code_Documentation/Tutorial_System_Developer_Guide.md).

## Quick Start: Adding a Tutorial

### Checklist

- [ ] Create definition file in `definitions/`
- [ ] Register in `registry.ts`
- [ ] Add i18n strings to `locale/en/tutorials.json`
- [ ] Add `data-testid` attributes to target elements
- [ ] Test on target page (press `?` to open Help Modal)

## Directory Structure

```
tutorials/
├── index.ts              # Public exports
├── registry.ts           # Central registry and types
├── definitions/          # Tutorial definitions by page
│   ├── playground.ts
│   ├── research-workspace.ts
│   ├── media.ts
│   ├── knowledge.ts
│   ├── characters.ts
│   ├── prompts.ts
│   ├── evaluations.ts
│   ├── notes.ts
│   ├── flashcards.ts
│   └── world-books.ts
└── README.md             # This file
```

## API Reference

### Types

```typescript
interface TutorialStep {
  target: string           // CSS selector (prefer data-testid)
  titleKey: string         // i18n key for title
  titleFallback: string    // Fallback text
  contentKey: string       // i18n key for content
  contentFallback: string  // Fallback text
  placement?: Placement    // "auto" | "top" | "bottom" | "left" | "right"
  disableBeacon?: boolean  // Hide pulsing dot (default: false)
  spotlightClicks?: boolean // Allow target clicks (default: false)
  isFixed?: boolean        // Fixed position target (default: false)
}

interface TutorialDefinition {
  id: string               // Unique ID (e.g., "playground-basics")
  routePattern: string     // Canonical route (e.g., "/chat")
  labelKey: string         // i18n key for name
  labelFallback: string
  descriptionKey: string   // i18n key for description
  descriptionFallback: string
  icon?: LucideIcon        // Display icon
  steps: TutorialStep[]
  prerequisites?: string[] // Required tutorial IDs
  priority?: number        // Sort order (lower = higher priority)
}
```

### Registry Functions

```typescript
// Get all tutorials for a route
getTutorialsForRoute(pathname: string): TutorialDefinition[]

// Get tutorial by ID
getTutorialById(tutorialId: string): TutorialDefinition | undefined

// Get primary tutorial for first-visit prompt
getPrimaryTutorialForRoute(pathname: string): TutorialDefinition | undefined

// Check if route has tutorials
hasTutorialsForRoute(pathname: string): boolean

// Get count of tutorials for route
getTutorialCountForRoute(pathname: string): number
```

### Store Hooks

```typescript
import { useTutorialStore, useActiveTutorial, useHelpModal } from "@/store/tutorials"

// Full store access
const store = useTutorialStore()

// Active tutorial controls
const {
  activeTutorialId,
  activeStepIndex,
  startTutorial,
  endTutorial,
  setStepIndex,
  markComplete
} = useActiveTutorial()

// Help modal controls
const { isOpen, open, close, toggle } = useHelpModal()
```

## Example: Adding a Page Tutorial

### 1. Create Definition (`definitions/my-page.ts`)

```typescript
import { FileText } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const myPageBasics: TutorialDefinition = {
  id: "my-page-basics",
  routePattern: "/my-page",
  labelKey: "tutorials:myPage.basics.label",
  labelFallback: "My Page Basics",
  descriptionKey: "tutorials:myPage.basics.description",
  descriptionFallback: "Learn how to use My Page",
  icon: FileText,
  priority: 1,
  steps: [
    {
      target: '[data-testid="my-page-header"]',
      titleKey: "tutorials:myPage.basics.headerTitle",
      titleFallback: "Welcome",
      contentKey: "tutorials:myPage.basics.headerContent",
      contentFallback: "This is the main header area.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="my-page-action"]',
      titleKey: "tutorials:myPage.basics.actionTitle",
      titleFallback: "Main Action",
      contentKey: "tutorials:myPage.basics.actionContent",
      contentFallback: "Click here to get started.",
      placement: "top"
    }
  ]
}

export const myPageTutorials: TutorialDefinition[] = [myPageBasics]
```

### 2. Register (`registry.ts`)

```typescript
import { myPageTutorials } from "./definitions/my-page"

export const TUTORIAL_REGISTRY: TutorialDefinition[] = [
  ...playgroundTutorials,
  ...myPageTutorials
]
```

### 3. Add i18n (`locale/en/tutorials.json`)

```json
{
  "myPage": {
    "basics": {
      "label": "My Page Basics",
      "description": "Learn how to use My Page",
      "headerTitle": "Welcome",
      "headerContent": "This is the main header area.",
      "actionTitle": "Main Action",
      "actionContent": "Click here to get started."
    }
  }
}
```

### 4. Add Target Attributes (Component)

```tsx
<header data-testid="my-page-header">
  <h1>My Page</h1>
</header>

<button data-testid="my-page-action">
  Get Started
</button>
```

## Route Patterns

| Pattern | Matches |
|---------|---------|
| `/my-page` | Exact path only |
| `/settings/*` | Any settings subpath |
| `/chat` | Canonical chat path |

## Debugging

```javascript
// Browser console
window.__tldw_useTutorialStore.getState()           // View state
window.__tldw_useTutorialStore.getState().resetProgress()  // Reset
window.__tldw_useTutorialStore.getState().startTutorial('my-page-basics')
```

Clear `localStorage.removeItem('tldw-tutorials')` to reset all progress.

# Writing Suite Phase 4: Live Feedback — Mood Detection + Echo Chamber

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real-time AI feedback while writing — mood detection (colored indicator), Echo Chamber (5 simulated reader reactions), and AI annotation marks in the TipTap editor.

**Architecture:** Frontend-only phase. A `useWritingFeedback` hook manages debounced LLM calls, opt-in state, and caching. A `FeedbackTab` inspector tab displays mood and reader reactions. An `AIAnnotationExtension` TipTap mark highlights AI-analyzed passages. All LLM calls go through the existing chat completions API.

**Tech Stack:** React hooks, Zustand (persistent storage), TipTap Mark extension, `/api/v1/chat/completions` API

---

## Task 1: useWritingFeedback Hook

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts`

Core hook managing mood detection and Echo Chamber state:

```typescript
import { useCallback, useEffect, useRef, useState } from "react"
import { useStorage } from "@plasmohq/storage/hook"

type Mood = "tense" | "romantic" | "melancholic" | "action" | "calm" | "mysterious" | "humorous" | null

type EchoReaction = {
  persona: string
  emoji: string
  message: string
  timestamp: number
}

type UseWritingFeedbackProps = {
  editorText: string
  isOnline: boolean
  isGenerating: boolean
  selectedModel?: string
}

type UseWritingFeedbackReturn = {
  // Mood
  moodEnabled: boolean
  setMoodEnabled: (enabled: boolean) => void
  currentMood: Mood
  moodAnalyzing: boolean
  // Echo Chamber
  echoEnabled: boolean
  setEchoEnabled: (enabled: boolean) => void
  echoReactions: EchoReaction[]
  echoAnalyzing: boolean
  // Stats
  charsSinceLastEcho: number
}
```

**Key behavior:**
- **Mood detection**: When `moodEnabled` and text changes, debounce 10s, then send last 500 chars to `/api/v1/chat/completions` with mood classification prompt. Rate limit: max 1 call per 10s. Store result in `currentMood`.
- **Echo Chamber**: When `echoEnabled` and `charsSinceLastEcho >= 500`, send last 1000 chars to chat API with personality-specific prompts. Rate limit: max 1 call per 30s. Append reaction to `echoReactions` array.
- **Opt-in toggles**: Persisted via `useStorage("writing:mood-enabled", false)` and `useStorage("writing:echo-enabled", false)`.
- **Don't run during generation** (`isGenerating`).

**Mood prompt:**
```
Classify the emotional mood of this text passage. Respond with ONLY one word from: tense, romantic, melancholic, action, calm, mysterious, humorous

Text: {last500chars}
```

**Echo Chamber personas and prompts:**
```typescript
const ECHO_PERSONAS = [
  { name: "Alex", emoji: "🧐", role: "The Analyst", prompt: "You are Alex, a sharp literary analyst. In 1-2 sentences, comment on the structure, foreshadowing, or plot mechanics of this passage. Be concise and insightful." },
  { name: "Sam", emoji: "😍", role: "The Shipper", prompt: "You are Sam, obsessed with character relationships and chemistry. In 1-2 sentences, react to any relationship dynamics, tension, or romantic potential in this passage." },
  { name: "Max", emoji: "🤨", role: "The Skeptic", prompt: "You are Max, a skeptical reader who questions motivations and calls out plot conveniences. In 1-2 sentences, point out anything that feels contrived or unmotivated." },
  { name: "Riley", emoji: "🎉", role: "The Hype", prompt: "You are Riley, an enthusiastic reader who gets excited about action and twists. In 1-2 sentences, react with energy to the most exciting or surprising element." },
  { name: "Jordan", emoji: "📚", role: "The Lore Keeper", prompt: "You are Jordan, a world-building enthusiast who tracks history, locations, and lore. In 1-2 sentences, comment on any world-building details, consistency, or missed opportunities." },
]
```

On each Echo trigger, pick one random persona (cycling through all 5), send their prompt + text, append to reactions.

**Commit:**
```bash
git commit -m "feat(writing): add useWritingFeedback hook with mood detection and Echo Chamber"
```

---

## Task 2: FeedbackTab Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/FeedbackTab.tsx`

Inspector tab with two sections:

```typescript
import { Divider, Empty, List, Segmented, Switch, Tag, Typography } from "antd"

// Mood section:
// - Toggle switch: "Mood Detection" [on/off]
// - When on: colored mood indicator badge
// - When analyzing: small spinner

// Echo Chamber section:
// - Toggle switch: "Echo Chamber" [on/off]  
// - When on: scrollable list of reactions
// - Each reaction: persona emoji + name + message + relative timestamp
// - Empty state: "Write 500+ characters to trigger reader reactions"
```

**Mood colors:**
```typescript
const MOOD_COLORS: Record<string, string> = {
  tense: "#ff4d4f",
  romantic: "#ff85c0",
  melancholic: "#597ef7",
  action: "#fa8c16",
  calm: "#52c41a",
  mysterious: "#722ed1",
  humorous: "#fadb14",
}
```

**Commit:**
```bash
git commit -m "feat(writing): add FeedbackTab with mood indicator and Echo Chamber display"
```

---

## Task 3: AIAnnotationExtension for TipTap

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/extensions/AIAnnotationExtension.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`

A TipTap Mark that highlights AI-generated or AI-suggested text:

```typescript
import { Mark, mergeAttributes } from "@tiptap/core"

export const AIAnnotationExtension = Mark.create({
  name: "aiAnnotation",

  addAttributes() {
    return {
      type: { default: "generated" }, // "generated" | "suggestion" | "feedback"
      confidence: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: "span[data-ai-annotation]" }]
  },

  renderHTML({ HTMLAttributes }) {
    const colors = {
      generated: "rgba(147, 51, 234, 0.1)",   // purple tint
      suggestion: "rgba(59, 130, 246, 0.1)",   // blue tint
      feedback: "rgba(250, 204, 21, 0.1)",     // yellow tint
    }
    const bg = colors[HTMLAttributes.type as keyof typeof colors] || colors.generated
    return [
      "span",
      mergeAttributes(HTMLAttributes, {
        "data-ai-annotation": HTMLAttributes.type,
        style: `background: ${bg}; border-radius: 2px;`,
        title: `AI ${HTMLAttributes.type}`,
      }),
      0,
    ]
  },
})
```

Register in WritingTipTapEditor.tsx's extensions array.

**Commit:**
```bash
git commit -m "feat(writing): add AIAnnotationExtension for TipTap"
```

---

## Task 4: Wire FeedbackTab + Mood Indicator into WritingPlayground

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`

### 4a: Add "feedback" to InspectorTabKey
```typescript
export type InspectorTabKey = "sampling" | "context" | "setup" | "inspect" | "characters" | "research" | "agent" | "feedback"
```

### 4b: Add tab to InspectorPanel
Add to TAB_DEFINITIONS, props, panelMap.

### 4c: Wire in index.tsx
- Import `FeedbackTab` and `useWritingFeedback`
- Call the hook: `const feedback = useWritingFeedback({ editorText, isOnline, isGenerating, selectedModel })`
- Create tab content: `const feedbackTabContent = <FeedbackTab {...feedback} />`
- Pass to inspector: `feedback={feedbackTabContent}`
- Add mood indicator to status bar (near token count): show colored dot when mood is detected
- Add tab label: `feedback: t("option:writingPlayground.sidebarFeedback", "Feedback")`

### 4d: Mood indicator in status bar
In the status bar area of index.tsx (near the generation stats), add:
```typescript
{feedback.moodEnabled && feedback.currentMood && (
  <Tag color={MOOD_COLORS[feedback.currentMood]} className="!text-xs !m-0">
    {feedback.currentMood}
  </Tag>
)}
```

**Commit:**
```bash
git commit -m "feat(writing): wire FeedbackTab, mood indicator, and 8-tab inspector into WritingPlayground"
```

---

## Verification Checklist

1. Inspector panel shows 8 tabs (sampling, context, setup, analysis, characters, research, agent, feedback)
2. Feedback tab has mood toggle and Echo Chamber toggle (both off by default)
3. Enabling mood detection shows colored mood badge in status bar after 10s of typing
4. Enabling Echo Chamber shows reader reactions after 500+ characters
5. AI annotations render as colored highlights in TipTap editor
6. Toggles persist across page reloads (stored in extension/browser storage)
7. No calls made during generation (`isGenerating`)
8. All existing frontend tests still pass

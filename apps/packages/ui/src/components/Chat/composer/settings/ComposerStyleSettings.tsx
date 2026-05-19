import React from "react"
import { useComposerVariantPreference } from "../hooks/useComposerVariantPreference"
import {
  COMPOSER_ENABLED_PREFERENCE_KEY,
  useComposerEnabledPreference,
} from "../hooks/useComposerEnabledPreference"
import type { ChatComposerVariant } from "../types"

/**
 * Re-export for callers that only need the storage key (PlaygroundForm
 * + Sidepanel form.tsx synchronously read localStorage before the hook
 * runs, so they don't pull in the full React hook).
 */
export const COMPOSER_ENABLED_STORAGE_KEY = COMPOSER_ENABLED_PREFERENCE_KEY

/**
 * Settings section for the chat composer variant preference. Renders
 * three radio-style cards (V1 / V3 / V5). Selection persists via
 * `useComposerVariantPreference()` — writes localStorage immediately
 * (no flicker) and fires a fire-and-forget PATCH to
 * `/api/v1/users/me/profile` so the choice follows the user across
 * surfaces (web + extension) and devices.
 *
 * Mounts inside the existing Chat settings page. The Playground
 * (`/chat`) and Sidepanel (`/__debug__/sidepanel-chat`) read the
 * same hook to drive `<ChatComposer variant=...>`, so picking V3
 * here re-renders both surfaces under V3 without further wiring.
 */

interface VariantOption {
  id: ChatComposerVariant
  name: string
  tag: string
  description: string
  /** Mini structural summary for the card body. */
  highlights: string[]
}

const VARIANTS: VariantOption[] = [
  {
    id: "v1",
    name: "Terminal Stack",
    tag: "Default",
    description:
      "A cleaned-up take on today's composer — visible source chip above the textarea, controls docked below. Closest to the familiar layout.",
    highlights: [
      "Source chip + toggle pills",
      "`>_` caret cyan",
      "Model + temp + OCR pills on bottom",
    ],
  },
  {
    id: "v3",
    name: "Split Brief",
    tag: "Structured",
    description:
      "Left pane is the brief (persona, sources, model, temp) as labelled field chips. Right pane is the question. Mirrors how researchers actually compose a prompt.",
    highlights: [
      "240px brief panel with labelled fields",
      "Token meter with cost estimate",
      "@ mentions + / slash commands inline",
    ],
  },
  {
    id: "v5",
    name: "Radial Command",
    tag: "Palette-first",
    description:
      "Everything collapses to a single line with a ⌘K cap. Typing `/` opens a full command palette that surfaces every composer capability — models, personas, prompts, tools.",
    highlights: [
      "Single-line pill composer",
      "Faceted meta row above",
      "Inline ⌘K command palette",
    ],
  },
]

export const ComposerStyleSettings: React.FC = () => {
  const [variant, setVariant] = useComposerVariantPreference()
  const [enabled, setEnabled] = useComposerEnabledPreference()
  const handleEnabledChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setEnabled(event.target.checked)
    },
    [setEnabled]
  )

  return (
    <section
      aria-label="Composer style"
      className="py-6"
      data-testid="composer-style-settings"
    >
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-text mb-1.5">
          Composer style
        </h3>
        <p className="text-sm text-text-muted leading-relaxed">
          Pick a layout for the chat composer. Your selection applies to the
          main chat and the extension sidepanel; individual features
          (slash commands, mentions, voice, attachments) work the same in
          all three.
        </p>

        <div className="mt-3 flex items-center gap-3 rounded-md border border-border/60 bg-surface2/40 px-3 py-2">
          <label className="flex items-center gap-2 cursor-pointer flex-1">
            <input
              type="checkbox"
              checked={enabled}
              onChange={handleEnabledChange}
              data-testid="composer-enabled-toggle"
              className="h-4 w-4 rounded border-border accent-primary"
              aria-describedby="composer-enabled-hint"
            />
            <span className="text-sm text-text font-medium">
              Enable new composer
            </span>
          </label>
          <span
            id="composer-enabled-hint"
            className="text-[11px] text-text-subtle"
          >
            Experimental · equivalent to{" "}
            <span className="font-mono text-primary">?nextgenComposer=1</span>
          </span>
        </div>
      </div>

      <div
        role="radiogroup"
        aria-label="Composer variant"
        className="grid grid-cols-1 md:grid-cols-3 gap-3"
      >
        {VARIANTS.map((opt) => {
          const selected = opt.id === variant
          return (
            <button
              key={opt.id}
              type="button"
              role="radio"
              aria-checked={selected}
              tabIndex={0}
              onClick={() => setVariant(opt.id)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" ||
                  event.key === " " ||
                  event.key === "Spacebar"
                ) {
                  event.preventDefault()
                  setVariant(opt.id)
                }
              }}
              className={
                "text-left p-4 rounded-lg border transition-colors " +
                (selected
                  ? "border-primary bg-primary/5 [box-shadow:var(--glow-primary)]"
                  : "border-border bg-surface hover:border-border-strong")
              }
              data-variant-option={opt.id}
            >
              <div className="flex items-baseline gap-2 mb-2">
                <span
                  className={
                    "font-mono text-[11px] tracking-wider " +
                    (selected ? "text-primary" : "text-text-subtle")
                  }
                >
                  {opt.id.toUpperCase()}
                </span>
                <span className="font-display font-semibold text-sm text-text">
                  {opt.name}
                </span>
                <span
                  className={
                    "ml-auto font-mono text-[10px] uppercase tracking-wider " +
                    (selected
                      ? "text-primary font-semibold"
                      : "text-text-subtle")
                  }
                  data-testid={
                    selected ? "composer-variant-active-badge" : undefined
                  }
                >
                  {selected ? "✓ Active" : opt.tag}
                </span>
              </div>
              <p className="text-xs text-text-muted leading-relaxed mb-2.5">
                {opt.description}
              </p>
              <ul className="flex flex-col gap-0.5">
                {opt.highlights.map((h) => (
                  <li
                    key={h}
                    className="font-mono text-[10px] text-text-subtle"
                  >
                    · {h}
                  </li>
                ))}
              </ul>
            </button>
          )
        })}
      </div>
    </section>
  )
}

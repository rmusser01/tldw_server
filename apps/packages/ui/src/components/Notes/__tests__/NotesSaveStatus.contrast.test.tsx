import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesSaveStatus from "../NotesSaveStatus"
import type { SaveIndicatorState } from "../notes-manager-types"
import { contrastRatio } from "@/themes/contrast"
import { getDefaultTheme } from "@/themes/presets"
import type { RGBTriple, ThemeColorTokens } from "@/themes/types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

// AX-07 (#3124): the editor status used fixed Tailwind shades, so "Saved 25m
// ago" (emerald-600) measured 3.76:1 and "Unsaved changes" (amber-600)
// 3.19:1 on white, and theme token fixes never reached them. Status text must
// use theme colours that stay at 4.5:1 on every surface in both themes.

const TOKEN_CLASSES: Record<string, keyof ThemeColorTokens> = {
  "text-text": "text",
  "text-text-muted": "textMuted",
  "text-text-subtle": "textSubtle",
  "text-success": "success",
  "text-warn": "warn",
  "text-danger": "danger",
  "text-primary": "primary",
  "text-primaryStrong": "primaryStrong"
}

const isColourClass = (name: string) =>
  /^text-/.test(name) &&
  !/^text-(\[|xs$|sm$|base$|lg$|xl$|\dxl$|left$|right$|center$|caption$|label$|body$)/.test(name)

/** The theme token a status state's text resolves to in `mode`. */
function statusTextToken(className: string, mode: "light" | "dark"): string {
  const classes = className.split(/\s+/).filter(Boolean)
  const light = classes.filter((name) => !name.includes(":") && isColourClass(name))
  const dark = classes
    .filter((name) => name.startsWith("dark:"))
    .map((name) => name.slice("dark:".length))
    .filter(isColourClass)
  const chosen = mode === "dark" && dark.length > 0 ? dark : light
  expect(chosen, `one text colour for ${mode} in "${className}"`).toHaveLength(1)
  return chosen[0]
}

const STATES: SaveIndicatorState[] = [
  "dirty",
  "saving",
  "saved",
  "offline",
  "retrying",
  "conflict",
  "error"
]

describe("NotesSaveStatus text contrast", () => {
  it.each(STATES.flatMap((state) => (["light", "dark"] as const).map((mode) => ({ state, mode }))))(
    "keeps the $state status readable on every $mode surface",
    ({ state, mode }) => {
      render(
        <NotesSaveStatus
          state={state}
          lastSavedAt={new Date(Date.now() - 25 * 60_000).toISOString()}
        />
      )
      const status = screen.getByTestId("notes-save-status")
      const colourClass = statusTextToken(status.className, mode)
      const token = TOKEN_CLASSES[colourClass]
      expect(token, `${colourClass} is a theme colour`).toBeDefined()

      const palette = getDefaultTheme().palette[mode]
      const colour = palette[token] as RGBTriple
      const failures = (["bg", "surface", "surface2", "elevated"] as const)
        .map((surface) => [surface, contrastRatio(colour, palette[surface] as RGBTriple)] as const)
        .filter(([, ratio]) => ratio < 4.5)
        .map(([surface, ratio]) => `${surface}=${ratio.toFixed(2)}`)
      expect(failures).toEqual([])
    }
  )
})

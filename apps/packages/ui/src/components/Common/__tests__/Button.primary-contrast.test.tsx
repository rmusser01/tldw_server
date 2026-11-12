import React from "react"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import postcss, { type Root } from "postcss"
import tailwindcss from "tailwindcss"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { Button } from "../Button"
import { applyThemeTokens, clearThemeTokens } from "@/themes/apply-theme"
import { contrastRatio } from "@/themes/contrast"
import { getBuiltinPresets, getDefaultTheme } from "@/themes/presets"

const require = createRequire(import.meta.url)
const clients = ["tldw-frontend", "extension"] as const
const utilities = new Map<string, Root>()
const css = postcss.parse(
  readFileSync(require.resolve("../../../assets/tailwind-shared.css"), "utf8")
)

// Resolve only emitted color declarations for classes actually on the rendered
// Button. This verifies the real Tailwind boundary, not jsdom's CSS-variable engine.
function emittedPair(
  button: HTMLElement,
  client: string,
  dark: boolean,
  state: "normal" | "hover" | "active"
) {
  const values = new Map<string, string>()
  for (const selector of dark ? [":root", ".dark"] : [":root"]) {
    css.walkRules(selector, (rule) => {
      rule.walkDecls((decl) => {
        values.set(decl.prop, decl.value)
      })
    })
  }
  const inline = document.documentElement.style
  for (const key of Array.from(inline))
    values.set(key, inline.getPropertyValue(key))
  utilities.get(client)!.walkRules((rule) => {
    // This endpoint check uses ordinary motion preference, not reduced-motion media.
    if (
      rule.parent?.type === "atrule" &&
      rule.parent.name === "media" &&
      rule.parent.params.includes("prefers-reduced-motion")
    )
      return
    const match = rule.selector.match(
      /^\.((?:\\.|[\w-])+)(?::(hover|active))?$/
    )
    if (!match || (match[2] && match[2] !== state)) return
    const name = match[1].replace(/\\/g, "")
    if (!button.classList.contains(name)) return
    rule.walkDecls((decl) => {
      values.set(decl.prop, decl.value)
    })
  })
  const resolve = (property: string) => {
    let value = values.get(property)!
    for (let step = 0; step < 4 && value.includes("var("); step++) {
      value = value.replace(
        /var\((--[\w-]+)\)/g,
        (_, key) => values.get(key) ?? "UNRESOLVED"
      )
    }
    expect(value, property).not.toContain("UNRESOLVED")
    return value
      .replace(/^rgb\(/, "")
      .split("/")[0]
      .replace(/\)$/, "")
      .trim()
  }
  return {
    foreground: resolve("color"),
    background: resolve("background-color"),
    transitionProperty: values.get("transition-property")
  }
}

beforeAll(async () => {
  const source = readFileSync(require.resolve("../Button.tsx"), "utf8")
  for (const client of clients) {
    const config = require(`../../../../../../${client}/tailwind.config.js`)
    const result = await postcss([
      tailwindcss({ ...config, content: [{ raw: source }] })
    ]).process("@tailwind utilities;", { from: undefined })
    utilities.set(client, result.root)
  }
})

afterEach(() => {
  cleanup()
  clearThemeTokens()
})

describe("shared primary Button color pairs", () => {
  it.each(clients)(
    "keeps default light/dark normal, hover and pressed labels readable in %s",
    (client) => {
      render(<Button variant="primary">Start chatting</Button>)
      const button = screen.getByRole("button", { name: "Start chatting" })
      for (const dark of [false, true]) {
        for (const state of ["normal", "hover", "active"] as const) {
          const { foreground, background } = emittedPair(
            button,
            client,
            dark,
            state
          )
          expect(
            contrastRatio(foreground, background),
            `${client}/${dark}/${state}`
          ).toBeGreaterThanOrEqual(4.5)
        }
      }
    }
  )

  it.each(clients)(
    "keeps all built-in palettes and opposite custom primary/strong edits readable in %s",
    (client) => {
      render(<Button type="primary">Start chatting</Button>)
      const button = screen.getByRole("button", { name: "Start chatting" })
      const themes = getBuiltinPresets()
      for (const theme of themes) {
        for (const tokens of Object.values(theme.palette)) {
          applyThemeTokens(tokens, theme)
          for (const state of ["normal", "hover", "active"] as const) {
            const { foreground, background } = emittedPair(
              button,
              client,
              false,
              state
            )
            expect(
              contrastRatio(foreground, background),
              `${client}/${theme.id}/${state}`
            ).toBeGreaterThanOrEqual(4.5)
          }
        }
      }
      const theme = getDefaultTheme()
      for (const [primary, primaryStrong] of [
        ["255 255 255", "0 0 0"],
        ["0 0 0", "255 255 255"]
      ]) {
        applyThemeTokens(
          { ...theme.palette.light, primary, primaryStrong },
          theme
        )
        for (const state of ["normal", "hover", "active"] as const) {
          const { foreground, background } = emittedPair(
            button,
            client,
            false,
            state
          )
          expect(contrastRatio(foreground, background)).toBeCloseTo(21, 5)
        }
      }
    }
  )

  it.each(clients)(
    "switches the primary pair together and keeps secondary transitions in %s",
    (client) => {
      render(
        <>
          <Button variant="primary">Begin</Button>
          <Button variant="secondary">Cancel</Button>
        </>
      )
      const primary = emittedPair(
        screen.getByRole("button", { name: "Begin" }),
        client,
        true,
        "normal"
      )
      const secondary = emittedPair(
        screen.getByRole("button", { name: "Cancel" }),
        client,
        true,
        "normal"
      )
      expect(primary.transitionProperty).toBe("none")
      expect(secondary.transitionProperty).toContain("background-color")
    }
  )

  it("retains primary action, submit, label and loading behavior", () => {
    const onClick = vi.fn()
    const { rerender } = render(
      <Button
        type="primary"
        htmlType="submit"
        ariaLabel="Begin"
        onClick={onClick}>
        Start chatting
      </Button>
    )
    const button = screen.getByRole("button", { name: "Begin" })
    fireEvent.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(button).toHaveAttribute("type", "submit")
    rerender(
      <Button variant="primary" loading ariaLabel="Begin" onClick={onClick}>
        Start chatting
      </Button>
    )
    fireEvent.click(button)
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute("aria-busy", "true")
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})

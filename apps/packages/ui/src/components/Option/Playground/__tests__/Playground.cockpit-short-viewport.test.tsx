// @vitest-environment jsdom
// AX-04 review: in short viewports (max-height 500px) a saved open mobile rail
// panel would push the composer out of the window. There, the panel is not
// restored on load, and one the user opens overlays the chat instead of taking
// space from it. The saved panel preference is left as it was.
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

const originalMatchMedia = window.matchMedia
const mockShortViewport = (short: boolean) => {
  window.matchMedia = ((query: string) => ({
    matches: short && query.includes("max-height: 500px"),
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false
  })) as typeof window.matchMedia
}

const renderShell = () => {
  const onMobilePanelChange = vi.fn()
  const Harness = () => {
    const [panel, setPanel] = React.useState<"context" | "runtime" | null>(
      "context"
    )
    return (
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        mobilePanel={panel}
        onMobilePanelChange={(next) => {
          onMobilePanelChange(next)
          setPanel(next)
        }}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )
  }
  render(<Harness />)
  return { onMobilePanelChange }
}

describe("cockpit mobile panels in short viewports", () => {
  afterEach(() => {
    window.matchMedia = originalMatchMedia
  })

  describe("short", () => {
    beforeEach(() => mockShortViewport(true))

    it("does not restore a saved open panel on load", () => {
      const { onMobilePanelChange } = renderShell()

      expect(
        screen.getByTestId("playground-cockpit-mobile-rails")
      ).toHaveAttribute("data-mobile-panel", "none")
      expect(
        screen.getByRole("button", { name: "Context" })
      ).toHaveAttribute("aria-expanded", "false")
      expect(onMobilePanelChange).not.toHaveBeenCalled()
    })

    it("opens a panel the user asks for as an overlay over the chat", () => {
      renderShell()
      fireEvent.click(screen.getByRole("button", { name: "Context" }))

      const panel = screen.getByRole("region", { name: "Context" })
      expect(panel).toBeVisible()
      expect(panel.className).toContain("[@media(max-height:500px)]:absolute")
      expect(
        screen.getByTestId("playground-cockpit-mobile-rails").className
      ).toContain("relative")
    })
  })

  it("restores the saved panel in taller viewports", () => {
    mockShortViewport(false)
    renderShell()

    expect(
      screen.getByTestId("playground-cockpit-mobile-rails")
    ).toHaveAttribute("data-mobile-panel", "context")
  })
})

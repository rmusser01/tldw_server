import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"
import type { ComposerModelReadiness } from "../hooks/useComposerModelReadiness"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))
vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { to: string }) => (
    <a href={to} {...rest}>{children}</a>
  )
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))

const readiness = (state: ComposerModelReadiness["state"]): ComposerModelReadiness => ({
  state,
  label: state === "working" ? "Working" : state === "unreachable" ? "Unreachable" : "Not checked",
  title: state === "working" ? "gpt-4o (OpenAI) answered" : "No reply yet",
  switchTarget: null,
  switchToTarget: () => undefined
})

const renderChip = (props: Partial<React.ComponentProps<typeof ChatModelSelectorDropdown>>) =>
  render(
    <ChatModelSelectorDropdown
      apiModelLabel="gpt-4o (OpenAI)"
      connectionStatusLabel="Server connected"
      modelDropdownOpen={false}
      modelDropdownMenuItems={[]}
      resolvedProviderKey="openai"
      selectedModel="openai:gpt-4o"
      setModelDropdownOpen={() => undefined}
      setModelSearchQuery={() => undefined}
      {...props}
    />
  )

// The chip shows the model's own readiness only with evidence (CC-01); until
// then it names the server connection, and a degraded server must still read
// as a warning, visually and in the accessible name.
describe("model chip connection status", () => {
  it("warns about a degraded server while the model has no evidence yet", () => {
    renderChip({
      connectionStatusLabel: "Degraded",
      connectionDegraded: true,
      modelReadiness: readiness("not_checked")
    })

    const chip = screen.getByTestId("model-selector")
    expect(chip).toHaveTextContent("Degraded")
    expect(chip).toHaveClass("border-warn/50")
    expect(chip).toHaveAccessibleName("gpt-4o (OpenAI). Degraded")
  })

  it("shows the model's own readiness instead of a degraded server once there is evidence", () => {
    renderChip({
      connectionStatusLabel: "Degraded",
      connectionDegraded: true,
      modelReadiness: readiness("working")
    })

    const chip = screen.getByTestId("model-selector")
    expect(chip).toHaveTextContent("Working")
    expect(chip).not.toHaveClass("border-warn/50")
  })

  it("shows Connecting… in the neutral style while the first check runs (CC-03)", () => {
    renderChip({
      connectionStatusLabel: "Connecting…",
      connectionStatusPending: true,
      modelReadiness: readiness("working")
    })

    const chip = screen.getByTestId("model-selector")
    expect(chip).toHaveTextContent("Connecting…")
    expect(chip).not.toHaveClass("border-warn/50")
    expect(chip).toHaveAccessibleName("gpt-4o (OpenAI). Connecting…")
  })

  it("still warns when the server is offline", () => {
    renderChip({
      connectionStatusLabel: "Offline",
      connectionStatusWarning: true,
      modelReadiness: readiness("working")
    })

    const chip = screen.getByTestId("model-selector")
    expect(chip).toHaveTextContent("Offline")
    expect(chip).toHaveClass("border-warn/50")
  })
})

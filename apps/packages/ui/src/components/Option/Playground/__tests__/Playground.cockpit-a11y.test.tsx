// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell"
import { PlaygroundModelCatalogControls } from "../PlaygroundModelCatalogControls"
import { PlaygroundStatusStrip } from "../PlaygroundStatusStrip"

const translate = vi.hoisted(() => (
  key: string,
  fallbackOrOptions?: unknown,
  maybeOptions?: Record<string, unknown>
) => {
  let template = key
  let options: Record<string, unknown> | undefined

  if (typeof fallbackOrOptions === "string") {
    template = fallbackOrOptions
    options = maybeOptions
  } else if (
    fallbackOrOptions &&
    typeof fallbackOrOptions === "object" &&
    "defaultValue" in (fallbackOrOptions as Record<string, unknown>)
  ) {
    template = String(
      (fallbackOrOptions as { defaultValue?: unknown }).defaultValue ?? key
    )
    options = fallbackOrOptions as Record<string, unknown>
  }

  if (!options) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = options?.[token]
    return value == null ? "" : String(value)
  })
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: translate })
}))

describe("Playground cockpit accessibility", () => {
  it("labels cockpit landmarks and exposes the layout toggle state", () => {
    const onModeChange = vi.fn()

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={onModeChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={
          <PlaygroundStatusStrip
            mode="cockpit"
            streaming={false}
            selectedModel="openai:gpt-4o"
            messageCount={0}
            sessionLabel="Local chat"
            hasContext={false}
          />
        }
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )

    expect(
      screen.getByRole("complementary", { name: "Chat cockpit context" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("playground-cockpit-mobile-rails")).toHaveTextContent(
      "Context"
    )
    expect(screen.getByTestId("playground-cockpit-mobile-rails")).toHaveTextContent(
      "Runtime"
    )

    const toggle = screen.getByRole("button", {
      name: "Enter focus chat"
    })
    expect(toggle).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(toggle)
    expect(onModeChange).toHaveBeenCalledWith("focus")
  })

  it("announces compact runtime state through one status region", () => {
    render(
      <PlaygroundStatusStrip
        mode="focus"
        streaming
        selectedModel="anthropic:claude-sonnet-4"
        messageCount={2}
        sessionLabel="Server chat"
        hasContext
      />
    )

    const status = screen.getByRole("status", { name: "Chat status" })
    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveAttribute("aria-atomic", "false")
    expect(status).toHaveTextContent("Streaming")
    expect(status).toHaveTextContent("Focus")
    expect(status).toHaveTextContent("Context active")
  })

  it("keeps model catalog controls labeled in the rendered DOM", () => {
    const setModelListScope = vi.fn()
    const setModelSearchQuery = vi.fn()
    const setModelSortMode = vi.fn()
    const { rerender } = render(
      <PlaygroundModelCatalogControls
        t={translate}
        modelListScope="configured"
        setModelListScope={setModelListScope}
        modelSearchQuery=""
        setModelSearchQuery={setModelSearchQuery}
        modelSortMode="provider"
        setModelSortMode={setModelSortMode}
      />
    )

    const toggle = screen.getByTestId("model-list-scope-toggle")
    expect(toggle).toHaveAttribute("aria-pressed", "false")
    expect(toggle).toHaveTextContent("Search all models")
    fireEvent.click(toggle)
    expect(setModelListScope).toHaveBeenCalledWith("catalog")

    const configuredSearch = screen.getByLabelText("Search models")
    expect(configuredSearch).toHaveAttribute("placeholder", "Search models")

    rerender(
      <PlaygroundModelCatalogControls
        t={translate}
        modelListScope="catalog"
        setModelListScope={setModelListScope}
        modelSearchQuery=""
        setModelSearchQuery={setModelSearchQuery}
        modelSortMode="provider"
        setModelSortMode={setModelSortMode}
      />
    )

    expect(screen.getByTestId("model-list-scope-toggle")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByLabelText("Search models")).toHaveAttribute(
      "placeholder",
      "Search all known models"
    )
  })
})

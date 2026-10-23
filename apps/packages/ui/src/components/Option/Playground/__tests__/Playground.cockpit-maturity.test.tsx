// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  PlaygroundCockpitShell,
  type PlaygroundCockpitMobilePanel,
} from "../PlaygroundCockpitShell"
import { PlaygroundContextRail } from "../PlaygroundContextRail"
import { PlaygroundRuntimeInspector } from "../PlaygroundRuntimeInspector"
import { PlaygroundStatusStrip } from "../PlaygroundStatusStrip"
import {
  formatCockpitMessageCount,
  getCockpitMessageCount,
} from "../playground-cockpit-state"

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

describe("Playground mature cockpit surfaces", () => {
  it("uses chat history as a fallback when rendered messages lag behind server sync", () => {
    expect(
      getCockpitMessageCount([], [
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi" }
      ])
    ).toBe(2)
    expect(getCockpitMessageCount([], [], 3)).toBe(3)
  })

  it("rejects stale translated message-count text", () => {
    expect(formatCockpitMessageCount("0 messages", 2)).toBe("2 messages")
    expect(formatCockpitMessageCount("{{count}} message", 1)).toBe("1 message")
  })

  it("renders context as an actionable source inventory", () => {
    const removeResearch = vi.fn()
    const removeFile = vi.fn()
    const toggleWebSearch = vi.fn()

    render(
      <PlaygroundContextRail
        hasContext
        contextSummary={[]}
        sessionLabel="Temporary chat"
        historyLinked={false}
        webSearch
        onToggleWebSearch={toggleWebSearch}
        temporaryChat
        onToggleTemporaryChat={vi.fn()}
        contextCounts={{ files: 1, knowledge: 0, media: 0, research: 1 }}
        onOpenSearchContext={vi.fn()}
        contextSources={[
          {
            id: "research-run-1",
            kind: "research",
            label: "Research",
            title: "Mature cockpit study",
            detail: "Run research-run-1",
            state: "active",
            onRemove: removeResearch
          },
          {
            id: "file-1",
            kind: "file",
            label: "File",
            title: "brief.pdf",
            detail: "Used on next reply",
            state: "active",
            onRemove: removeFile
          }
        ]}
      />
    )

    const inventory = screen.getByRole("list", {
      name: "Context sources"
    })
    expect(within(inventory).getByText("Mature cockpit study")).toBeInTheDocument()
    expect(within(inventory).getByText("Run research-run-1")).toBeInTheDocument()
    expect(within(inventory).getByText("brief.pdf")).toBeInTheDocument()
    expect(screen.getByText("2 selected sources")).toBeInTheDocument()

    fireEvent.click(
      within(inventory).getByRole("button", {
        name: "Remove Mature cockpit study"
      })
    )
    fireEvent.click(
      within(inventory).getByRole("button", {
        name: "Remove brief.pdf"
      })
    )
    expect(removeResearch).toHaveBeenCalledTimes(1)
    expect(removeFile).toHaveBeenCalledTimes(1)
  })

  it("renders runtime diagnostics, scoped settings, and tool entry points", () => {
    const openModel = vi.fn()
    const openTools = vi.fn()

    render(
      <PlaygroundRuntimeInspector
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        providerRouteLabel="openai:gpt-4.1-mini"
        runtimeStatus="ready"
        runtimeStatusDetail={null}
        messageCount={4}
        threadSearchOpen={false}
        onOpenModelSettings={openModel}
        settingSummaries={[
          { label: "Temperature", value: "0.7" },
          { label: "Context", value: "8k" }
        ]}
        toolSummary={{
          state: "available",
          label: "MCP tools",
          detail: "3 chat tools available",
          onOpen: openTools
        }}
      />
    )

    expect(screen.getByRole("heading", { name: "Model & settings" })).toBeInTheDocument()
    expect(screen.getByText("Route openai:gpt-4.1-mini")).toBeInTheDocument()
    expect(screen.getByText("Temperature")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Default settings"))
    expect(screen.getByText("0.7")).toBeInTheDocument()
    expect(screen.getByText("MCP tools")).toBeInTheDocument()
    expect(screen.getByText("Chat tool access")).toBeInTheDocument()
    expect(screen.getByText("3 chat tools available")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open MCP tools" }))
    expect(openTools).toHaveBeenCalledTimes(1)
  })

  it("prioritizes status and exposes recovery actions in the strip", () => {
    const stop = vi.fn()
    const openContext = vi.fn()

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={3}
        sessionLabel="Server chat"
        hasContext
        contextSummary={["1 file"]}
        temporaryChat={false}
        onStopStreaming={stop}
        onOpenSearchContext={openContext}
      />
    )

    const status = screen.getByRole("status", { name: "Chat status" })
    expect(within(status).getByText("Streaming")).toBeInTheDocument()
    expect(within(status).getByText("openai:gpt-4.1-mini")).toBeInTheDocument()
    expect(within(status).queryByText("Context active")).toBeNull()

    fireEvent.click(
      within(status).getByRole("button", { name: "Stop generation" })
    )
    fireEvent.click(
      within(status).getByRole("button", { name: "Open Search & Context" })
    )
    expect(stop).toHaveBeenCalledTimes(1)
    expect(openContext).toHaveBeenCalledTimes(1)
  })

  it("keeps the chat surface visible without adding a bottom status rail when both cockpit rails are collapsed", () => {
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={false}
        rightRailVisible={false}
        leftRail={<div>Context controls</div>}
        rightRail={<div>Runtime controls</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )

    expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull()
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull()
    expect(screen.getByTestId("playground-cockpit-mode-summary")).toHaveTextContent(
      "Cockpit rails hidden. Chat and composer remain active."
    )
    expect(
      screen.queryByTestId("playground-cockpit-status-strip")
    ).not.toBeInTheDocument()
    expect(screen.getByText("Chat transcript")).toBeInTheDocument()
  })

  it("uses controlled mobile disclosures with actual expanded state and local close", () => {
    const onMobilePanelChange = vi.fn()
    const onLeftRailVisibleChange = vi.fn()
    const onRightRailVisibleChange = vi.fn()
    function ControlledShell() {
      const [mobilePanel, setMobilePanel] =
        React.useState<PlaygroundCockpitMobilePanel>("context")
      return (
        <PlaygroundCockpitShell
          mode="cockpit"
          leftRailVisible
          rightRailVisible
          onLeftRailVisibleChange={onLeftRailVisibleChange}
          onRightRailVisibleChange={onRightRailVisibleChange}
          mobilePanel={mobilePanel}
          onMobilePanelChange={(panel) => {
            onMobilePanelChange(panel)
            setMobilePanel(panel)
          }}
          leftRail={<div>Context controls</div>}
          rightRail={<div>Runtime controls</div>}
        >
          <div>Chat transcript</div>
        </PlaygroundCockpitShell>
      )
    }
    render(<ControlledShell />)

    const mobilePanels = screen.getByTestId("playground-cockpit-mobile-rails")
    const panels = within(mobilePanels)
    const contextButton = panels.getByRole("button", { name: "Context" })
    const runtimeButton = panels.getByRole("button", { name: "Runtime" })
    const contextPanel = document.getElementById(
      contextButton.getAttribute("aria-controls") ?? "",
    )
    const runtimePanel = document.getElementById(
      runtimeButton.getAttribute("aria-controls") ?? "",
    )

    expect(panels.getAllByRole("button")).toHaveLength(3)
    expect(panels.queryByRole("tablist")).not.toBeInTheDocument()
    expect(contextPanel).toBe(panels.getByRole("region", { name: "Context" }))
    expect(contextPanel).toHaveAttribute("aria-labelledby", contextButton.id)
    expect(runtimePanel).toHaveAttribute("aria-labelledby", runtimeButton.id)
    expect(contextPanel).toHaveTextContent("Context controls")
    expect(contextPanel).toBeVisible()
    expect(runtimePanel).toHaveTextContent("Runtime controls")
    expect(runtimePanel).not.toBeVisible()
    expect(contextButton).toHaveAttribute("aria-expanded", "true")
    expect(runtimeButton).toHaveAttribute("aria-expanded", "false")

    fireEvent.click(runtimeButton)
    expect(onMobilePanelChange).toHaveBeenLastCalledWith("runtime")
    expect(runtimePanel).toBe(panels.getByRole("region", { name: "Runtime" }))
    expect(contextPanel).not.toBeVisible()
    expect(runtimePanel).toBeVisible()
    expect(contextButton).toHaveAttribute("aria-expanded", "false")
    expect(runtimeButton).toHaveAttribute("aria-expanded", "true")

    fireEvent.click(runtimeButton)
    expect(onMobilePanelChange).toHaveBeenLastCalledWith(null)
    expect(mobilePanels).toHaveAttribute("data-mobile-panel", "none")
    expect(panels.queryAllByRole("region")).toHaveLength(0)
    expect(contextButton).toHaveAttribute("aria-expanded", "false")
    expect(runtimeButton).toHaveAttribute("aria-expanded", "false")
    expect(onLeftRailVisibleChange).not.toHaveBeenCalled()
    expect(onRightRailVisibleChange).not.toHaveBeenCalled()
  })

  it("keeps mobile cockpit panel state explicit while preserving the edited draft surface", () => {
    const shell = (mobilePanel: PlaygroundCockpitMobilePanel) => (
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        mobilePanel={mobilePanel}
        leftRail={<div>Context controls</div>}
        rightRail={<div>Runtime controls</div>}
      >
        <label htmlFor="mobile-draft">Message</label>
        <textarea
          id="mobile-draft"
          data-testid="mobile-draft"
          defaultValue="Draft stays here"
        />
      </PlaygroundCockpitShell>
    )
    const { rerender } = render(shell("context"))
    const mobilePanels = screen.getByTestId("playground-cockpit-mobile-rails")
    const panels = within(mobilePanels)
    const draft = screen.getByRole("textbox", { name: "Message" })
    expect(mobilePanels).toHaveAttribute("data-mobile-panel", "context")
    expect(draft).toHaveValue("Draft stays here")
    expect(
      panels.getByTestId("playground-cockpit-mobile-panel-summary"),
    ).toHaveTextContent(
      "Context panel active. Composer draft remains available below.",
    )
    for (const name of ["Context", "Runtime"]) {
      expect(panels.getByRole("button", { name })).toHaveClass("min-h-[44px]")
    }

    fireEvent.change(draft, { target: { value: "Edited draft stays here" } })
    draft.focus()
    rerender(shell("runtime"))
    expect(mobilePanels).toHaveAttribute("data-mobile-panel", "runtime")
    expect(panels.getByRole("region", { name: "Runtime" })).toBeVisible()
    expect(panels.getByRole("button", { name: "Runtime" })).toHaveAttribute(
      "aria-expanded",
      "true",
    )
    expect(
      panels.getByTestId("playground-cockpit-mobile-panel-summary"),
    ).toHaveTextContent(
      "Runtime panel active. Composer draft remains available below.",
    )

    rerender(shell(null))
    expect(mobilePanels).toHaveAttribute("data-mobile-panel", "none")
    expect(panels.queryAllByRole("region")).toHaveLength(0)
    expect(
      panels.getByTestId("playground-cockpit-mobile-panel-summary"),
    ).toHaveTextContent(
      "Cockpit panels hidden. Composer draft remains available below.",
    )
    expect(screen.getByRole("textbox", { name: "Message" })).toBe(draft)
    expect(draft).toHaveValue("Edited draft stays here")
    expect(draft).toHaveFocus()
    expect(
      panels.queryByRole("button", { name: "Return to focus chat" }),
    ).not.toBeInTheDocument()
  })
})

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
    expect(screen.getByText("2 active sources")).toBeInTheDocument()

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
        selectedCharacterName="Mira Vale"
        onOpenModelSettings={openModel}
        onOpenCharacterSettings={vi.fn()}
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

    expect(screen.getByText("Provider route")).toBeInTheDocument()
    expect(screen.getByText("openai:gpt-4.1-mini")).toBeInTheDocument()
    expect(screen.getByText("Scoped settings")).toBeInTheDocument()
    expect(screen.getByText("Temperature")).toBeInTheDocument()
    expect(screen.getByText("0.7")).toBeInTheDocument()
    expect(screen.getByText("MCP tools")).toBeInTheDocument()
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
    expect(within(status).getByText("Context active")).toBeInTheDocument()

    fireEvent.click(
      within(status).getByRole("button", { name: "Stop generation" })
    )
    fireEvent.click(
      within(status).getByRole("button", { name: "Open Search & Context" })
    )
    expect(stop).toHaveBeenCalledTimes(1)
    expect(openContext).toHaveBeenCalledTimes(1)
  })

  it("uses a controlled mobile cockpit panel instead of independent details", () => {
    const onMobilePanelChange = vi.fn()

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={vi.fn()}
        leftRailVisible
        rightRailVisible
        mobilePanel={"context" satisfies PlaygroundCockpitMobilePanel}
        onMobilePanelChange={onMobilePanelChange}
        leftRail={<div>Context controls</div>}
        rightRail={<div>Runtime controls</div>}
        statusStrip={<div>Ready</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )

    const mobilePanels = screen.getByTestId("playground-cockpit-mobile-rails")
    expect(within(mobilePanels).queryByRole("tab", { name: "Context" })).toBeInTheDocument()
    expect(within(mobilePanels).queryByText("Context controls")).toBeInTheDocument()
    expect(within(mobilePanels).queryByText("Runtime controls")).toBeNull()
    expect(within(mobilePanels).queryByRole("tab", { name: "Runtime" })).toHaveAttribute(
      "aria-selected",
      "false"
    )

    fireEvent.click(within(mobilePanels).getByRole("tab", { name: "Runtime" }))
    expect(onMobilePanelChange).toHaveBeenCalledWith("runtime")
  })
})

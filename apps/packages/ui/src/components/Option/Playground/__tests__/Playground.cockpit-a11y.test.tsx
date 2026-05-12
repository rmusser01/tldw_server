// @vitest-environment jsdom
import fs from "node:fs"
import path from "node:path"
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell"
import { PlaygroundStatusStrip } from "../PlaygroundStatusStrip"

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
            serverChatId={null}
            temporaryChat={false}
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
        serverChatId="chat-1"
        temporaryChat={false}
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

  it("keeps model catalog controls labeled", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('data-testid="model-list-scope-toggle"')
    expect(source).toContain("aria-pressed={modelListScope === \"catalog\"}")
    expect(source).toContain("playground:composer.modelSearchLabel")
    expect(source).toContain("\"Search models\"")
    expect(source).toContain("Search all known models")
  })
})

// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { buildQueuedRequest } from "@/utils/chat-request-queue"
import { ChatQueuePanel } from "../ChatQueuePanel"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string,
      options?: Record<string, unknown>
    ) => {
      const template = fallback || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@/design-system", () => ({
  getDesignSystemState: vi.fn((key: string) => ({
    key,
    label: key === "blocked" ? "Blocked via registry" : key
  }))
}))

describe("ChatQueuePanel", () => {
  it("shows a queue summary and lets the user edit a queued request", async () => {
    const user = userEvent.setup()
    const queued = buildQueuedRequest({
      promptText: "Summarize the prior answer",
      snapshot: {
        selectedModel: "gpt-4o-mini",
        chatMode: "normal"
      }
    })
    const onUpdate = vi.fn()

    render(
      <ChatQueuePanel
        queue={[queued]}
        isConnectionReady
        isStreaming={false}
        onRunNext={vi.fn()}
        onRunNow={vi.fn()}
        onDelete={vi.fn()}
        onMove={vi.fn()}
        onUpdate={onUpdate}
        onClearAll={vi.fn()}
      />
    )

    expect(screen.getByText("1 queued")).toBeInTheDocument()
    expect(screen.getByText(/Summarize the prior answer/)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "View queue" }))
    await user.click(screen.getByRole("button", { name: "Edit" }))

    const input = screen.getByLabelText("Edit queued request")
    await user.clear(input)
    await user.type(input, "Turn that into bullets")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onUpdate).toHaveBeenCalledWith(queued.id, "Turn that into bullets")
  })

  it("routes row actions through the provided callbacks", async () => {
    const user = userEvent.setup()
    const first = buildQueuedRequest({ promptText: "first draft" })
    const second = buildQueuedRequest({ promptText: "second draft" })
    const onRunNext = vi.fn()
    const onRunNow = vi.fn()
    const onDelete = vi.fn()
    const onMove = vi.fn()
    const onClearAll = vi.fn()

    render(
      <ChatQueuePanel
        queue={[first, second]}
        isConnectionReady
        isStreaming
        onRunNext={onRunNext}
        onRunNow={onRunNow}
        onDelete={onDelete}
        onMove={onMove}
        onUpdate={vi.fn()}
        onClearAll={onClearAll}
      />
    )

    await user.click(screen.getByRole("button", { name: "Run next" }))
    expect(onRunNext).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "View queue" }))
    await user.click(screen.getAllByRole("button", { name: "Move down" })[0])
    await user.click(
      screen.getAllByRole("button", { name: "Cancel current & run now" })[1]
    )
    await user.click(screen.getAllByRole("button", { name: "Delete" })[0])
    await user.click(screen.getByRole("button", { name: "Clear all" }))

    expect(onMove).toHaveBeenCalledWith(first.id, "down")
    expect(onRunNow).toHaveBeenCalledWith(second.id)
    expect(onDelete).toHaveBeenCalledWith(first.id)
    expect(onClearAll).toHaveBeenCalledTimes(1)
  })

  it("disables edit and reorder controls for requests already sending", async () => {
    const user = userEvent.setup()
    const sending = buildQueuedRequest({
      promptText: "sending now",
      status: "sending"
    })
    const queued = buildQueuedRequest({ promptText: "queued next" })

    render(
      <ChatQueuePanel
        queue={[sending, queued]}
        isConnectionReady
        isStreaming={false}
        onRunNext={vi.fn()}
        onRunNow={vi.fn()}
        onDelete={vi.fn()}
        onMove={vi.fn()}
        onUpdate={vi.fn()}
        onClearAll={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: "View queue" }))

    expect(screen.getAllByRole("button", { name: "Edit" })[0]).toBeDisabled()
    expect(
      screen.getAllByRole("button", { name: "Move down" })[0]
    ).toBeDisabled()
  })

  it("uses the design-system blocked label when no blocked reason is provided", () => {
    const blocked = buildQueuedRequest({
      promptText: "retry when the backend is ready",
      status: "blocked"
    })

    render(
      <ChatQueuePanel
        queue={[blocked]}
        isConnectionReady
        isStreaming={false}
        onRunNext={vi.fn()}
        onRunNow={vi.fn()}
        onDelete={vi.fn()}
        onMove={vi.fn()}
        onUpdate={vi.fn()}
        onClearAll={vi.fn()}
      />
    )

    expect(screen.getByText("Blocked via registry")).toBeInTheDocument()
  })

  it("keeps custom blocked reasons unchanged", () => {
    const blocked = buildQueuedRequest({
      promptText: "retry after reviewing diagnostics",
      status: "blocked",
      blockedReason: "Review diagnostics before retrying"
    })

    render(
      <ChatQueuePanel
        queue={[blocked]}
        isConnectionReady
        isStreaming={false}
        onRunNext={vi.fn()}
        onRunNow={vi.fn()}
        onDelete={vi.fn()}
        onMove={vi.fn()}
        onUpdate={vi.fn()}
        onClearAll={vi.fn()}
      />
    )

    expect(
      screen.getByText("Review diagnostics before retrying")
    ).toBeInTheDocument()
  })
})

import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ContextStagingCard } from "../ContextStagingCard"
import type { StagedWorkspaceSource } from "../types"

const staged: StagedWorkspaceSource[] = [
  {
    sourceId: "s1",
    mediaId: 1,
    title: "Operator Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "ready"
  }
]

describe("ContextStagingCard", () => {
  it("renders staged sources as not sent", () => {
    render(
      <ContextStagingCard
        sources={staged}
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(screen.getByLabelText("Staged context")).toBeInTheDocument()
    expect(screen.getByText("Context staged - not sent")).toBeInTheDocument()
    expect(screen.getByText("Operator Notes")).toBeInTheDocument()
    expect(screen.getByText("ready")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Clear staged context" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Insert context summary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Send with staged context" })
    ).toBeInTheDocument()
  })

  it("shows unavailable warnings", () => {
    render(
      <ContextStagingCard
        sources={[
          { ...staged[0], availability: "error", statusMessage: "Source failed" }
        ]}
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(screen.getByText("Source failed")).toBeInTheDocument()
    expect(screen.getByText("error")).toBeInTheDocument()
  })

  it("calls clear, insert, and send actions", () => {
    const onClear = vi.fn()
    const onInsert = vi.fn()
    const onSend = vi.fn()

    render(
      <ContextStagingCard
        sources={staged}
        onClear={onClear}
        onInsert={onInsert}
        onSend={onSend}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Clear staged context" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Insert context summary" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Send with staged context" })
    )

    expect(onClear).toHaveBeenCalledTimes(1)
    expect(onInsert).toHaveBeenCalledTimes(1)
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it("disables send and labels the sending state without relying on color", () => {
    const onSend = vi.fn()

    render(
      <ContextStagingCard
        sources={staged}
        isSending
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={onSend}
      />
    )

    const sendButton = screen.getByRole("button", {
      name: "Send with staged context"
    })

    expect(sendButton).toBeDisabled()
    expect(screen.getByText("Sending staged context")).toBeInTheDocument()
    expect(screen.getByText("Sending with staged context")).toBeInTheDocument()

    fireEvent.click(sendButton)

    expect(onSend).not.toHaveBeenCalled()
  })

  it("disables send and shows an explicit empty state without staged sources", () => {
    const onSend = vi.fn()

    render(
      <ContextStagingCard
        sources={[]}
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={onSend}
      />
    )

    const sendButton = screen.getByRole("button", {
      name: "Send with staged context"
    })

    expect(screen.getByText("No context staged")).toBeInTheDocument()
    expect(sendButton).toBeDisabled()

    fireEvent.click(sendButton)

    expect(onSend).not.toHaveBeenCalled()
  })

  it("wraps long staged source text within the card", () => {
    const longToken = "x".repeat(160)

    render(
      <ContextStagingCard
        sources={[
          {
            ...staged[0],
            title: longToken,
            statusMessage: longToken
          }
        ]}
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={vi.fn()}
      />
    )

    const longTextNodes = screen.getAllByText(longToken)

    expect(longTextNodes[0]).toHaveClass("min-w-0")
    expect(longTextNodes[0]).toHaveClass("break-words")
    expect(longTextNodes[1]).toHaveClass("break-words")
  })
})

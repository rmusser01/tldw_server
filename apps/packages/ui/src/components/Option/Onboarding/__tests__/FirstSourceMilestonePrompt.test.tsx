// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("FirstSourceMilestonePrompt", () => {
  it("defaults to Web URL and adds that source kind", async () => {
    const onAddSource = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="idle"
        onAddSource={onAddSource}
        onDismiss={vi.fn()}
      />
    )

    expect(
      screen.getByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: /web url/i })).toBeChecked()
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(onAddSource).toHaveBeenCalledWith("web_url")
  })

  it("lets users select file upload and paste text source kinds", async () => {
    const onAddSource = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="idle"
        onAddSource={onAddSource}
        onDismiss={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("radio", { name: /file/i }))
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))
    fireEvent.click(screen.getByRole("radio", { name: /paste/i }))
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(onAddSource).toHaveBeenNthCalledWith(1, "file_upload")
    expect(onAddSource).toHaveBeenNthCalledWith(2, "paste_text")
  })

  it("shows processing state without the picker", async () => {
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="processing"
        onAddSource={vi.fn()}
        onDismiss={vi.fn()}
      />
    )

    expect(screen.getByText(/processing your source/i)).toBeInTheDocument()
    expect(screen.queryByRole("radio", { name: /web url/i })).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /add source/i })
    ).not.toBeInTheDocument()
  })

  it("shows retry action after ingest errors", async () => {
    const onRetry = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="error"
        errorMessage="Upload failed"
        onAddSource={vi.fn()}
        onRetry={onRetry}
        onDismiss={vi.fn()}
      />
    )

    expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /retry/i }))

    expect(onRetry).toHaveBeenCalled()
  })

  it("does not offer a hidden add-source action in error copy", async () => {
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="error"
        onAddSource={vi.fn()}
        onRetry={vi.fn()}
        onDismiss={vi.fn()}
      />
    )

    expect(screen.getByText(/source ingest did not finish/i)).toHaveTextContent(
      /retry/i
    )
    expect(screen.getByText(/source ingest did not finish/i)).not.toHaveTextContent(
      /add a different source/i
    )
    expect(
      screen.queryByRole("button", { name: /add source/i })
    ).not.toBeInTheDocument()
  })

  it("exposes a focus-visible ring on source kind cards", async () => {
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="idle"
        onAddSource={vi.fn()}
        onDismiss={vi.fn()}
      />
    )

    const pasteCard = screen
      .getByRole("radio", { name: /paste/i })
      .closest("label")

    expect(pasteCard?.className).toContain("focus-within:ring-2")
  })

  it("shows grounded chat action only when a ready source handler is provided", async () => {
    const onAskAboutSource = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    const { rerender } = render(
      <FirstSourceMilestonePrompt
        readinessStatus="ready"
        lastSourceLabel="Example article"
        onAddSource={vi.fn()}
        onDismiss={vi.fn()}
      />
    )

    expect(
      screen.queryByRole("button", { name: /ask a question about this source/i })
    ).not.toBeInTheDocument()

    rerender(
      <FirstSourceMilestonePrompt
        readinessStatus="ready"
        lastSourceLabel="Example article"
        onAddSource={vi.fn()}
        onAskAboutSource={onAskAboutSource}
        onDismiss={vi.fn()}
      />
    )

    expect(screen.getByText(/example article/i)).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: /ask a question about this source/i })
    )

    expect(onAskAboutSource).toHaveBeenCalled()
  })

  it("lets users dismiss the post-onboarding source milestone", async () => {
    const onDismiss = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt
        readinessStatus="idle"
        onAddSource={vi.fn()}
        onDismiss={onDismiss}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }))

    expect(onDismiss).toHaveBeenCalled()
  })
})

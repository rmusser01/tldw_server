// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("FirstSourceMilestonePrompt", () => {
  it("offers adding a first source immediately after onboarding completion", async () => {
    const onAddSource = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt onAddSource={onAddSource} onDismiss={vi.fn()} />
    )

    expect(
      screen.getByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(onAddSource).toHaveBeenCalled()
  })

  it("lets users dismiss the post-onboarding source milestone", async () => {
    const onDismiss = vi.fn()
    const { FirstSourceMilestonePrompt } = await import(
      "../FirstSourceMilestonePrompt"
    )

    render(
      <FirstSourceMilestonePrompt onAddSource={vi.fn()} onDismiss={onDismiss} />
    )

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }))

    expect(onDismiss).toHaveBeenCalled()
  })
})

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"

import { FirstRunGate } from "../FirstRunGate"

// Mock useFirstRunCheck
const mockUseFirstRunCheck = vi.fn()
vi.mock("@/hooks/useFirstRunCheck", () => ({
  useFirstRunCheck: () => mockUseFirstRunCheck()
}))

describe("FirstRunGate", () => {
  const onStartSetup = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it("renders children while loading", () => {
    mockUseFirstRunCheck.mockReturnValue({
      shouldShowSetup: false,
      resumeStep: null,
      loading: true
    })

    render(
      <FirstRunGate onStartSetup={onStartSetup}>
        <div data-testid="child-content">Hello</div>
      </FirstRunGate>
    )

    expect(screen.getByTestId("child-content")).toBeInTheDocument()
    expect(screen.queryByTestId("first-run-gate-overlay")).not.toBeInTheDocument()
  })

  it("renders children when shouldShowSetup is false", () => {
    mockUseFirstRunCheck.mockReturnValue({
      shouldShowSetup: false,
      resumeStep: null,
      loading: false
    })

    render(
      <FirstRunGate onStartSetup={onStartSetup}>
        <div data-testid="child-content">Hello</div>
      </FirstRunGate>
    )

    expect(screen.getByTestId("child-content")).toBeInTheDocument()
    expect(screen.queryByTestId("first-run-gate-overlay")).not.toBeInTheDocument()
  })

  it("renders the overlay when shouldShowSetup is true", () => {
    mockUseFirstRunCheck.mockReturnValue({
      shouldShowSetup: true,
      resumeStep: null,
      loading: false
    })

    render(
      <FirstRunGate onStartSetup={onStartSetup}>
        <div data-testid="child-content">Hello</div>
      </FirstRunGate>
    )

    expect(screen.getByTestId("first-run-gate-overlay")).toBeInTheDocument()
    expect(screen.getByText("Build Your Assistant")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Set up a personalized AI assistant to help you get more out of your workflow."
      )
    ).toBeInTheDocument()
    expect(screen.queryByTestId("child-content")).not.toBeInTheDocument()
  })

  it("calls onStartSetup when Get Started is clicked", () => {
    mockUseFirstRunCheck.mockReturnValue({
      shouldShowSetup: true,
      resumeStep: null,
      loading: false
    })

    render(
      <FirstRunGate onStartSetup={onStartSetup}>
        <div>Hello</div>
      </FirstRunGate>
    )

    fireEvent.click(screen.getByTestId("first-run-get-started"))
    expect(onStartSetup).toHaveBeenCalledTimes(1)
    expect(screen.queryByTestId("first-run-gate-overlay")).not.toBeInTheDocument()
  })

  it("dismisses the overlay and sets localStorage when Skip is clicked", () => {
    mockUseFirstRunCheck.mockReturnValue({
      shouldShowSetup: true,
      resumeStep: null,
      loading: false
    })

    render(
      <FirstRunGate onStartSetup={onStartSetup}>
        <div data-testid="child-content">Hello</div>
      </FirstRunGate>
    )

    expect(screen.getByTestId("first-run-gate-overlay")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("first-run-skip"))

    expect(screen.queryByTestId("first-run-gate-overlay")).not.toBeInTheDocument()
    expect(screen.getByTestId("child-content")).toBeInTheDocument()
    expect(localStorage.getItem("assistant_setup_dismissed")).toBe("true")
  })
})

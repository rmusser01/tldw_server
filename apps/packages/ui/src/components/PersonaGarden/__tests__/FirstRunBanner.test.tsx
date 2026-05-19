import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { FirstRunBanner } from "../FirstRunBanner"

describe("FirstRunBanner", () => {
  describe("resume variant", () => {
    it("renders amber banner with resume text", () => {
      const onResume = vi.fn()
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner
          variant="resume"
          resumeStep="commands"
          onResume={onResume}
          onDismiss={onDismiss}
        />
      )

      expect(screen.getByTestId("first-run-banner-resume")).toBeInTheDocument()
      expect(
        screen.getByText("Continue setting up your assistant?")
      ).toBeInTheDocument()
    })

    it("calls onResume when Resume button is clicked", () => {
      const onResume = vi.fn()
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner
          variant="resume"
          onResume={onResume}
          onDismiss={onDismiss}
        />
      )

      fireEvent.click(screen.getByTestId("first-run-banner-resume-btn"))
      expect(onResume).toHaveBeenCalledTimes(1)
    })

    it("calls onDismiss when dismiss button is clicked", () => {
      const onResume = vi.fn()
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner
          variant="resume"
          onResume={onResume}
          onDismiss={onDismiss}
        />
      )

      fireEvent.click(screen.getByTestId("first-run-banner-dismiss"))
      expect(onDismiss).toHaveBeenCalledTimes(1)
    })

    it("omits Resume button when onResume is not provided", () => {
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner variant="resume" onDismiss={onDismiss} />
      )

      expect(
        screen.queryByTestId("first-run-banner-resume-btn")
      ).not.toBeInTheDocument()
    })
  })

  describe("nudge variant", () => {
    it("renders muted banner with nudge text", () => {
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner variant="nudge" onDismiss={onDismiss} />
      )

      expect(screen.getByTestId("first-run-banner-nudge")).toBeInTheDocument()
      expect(
        screen.getByText(/Set up an assistant to get more out of this/i)
      ).toBeInTheDocument()
    })

    it("calls onResume when Set up link is clicked", () => {
      const onResume = vi.fn()
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner
          variant="nudge"
          onResume={onResume}
          onDismiss={onDismiss}
        />
      )

      fireEvent.click(screen.getByTestId("first-run-banner-setup-link"))
      expect(onResume).toHaveBeenCalledTimes(1)
    })

    it("calls onDismiss when dismiss button is clicked", () => {
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner variant="nudge" onDismiss={onDismiss} />
      )

      fireEvent.click(screen.getByTestId("first-run-banner-dismiss"))
      expect(onDismiss).toHaveBeenCalledTimes(1)
    })

    it("omits Set up link when onResume is not provided", () => {
      const onDismiss = vi.fn()

      render(
        <FirstRunBanner variant="nudge" onDismiss={onDismiss} />
      )

      expect(
        screen.queryByTestId("first-run-banner-setup-link")
      ).not.toBeInTheDocument()
    })
  })
})

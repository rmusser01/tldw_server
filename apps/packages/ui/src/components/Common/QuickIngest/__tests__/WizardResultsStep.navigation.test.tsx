// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import React from "react"
import type { WizardResultItem } from "../types"

const wizardHarness = vi.hoisted(() => ({
  results: [] as WizardResultItem[],
  reset: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: any) => {
      if (typeof opts === "string") return opts
      return opts?.defaultValue ?? key
    },
  }),
}))

vi.mock("../IngestWizardContext", () => ({
  useIngestWizard: () => ({
    state: {
      results: wizardHarness.results,
      processingState: { elapsed: 10 },
    },
    reset: wizardHarness.reset,
  }),
}))

import { WizardResultsStep } from "../WizardResultsStep"

describe("WizardResultsStep navigation buttons", () => {
  const setSinglePdfResult = (overrides: Partial<WizardResultItem> = {}) => {
    wizardHarness.results = [
      {
        id: "test-1",
        status: "ok" as const,
        type: "pdf",
        title: "My Test PDF",
        mediaId: 42,
        persisted: true,
        ...overrides,
      },
    ]
  }

  beforeEach(() => {
    wizardHarness.reset.mockReset()
    setSinglePdfResult()
  })

  it("renders Search in Knowledge button when onSearchKnowledge is provided", () => {
    const onSearchKnowledge = vi.fn()
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onSearchKnowledge={onSearchKnowledge}
      />
    )
    const btn = screen.getByText("Search in Knowledge")
    expect(btn).toBeTruthy()
    fireEvent.click(btn)
    expect(onSearchKnowledge).toHaveBeenCalledTimes(1)
  })

  it("renders Open in Workspace button when onOpenWorkspace provided and PDF ingested", () => {
    const onOpenWorkspace = vi.fn()
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenWorkspace={onOpenWorkspace}
      />
    )
    const btn = screen.getByText("Open in Workspace")
    expect(btn).toBeTruthy()
    fireEvent.click(btn)
    expect(onOpenWorkspace).toHaveBeenCalledTimes(1)
    expect(onOpenWorkspace).toHaveBeenCalledWith(
      expect.objectContaining({ type: "pdf", mediaId: 42 })
    )
  })

  it("does not render Open in Workspace when the original file was not persisted", () => {
    setSinglePdfResult({ persisted: false })
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenWorkspace={vi.fn()}
      />
    )

    expect(screen.queryByText("Open in Workspace")).toBeNull()
  })

  it("does not render navigation buttons when callbacks are not provided", () => {
    render(<WizardResultsStep onClose={vi.fn()} />)
    expect(screen.queryByText("Search in Knowledge")).toBeNull()
    expect(screen.queryByText("Open in Workspace")).toBeNull()
  })
})

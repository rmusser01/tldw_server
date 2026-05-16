// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent, within } from "@testing-library/react"
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

  it("renders Open in Media for persisted results when onOpenMedia is provided", () => {
    const onOpenMedia = vi.fn()
    setSinglePdfResult({ mediaId: 42, persisted: true })
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenMedia={onOpenMedia}
      />
    )

    const completedSection = screen.getByRole("region", { name: "Completed items" })
    const btn = within(completedSection).getByRole("button", {
      name: /open/i,
    })
    fireEvent.click(btn)

    expect(onOpenMedia).toHaveBeenCalledTimes(1)
    expect(onOpenMedia).toHaveBeenCalledWith(
      expect.objectContaining({ mediaId: 42, persisted: true })
    )
  })

  it("does not render Open in Media for unpersisted results without mediaId", () => {
    setSinglePdfResult({ mediaId: null, persisted: false })
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenMedia={vi.fn()}
      />
    )

    const completedSection = screen.getByRole("region", { name: "Completed items" })
    expect(
      within(completedSection).queryByRole("button", { name: /open/i })
    ).toBeNull()
  })

  it("renders Open in Media for completed URL results with a mediaId", () => {
    const onOpenMedia = vi.fn()
    setSinglePdfResult({
      type: "html",
      title: "Captured article",
      mediaId: "url-media-42",
      persisted: false,
    })

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenMedia={onOpenMedia}
      />
    )

    const completedSection = screen.getByRole("region", { name: "Completed items" })
    fireEvent.click(
      within(completedSection).getByRole("button", { name: /open .* media/i })
    )

    expect(onOpenMedia).toHaveBeenCalledWith(
      expect.objectContaining({ mediaId: "url-media-42" })
    )
  })

  it("does not render Remove for errors when no real remove callback exists", () => {
    wizardHarness.results = [
      {
        id: "err-1",
        status: "error",
        outcome: "failed",
        error: "Network failed",
        url: "https://example.com",
        type: "web",
      },
    ]

    render(<WizardResultsStep onClose={vi.fn()} />)

    expect(screen.queryByRole("button", { name: /remove/i })).toBeNull()
  })

  it("describes local skipped duplicates as already queued", () => {
    wizardHarness.results = [
      {
        id: "skipped-local-1",
        status: "ok",
        outcome: "skipped",
        message: "Duplicate URL",
        url: "https://example.com",
        type: "web",
      },
    ]

    render(<WizardResultsStep onClose={vi.fn()} />)

    expect(screen.getByText(/already queued/i)).toBeTruthy()
  })

  it("describes backend skipped duplicates as already in library with overwrite recovery", () => {
    wizardHarness.results = [
      {
        id: "skipped-library-1",
        status: "ok",
        outcome: "skipped",
        message: "This item already exists in your library. Use the \u2018Deep\u2019 preset to overwrite.",
        url: "https://example.com",
        type: "web",
      },
    ]

    render(<WizardResultsStep onClose={vi.fn()} />)

    expect(screen.getByText(/already in library/i)).toBeTruthy()
    expect(screen.getByText(/overwrite existing/i)).toBeTruthy()
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

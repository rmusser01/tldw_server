// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent, within } from "@testing-library/react"
import React from "react"
import type { WizardResultItem } from "../types"

const wizardHarness = vi.hoisted(() => ({
  results: [] as WizardResultItem[],
  reset: vi.fn(),
}))

const sessionHarness = vi.hoisted(() => ({
  tracking: undefined as any,
}))

const capabilitiesHarness = vi.hoisted(() => ({
  capabilities: { hasKnowledgeQaMediaScope: false } as any,
  loading: false,
  refresh: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: any) => {
      if (typeof opts === "string") return opts
      const value = opts?.defaultValue ?? key
      return value.replace(/\{\{(\w+)\}\}/g, (_match: string, token: string) =>
        opts?.[token] == null ? `{{${token}}}` : String(opts[token])
      )
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

vi.mock("@/store/quick-ingest-session", () => ({
  useQuickIngestSessionStore: (selector: any) =>
    selector({
      session: sessionHarness.tracking
        ? { tracking: sessionHarness.tracking }
        : null,
    }),
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: capabilitiesHarness.capabilities,
    loading: capabilitiesHarness.loading,
    refresh: capabilitiesHarness.refresh,
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
    sessionHarness.tracking = undefined
    capabilitiesHarness.capabilities = { hasKnowledgeQaMediaScope: false }
    capabilitiesHarness.loading = false
    capabilitiesHarness.refresh.mockReset()
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

  it("groups conference outcomes and opens the durable collection", () => {
    const onOpenCollection = vi.fn()
    wizardHarness.results = [
      {
        id: "ok-1",
        status: "ok" as const,
        type: "video",
        title: "Opening Keynote",
        mediaId: 101,
        collectionItemId: "11",
      } as any,
      {
        id: "skip-1",
        status: "ok" as const,
        outcome: "skipped",
        type: "video",
        title: "Existing Talk",
        mediaId: 102,
        collectionItemId: "12",
      } as any,
      {
        id: "submit-1",
        status: "error" as const,
        outcome: "submit_failed",
        type: "video",
        url: "https://example.com/submit",
        title: "Submit Blocked",
        error: "Queue unavailable",
        collectionItemId: "13",
      } as any,
      {
        id: "failed-1",
        status: "error" as const,
        outcome: "failed",
        type: "video",
        url: "https://example.com/fail",
        title: "Bad Video",
        error: "Download failed",
        collectionItemId: "14",
        retryAttempt: 2,
      } as any,
      {
        id: "cancel-1",
        status: "error" as const,
        outcome: "cancelled",
        type: "video",
        url: "https://example.com/cancel",
        title: "Cancelled Talk",
        error: "Cancelled by user",
        collectionItemId: "15",
      } as any,
    ]
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["11", "12", "13", "14", "15"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenCollection={onOpenCollection}
      />
    )

    expect(screen.getByText("Succeeded (1)")).toBeTruthy()
    expect(screen.getByText("Skipped existing (1)")).toBeTruthy()
    expect(screen.getByText("Not submitted (1)")).toBeTruthy()
    expect(screen.getByText("Failed during processing (1)")).toBeTruthy()
    expect(screen.getByText("Cancelled (1)")).toBeTruthy()

    fireEvent.click(
      screen.getByRole("button", { name: "Open collection 7" })
    )
    expect(onOpenCollection).toHaveBeenCalledWith("7")
  })

  it("only exposes Ask this collection when media-scope QA is supported and collection content is ready", () => {
    const onSearchKnowledge = vi.fn()
    capabilitiesHarness.capabilities = { hasKnowledgeQaMediaScope: true }
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["11"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }
    setSinglePdfResult({
      type: "video",
      mediaId: 101,
      persisted: true,
      collectionItemId: "11",
    } as any)

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onSearchKnowledge={onSearchKnowledge}
      />
    )

    const askButton = screen.getByRole("button", {
      name: "Ask this collection",
    })
    fireEvent.click(askButton)
    expect(onSearchKnowledge).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("Search in Knowledge")).toBeNull()
  })

  it("keeps the collection handoff available when every item failed", () => {
    const onOpenCollection = vi.fn()
    wizardHarness.results = [
      {
        id: "failed-1",
        status: "error" as const,
        outcome: "failed",
        type: "video",
        title: "Bad Video",
        error: "Download failed",
        collectionItemId: "14",
      } as any,
    ]
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["14"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenCollection={onOpenCollection}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Open collection 7" })
    )
    expect(onOpenCollection).toHaveBeenCalledWith("7")
  })

  it("allows collection-scoped QA when only skipped existing content is ready", () => {
    const onSearchKnowledge = vi.fn()
    capabilitiesHarness.capabilities = { hasKnowledgeQaMediaScope: true }
    wizardHarness.results = [
      {
        id: "skip-1",
        status: "ok" as const,
        outcome: "skipped",
        type: "video",
        title: "Existing Talk",
        mediaId: 102,
        collectionItemId: "12",
      } as any,
    ]
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["12"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onSearchKnowledge={onSearchKnowledge}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Ask this collection" })
    )
    expect(onSearchKnowledge).toHaveBeenCalledTimes(1)
  })

  it("does not expose collection-scoped QA when the server lacks media-scope support", () => {
    const onSearchKnowledge = vi.fn()
    capabilitiesHarness.capabilities = { hasKnowledgeQaMediaScope: false }
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["11"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }
    setSinglePdfResult({
      type: "video",
      mediaId: 101,
      persisted: true,
      collectionItemId: "11",
    } as any)

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onSearchKnowledge={onSearchKnowledge}
      />
    )

    expect(screen.queryByText("Ask this collection")).toBeNull()
  })
})

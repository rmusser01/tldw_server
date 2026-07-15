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

const virtualizerHarness = vi.hoisted(() => ({
  scrollToIndex: vi.fn(),
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

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count, getItemKey }: any) => {
    const [start, setStart] = React.useState(0)
    const mountedCount = Math.min(count, 12)
    const boundedStart = Math.min(start, Math.max(0, count - mountedCount))
    return {
      getTotalSize: () => count * 64,
      getVirtualItems: () =>
        Array.from({ length: mountedCount }, (_, offset) => {
          const index = boundedStart + offset
          return {
            index,
            start: index * 64,
            size: 64,
            key: getItemKey?.(index) ?? index,
          }
        }),
      measureElement: vi.fn(),
      scrollToIndex: (index: number) => {
        virtualizerHarness.scrollToIndex(index)
        setStart(Math.max(0, index - mountedCount + 1))
      },
    }
  },
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
    virtualizerHarness.scrollToIndex.mockReset()
    setSinglePdfResult()
  })

  it("delegates Start over to durable session replacement when provided", () => {
    const onStartOver = vi.fn()
    render(
      <WizardResultsStep onClose={vi.fn()} onStartOver={onStartOver} />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Start a new ingest" })
    )

    expect(onStartOver).toHaveBeenCalledTimes(1)
    expect(wizardHarness.reset).not.toHaveBeenCalled()
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

  it("renders all eight terminal outcomes as separate truthful groups", () => {
    const outcomes = [
      ["completed", "Completed talk"],
      ["included_existing", "Included talk"],
      ["metadata_updated", "Metadata-updated talk"],
      ["skipped_existing", "Skipped talk"],
      ["submit_failed", "Not-submitted talk"],
      ["processing_failed", "Processing-failed talk"],
      ["metadata_update_failed", "Metadata-failed talk"],
      ["cancelled", "Cancelled talk"],
    ] as const
    wizardHarness.results = outcomes.map(([terminalOutcome, title], index) => ({
      id: `outcome-${index + 1}`,
      status:
        terminalOutcome === "completed" ||
        terminalOutcome === "included_existing" ||
        terminalOutcome === "metadata_updated" ||
        terminalOutcome === "skipped_existing"
          ? "ok"
          : "error",
      outcome:
        terminalOutcome === "completed"
          ? "processed"
          : terminalOutcome === "included_existing" ||
              terminalOutcome === "skipped_existing"
            ? "skipped"
            : terminalOutcome === "submit_failed"
              ? "submit_failed"
              : terminalOutcome === "cancelled"
                ? "cancelled"
                : "failed",
      terminalOutcome,
      type: "video",
      title,
      mediaId:
        terminalOutcome === "completed" || terminalOutcome === "included_existing"
          ? index + 1
          : null,
      error:
        terminalOutcome === "submit_failed" ||
        terminalOutcome === "processing_failed" ||
        terminalOutcome === "metadata_update_failed" ||
        terminalOutcome === "cancelled"
          ? title
          : undefined,
    })) as any

    render(<WizardResultsStep onClose={vi.fn()} />)

    expect(screen.getByText("Completed (1)")).toBeInTheDocument()
    expect(screen.getByText("Included existing (1)")).toBeInTheDocument()
    expect(screen.getByText("Metadata updated (1)")).toBeInTheDocument()
    expect(screen.getByText("Skipped existing (1)")).toBeInTheDocument()
    expect(screen.getByText("Not submitted (1)")).toBeInTheDocument()
    expect(screen.getByText("Failed during processing (1)")).toBeInTheDocument()
    expect(screen.getByText("Metadata update failed (1)")).toBeInTheDocument()
    expect(screen.getByText("Cancelled (1)")).toBeInTheDocument()
  })

  it("bounds 500 terminal rows, exposes outcomes, and preserves keyboard focus", async () => {
    const outcomes = [
      "completed",
      "included_existing",
      "metadata_updated",
      "skipped_existing",
      "submit_failed",
      "processing_failed",
      "metadata_update_failed",
      "cancelled",
    ] as const
    wizardHarness.results = Array.from({ length: 500 }, (_, index) => {
      const terminalOutcome = outcomes[index % outcomes.length]
      const ordinal = index + 1
      return {
        id: `terminal-${ordinal}`,
        status:
          terminalOutcome === "completed" ||
          terminalOutcome === "included_existing" ||
          terminalOutcome === "metadata_updated" ||
          terminalOutcome === "skipped_existing"
            ? "ok"
            : "error",
        outcome:
          terminalOutcome === "completed"
            ? "processed"
            : terminalOutcome === "included_existing" ||
                terminalOutcome === "skipped_existing"
              ? "skipped"
              : terminalOutcome === "submit_failed"
                ? "submit_failed"
                : terminalOutcome === "cancelled"
                  ? "cancelled"
                  : "failed",
        terminalOutcome,
        type: "video",
        title: `${ordinal}. Terminal talk ${ordinal}`,
        error: terminalOutcome.endsWith("failed") ? "Server-reported failure" : undefined,
      }
    }) as any

    render(<WizardResultsStep onClose={vi.fn()} />)

    const list = screen.getByRole("list", { name: "Terminal results" })
    const initialRows = within(list).getAllByRole("listitem")
    expect(initialRows).toHaveLength(12)
    expect(initialRows[0]).toHaveAttribute(
      "aria-setsize",
      "500"
    )
    expect(initialRows[0]).toHaveAttribute(
      "aria-posinset",
      "1"
    )
    const labels = [
      "Completed",
      "Included existing",
      "Metadata updated",
      "Skipped existing",
      "Not submitted",
      "Failed during processing",
      "Metadata update failed",
      "Cancelled",
    ]
    initialRows.forEach((row, index) => {
      expect(within(row).getByText(labels[index % labels.length])).toBeVisible()
    })

    initialRows[0].focus()
    fireEvent.keyDown(initialRows[0], { key: "ArrowDown" })
    expect(initialRows[1]).toHaveFocus()
    fireEvent.keyDown(initialRows[1], { key: "End" })
    expect(virtualizerHarness.scrollToIndex).toHaveBeenCalledWith(499)
    expect(await screen.findByText("500. Terminal talk 500")).toBeInTheDocument()
    expect(within(list).getAllByRole("listitem").at(-1)).toHaveFocus()

    fireEvent.change(screen.getByRole("combobox", { name: "Filter results by outcome" }), {
      target: { value: "metadata_update_failed" },
    })
    expect(screen.getByText("Metadata update failed (62)")).toBeInTheDocument()
    expect(within(list).getAllByRole("listitem")[0]).toHaveFocus()
    for (const row of within(list).getAllByRole("listitem")) {
      expect(row).toHaveAttribute("data-terminal-outcome", "metadata_update_failed")
      expect(within(row).getByText("Metadata update failed")).toBeVisible()
    }
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

  it("retries durable failed collection items with retry request metadata", () => {
    const onRetryItems = vi.fn()
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
        id: "submit-1",
        status: "error" as const,
        outcome: "submit_failed",
        type: "video",
        title: "Submit Blocked",
        error: "timed out",
        collectionItemId: "13",
      } as any,
      {
        id: "failed-1",
        status: "error" as const,
        outcome: "failed",
        type: "video",
        title: "Bad Video",
        error: "timed out",
        collectionItemId: "14",
        retryAttempt: 2,
      } as any,
      {
        id: "cancel-1",
        status: "error" as const,
        outcome: "cancelled",
        type: "video",
        title: "Cancelled Talk",
        error: "cancelled",
        collectionItemId: "15",
      } as any,
      {
        id: "legacy-failed",
        status: "error" as const,
        outcome: "failed",
        type: "video",
        title: "Legacy Failed",
        error: "timed out",
      } as any,
    ]
    sessionHarness.tracking = {
      mode: "webui-direct",
      collectionId: "7",
      plannedItemIds: ["11", "13", "14", "15"],
      durableMode: "durable_collection",
      startedAt: "2026-05-16T12:00:00Z",
    }

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onRetryItems={onRetryItems}
      />
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: "Retry all 3 retryable errors",
      })
    )

    expect(onRetryItems).toHaveBeenCalledWith(
      ["13", "14", "15"],
      [
        {
          resultId: "submit-1",
          collectionItemId: "13",
          retryAttempt: 1,
          idempotencyKey: "conference-retry-13-1",
        },
        {
          resultId: "failed-1",
          collectionItemId: "14",
          retryAttempt: 3,
          idempotencyKey: "conference-retry-14-3",
        },
        {
          resultId: "cancel-1",
          collectionItemId: "15",
          retryAttempt: 1,
          idempotencyKey: "conference-retry-15-1",
        },
      ]
    )
  })

  it("uses canonical retry eligibility before legacy error classification", () => {
    const onRetryItems = vi.fn()
    wizardHarness.results = [
      {
        id: "server-denied-retry",
        status: "error",
        outcome: "failed",
        terminalOutcome: "processing_failed",
        retryable: false,
        type: "video",
        title: "Server denied retry",
        error: "Network timed out",
      },
      {
        id: "server-allowed-retry",
        status: "error",
        outcome: "failed",
        terminalOutcome: "processing_failed",
        retryable: true,
        type: "video",
        title: "Server allowed retry",
        error: "Permanent validation failure",
      },
    ] as any

    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onRetryItems={onRetryItems}
      />
    )

    expect(
      screen.queryByRole("button", { name: "Retry Server denied retry" })
    ).not.toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Retry Server allowed retry" })
    )
    expect(onRetryItems).toHaveBeenCalledWith(["server-allowed-retry"])
  })

  it("uses canonical retryability for durable conference retries", () => {
    const onRetryItems = vi.fn()
    wizardHarness.results = [
      {
        id: "conference-denied-retry",
        status: "error",
        outcome: "failed",
        terminalOutcome: "processing_failed",
        retryable: false,
        type: "video",
        title: "Conference denied retry",
        error: "Transient-looking timeout",
        collectionItemId: "21",
      },
      {
        id: "conference-allowed-retry",
        status: "error",
        outcome: "failed",
        terminalOutcome: "processing_failed",
        retryable: true,
        type: "video",
        title: "Conference allowed retry",
        error: "Permanent-looking validation error",
        collectionItemId: "22",
      },
      {
        id: "conference-second-allowed-retry",
        status: "error",
        outcome: "failed",
        terminalOutcome: "processing_failed",
        retryable: true,
        type: "video",
        title: "Conference second allowed retry",
        error: "Another retryable failure",
        collectionItemId: "23",
      },
    ] as any
    sessionHarness.tracking = {
      mode: "webui-direct",
      runId: "run-conference-canonical-retry",
      collectionId: "7",
      plannedItemIds: ["21", "22", "23"],
      durableMode: "durable_collection",
      startedAt: Date.now(),
    }

    render(
      <WizardResultsStep onClose={vi.fn()} onRetryItems={onRetryItems} />
    )

    expect(
      screen.queryByRole("button", { name: "Retry Conference denied retry" })
    ).not.toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Retry Conference allowed retry" })
    )
    expect(onRetryItems).toHaveBeenCalledWith(
      ["22"],
      [
        expect.objectContaining({
          resultId: "conference-allowed-retry",
          collectionItemId: "22",
        }),
      ]
    )

    onRetryItems.mockClear()
    fireEvent.click(
      screen.getByRole("button", { name: /Retry all .* retryable errors/i })
    )
    expect(onRetryItems).toHaveBeenCalledWith(
      ["22", "23"],
      [
        expect.objectContaining({
          resultId: "conference-allowed-retry",
          collectionItemId: "22",
        }),
        expect.objectContaining({
          resultId: "conference-second-allowed-retry",
          collectionItemId: "23",
        }),
      ]
    )
  })
})

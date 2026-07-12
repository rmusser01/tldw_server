import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  startQuickIngestSession: vi.fn(),
  submitQuickIngestBatch: vi.fn(),
  cancelQuickIngestSession: vi.fn(),
  reattachQuickIngestSession: vi.fn(),
  initialize: vi.fn(),
  getQuickIngestAnalysisProviderWarning: vi.fn(),
  checkConnection: vi.fn(),
  navigate: vi.fn(),
  runtimeListeners: [] as Array<(message: any) => void>,
  modalProps: [] as any[],
  afterCancelProcessing: null as null | (() => void),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [k: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      return defaultValueOrOptions?.defaultValue || key
    },
  }),
}))

vi.mock("antd", () => ({
  Modal: Object.assign(
    (props: any) => {
      mocks.modalProps.push(props)
      const { children, open, onCancel, className, title } = props
      return open ? (
        <div role="dialog" className={className}>
          <div className="ant-modal-content">
            <h2>{title}</h2>
            <button onClick={onCancel}>Close</button>
            {children}
          </div>
        </div>
      ) : null
    },
    {
      confirm: vi.fn(),
      destroyAll: vi.fn(),
    }
  ),
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
  Switch: ({ checked, onChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={checked}
      onChange={(event) => onChange?.(event.target.checked)}
      {...props}
    />
  ),
  Select: ({ value, onChange, options, ...props }: any) => (
    <select value={value} onChange={(event) => onChange?.(event.target.value)} {...props}>
      {(options || []).map((option: any) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Radio: Object.assign(
    ({ children, value, checked, onChange, ...props }: any) => (
      <label>
        <input
          type="radio"
          value={value}
          checked={checked}
          onChange={onChange}
          {...props}
        />
        {children}
      </label>
    ),
    {
      Group: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    }
  ),
  Collapse: ({ items }: any) => (
    <div>{items?.map((item: any) => <div key={item.key}>{item.children}</div>)}</div>
  ),
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return { ...actual, useNavigate: () => mocks.navigate }
})

vi.mock("@/routes/route-paths", () => ({
  DOCUMENT_WORKSPACE_PATH: "/document-workspace",
  buildMediaCollectionReviewPath: (collectionId: string | number) =>
    `/media-collections/${collectionId}`,
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (selector: any) =>
    selector({
      state: {
        phase: "connected",
        isConnected: true,
        isChecking: false,
        lastError: null,
        offlineBypass: false,
      },
      checkOnce: mocks.checkConnection,
    }),
}))

vi.mock("lucide-react", () => {
  const icon = (name: string) => (props: any) => (
    <span data-icon={name} aria-hidden={props?.["aria-hidden"]} />
  )
  return {
    ArrowLeft: icon("ArrowLeft"),
    ArrowRight: icon("ArrowRight"),
    ChevronDown: icon("ChevronDown"),
    Minimize2: icon("Minimize2"),
    XCircle: icon("XCircle"),
    Info: icon("Info"),
  }
})

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      onMessage: {
        addListener: (listener: (message: any) => void) => {
          mocks.runtimeListeners.push(listener)
        },
        removeListener: (listener: (message: any) => void) => {
          const index = mocks.runtimeListeners.indexOf(listener)
          if (index >= 0) {
            mocks.runtimeListeners.splice(index, 1)
          }
        },
      },
    },
  },
}))

vi.mock("@/services/tldw/quick-ingest-batch", () => ({
  startQuickIngestSession: (...args: unknown[]) => mocks.startQuickIngestSession(...args),
  submitQuickIngestBatch: (...args: unknown[]) => mocks.submitQuickIngestBatch(...args),
  cancelQuickIngestSession: (...args: unknown[]) => mocks.cancelQuickIngestSession(...args),
  getQuickIngestAnalysisProviderWarning: (...args: unknown[]) =>
    mocks.getQuickIngestAnalysisProviderWarning(...args),
}))

vi.mock("@/services/tldw/quick-ingest-session-reattach", () => ({
  reattachQuickIngestSession: (...args: unknown[]) =>
    mocks.reattachQuickIngestSession(...args),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
  },
}))

vi.mock("@/components/Common/QuickIngest/IngestWizardStepper", () => ({
  IngestWizardStepper: () => <div data-testid="wizard-stepper" />,
}))

vi.mock("@/components/Common/QuickIngest/AddContentStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    AddContentStep: ({
      onQuickProcess,
      quickProcessWarning,
    }: {
      onQuickProcess?: () => void
      quickProcessWarning?: string | null
    }) => {
      const context = actual.useIngestWizard() as any
      const { state, setQueueItems } = context
      return (
        <div>
          {quickProcessWarning ? <div role="alert">{quickProcessWarning}</div> : null}
          <button
            onClick={() => {
              setQueueItems([
                {
                  id: "queued-url-1",
                  url: "https://example.com/article",
                  detectedType: "web",
                  icon: "Globe",
                  fileSize: 0,
                  validation: { valid: true },
                },
              ])
              onQuickProcess?.()
            }}
          >
            Queue And Process
          </button>
          <button
            onClick={() => {
              context.setConferenceBatchMetadata({
                collectionName: "Strange Loop 2012",
                conferenceName: "Strange Loop",
                eventYear: "2012",
                sharedTags: ["conference", "clojure"],
                sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf",
              })
              setQueueItems([
                {
                  id: "conference-talk-1",
                  url: "https://youtube.com/watch?v=talk-1",
                  detectedType: "video",
                  icon: "Film",
                  fileSize: 0,
                  validation: { valid: true },
                  playlist: {
                    playlistId: "PL-conf",
                    playlistTitle: "Strange Loop 2012",
                    ordinal: 1,
                    normalizedSourceId: "youtube:video:talk-1",
                    duplicateStatus: "new",
                  },
                  conferenceOverride: {
                    selected: true,
                    title: "Simplicity Matters",
                    speaker: "Rich Hickey",
                    tags: ["keynote"],
                  },
                },
              ])
              onQuickProcess?.()
            }}
          >
            Queue Conference And Process
          </button>
          {state.queueItems.map((item) => (
            <div key={item.id} data-testid={`queued-item-${item.id}`}>
              <span>{item.fileName || item.url || item.id}</span>
              {item.validation.warnings?.map((warning) => (
                <span key={`${item.id}-${warning}`}>{warning}</span>
              ))}
            </div>
          ))}
        </div>
      )
    },
  }
})

vi.mock("@/components/Common/QuickIngest/ReviewStep", () => ({
  ReviewStep: () => <div data-testid="wizard-review" />,
}))

vi.mock("@/components/Common/QuickIngest/WizardConfigureStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    WizardConfigureStep: ({
      analysisProviderWarning,
      focusAnalysisProvider,
    }: {
      analysisProviderWarning?: string | null
      focusAnalysisProvider?: boolean
    }) => {
      const { state, setCustomOptions } = actual.useIngestWizard()
      const helpId = "analysis-provider-help"
      const warningId = "analysis-provider-warning"
      return (
        <div data-testid="wizard-configure">
          <label htmlFor="analysis-provider">Analysis provider</label>
          <input
            id="analysis-provider"
            role="combobox"
            aria-describedby={`${helpId}${analysisProviderWarning ? ` ${warningId}` : ""}`}
            autoFocus={focusAnalysisProvider}
            value={String(state.presetConfig.advancedValues?.api_name || "")}
            onChange={(event) =>
              setCustomOptions({
                advancedValues: {
                  api_name: event.target.value || undefined,
                },
              })
            }
          />
          <p id={helpId}>For this ingest</p>
          {analysisProviderWarning ? (
            <p id={warningId} role="alert" aria-live="assertive">
              {analysisProviderWarning}
            </p>
          ) : null}
        </div>
      )
    },
  }
})

vi.mock("@/components/Common/QuickIngest/ProcessingStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    ProcessingStep: ({ onCancelAll }: { onCancelAll?: () => void }) => {
      const { state, cancelProcessing } = actual.useIngestWizard()
      return (
        <div data-testid="wizard-processing">
          {state.processingState.status}:{state.processingState.perItemProgress.length}
          <button
            onClick={() => {
              if (onCancelAll) {
                onCancelAll()
              } else {
                cancelProcessing()
              }
              mocks.afterCancelProcessing?.()
            }}
          >
            Cancel Processing
          </button>
        </div>
      )
    },
  }
})

vi.mock("@/components/Common/QuickIngest/WizardResultsStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    WizardResultsStep: ({
      onOpenCollection,
      onIngestMore,
    }: {
      onOpenCollection?: (collectionId: string) => void
      onIngestMore?: () => void
    }) => {
      const { state, reset } = actual.useIngestWizard()
      return (
        <div data-testid="wizard-results">
          {state.processingState.status}:{state.results.length}
          {state.results.map((item) => (
            <div key={item.id} data-testid={`wizard-result-${item.id}`}>
              {item.id}:{item.outcome}:{item.message || ""}
            </div>
          ))}
          {onOpenCollection ? (
            <button
              type="button"
              onClick={() => onOpenCollection("7")}
            >
              Open collection
            </button>
          ) : null}
          <button type="button" onClick={onIngestMore || reset}>
            Start over
          </button>
        </div>
      )
    },
  }
})

vi.mock("@/components/Common/QuickIngest/FloatingProgressWidget", () => ({
  FloatingProgressWidget: () => null,
}))

import { QuickIngestWizardModal } from "@/components/Common/QuickIngestWizardModal"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"

const emitRuntimeMessage = (message: any) => {
  for (const listener of [...mocks.runtimeListeners]) {
    listener(message)
  }
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const SessionBackedQuickIngestModal = () => {
  const open = useQuickIngestSessionStore(
    (store) => store.session?.visibility === "visible"
  )
  return (
    <QuickIngestWizardModal
      open={open}
      onClose={() => useQuickIngestSessionStore.getState().hideSession()}
    />
  )
}

describe("QuickIngestWizardModal session runtime", () => {
  beforeEach(() => {
    mocks.runtimeListeners.splice(0, mocks.runtimeListeners.length)
    mocks.startQuickIngestSession.mockReset()
    mocks.submitQuickIngestBatch.mockReset()
    mocks.cancelQuickIngestSession.mockReset()
    mocks.reattachQuickIngestSession.mockReset()
    mocks.initialize.mockReset()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getQuickIngestAnalysisProviderWarning.mockReset()
    mocks.getQuickIngestAnalysisProviderWarning.mockReturnValue(null)
    mocks.checkConnection.mockReset()
    mocks.navigate.mockReset()
    mocks.modalProps.splice(0, mocks.modalProps.length)
    mocks.afterCancelProcessing = null
    mocks.cancelQuickIngestSession.mockResolvedValue({ ok: true })
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("submits the queued wizard batch through the authenticated quick-ingest transport", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-test",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/article",
          type: "html",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(screen.getByRole("dialog")).toHaveClass(
      "quick-ingest-modal",
      "quick-ingest-wizard-modal"
    )

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })

    expect(mocks.submitQuickIngestBatch).toHaveBeenCalledWith(
      expect.objectContaining({
        __quickIngestSessionId: "qi-direct-test",
        common: expect.objectContaining({
          perform_chunking: true,
          chunking_mode: "auto",
          auto_chunking_goal: "balanced",
          auto_chunking_use_llm: false,
        }),
        entries: [
          expect.objectContaining({
            id: "queued-url-1",
            url: "https://example.com/article",
            type: "html",
          }),
        ],
      })
    )

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("keeps AntD modal portal props stable while results land", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-stable-modal",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          outcome: "skipped",
          url: "https://example.com/article",
          type: "html",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })

    const renderedModalProps = mocks.modalProps.filter((props) => props.open)
    expect(renderedModalProps.length).toBeGreaterThan(1)
    expect(renderedModalProps.every((props) => props.getContainer === false)).toBe(
      true
    )
    expect(new Set(renderedModalProps.map((props) => props.styles)).size).toBe(1)
    expect(renderedModalProps[0].styles.body).toEqual({
      padding: "0 16px 16px",
      maxHeight: "calc(100vh - 180px)",
      overflowY: "auto",
    })
  })

  it("starts Ingest More in a new persisted session", async () => {
    const user = userEvent.setup()
    const firstSession = useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-first-run",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/article",
          type: "html",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await screen.findByTestId("wizard-results")

    await user.click(screen.getByRole("button", { name: "Start over" }))

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.id).not.toBe(
        firstSession.id
      )
    })
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "draft",
      currentStep: 1,
      tracking: undefined,
    })
  })

  it.each(["standard", "deep"] as const)(
    "processes the configured %s preset with its analysis provider",
    async (preset) => {
      const user = userEvent.setup()
      useQuickIngestSessionStore.getState().createDraftSession({
        selectedPreset: preset,
        customBasePreset: preset,
        presetConfig: {
          ...resolvePresetMap()[preset],
          advancedValues: { api_name: "openai" },
        },
      })
      mocks.getQuickIngestAnalysisProviderWarning.mockImplementation(
        ({ advancedValues }: any) =>
          advancedValues?.api_name ? null : "missing-provider"
      )
      mocks.startQuickIngestSession.mockResolvedValue({
        ok: true,
        sessionId: `qi-${preset}`,
      })
      mocks.submitQuickIngestBatch.mockResolvedValue({
        ok: true,
        results: [
          {
            id: "queued-url-1",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
          },
        ],
      })

      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await user.click(screen.getByRole("button", { name: "Queue And Process" }))

      await waitFor(() => {
        expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
      })
      expect(mocks.startQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          advancedValues: expect.objectContaining({ api_name: "openai" }),
        })
      )
    }
  )

  it.each(["standard", "deep"] as const)(
    "routes the %s preset to Configure when analysis needs a provider",
    async (preset) => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession({
      selectedPreset: preset,
      customBasePreset: preset,
      presetConfig: resolvePresetMap()[preset],
    })
    mocks.getQuickIngestAnalysisProviderWarning.mockImplementation(
      ({ advancedValues }: any) =>
        advancedValues?.api_name ? null : "missing-provider"
    )

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    expect(screen.getByTestId("wizard-configure")).toBeInTheDocument()
    expect(provider).toHaveFocus()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Choose an analysis provider before running ingest analysis."
    )
    expect(screen.queryByTestId("wizard-processing")).not.toBeInTheDocument()
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      currentStep: 2,
      lifecycle: "draft",
      processingState: { status: "idle" },
    })
    await user.type(provider, "openai")
    await waitFor(() => {
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    })
    }
  )

  it("does not enter processing when auto-process lacks an analysis provider", async () => {
    mocks.getQuickIngestAnalysisProviderWarning.mockReturnValue("missing-provider")
    useQuickIngestSessionStore.getState().createDraftSession({
      queueItems: [
        {
          id: "queued-url-1",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
    })

    render(
      <QuickIngestWizardModal
        open
        autoProcessQueued
        onClose={vi.fn()}
      />
    )

    expect(
      await screen.findByRole("combobox", { name: "Analysis provider" })
    ).toHaveFocus()
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      currentStep: 2,
      lifecycle: "draft",
      processingState: { status: "idle" },
    })
  })

  it("restores a hidden processing session when the late analysis provider guard blocks startRun", async () => {
    mocks.getQuickIngestAnalysisProviderWarning.mockReturnValue("missing-provider")
    useQuickIngestSessionStore.getState().createDraftSession({
      visibility: "hidden",
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "late-guard-url-1",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
      processingState: {
        status: "running",
        perItemProgress: [
          {
            id: "late-guard-url-1",
            status: "processing",
            progressPercent: 10,
            currentStage: "Processing",
            estimatedRemaining: 0,
          },
        ],
        elapsed: 0,
        estimatedRemaining: 0,
      },
    })

    render(<SessionBackedQuickIngestModal />)

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Choose an analysis provider before running ingest analysis."
      )
    })

    const session = useQuickIngestSessionStore.getState().session
    expect(session?.visibility).toBe("visible")
    expect(session?.currentStep).toBe(2)
    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    expect(provider).toHaveFocus()
    expect(provider.getAttribute("aria-describedby")).toContain(
      "analysis-provider-warning"
    )
    expect(screen.queryByTestId("wizard-processing")).not.toBeInTheDocument()
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
  })

  it("preserves first-source open detail while syncing wizard state", async () => {
    const user = userEvent.setup()
    const firstSourceDetail = {
      source: "first_source_milestone" as const,
      preferredPreset: "quick" as const,
      firstSource: true,
      firstSourceKind: "file_upload" as const,
    }
    useQuickIngestSessionStore.getState().createDraftSession({
      openDetail: firstSourceDetail,
    })
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-first-source",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/article",
          type: "html",
          mediaId: "42",
          title: "Example article",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
    expect(useQuickIngestSessionStore.getState().session?.openDetail).toEqual(
      firstSourceDetail
    )
  })

  it("syncs cleared first-source add mode from wizard reset", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession({
      firstSourceAddMode: "paste_text",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await screen.findByTestId("wizard-results")
    await user.click(screen.getByRole("button", { name: "Start over" }))

    await waitFor(() => {
      expect(
        useQuickIngestSessionStore.getState().session?.firstSourceAddMode,
      ).toBeNull()
    })
  })

  it("submits conference batch metadata and item overrides through the session payload", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-conference",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "conference-talk-1",
          status: "ok",
          url: "https://youtube.com/watch?v=talk-1",
          type: "video",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(
      screen.getByRole("button", { name: "Queue Conference And Process" })
    )

    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })
    expect(mocks.submitQuickIngestBatch).toHaveBeenCalledWith(
      expect.objectContaining({
        __quickIngestSessionId: "qi-direct-conference",
        conferenceBatchMetadata: {
          collectionName: "Strange Loop 2012",
          conferenceName: "Strange Loop",
          eventYear: "2012",
          sharedTags: ["conference", "clojure"],
          sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf",
        },
        entries: [
          expect.objectContaining({
            id: "conference-talk-1",
            url: "https://youtube.com/watch?v=talk-1",
            type: "video",
            playlist: expect.objectContaining({
              playlistId: "PL-conf",
              ordinal: 1,
              normalizedSourceId: "youtube:video:talk-1",
            }),
            conferenceOverride: expect.objectContaining({
              selected: true,
              title: "Simplicity Matters",
              speaker: "Rich Hickey",
              tags: ["keynote"],
            }),
          }),
        ],
      })
    )
  })

  it("does not pre-seed direct tracking item identities before backend submissions are acknowledged", async () => {
    let resolveBatch: ((value: any) => void) | null = null
    const batchPromise = new Promise((resolve) => {
      resolveBatch = resolve
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article-1",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
        {
          id: "queued-url-2",
          kind: "url",
          url: "https://example.com/article-2",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 0,
        estimatedRemaining: 0,
      },
    })

    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-tracking-preseed",
    })
    mocks.submitQuickIngestBatch.mockImplementation(() => batchPromise)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })

    const tracking = useQuickIngestSessionStore.getState().session?.tracking
    expect(tracking?.mode).toBe("webui-direct")
    expect(tracking?.sessionId).toBe("qi-direct-tracking-preseed")
    expect(tracking?.submittedItemIds).toBeUndefined()
    expect(tracking?.itemIds).toBeUndefined()

    resolveBatch?.({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          type: "html",
        },
      ],
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("starts persisted direct-job reattach when tracking metadata arrives after the run begins", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()

    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-late-tracking",
    })
    mocks.submitQuickIngestBatch.mockImplementation((payload: any) => {
      payload?.onTrackingMetadata?.({
        mode: "webui-direct",
        sessionId: "qi-direct-late-tracking",
        batchId: "batch-77",
        batchIds: ["batch-77"],
        jobIds: [77],
        collectionId: "7",
        plannedItemIds: ["11"],
        itemIds: ["queued-url-1"],
        submittedItemIds: ["queued-url-1"],
        jobIdToItemId: { "77": "queued-url-1" },
        jobIdToCollectionItemId: { "77": "11" },
        durableMode: "durable_collection",
        startedAt: Date.now(),
      })
      return new Promise(() => {})
    })
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "completed",
      jobs: [
        {
          jobId: 77,
          status: "completed",
          sourceItemId: "queued-url-1",
          result: { media_id: "media-77", title: "Recovered Result" },
        },
      ],
      errorMessage: null,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "webui-direct",
          sessionId: "qi-direct-late-tracking",
          batchIds: ["batch-77"],
          jobIds: [77],
          collectionId: "7",
          plannedItemIds: ["11"],
          jobIdToCollectionItemId: { "77": "11" },
          durableMode: "durable_collection",
        })
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("keeps cancellation terminal when runtime completion arrives in the cancel click", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-cancel-completion-race",
    })
    mocks.afterCancelProcessing = () => {
      emitRuntimeMessage({
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "qi-runtime-cancel-completion-race",
          results: [
            {
              id: "queued-url-1",
              status: "ok",
              url: "https://example.com/article",
              type: "html",
            },
          ],
        },
      })
    }

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
    })
    expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
      "queued-url-1:cancelled"
    )
  })

  it("ignores runtime progress emitted in the cancel click", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-cancel-progress-race",
    })
    mocks.afterCancelProcessing = () => {
      emitRuntimeMessage({
        type: "tldw:quick-ingest/progress",
        payload: {
          sessionId: "qi-runtime-cancel-progress-race",
          result: {
            id: "queued-url-1",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
          },
        },
      })
    }

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
    })
    expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
      "queued-url-1:cancelled"
    )
  })

  it("cancels an extension session acknowledged after cancellation", async () => {
    const user = userEvent.setup()
    const startAck = deferred<any>()
    const cancelError = new Error("cancel transport unavailable")
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockReturnValue(startAck.promise)
    mocks.cancelQuickIngestSession.mockRejectedValueOnce(cancelError)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))
    startAck.resolve({ ok: true, sessionId: "qi-runtime-late-ack" })

    await waitFor(() => {
      expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionId: "qi-runtime-late-ack",
          reason: "user_cancelled",
        })
      )
    })
    await waitFor(() => {
      expect(warnSpy).toHaveBeenCalledWith(
        "[QuickIngest] Failed to cancel session.",
        {
          sessionId: "qi-runtime-late-ack",
          error: cancelError,
        }
      )
    })
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
  })

  it("does not submit a direct session acknowledged after cancellation", async () => {
    const user = userEvent.setup()
    const startAck = deferred<any>()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockReturnValue(startAck.promise)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))
    startAck.resolve({ ok: true, sessionId: "qi-direct-late-ack" })

    await act(async () => {
      await startAck.promise
      await Promise.resolve()
    })

    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
  })

  it("does not start a session when setup resumes after cancellation", async () => {
    const user = userEvent.setup()
    const setup = deferred<void>()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.initialize.mockReturnValue(setup.promise)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.initialize).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))
    setup.resolve()

    await act(async () => {
      await setup.promise
      await Promise.resolve()
    })

    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
  })

  it("keeps cancellation terminal when start acknowledgement rejects", async () => {
    const user = userEvent.setup()
    const startAck = deferred<any>()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockReturnValue(startAck.promise)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))
    startAck.reject(new Error("late start failure"))

    await act(async () => {
      try {
        await startAck.promise
      } catch {
        // startRun owns the rejection; this await only flushes the deferred promise.
      }
      await Promise.resolve()
    })

    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
    expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
      "queued-url-1:cancelled"
    )
  })

  it("uses runtime completion events for extension-backed sessions instead of calling the broken SSE path", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-test",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(screen.getByRole("dialog")).toHaveClass(
      "quick-ingest-modal",
      "quick-ingest-wizard-modal"
    )

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-runtime-test",
        results: [
          {
            id: "queued-url-1",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
          },
        ],
      },
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("normalizes runtime duplicate results from db_message into skipped items", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-duplicate",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-runtime-duplicate",
        results: [
          {
            id: "queued-url-1",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
            data: {
              db_message:
                "Media 'https://example.com/article' already exists. Overwrite not enabled.",
            },
          },
        ],
      },
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
        "queued-url-1:skipped"
      )
    })
    expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
      "already exists in your library"
    )
  })

  it("normalizes runtime ok results with error payloads into failed items", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-error-payload",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-runtime-error-payload",
        results: [
          {
            id: "queued-url-1",
            status: "ok",
            url: "http://127.0.0.1:3000/e2e/quick-ingest-source.html",
            type: "html",
            data: {
              status: "Error",
              error: "File preparation/download failed: Port not allowed: 3000",
            },
          },
        ],
      },
    })

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.results).toEqual([
        expect.objectContaining({
          id: "queued-url-1",
          status: "error",
          outcome: "failed",
          error: "File preparation/download failed: Port not allowed: 3000",
        }),
      ])
    })
  })

  it("rehydrates a hidden processing session when the modal is reopened", () => {
    const onClose = vi.fn()

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      visibility: "hidden",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [
          {
            id: "queued-url-1",
            status: "processing",
            progressPercent: 40,
            currentStage: "Processing",
            estimatedRemaining: 12,
          },
        ],
        elapsed: 5,
        estimatedRemaining: 12,
      },
    })

    const { rerender } = render(
      <QuickIngestWizardModal open={false} onClose={onClose} />
    )

    rerender(<QuickIngestWizardModal open onClose={onClose} />)

    expect(screen.getByTestId("wizard-processing")).toHaveTextContent("running:1")
  })

  it("rehydrates a completed session with results after a remount", () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "completed",
      currentStep: 5,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "complete",
        perItemProgress: [
          {
            id: "queued-url-1",
            status: "complete",
            progressPercent: 100,
            currentStage: "Complete",
            estimatedRemaining: 0,
          },
        ],
        elapsed: 4,
        estimatedRemaining: 0,
      },
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/article",
          type: "html",
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
  })

  it("restores persisted file stubs with a reattach-required warning", () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft",
      currentStep: 1,
      queueItems: [
        {
          id: "queued-file-1",
          kind: "file",
          fileName: "clip.mkv",
          detectedType: "video",
          icon: "Film",
          fileSize: 1024,
          mimeType: "video/x-matroska",
          validation: {
            valid: false,
            warnings: ["Reattach this file after refresh to process it."],
          },
          fileStub: {
            key: "clip.mkv::1024::1700000000000",
            lastModified: 1700000000000,
          },
        } as any,
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(screen.getByTestId("queued-item-queued-file-1")).toHaveTextContent("clip.mkv")
    expect(screen.getByText("Reattach this file after refresh to process it.")).toBeVisible()
  })

  it("reattaches persisted direct-ingest jobs after refresh", async () => {
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "completed",
      jobs: [
        {
          jobId: 77,
          status: "completed",
          result: {
            media_id: "media-77",
            title: "Recovered Result",
          },
        },
      ],
      errorMessage: null,
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [
          {
            id: "queued-url-1",
            status: "processing",
            progressPercent: 30,
            currentStage: "Processing",
            estimatedRemaining: 20,
          },
        ],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        batchId: "batch-77",
        jobIds: [77],
        startedAt: Date.now(),
      },
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "webui-direct",
          batchId: "batch-77",
          jobIds: [77],
        })
      )
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("maps refreshed file-backed reattach results back to the original queued item id", async () => {
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "completed",
      jobs: [
        {
          jobId: 77,
          status: "completed",
          result: {
            media_id: "media-file-77",
            title: "Recovered MKV Result",
          },
        },
      ],
      errorMessage: null,
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-file-1",
          kind: "file",
          fileName: "clip.mkv",
          detectedType: "video",
          icon: "Film",
          fileSize: 1024,
          mimeType: "video/x-matroska",
          validation: {
            valid: false,
            warnings: ["Reattach this file after refresh to process it."],
          },
          fileStub: {
            key: "clip.mkv::1024::1700000000000",
            lastModified: 1700000000000,
          },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-file-refresh",
        batchId: "batch-file-77",
        batchIds: ["batch-file-77"],
        jobIds: [77],
        itemIds: ["queued-file-1"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })

    expect(useQuickIngestSessionStore.getState().session?.results).toEqual([
      expect.objectContaining({
        id: "queued-file-1",
        fileName: "clip.mkv",
        mediaId: "media-file-77",
      }),
    ])
  })

  it("does not run persisted direct-job reattach for extension runtime sessions", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-runtime-refresh",
        itemIds: ["queued-url-1"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-runtime-refresh",
        results: [
          {
            id: "queued-url-1",
            status: "ok",
            url: "https://example.com/article",
            type: "html",
          },
        ],
      },
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("restarts direct processing after refresh when tracking exists without persisted job ids", async () => {
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-restarted",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: true,
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/article",
          type: "html",
        },
      ],
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-ack-only",
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
  })

  it("cancels a refreshed direct session using persisted tracking metadata", async () => {
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [{ jobId: 77, status: "processing" }],
      errorMessage: null,
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-refresh",
        batchId: "batch-77",
        batchIds: ["batch-77"],
        jobIds: [77],
        itemIds: ["queued-url-1"],
        startedAt: Date.now(),
      } as any,
    })

    const user = userEvent.setup()
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalled()
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))

    await waitFor(() => {
      expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionId: "qi-direct-refresh",
          batchIds: ["batch-77"],
          reason: "user_cancelled",
        })
      )
    })
  })

  it("ignores late persisted reattach processing after cancellation", async () => {
    vi.useFakeTimers()
    const reattach = deferred<any>()
    mocks.reattachQuickIngestSession.mockReturnValue(reattach.promise)

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-late-processing",
        batchIds: ["batch-77"],
        jobIds: [77],
        itemIds: ["queued-url-1"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Cancel Processing" }))
    reattach.resolve({
      lifecycle: "processing",
      jobs: [{ jobId: 77, status: "processing" }],
      errorMessage: null,
    })
    await act(async () => {
      await reattach.promise
      await Promise.resolve()
    })

    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)
  })

  it("ignores late persisted reattach completion after cancellation", async () => {
    const user = userEvent.setup()
    const reattach = deferred<any>()
    mocks.reattachQuickIngestSession.mockReturnValue(reattach.promise)

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-late-completion",
        batchIds: ["batch-77"],
        jobIds: [77],
        itemIds: ["queued-url-1"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))
    reattach.resolve({
      lifecycle: "completed",
      jobs: [
        {
          jobId: 77,
          status: "completed",
          result: { media_id: "media-77", title: "Late completion" },
        },
      ],
      errorMessage: null,
    })
    await act(async () => {
      await reattach.promise
      await Promise.resolve()
    })

    expect(screen.getByTestId("wizard-results")).toHaveTextContent("cancelled:1")
    expect(screen.getByTestId("wizard-result-queued-url-1")).toHaveTextContent(
      "queued-url-1:cancelled"
    )
  })

  it("reruns persisted direct-session reattach when item mapping metadata arrives later", async () => {
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [{ jobId: 77, status: "processing" }],
      errorMessage: null,
    })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-refresh-signature",
        batchId: "batch-77",
        batchIds: ["batch-77"],
        jobIds: [77],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    const existingSession = useQuickIngestSessionStore.getState().session
    expect(existingSession).toBeTruthy()

    useQuickIngestSessionStore.getState().upsertSession({
      ...existingSession!,
      tracking: {
        ...existingSession!.tracking,
        itemIds: ["queued-url-1"],
        jobIdToItemId: { "77": "queued-url-1" },
      } as any,
    })

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(2)
    })
  })

  it("preserves already completed item results when cancellation finalizes pending items", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/already-complete",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
        {
          id: "queued-url-2",
          kind: "url",
          url: "https://example.com/pending",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [
          {
            id: "queued-url-1",
            status: "complete",
            progressPercent: 100,
            currentStage: "Complete",
            estimatedRemaining: 0,
          },
          {
            id: "queued-url-2",
            status: "processing",
            progressPercent: 50,
            currentStage: "Processing",
            estimatedRemaining: 12,
          },
        ],
        elapsed: 3,
        estimatedRemaining: 12,
      },
      results: [
        {
          id: "queued-url-1",
          status: "ok",
          url: "https://example.com/already-complete",
          type: "html",
        } as any,
      ],
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-runtime-cancel-preserve",
        itemIds: ["queued-url-1", "queued-url-2"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Cancel Processing" }))

    await waitFor(() => {
      const sessionResults = useQuickIngestSessionStore.getState().session?.results || []
      expect(sessionResults).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "queued-url-1",
            status: "ok",
          }),
          expect.objectContaining({
            id: "queued-url-2",
            status: "error",
            outcome: "cancelled",
          }),
        ])
      )
    })
  })

  it("opens durable conference collections from terminal results", async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "completed",
      currentStep: 5,
      processingState: {
        status: "complete",
        perItemProgress: [],
        elapsed: 7,
        estimatedRemaining: 0,
      },
      results: [
        {
          id: "conference-talk-1",
          status: "ok",
          url: "https://youtube.com/watch?v=talk-1",
          type: "video",
          mediaId: "101",
        } as any,
      ],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-conference",
        collectionId: "7",
        durableMode: "durable_collection",
        startedAt: 1234,
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={onClose} />)

    await user.click(screen.getByRole("button", { name: "Open collection" }))

    expect(onClose).toHaveBeenCalledTimes(1)
    await waitFor(() => {
      expect(mocks.navigate).toHaveBeenCalledWith("/media-collections/7")
    })
  })

  it("keeps polling persisted direct-job reattach until the resumed session reaches a terminal state", async () => {
    vi.useFakeTimers()
    mocks.reattachQuickIngestSession
      .mockResolvedValueOnce({
        lifecycle: "processing",
        jobs: [{ jobId: 77, status: "processing" }],
        errorMessage: null,
      })
      .mockResolvedValueOnce({
        lifecycle: "completed",
        jobs: [
          {
            jobId: 77,
            status: "completed",
            result: { media_id: "media-77", title: "Recovered Result" },
          },
        ],
        errorMessage: null,
      })

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-refresh-loop",
        batchId: "batch-77",
        batchIds: ["batch-77"],
        jobIds: [77],
        itemIds: ["queued-url-1"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })

    expect(mocks.reattachQuickIngestSession).toHaveBeenCalledTimes(2)
    expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
  })
})

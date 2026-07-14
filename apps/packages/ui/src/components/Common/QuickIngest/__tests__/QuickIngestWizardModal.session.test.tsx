import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  startQuickIngestSession: vi.fn(),
  submitQuickIngestBatch: vi.fn(),
  cancelQuickIngestSession: vi.fn(),
  queryQuickIngestSession: vi.fn(),
  acknowledgeQuickIngestSessionReplay: vi.fn(),
  reattachQuickIngestSession: vi.fn(),
  initialize: vi.fn(),
  getQuickIngestAnalysisProviderWarning: vi.fn(),
  checkConnection: vi.fn(),
  navigate: vi.fn(),
  runtimeListeners: [] as Array<(message: any) => void>,
  modalProps: [] as any[],
  afterCancelProcessing: null as null | (() => void),
  connectionState: {
    phase: "connected",
    isConnected: true,
    isChecking: false,
    lastError: null as string | null,
    offlineBypass: false,
  },
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
    <div>
      {items?.map((item: any) => (
        <div key={item.key}>{item.children}</div>
      ))}
    </div>
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
      state: mocks.connectionState,
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
  queryQuickIngestSession: (...args: unknown[]) => mocks.queryQuickIngestSession(...args),
  acknowledgeQuickIngestSessionReplay: (...args: unknown[]) =>
    mocks.acknowledgeQuickIngestSessionReplay(...args),
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
              setQueueItems([
                {
                  id: "queued-partial-accepted",
                  kind: "url",
                  url: "https://example.com/accepted",
                  sourceRef: {
                    kind: "direct_url",
                    occurrenceId: "queued-partial-accepted",
                    url: "https://example.com/accepted",
                  },
                  detectedType: "web",
                  icon: "Globe",
                  fileSize: 0,
                  validation: { valid: true },
                },
                {
                  id: "queued-partial-unsent",
                  kind: "url",
                  url: "https://example.com/unsent",
                  sourceRef: {
                    kind: "direct_url",
                    occurrenceId: "queued-partial-unsent",
                    url: "https://example.com/unsent",
                  },
                  detectedType: "web",
                  icon: "Globe",
                  fileSize: 0,
                  validation: { valid: true },
                },
              ])
              onQuickProcess?.()
            }}
          >
            Queue Partial Submission And Process
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
                  kind: "url",
                  url: "https://youtube.com/watch?v=talk-1",
                  sourceRef: {
                    kind: "materialized_playlist_item",
                    materializationId: "conference-materialization",
                    occurrenceId: "conference-talk-1",
                  },
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
                    materializationExpiresAt: "2099-01-01T00:00:00Z",
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
          <button
            onClick={() => {
              setQueueItems([
                {
                  id: "occ-persisted-playlist",
                  kind: "url",
                  url: "https://cached.example.invalid/watch?v=display-only",
                  sourceRef: {
                    kind: "materialized_playlist_item",
                    materializationId: "opaque-owner-bound-materialization",
                    occurrenceId: "occ-persisted-playlist",
                  },
                  detectedType: "video",
                  icon: "Film",
                  fileSize: 0,
                  validation: { valid: true },
                  playlist: {
                    title: "Persisted playlist row",
                    playlistTitle: "Persisted playlist",
                    ordinal: 7,
                    materializationExpiresAt: "2099-01-01T00:00:00Z",
                  },
                  playlistReview: {
                    selected: true,
                    duplicatePolicy: "overwrite",
                    editedFields: ["title"],
                    metadataPatch: { title: "Edited persisted title" },
                  },
                },
              ])
            }}
          >
            Queue Playlist Draft
          </button>
          <button
            onClick={() => {
              setQueueItems([
                {
                  id: "occ-blocked-quick",
                  kind: "url",
                  url: "https://cached.example.invalid/watch?v=blocked-quick",
                  sourceRef: {
                    kind: "materialized_playlist_item",
                    materializationId: "blocked-quick-materialization",
                    occurrenceId: "occ-blocked-quick",
                  },
                  detectedType: "video",
                  icon: "Film",
                  fileSize: 0,
                  validation: { valid: true },
                  playlist: {
                    duplicateStatus: "duplicate_existing",
                    materializationExpiresAt: "2099-01-01T00:00:00Z",
                  },
                  playlistReview: { selected: true },
                },
              ])
              onQuickProcess?.()
            }}
          >
            Queue Blocked Duplicate And Process
          </button>
          <button
            onClick={() => {
              setQueueItems([
                {
                  id: "direct-review-reload",
                  kind: "url",
                  url: "https://example.com/direct-review-reload",
                  sourceRef: {
                    kind: "direct_url",
                    occurrenceId: "direct-review-reload",
                    url: "https://example.com/direct-review-reload",
                  },
                  detectedType: "web",
                  icon: "Globe",
                  fileSize: 0,
                  validation: { valid: true },
                },
              ])
              context.applyPlaylistReviewRequired([
                {
                  occurrenceId: "direct-review-reload",
                  reason: "duplicate_action_required",
                  evidence: {
                    kind: "library",
                    existingMediaId: 42,
                    duplicateOfOccurrenceId: null,
                  },
                  allowedActions: ["skip", "include_existing", "overwrite"],
                },
              ])
            }}
          >
            Queue Direct Duplicate Review
          </button>
          {state.queueItems.map((item) => (
            <div key={item.id}
              data-testid={`queued-item-${item.id}`}
              data-kind={item.kind ?? ""}
              data-url={item.url ?? ""}
              data-file-name={item.fileName ?? ""}
              data-source-ref={JSON.stringify(item.sourceRef ?? null)}
              data-playlist={JSON.stringify(item.playlist ?? null)}
              data-playlist-review={JSON.stringify(item.playlistReview ?? null)}
              data-validation={JSON.stringify(item.validation)}
            >
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

vi.mock("@/components/Common/QuickIngest/ReviewStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    ReviewStep: () => {
      const { state, startProcessing, updateQueueItems } = actual.useIngestWizard()
      return (
        <div
          data-testid="wizard-review"
          data-processing-block={state.processingBlock?.code ?? ""}
        >
          <button
            onClick={() =>
              updateQueueItems((current) =>
                current.map((item) => ({
                  ...item,
                  playlistReview: {
                    selected: item.playlistReview?.selected ?? true,
                    ...(item.playlistReview || {}),
                    duplicatePolicy: "skip",
                  },
                }))
              )
            }
          >
            Resolve duplicate
          </button>
          <button onClick={startProcessing}>Start reviewed processing</button>
          {state.queueItems.map((item) => (
            <div
              key={item.id}
              data-testid={`review-item-${item.id}`}
              data-source-ref={JSON.stringify(item.sourceRef ?? null)}
              data-playlist={JSON.stringify(item.playlist ?? null)}
              data-validation={JSON.stringify(item.validation)}
            />
          ))}
        </div>
      )
    },
  }
})

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
      const inputRef = React.useRef<HTMLInputElement>(null)
      React.useEffect(() => {
        if (focusAnalysisProvider) {
          inputRef.current?.focus()
        }
      }, [focusAnalysisProvider])
      return (
        <div data-testid="wizard-configure">
          <label htmlFor="analysis-provider">Analysis provider</label>
          <input
            ref={inputRef}
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
              onClick={() => onOpenCollection("7")}>
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
import { createQuickIngestSessionRuntime } from "@/entries/shared/quick-ingest-session-runtime"
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
    mocks.queryQuickIngestSession.mockReset()
    mocks.acknowledgeQuickIngestSessionReplay.mockReset()
    mocks.reattachQuickIngestSession.mockReset()
    mocks.initialize.mockReset()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getQuickIngestAnalysisProviderWarning.mockReset()
    mocks.getQuickIngestAnalysisProviderWarning.mockReturnValue(null)
    mocks.checkConnection.mockReset()
    mocks.navigate.mockReset()
    mocks.modalProps.splice(0, mocks.modalProps.length)
    mocks.afterCancelProcessing = null
    Object.assign(mocks.connectionState, {
      phase: "connected",
      isConnected: true,
      isChecking: false,
      lastError: null,
      offlineBypass: false,
    })
    mocks.cancelQuickIngestSession.mockResolvedValue({ ok: true })
    mocks.queryQuickIngestSession.mockResolvedValue({ ok: true, active: true, event: null })
    mocks.acknowledgeQuickIngestSessionReplay.mockResolvedValue({ ok: true })
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("routes a blocked manual quick process to visible Review feedback", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(
      screen.getByRole("button", { name: "Queue Blocked Duplicate And Process" })
    )

    expect(await screen.findByTestId("wizard-review")).toHaveAttribute(
      "data-processing-block",
      "review_required"
    )
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
  })

  it("keeps blocked auto-process retry available until Review resolves the block", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession({
      queueItems: [
        {
          id: "auto-blocked-duplicate",
          kind: "url",
          url: "https://cached.example.invalid/watch?v=auto-blocked",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "auto-blocked-materialization",
            occurrenceId: "auto-blocked-duplicate",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            duplicateStatus: "duplicate_existing",
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
          playlistReview: { selected: true },
        },
      ],
    } as never)
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "auto-blocked-session",
    })
    mocks.submitQuickIngestBatch.mockReturnValue(new Promise(() => {}))

    render(<QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />)

    const review = await screen.findByTestId("wizard-review")
    expect(review).toHaveAttribute("data-processing-block", "review_required")
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Resolve duplicate" }))

    await waitFor(() => expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1))
    expect(screen.getByTestId("wizard-processing")).toHaveTextContent("running:1")
  })

  it("reloads fresh duplicate review metadata without invalidating direct URL authority", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    const mounted = render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue Direct Duplicate Review" }))
    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.queueItems[0]).toMatchObject({
        sourceRef: {
          kind: "direct_url",
          occurrenceId: "direct-review-reload",
          url: "https://example.com/direct-review-reload",
        },
        validation: { valid: true },
        playlist: { duplicateStatus: "duplicate_existing" },
      })
    })

    mounted.unmount()
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    const restored = screen.getByTestId("review-item-direct-review-reload")
    expect(JSON.parse(restored.getAttribute("data-source-ref") || "null")).toEqual({
      kind: "direct_url",
      occurrenceId: "direct-review-reload",
      url: "https://example.com/direct-review-reload",
    })
    expect(JSON.parse(restored.getAttribute("data-playlist") || "null")).toMatchObject({
      duplicateStatus: "duplicate_existing",
    })
    expect(JSON.parse(restored.getAttribute("data-validation") || "null")).toMatchObject({
      valid: true,
    })
  })

  it("hydrates orphaned materialization cues as an invalid display-only row", () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft",
      currentStep: 1,
      queueItems: [
        {
          id: "hydrate-orphaned-materialized-cues",
          kind: "url",
          url: "https://cached.example.invalid/hydrate-orphaned-cues",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            sourceUrl: "https://cached.example.invalid/hydrate-orphaned-cues",
            playlistTitle: "Lost materialization",
          },
        },
      ],
    } as never)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    const row = screen.getByTestId("queued-item-hydrate-orphaned-materialized-cues")
    expect(row).toHaveAttribute(
      "data-url",
      "https://cached.example.invalid/hydrate-orphaned-cues"
    )
    expect(row).toHaveAttribute("data-source-ref", "null")
    expect(JSON.parse(row.getAttribute("data-validation") || "null")).toMatchObject({
      valid: false,
      errors: ["Reattach this source before processing."],
    })
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
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
        pendingRunRequest: {
          inputs: [
            {
              inputKind: "direct_url",
              occurrenceId: "queued-url-1",
              url: "https://example.com/article",
              displayMetadata: { title: null },
            },
          ],
        },
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

  it.each([2, 3] as const)(
    "routes an auto-process provider warning from persisted step %s to Configure",
    async (currentStep) => {
    mocks.getQuickIngestAnalysisProviderWarning.mockReturnValue("missing-provider")
    useQuickIngestSessionStore.getState().createDraftSession({
      currentStep,
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
      <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
    )

    expect(
      await screen.findByRole("combobox", { name: "Analysis provider" })
    ).toHaveFocus()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      currentStep: 2,
      lifecycle: "draft",
      processingState: { status: "idle" },
    })
    expect(screen.queryByTestId("wizard-review")).not.toBeInTheDocument()
    }
  )

  it("retries auto-process after closing and reopening a provider-blocked draft", async () => {
    const user = userEvent.setup()
    mocks.getQuickIngestAnalysisProviderWarning.mockImplementation(
      ({ advancedValues }: any) =>
        advancedValues?.api_name ? null : "missing-provider"
    )
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-reopened-provider",
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

    const { rerender } = render(
      <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
    )
    const provider = await screen.findByRole("combobox", {
      name: "Analysis provider",
    })
    await user.type(provider, "openai")

    rerender(
      <QuickIngestWizardModal open={false} autoProcessQueued onClose={vi.fn()} />
    )
    rerender(
      <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
    )

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })
  })

  it("waits for an in-flight connection check before consuming auto-process", async () => {
    mocks.connectionState.isChecking = true
    useQuickIngestSessionStore.getState().createDraftSession({
      presetConfig: {
        ...resolvePresetMap().standard,
        advancedValues: { api_name: "openai" },
      },
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
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-after-connection-check",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [] })

    const { rerender } = render(
      <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
    )
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()

    mocks.connectionState.isChecking = false
    rerender(
      <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
    )

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
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
    expect(useQuickIngestSessionStore.getState().session?.openDetail).toEqual(firstSourceDetail)
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
      expect(useQuickIngestSessionStore.getState().session?.firstSourceAddMode).toBeNull()
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

    await user.click(screen.getByRole("button", { name: "Queue Conference And Process" }))

    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })
    const pendingRunRequest = {
      inputs: [
        {
          inputKind: "materialized_playlist_item",
          occurrenceId: "conference-talk-1",
          materializationId: "conference-materialization",
        },
      ],
    }
    expect(mocks.startQuickIngestSession).toHaveBeenCalledWith(
      expect.objectContaining({
        entries: [],
        pendingRunRequest,
      })
    )
    expect(mocks.submitQuickIngestBatch).toHaveBeenCalledWith(
      expect.objectContaining({
        __quickIngestSessionId: "qi-direct-conference",
        pendingRunRequest,
        conferenceBatchMetadata: {
          collectionName: "Strange Loop 2012",
          conferenceName: "Strange Loop",
          eventYear: "2012",
          sharedTags: ["conference", "clojure"],
          sourcePlaylistUrl: "https://youtube.com/playlist?list=PL-conf",
        },
        conferenceItemMetadata: {
          "conference-talk-1": {
            playlist: expect.objectContaining({
              playlistId: "PL-conf",
              normalizedSourceId: "youtube:video:talk-1",
            }),
            conferenceOverride: {
              selected: true,
              title: "Simplicity Matters",
              speaker: "Rich Hickey",
              tags: ["keynote"],
            },
          },
        },
        entries: [],
      })
    )
    expect(JSON.stringify(mocks.submitQuickIngestBatch.mock.calls)).not.toContain(
      "https://youtube.com/watch?v=talk-1"
    )
  })

  it("keeps an accepted version-2 run processing until run status is terminal", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-run-accepted",
    })
    mocks.submitQuickIngestBatch.mockImplementation(async (payload: any) => {
      payload.onTrackingMetadata({
        mode: "webui-direct",
        runId: "run-accepted",
        jobIds: [801],
        submittedItemIds: ["conference-talk-1"],
        startedAt: Date.now(),
      })
      return { ok: true, accepted: true, runId: "run-accepted" }
    })
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 801,
          status: "running",
          sourceItemId: "conference-talk-1",
        },
      ],
      errorMessage: null,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue Conference And Process" }))

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({ runId: "run-accepted" })
      )
    })
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "processing",
      tracking: { runId: "run-accepted" },
    })
  })

  it("returns a version-2 review-required response to Review without tracking a run", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-review-required",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: false,
      error: "Review the updated duplicate choices before continuing.",
      reviewRequired: [
        {
          occurrenceId: "conference-talk-1",
          reason: "duplicate_action_required",
          evidence: {
            kind: "library",
            existingMediaId: 42,
            duplicateOfOccurrenceId: null,
          },
          allowedActions: ["skip", "include_existing", "overwrite"],
        },
      ],
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue Conference And Process" }))

    expect(await screen.findByTestId("wizard-review")).toHaveAttribute(
      "data-processing-block",
      "review_required"
    )
    expect(useQuickIngestSessionStore.getState().session?.tracking).toBeUndefined()
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
  })

  it("returns structured extension review-required recovery to Review", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-extension-review-required",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue Conference And Process" }))
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/review-required",
      payload: {
        sessionId: "qi-extension-review-required",
        reviewRequired: [
          {
            occurrenceId: "conference-talk-1",
            reason: "duplicate_action_required",
            evidence: {
              kind: "library",
              existingMediaId: 42,
              duplicateOfOccurrenceId: null,
            },
            allowedActions: ["skip", "include_existing", "overwrite"],
          },
        ],
      },
    })

    expect(await screen.findByTestId("wizard-review")).toHaveAttribute(
      "data-processing-block",
      "review_required"
    )
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
  })

  it("clears extension review authority and starts exactly one corrected retry", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession
      .mockResolvedValueOnce({
        ok: true,
        sessionId: "qi-extension-review-first",
      })
      .mockResolvedValueOnce({
        ok: true,
        sessionId: "qi-extension-review-corrected",
      })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(
      screen.getByRole("button", { name: "Queue Conference And Process" })
    )
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/review-required",
      payload: {
        sessionId: "qi-extension-review-first",
        reviewRequired: [
          {
            occurrenceId: "conference-talk-1",
            reason: "duplicate_action_required",
            evidence: {
              kind: "library",
              existingMediaId: 42,
              duplicateOfOccurrenceId: null,
            },
            allowedActions: ["skip", "include_existing", "overwrite"],
          },
        ],
      },
    })

    expect(await screen.findByTestId("wizard-review")).toHaveAttribute(
      "data-processing-block",
      "review_required"
    )
    expect(useQuickIngestSessionStore.getState().session?.tracking).toBeUndefined()

    await user.click(screen.getByRole("button", { name: "Resolve duplicate" }))
    await user.click(
      screen.getByRole("button", { name: "Start reviewed processing" })
    )

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(2)
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(2)
    await expect(
      mocks.startQuickIngestSession.mock.results[1]?.value
    ).resolves.toMatchObject({
      sessionId: "qi-extension-review-corrected",
    })
  })

  it("does not let stale Provider processing state overwrite durable Review", async () => {
    const user = userEvent.setup()
    const storageKey = "tldw-quick-ingest-session"
    window.sessionStorage.clear()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-review-provider-guard",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(
      screen.getByRole("button", { name: "Queue Conference And Process" })
    )
    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.tracking).toMatchObject({
        sessionId: "qi-review-provider-guard",
      })
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/review-required",
      payload: {
        sessionId: "qi-review-provider-guard",
        reviewRequired: [
          {
            occurrenceId: "conference-talk-1",
            reason: "duplicate_action_required",
            evidence: {
              kind: "library",
              existingMediaId: 42,
              duplicateOfOccurrenceId: null,
            },
            allowedActions: ["skip", "overwrite"],
          },
        ],
      },
    })

    expect(await screen.findByTestId("wizard-review")).toBeInTheDocument()
    await act(async () => {
      await Promise.resolve()
    })
    const persisted = JSON.parse(
      window.sessionStorage.getItem(storageKey) || "null"
    )?.state?.session
    expect(persisted).toMatchObject({ currentStep: 3 })
    expect(persisted?.tracking).toBeUndefined()
  })

  it("consumes oversized runtime fallback outcomes into the exact result groups", async () => {
    const sessionId = "qi-runtime-oversized-consumer"
    const runId = "run-runtime-oversized-consumer"
    const representativeOutcomes = [
      "processed",
      "skipped_existing",
      "submit_failed",
      "cancelled",
    ] as const
    const occurrenceIds = Array.from(
      { length: 500 },
      (_, index) => `occ-runtime-consumer-${index}`
    )
    let terminalMessage: any = null
    const runtime = createQuickIngestSessionRuntime({
      run: vi.fn(),
      emit: vi.fn((type: string, payload: Record<string, unknown>) => {
        if (type === "tldw:quick-ingest/completed") {
          terminalMessage = { type, payload }
        }
      }),
      loadRunSessions: vi.fn().mockResolvedValue([
        {
          version: 1,
          kind: "run",
          sessionId,
          runId,
          generation: "generation-runtime-oversized-consumer",
          attemptToken: "attempt-runtime-oversized-consumer",
          occurrenceIds,
          jobIdToItemId: {},
          startedAt: Date.now(),
        },
      ]),
      saveRunSession: vi.fn().mockResolvedValue(true),
      reattachRun: vi.fn().mockResolvedValue({
        lifecycle: "completed",
        jobs: occurrenceIds.map((occurrenceId, index) => {
          const outcome = representativeOutcomes[index] || "processed"
          return {
            jobId: index + 1,
            status:
              outcome === "cancelled"
                ? "cancelled"
                : outcome === "submit_failed"
                  ? "failed"
                  : "completed",
            sourceItemId: occurrenceId,
            result: {
              outcome,
              title: `Oversized consumer ${index} ${"t".repeat(2_000)}`,
            },
            error:
              outcome === "cancelled" || outcome === "submit_failed"
                ? `${outcome} ${"e".repeat(2_000)}`
                : null,
          }
        }),
        errorMessage: null,
      }),
    } as any)

    await runtime.restore()
    expect(terminalMessage).toBeTruthy()
    expect(terminalMessage.payload.results).toHaveLength(500)
    expect(terminalMessage.payload.results[0]).not.toHaveProperty("title")

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: occurrenceIds.slice(0, 4).map((id) => ({
        id,
        kind: "url",
        url: `https://example.com/${id}`,
        detectedType: "web",
        icon: "Globe",
        fileSize: 0,
        validation: { valid: true },
      })) as any,
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 20,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId,
        runId,
        itemIds: occurrenceIds.slice(0, 4),
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await waitFor(() => {
      expect(mocks.runtimeListeners.length).toBeGreaterThan(0)
    })
    emitRuntimeMessage(terminalMessage)

    expect(await screen.findByTestId("wizard-result-occ-runtime-consumer-0")).toHaveTextContent(
      "occ-runtime-consumer-0:processed"
    )
    expect(screen.getByTestId("wizard-result-occ-runtime-consumer-1")).toHaveTextContent(
      "occ-runtime-consumer-1:skipped"
    )
    expect(screen.getByTestId("wizard-result-occ-runtime-consumer-2")).toHaveTextContent(
      "occ-runtime-consumer-2:submit_failed"
    )
    expect(screen.getByTestId("wizard-result-occ-runtime-consumer-3")).toHaveTextContent(
      "occ-runtime-consumer-3:cancelled"
    )
  })

  it("retains an indeterminate accepted extension identity as interrupted instead of terminalizing locally", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: false,
      indeterminate: true,
      sessionId: "qi-extension-ambiguous-start",
      error: "The extension accepted the start but both responses timed out.",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        lifecycle: "interrupted",
        tracking: {
          mode: "extension-runtime",
          sessionId: "qi-extension-ambiguous-start",
          submissionOccurrenceIds: ["queued-url-1"],
        },
        errorMessage: expect.stringMatching(/accepted|response|timed out/i),
      })
    })
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
  })

  it("surfaces a stopped version-2 submission instead of leaving it accepted", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-rate-limited",
    })
    mocks.submitQuickIngestBatch.mockResolvedValue({
      ok: false,
      accepted: false,
      runId: "run-rate-limited",
      retryAfterMs: 3_000,
      unsentOccurrenceIds: ["conference-talk-1"],
      error: "Playlist ingest submission was rate limited. Try again in 3 seconds.",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue Conference And Process" }))

    expect(await screen.findByTestId("wizard-results")).toHaveTextContent("error:1")
    expect(useQuickIngestSessionStore.getState().session?.results).toEqual([
      expect.objectContaining({
        id: "conference-talk-1",
        status: "error",
        error: expect.stringMatching(/try again in 3 seconds/i),
      }),
    ])
  })

  it("keeps accepted chunks processing while marking only unsent occurrences failed", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-partial-submission",
    })
    mocks.submitQuickIngestBatch.mockImplementation(async (payload: any) => {
      payload.onTrackingMetadata({
        mode: "webui-direct",
        runId: "run-partial-submission",
        batchIds: ["batch-partial-submission"],
        jobIds: [801],
        submittedItemIds: [
          "queued-partial-accepted",
          "queued-partial-unsent",
        ],
        jobIdToItemId: { "801": "queued-partial-accepted" },
        startedAt: Date.now(),
      })
      return {
        ok: false,
        accepted: true,
        submissionBlocked: true,
        runId: "run-partial-submission",
        retryAfterMs: 3_000,
        unsentOccurrenceIds: ["queued-partial-unsent"],
        error: "Playlist ingest submission was rate limited. Try again in 3 seconds.",
      }
    })
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 801,
          status: "running",
          sourceItemId: "queued-partial-accepted",
        },
        {
          jobId: null,
          status: "cancelled",
          sourceItemId: "queued-partial-unsent",
        },
      ],
      errorMessage: null,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(
      screen.getByRole("button", { name: "Queue Partial Submission And Process" })
    )

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({ runId: "run-partial-submission" })
      )
    })
    expect(screen.getByTestId("wizard-processing")).toBeInTheDocument()
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "processing",
      tracking: { runId: "run-partial-submission" },
      results: [
        expect.objectContaining({
          id: "queued-partial-unsent",
          status: "error",
          error: expect.stringMatching(/try again in 3 seconds/i),
        }),
      ],
    })
  })

  it("interrupts instead of reattaching forever when unsent cleanup fails", async () => {
    const user = userEvent.setup()
    let resolveCleanupFailure: ((value: any) => void) | null = null
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-cleanup-failure",
    })
    mocks.submitQuickIngestBatch.mockImplementation((payload: any) => {
      payload.onTrackingMetadata({
        mode: "webui-direct",
        runId: "run-cleanup-failure",
        batchIds: ["batch-cleanup-failure"],
        jobIds: [901],
        submittedItemIds: [
          "queued-partial-accepted",
          "queued-partial-unsent",
        ],
        jobIdToItemId: { "901": "queued-partial-accepted" },
        startedAt: Date.now(),
      })
      return new Promise((resolve) => {
        resolveCleanupFailure = resolve
      })
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(
      screen.getByRole("button", { name: "Queue Partial Submission And Process" })
    )

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.tracking).toMatchObject({
        runId: "run-cleanup-failure",
      })
    })
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 10))
    })
    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()
    expect(screen.getByTestId("wizard-processing")).toHaveTextContent("running:2")

    await act(async () => {
      resolveCleanupFailure?.({
        ok: false,
        accepted: true,
        submissionBlocked: true,
        submissionCleanupFailed: true,
        runId: "run-cleanup-failure",
        unsentOccurrenceIds: ["queued-partial-unsent"],
        error:
          "Submission stopped, but the server could not cancel the unsent occurrences. Retry cancellation before reconnecting.",
      })
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByTestId("wizard-processing")).toHaveTextContent("error:2")
    })
    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "interrupted",
      tracking: { runId: "run-cleanup-failure" },
      results: [
        expect.objectContaining({
          id: "queued-partial-unsent",
          status: "error",
        }),
      ],
    })
    expect(
      useQuickIngestSessionStore.getState().session?.results.some(
        (result) => result.id === "queued-partial-accepted"
      )
    ).toBe(false)
  })

  it("interrupts an unresolved run when first-chunk submission and cleanup both fail", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-first-cleanup-failure",
    })
    mocks.submitQuickIngestBatch.mockImplementation(async (payload: any) => {
      payload.onTrackingMetadata({
        mode: "webui-direct",
        submissionState: "cleanup_required",
        runId: "run-first-cleanup-failure",
        submittedItemIds: ["queued-url-1"],
        startedAt: Date.now(),
      })
      return {
        ok: false,
        accepted: false,
        submissionBlocked: true,
        submissionCleanupFailed: true,
        runId: "run-first-cleanup-failure",
        unsentOccurrenceIds: ["queued-url-1"],
        error:
          "Submission stopped, but the server could not cancel the unsent occurrence.",
      }
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(screen.getByRole("button", { name: "Queue And Process" }))

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        lifecycle: "interrupted",
        currentStep: 4,
        tracking: {
          submissionState: "cleanup_required",
          runId: "run-first-cleanup-failure",
        },
        results: [],
      })
    })
    expect(screen.getByTestId("wizard-processing")).toHaveTextContent("error:1")
    expect(screen.queryByTestId("wizard-results")).not.toBeInTheDocument()
    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()
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
    expect(useQuickIngestSessionStore.getState().session?.tracking).toMatchObject({
      mode: "extension-runtime",
      sessionId: "qi-runtime-test",
      submissionOccurrenceIds: ["queued-url-1"],
    })

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

  it("consumes restored progress and preserves per-item results from a failed runtime", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-runtime-partial-failure",
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await user.click(
      screen.getByRole("button", { name: "Queue Partial Submission And Process" })
    )
    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/progress",
      payload: {
        sessionId: "qi-runtime-partial-failure",
        occurrenceId: "queued-partial-accepted",
        status: "running",
        result: {
          id: "queued-partial-accepted",
          status: "running",
          type: "item",
        },
      },
    })

    expect(useQuickIngestSessionStore.getState().session?.results).toEqual([])
    expect(
      useQuickIngestSessionStore
        .getState()
        .session?.processingState.perItemProgress.find(
          (item) => item.id === "queued-partial-accepted"
        )
    ).toMatchObject({ status: "processing" })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/failed",
      payload: {
        sessionId: "qi-runtime-partial-failure",
        error: "One item failed.",
        results: [
          {
            id: "queued-partial-accepted",
            status: "ok",
            type: "video",
          },
          {
            id: "queued-partial-unsent",
            status: "error",
            type: "video",
            error: "Submission failed.",
          },
        ],
      },
    })

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.results).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "queued-partial-accepted",
            status: "ok",
          }),
          expect.objectContaining({
            id: "queued-partial-unsent",
            status: "error",
            error: "Submission failed.",
          }),
        ])
      )
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

    const { rerender } = render(<QuickIngestWizardModal open={false} onClose={onClose} />)

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

  it("persists and rehydrates playlist materialization authority, expiry, and review choices", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().createDraftSession()
    const mounted = render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await user.click(screen.getByRole("button", { name: "Queue Playlist Draft" }))

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.queueItems).toEqual([
        expect.objectContaining({
          id: "occ-persisted-playlist",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "opaque-owner-bound-materialization",
            occurrenceId: "occ-persisted-playlist",
          },
          playlist: expect.objectContaining({
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          }),
          playlistReview: expect.objectContaining({
            selected: true,
            duplicatePolicy: "overwrite",
            editedFields: ["title"],
            metadataPatch: { title: "Edited persisted title" },
          }),
        }),
      ])
    })

    mounted.unmount()
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    const row = screen.getByTestId("queued-item-occ-persisted-playlist")
    expect(JSON.parse(row.getAttribute("data-source-ref") || "null")).toEqual({
      kind: "materialized_playlist_item",
      materializationId: "opaque-owner-bound-materialization",
      occurrenceId: "occ-persisted-playlist",
    })
    expect(JSON.parse(row.getAttribute("data-playlist") || "null")).toEqual(
      expect.objectContaining({
        title: "Persisted playlist row",
        ordinal: 7,
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      })
    )
    expect(JSON.parse(row.getAttribute("data-playlist-review") || "null")).toEqual(
      expect.objectContaining({
        duplicatePolicy: "overwrite",
        editedFields: ["title"],
      })
    )
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

  it("hydrates corrupt persisted display fields from their source authority", () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft",
      currentStep: 1,
      queueItems: [
        {
          id: "hydrate-direct",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "hydrate-direct",
            url: "https://example.com/hydrate-direct",
          },
          kind: "file",
          url: "https://cached.example.invalid/hydrate-direct",
          fileName: "wrong-direct.txt",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
        {
          id: "hydrate-file",
          sourceRef: { kind: "file_stub", occurrenceId: "hydrate-file" },
          kind: "url",
          url: "https://example.com/not-a-file",
          name: "hydrate-file.pdf",
          detectedType: "pdf",
          icon: "FileText",
          fileSize: 32,
          validation: { valid: true },
        },
        {
          id: "hydrate-materialized",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "hydrate-materialization",
            occurrenceId: "hydrate-materialized",
          },
          kind: "file",
          url: "https://cached.example.invalid/hydrate-materialized",
          fileName: "wrong-materialized.txt",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: { materializationExpiresAt: "2099-01-01T00:00:00Z" },
        },
      ],
    } as never)

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    const direct = screen.getByTestId("queued-item-hydrate-direct")
    expect(direct).toHaveAttribute("data-kind", "url")
    expect(direct).toHaveAttribute("data-url", "https://example.com/hydrate-direct")
    expect(direct).toHaveAttribute("data-file-name", "")

    const file = screen.getByTestId("queued-item-hydrate-file")
    expect(file).toHaveAttribute("data-kind", "file")
    expect(file).toHaveAttribute("data-url", "")
    expect(file).toHaveAttribute("data-file-name", "hydrate-file.pdf")
    expect(JSON.parse(file.getAttribute("data-validation") || "null")).toMatchObject({
      valid: false,
    })

    const materialized = screen.getByTestId("queued-item-hydrate-materialized")
    expect(materialized).toHaveAttribute("data-kind", "url")
    expect(materialized).toHaveAttribute(
      "data-url",
      "https://cached.example.invalid/hydrate-materialized"
    )
    expect(materialized).toHaveAttribute("data-file-name", "")
  })

  it("interrupts an ambiguous persisted pre-create marker without creating another run", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-restored-pre-create",
          kind: "url",
          url: "https://cached.example.invalid/must-not-submit",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-restored-pre-create",
            occurrenceId: "occ-restored-pre-create",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 1,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-restored-pre-create",
        submissionState: "creating_run",
        submissionOccurrenceIds: ["occ-restored-pre-create"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        lifecycle: "interrupted",
        currentStep: 4,
        tracking: {
          submissionState: "creating_run",
          submissionOccurrenceIds: ["occ-restored-pre-create"],
        },
      })
    })
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(screen.getByTestId("wizard-processing")).toHaveTextContent("error")
  })

  it("never falls an expired restored pre-create marker back to its cached URL", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-expired-pre-create",
          kind: "url",
          url: "https://cached.example.invalid/expired-must-not-submit",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-expired-pre-create",
            occurrenceId: "occ-expired-pre-create",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            materializationExpiresAt: "2020-01-01T00:00:00Z",
          },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 1,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-expired-pre-create",
        submissionState: "creating_run",
        submissionOccurrenceIds: ["occ-expired-pre-create"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(useQuickIngestSessionStore.getState().session?.lifecycle).toBe(
        "interrupted"
      )
    })
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
    expect(JSON.stringify(mocks.submitQuickIngestBatch.mock.calls)).not.toContain(
      "cached.example.invalid"
    )
  })

  it("reattaches a post-create run marker without restarting submission", async () => {
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: null,
          status: "queued",
          sourceItemId: "occ-restored-run-created",
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
          id: "occ-restored-run-created",
          kind: "url",
          url: "https://cached.example.invalid/must-not-restart",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "running",
        perItemProgress: [],
        elapsed: 2,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-restored-run-created",
        submissionState: "run_created",
        runId: "run-restored-created",
        submittedItemIds: ["occ-restored-run-created"],
        startedAt: Date.now(),
      } as any,
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.reattachQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          submissionState: "run_created",
          runId: "run-restored-created",
        })
      )
    })
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
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
        submissionState: "submitting",
        runId: "run-partial-refresh",
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
          submissionState: "submitting",
          runId: "run-partial-refresh",
          batchId: "batch-77",
          jobIds: [77],
        })
      )
    })

    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()

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

  it("queries and replays a persisted extension session without restarting it", async () => {
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
    mocks.queryQuickIngestSession.mockResolvedValue({
      ok: true,
      active: false,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "qi-runtime-refresh",
          runId: "run-runtime-refresh",
          results: [
            {
              id: "queued-url-1",
              status: "ok",
              url: "https://example.com/article",
              type: "html",
            },
          ],
        },
      },
      replayAck: {
        runId: "run-runtime-refresh",
        generation: "generation-runtime-refresh",
      },
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(mocks.queryQuickIngestSession).toHaveBeenCalledWith(
        "qi-runtime-refresh"
      )
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
    expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
  })

  it("returns a recreated worker review tombstone to Review without restarting", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "interrupted",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-recreated-review",
          kind: "url",
          url: "https://cached.example.invalid/recreated-review",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-recreated-review",
            occurrenceId: "occ-recreated-review",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: {
            materializationExpiresAt: "2099-01-01T00:00:00Z",
          },
          playlistReview: { selected: true },
        } as any,
      ],
      processingState: {
        status: "error",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-recreated-review",
        itemIds: ["occ-recreated-review"],
        startedAt: Date.now(),
      } as any,
    })
    mocks.queryQuickIngestSession.mockResolvedValue({
      ok: true,
      active: false,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "qi-recreated-review",
          reviewRequired: [
            {
              occurrenceId: "occ-recreated-review",
              reason: "duplicate_action_required",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "overwrite"],
            },
          ],
        },
      },
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    expect(await screen.findByTestId("wizard-review")).toHaveAttribute(
      "data-processing-block",
      "review_required"
    )
    expect(mocks.queryQuickIngestSession).toHaveBeenCalledWith(
      "qi-recreated-review"
    )
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
  })

  it("never persists a crash boundary without Review state or extension replay authority", async () => {
    const storageKey = "tldw-quick-ingest-session"
    window.sessionStorage.clear()
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "interrupted",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-review-crash-boundary",
          kind: "url",
          url: "https://cached.example.invalid/review-crash-boundary",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-review-crash-boundary",
            occurrenceId: "occ-review-crash-boundary",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: { materializationExpiresAt: "2099-01-01T00:00:00Z" },
          playlistReview: { selected: true },
        } as any,
      ],
      processingState: {
        status: "error",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-review-crash-boundary",
        itemIds: ["occ-review-crash-boundary"],
        startedAt: Date.now(),
      } as any,
    })
    const commitReviewHandoff =
      useQuickIngestSessionStore.getState().commitReviewHandoff
    useQuickIngestSessionStore.setState({
      commitReviewHandoff: () => {
        throw new Error("simulated crash during durable Review commit")
      },
    })

    try {
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await waitFor(() => {
        expect(mocks.runtimeListeners.length).toBeGreaterThan(0)
      })
      let boundaryError: unknown
      try {
        emitRuntimeMessage({
          type: "tldw:quick-ingest/review-required",
          payload: {
            sessionId: "qi-review-crash-boundary",
            reviewRequired: [
              {
                occurrenceId: "occ-review-crash-boundary",
                reason: "duplicate_action_required",
                evidence: {
                  kind: "library",
                  existingMediaId: 42,
                  duplicateOfOccurrenceId: null,
                },
                allowedActions: ["skip", "overwrite"],
              },
            ],
          },
        })
      } catch (error) {
        boundaryError = error
      }

      const persisted = JSON.parse(
        window.sessionStorage.getItem(storageKey) || "null"
      )?.state?.session
      expect(boundaryError).toBeUndefined()
      expect(
        persisted?.currentStep === 3 ||
          persisted?.tracking?.sessionId === "qi-review-crash-boundary"
      ).toBe(true)
      expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
    } finally {
      useQuickIngestSessionStore.setState({ commitReviewHandoff })
    }
  })

  it("leaves extension replay identity untouched when durable Review commit fails", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-review-write-failure",
          kind: "url",
          url: "https://cached.example.invalid/review-write-failure",
          sourceRef: {
            kind: "materialized_playlist_item",
            materializationId: "materialization-review-write-failure",
            occurrenceId: "occ-review-write-failure",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: { materializationExpiresAt: "2099-01-01T00:00:00Z" },
          playlistReview: { selected: true },
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
        sessionId: "qi-review-write-failure",
        itemIds: ["occ-review-write-failure"],
        startedAt: Date.now(),
      } as any,
    })
    const commitReviewHandoff =
      useQuickIngestSessionStore.getState().commitReviewHandoff
    useQuickIngestSessionStore.setState({
      commitReviewHandoff: () => false,
    })

    try {
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await waitFor(() => {
        expect(mocks.runtimeListeners.length).toBeGreaterThan(0)
      })
      emitRuntimeMessage({
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "qi-review-write-failure",
          reviewRequired: [
            {
              occurrenceId: "occ-review-write-failure",
              reason: "duplicate_action_required",
              evidence: {
                kind: "library",
                existingMediaId: 42,
                duplicateOfOccurrenceId: null,
              },
              allowedActions: ["skip", "overwrite"],
            },
          ],
        },
      })

      await act(async () => {
        await Promise.resolve()
      })
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        lifecycle: "processing",
        currentStep: 4,
        tracking: { sessionId: "qi-review-write-failure" },
      })
      expect(screen.queryByTestId("wizard-review")).not.toBeInTheDocument()
      expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
    } finally {
      useQuickIngestSessionStore.setState({ commitReviewHandoff })
    }
  })

  it("queries an interrupted extension session after reopen and reconciles retained terminal replay", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "interrupted",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-interrupted-replay",
          kind: "url",
          url: "https://example.com/interrupted-replay",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "error",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-runtime-interrupted-replay",
        runId: "run-runtime-interrupted-replay",
        itemIds: ["queued-interrupted-replay"],
        startedAt: Date.now(),
      } as any,
    })
    mocks.queryQuickIngestSession.mockResolvedValue({
      ok: true,
      active: false,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "qi-runtime-interrupted-replay",
          runId: "run-runtime-interrupted-replay",
          results: [
            {
              id: "queued-interrupted-replay",
              status: "ok",
              type: "html",
            },
          ],
        },
      },
      replayAck: {
        runId: "run-runtime-interrupted-replay",
        generation: "generation-runtime-interrupted-replay",
      },
    })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)

    await waitFor(() => {
      expect(mocks.queryQuickIngestSession).toHaveBeenCalledWith(
        "qi-runtime-interrupted-replay"
      )
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
    })
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
  })

  it("retries transient replay failures with bounded backoff and keeps the tombstone unacknowledged", async () => {
    vi.useFakeTimers()
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "interrupted",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-replay-retry",
          kind: "url",
          url: "https://example.com/replay-retry",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        } as any,
      ],
      processingState: {
        status: "error",
        perItemProgress: [],
        elapsed: 3,
        estimatedRemaining: 0,
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-runtime-replay-retry",
        runId: "run-runtime-replay-retry",
        itemIds: ["queued-replay-retry"],
        startedAt: Date.now(),
      } as any,
    })
    mocks.queryQuickIngestSession
      .mockResolvedValueOnce({ ok: false, error: "storage temporarily unavailable" })
      .mockResolvedValueOnce({
        ok: true,
        active: false,
        event: {
          type: "tldw:quick-ingest/completed",
          payload: {
            sessionId: "qi-runtime-replay-retry",
            runId: "run-runtime-replay-retry",
            results: [
              {
                id: "queued-replay-retry",
                status: "ok",
                type: "html",
              },
            ],
          },
        },
        replayAck: {
          runId: "run-runtime-replay-retry",
          generation: "generation-runtime-replay-retry",
        },
      })

    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })

    expect(mocks.queryQuickIngestSession).toHaveBeenCalledTimes(2)
    expect(useQuickIngestSessionStore.getState().session?.lifecycle).toBe("completed")
    expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
  })

  it("runs a fresh bounded replay cycle on each open and never consumes recovery while hidden", async () => {
    vi.useFakeTimers()
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-replay-open-cycle",
          kind: "url",
          url: "https://example.com/replay-open-cycle",
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
        sessionId: "qi-runtime-replay-open-cycle",
        runId: "run-runtime-replay-open-cycle",
        itemIds: ["queued-replay-open-cycle"],
        startedAt: Date.now(),
      } as any,
    })
    mocks.queryQuickIngestSession.mockResolvedValue({
      ok: false,
      error: "Extension recovery is temporarily unavailable.",
    })

    const view = render(<QuickIngestWizardModal open={false} onClose={vi.fn()} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    expect(mocks.queryQuickIngestSession).not.toHaveBeenCalled()

    view.rerender(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    expect(mocks.queryQuickIngestSession).toHaveBeenCalledTimes(3)
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "interrupted",
      errorMessage: expect.stringMatching(/reopen|try again|recovery/i),
    })

    view.rerender(<QuickIngestWizardModal open={false} onClose={vi.fn()} />)
    view.rerender(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    expect(mocks.queryQuickIngestSession).toHaveBeenCalledTimes(6)
  })

  it("does not acknowledge terminal replay when session storage rejects terminal persistence", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "queued-replay-quota",
          kind: "url",
          url: "https://example.com/replay-quota",
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
        sessionId: "qi-runtime-replay-quota",
        runId: "run-runtime-replay-quota",
        itemIds: ["queued-replay-quota"],
        startedAt: Date.now(),
      } as any,
    })
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("Quota exceeded", "QuotaExceededError")
      })
    mocks.queryQuickIngestSession.mockResolvedValue({
      ok: true,
      active: false,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "qi-runtime-replay-quota",
          runId: "run-runtime-replay-quota",
          results: [
            {
              id: "queued-replay-quota",
              status: "ok",
              type: "html",
            },
          ],
        },
      },
      replayAck: {
        runId: "run-runtime-replay-quota",
        generation: "generation-runtime-replay-quota",
      },
    })

    try {
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)

      await waitFor(() => {
        expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:1")
      })
      expect(mocks.acknowledgeQuickIngestSessionReplay).not.toHaveBeenCalled()
    } finally {
      setItem.mockRestore()
    }
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
        runId: "run-refresh-77",
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
          tracking: expect.objectContaining({
            sessionId: "qi-direct-refresh",
            runId: "run-refresh-77",
            batchIds: ["batch-77"],
          }),
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

  it.each([
    {
      terminalType: "tldw:quick-ingest/completed",
      terminalPayload: {
        results: [
          {
            id: "queued-authoritative-cancel",
            status: "ok",
            type: "html",
          },
        ],
      },
      expectedLifecycle: "completed",
      expectedStatus: "complete",
    },
    {
      terminalType: "tldw:quick-ingest/failed",
      terminalPayload: {
        error: "The run finished with an item failure.",
        results: [
          {
            id: "queued-authoritative-cancel",
            status: "error",
            type: "html",
            error: "Item failed after cancellation was requested.",
          },
        ],
      },
      expectedLifecycle: "partial_failure",
      expectedStatus: "error",
    },
  ])(
    "keeps authoritative cancellation nonterminal until $terminalType arrives",
    async ({
      terminalType,
      terminalPayload,
      expectedLifecycle,
      expectedStatus,
    }) => {
      const user = userEvent.setup()
      useQuickIngestSessionStore.getState().upsertSession({
        ...createEmptyQuickIngestSession(),
        lifecycle: "processing",
        currentStep: 4,
        queueItems: [
          {
            id: "queued-authoritative-cancel",
            kind: "url",
            url: "https://example.com/authoritative-cancel",
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
              id: "queued-authoritative-cancel",
              status: "processing",
              progressPercent: 50,
              currentStage: "Processing",
              estimatedRemaining: 12,
            },
          ],
          elapsed: 3,
          estimatedRemaining: 12,
        },
        tracking: {
          mode: "extension-runtime",
          sessionId: "qi-runtime-authoritative-cancel",
          runId: "run-authoritative-cancel",
          itemIds: ["queued-authoritative-cancel"],
          startedAt: Date.now(),
        } as any,
      })

      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await user.click(screen.getByRole("button", { name: "Cancel Processing" }))

      await waitFor(() => {
        expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith(
          expect.objectContaining({
            sessionId: "qi-runtime-authoritative-cancel",
            tracking: expect.objectContaining({
              runId: "run-authoritative-cancel",
            }),
          })
        )
      })
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        lifecycle: "processing",
        completedAt: null,
        processingState: {
          status: "running",
          perItemProgress: [
            expect.objectContaining({
              id: "queued-authoritative-cancel",
              status: "processing",
              currentStage: expect.stringMatching(/cancellation requested/i),
            }),
          ],
        },
      })

      emitRuntimeMessage({
        type: "tldw:quick-ingest/progress",
        payload: {
          sessionId: "qi-runtime-authoritative-cancel",
          runId: "run-authoritative-cancel",
          occurrenceId: "queued-authoritative-cancel",
          status: "running",
          result: {
            id: "queued-authoritative-cancel",
            status: "running",
            type: "html",
          },
        },
      })
      await waitFor(() => {
        expect(
          useQuickIngestSessionStore
            .getState()
            .session?.processingState.perItemProgress[0]
        ).toMatchObject({
          status: "processing",
          currentStage: expect.stringMatching(/cancellation requested/i),
        })
      })

      emitRuntimeMessage({
        type: terminalType,
        payload: {
          sessionId: "qi-runtime-authoritative-cancel",
          runId: "run-authoritative-cancel",
          ...terminalPayload,
        },
      })

      await waitFor(() => {
        expect(useQuickIngestSessionStore.getState().session).toMatchObject({
          lifecycle: expectedLifecycle,
          processingState: { status: expectedStatus },
        })
      })
    }
  )

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
      expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({ sessionId: "qi-runtime-cancel-preserve" })
      )
    })
    emitRuntimeMessage({
      type: "tldw:quick-ingest/cancelled",
      payload: {
        sessionId: "qi-runtime-cancel-preserve",
        reason: "Cancelled by user.",
      },
    })

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

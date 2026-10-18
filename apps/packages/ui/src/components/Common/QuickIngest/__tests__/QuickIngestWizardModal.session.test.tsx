import React from "react"
import { flushSync } from "react-dom"
import i18n from "i18next"
import { File as NodeFile } from "node:buffer"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useQuickIngestEvents } from "@/components/Layouts/QuickIngestButton"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import { getEligibleQueueItems } from "../queue-items"
import { useMediaSearch } from "@/components/Review/hooks/useMediaSearch"

const mocks = vi.hoisted(() => ({
  startQuickIngestSession: vi.fn(),
  submitQuickIngestBatch: vi.fn(),
  cancelQuickIngestSession: vi.fn(),
  reattachQuickIngestSession: vi.fn(),
  initialize: vi.fn(),
  getQuickIngestAnalysisProviderWarning: vi.fn(),
  checkConnection: vi.fn(),
  navigate: vi.fn(),
  createTab: vi.fn().mockResolvedValue({}),
  runtimeListeners: [] as Array<(message: any) => void>,
  modalProps: [] as any[],
  afterCancelProcessing: null as null | (() => void),
  useActualModal: false,
  useActualAddContentStep: false,
  useActualProcessingStep: false,
  useActualResultsStep: false,
  useActualReviewStep: false,
  useActualConfigureStep: false,
  queueSourceFile: false,
  queuedReviewFile: null as File | null,
  reviewBatches: new Map<string, Record<string, unknown>>(),
  reviewDrafts: new Map<string, Record<string, unknown>>(),
  reviewFiles: new Map<string, File>(),
  failReviewWrite: false,
  reviewSelectionMirrorWait: null as Promise<void> | null,
  failReviewSelectionMirror: false,
  reviewSelectionMirrorStarted: vi.fn(),
  reviewWriteWait: null as Promise<void> | null,
  reviewWriteStarted: vi.fn(),
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  connectionState: {
    phase: "connected",
    isConnected: true,
    isChecking: false,
    lastError: null as string | null,
    offlineBypass: false,
  }
}))

vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [undefined, vi.fn(), { isLoading: false }] }))

vi.mock("@/db/dexie/drafts", () => ({
  DRAFT_STORAGE_CAP_BYTES: 100 * 1024 * 1024,
  withDraftTransaction: async (operation: { assertCurrent: () => void }, run: () => Promise<unknown>) => { operation.assertCurrent(); return run() },
  getDraftBatchById: async (id: string) => mocks.reviewBatches.get(id),
  getDraftsByBatch: async (id: string) =>
    [...mocks.reviewDrafts.values()].filter((row) => row.batchId === id),
  upsertDraftBatch: async (batch: Record<string, unknown>, operation: { authorityKey: string }) => {
    mocks.reviewWriteStarted()
    await mocks.reviewWriteWait
    if (mocks.failReviewWrite) throw new Error("Review disk is full")
    mocks.reviewBatches.set(String(batch.id), { ...batch, ownerScope: operation.authorityKey })
  },
  upsertContentDraft: async (draft: Record<string, unknown>, operation: { authorityKey: string }) => {
    mocks.reviewDrafts.set(String(draft.id), { ...draft, ownerScope: operation.authorityKey })
  },
  storeDraftAsset: async (id: string, file: File) => { mocks.reviewFiles.set(id, file); return { asset: { id: "asset-" + id }, stored: true } }
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return { ...actual, setSetting: async (setting: any, value: unknown) => {
    if (setting.key === "media-review-selection") {
      mocks.reviewSelectionMirrorStarted()
      await mocks.reviewSelectionMirrorWait
      if (mocks.failReviewSelectionMirror) throw new Error("Selection mirror unavailable")
    }
    return actual.setSetting(setting, value)
  } }
})

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
      return (defaultValueOrOptions?.defaultValue || key).replace(/\{\{(\w+)\}\}/g, (_match, token) => String(defaultValueOrOptions?.[token] ?? _match))
    },
  }),
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: {} }),
}))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return {
    ...actual,
    Modal: Object.assign(
      (props: any) => {
        if (mocks.useActualModal) return <actual.Modal {...props} />
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
        destroyAll: vi.fn()
      }
    ),
    Button: ({ children, onClick, disabled, ...props }: any) => mocks.useActualAddContentStep ? <actual.Button onClick={onClick} disabled={disabled} {...props}>{children}</actual.Button> : (
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
        Group: ({ children, ...props }: any) => <div {...props}>{children}</div>
      }
    ),
    Collapse: ({ items }: any) => (
      <div>
        {items?.map((item: any) => (
          <div key={item.key}>{item.children}</div>
        ))}
      </div>
    )
  }
})

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
  useConnectionStore: Object.assign((selector: any) =>
    selector({
      state: mocks.connectionState,
      checkOnce: mocks.checkConnection,
    }), { subscribe: () => () => {} }),
}))

vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>()
  const icon = (name: string) => (props: any) => (
    <span data-icon={name} aria-hidden={props?.["aria-hidden"]} />
  )
  return {
    ...actual,
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
    tabs: { create: (...args: unknown[]) => mocks.createTab(...args) },
    runtime: {
      getURL: (path: string) => `chrome-extension://test${path}`,
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
    getProvidersStatus: vi.fn().mockResolvedValue({ providers: [], any_configured: false }),
    getTranscriptionModels: vi.fn().mockResolvedValue({ all_models: [] }),
    ensureConfigForRequest: async () => JSON.parse(localStorage.getItem("tldwConfig") || "null"),
  },
}))

vi.mock("@/components/Common/QuickIngest/IngestWizardStepper", () => ({
  IngestWizardStepper: () => <div data-testid="wizard-stepper" />,
}))

vi.mock("@/components/Common/QuickIngest/AddContentStep", async () => {
  const { AddContentStep: ActualAddContentStep } = await vi.importActual<typeof import("@/components/Common/QuickIngest/AddContentStep")>("@/components/Common/QuickIngest/AddContentStep")
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
      if (mocks.useActualAddContentStep) return <ActualAddContentStep onQuickProcess={onQuickProcess} quickProcessWarning={quickProcessWarning} />
      const context = actual.useIngestWizard() as any
      const { state, setQueueItems } = context
      return (
        <div>
          <output data-testid="eligible-count">
            {getEligibleQueueItems(state.queueItems).length}
          </output>
          <button onClick={() => setQueueItems([{ id: "attached-file", kind: "file", fileName: mocks.queuedReviewFile!.name, file: mocks.queuedReviewFile!, detectedType: "document", icon: "FileText", fileSize: mocks.queuedReviewFile!.size, validation: { valid: true } }])}>
            Attach source file
          </button>
          <button onClick={onQuickProcess}>Submit current queue</button>
          {quickProcessWarning ? (
            <div role="alert">{quickProcessWarning}</div>
          ) : null}
          <button
            onClick={() => {
              setQueueItems([
                mocks.queueSourceFile ? {
                  id: "queued-url-1", kind: "file", fileName: "source.txt",
                  file: mocks.queuedReviewFile || { name: "source.txt", size: 1, arrayBuffer: async () => new Uint8Array([65]).buffer },
                  detectedType: "document", icon: "FileText", fileSize: 1,
                  validation: { valid: true },
                } : {
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
    }
  }
})

vi.mock("@/components/Common/QuickIngest/ReviewStep", async () => {
  const { ReviewStep } = await vi.importActual<typeof import("@/components/Common/QuickIngest/ReviewStep")>("@/components/Common/QuickIngest/ReviewStep")
  return { ReviewStep: (props: React.ComponentProps<typeof ReviewStep>) => mocks.useActualReviewStep ? <ReviewStep {...props} /> : <div data-testid="wizard-review" /> }
})

vi.mock("@/components/Common/QuickIngest/WizardConfigureStep", async () => {
  const { WizardConfigureStep: ActualConfigureStep } = await vi.importActual<typeof import("@/components/Common/QuickIngest/WizardConfigureStep")>("@/components/Common/QuickIngest/WizardConfigureStep")
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  return {
    WizardConfigureStep: (props: React.ComponentProps<typeof ActualConfigureStep>) => {
      if (mocks.useActualConfigureStep) return <ActualConfigureStep {...props} />
      const { analysisProviderWarning, focusAnalysisProvider } = props
      return <StubConfigureStep analysisProviderWarning={analysisProviderWarning} focusAnalysisProvider={focusAnalysisProvider} />
    },
  }

  function StubConfigureStep({
      analysisProviderWarning,
      focusAnalysisProvider,
    }: {
      analysisProviderWarning?: string | null
      focusAnalysisProvider?: boolean
    }) {
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
    }
})

vi.mock("@/components/Common/QuickIngest/ProcessingStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  const { ProcessingStep: ActualProcessingStep } = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/ProcessingStep")
  >("@/components/Common/QuickIngest/ProcessingStep")
  return {
    ProcessingStep: (props: React.ComponentProps<typeof ActualProcessingStep>) => {
      if (mocks.useActualProcessingStep) return <ActualProcessingStep {...props} />
      return <StubProcessingStep {...props} />
    },
  }

  function StubProcessingStep({ onCancelAll }: { onCancelAll?: () => void }) {
    const { state, cancelProcessing } = actual.useIngestWizard()
    return (
      <div data-testid="wizard-processing">
        {state.processingState.status}:
        {state.processingState.perItemProgress.length}
        <output data-testid="retry-item-status">
          {state.processingState.perItemProgress
            .map((item) => item.status)
            .join(",")}
        </output>
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
  }
})

vi.mock("@/components/Common/QuickIngest/WizardResultsStep", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/IngestWizardContext")
  >("@/components/Common/QuickIngest/IngestWizardContext")
  const { WizardResultsStep: ActualResultsStep } = await vi.importActual<
    typeof import("@/components/Common/QuickIngest/WizardResultsStep")
  >("@/components/Common/QuickIngest/WizardResultsStep")
  return {
    WizardResultsStep: (props: React.ComponentProps<typeof ActualResultsStep>) => {
      if (mocks.useActualResultsStep) return <ActualResultsStep {...props} />
      return <StubResultsStep {...props} />
    },
  }

  function StubResultsStep({ onOpenCollection, onIngestMore }: React.ComponentProps<typeof ActualResultsStep>) {
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

vi.mock("@plasmohq/storage", async () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1 }) } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
import { quickIngestAuthority } from "@/services/tldw/quick-ingest-authority"
let releaseAuthority: (() => void) | undefined

function ReadyEventModalHost() {
  const events = useQuickIngestEvents()
  if (!events.quickIngestReady || (!events.quickIngestOpen && !events.hasQuickIngestSession)) return null
  return <QuickIngestWizardModal open={events.quickIngestOpen} openRevision={events.openRevision} presetMap={events.presetMap} onClose={events.closeQuickIngest} />
}

const catalogueT = (key: string, opts?: Record<string, unknown>) => String(opts?.defaultValue ?? key)
const catalogueMessage = { error: vi.fn(), warning: vi.fn() }
function CatalogueProbe() {
  const search = useMediaSearch({ t: catalogueT, message: catalogueMessage })
  return (
    <div data-testid="actual-catalogue">
      {search.results.map(item => item.title).join(',')} / {search.mediaTotal}
    </div>
  )
}

describe("QuickIngestWizardModal session runtime", () => {
  it("keeps the attached live File eligible when an ordinary URL handoff remounts the real modal", async () => {
    const file = new NodeFile(["Attached source"], "source.txt", { type: "text/plain" }) as unknown as File
    mocks.queuedReviewFile = file
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-handoff" })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [{ id: "attached-file", type: "document", status: "ok" }] })
    function RealModalHost() {
      const events = useQuickIngestEvents()
      return (
        <QuickIngestWizardModal open={events.quickIngestOpen} openRevision={events.openRevision} presetMap={events.presetMap} onClose={events.closeQuickIngest} />
      )
    }
    useQuickIngestSessionStore.getState().createDraftSession()
    render(<RealModalHost />)
    await screen.findByRole("button", { name: "Attach source file" })
    fireEvent.click(screen.getByRole("button", { name: "Attach source file" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.queueItems).toHaveLength(1))
    act(() => { requestQuickIngestOpen({ source: "manual", url: "https://example.com/article" }) })
    await waitFor(() => expect(screen.getByTestId("eligible-count")).toHaveTextContent("2"))
    fireEvent.click(screen.getByRole("button", { name: "Submit current queue" }))
    await waitFor(() => expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1))
    expect(mocks.submitQuickIngestBatch.mock.calls[0][0]).toMatchObject({ files: [{ id: "attached-file", name: "source.txt", data: Array.from(new Uint8Array(await file.arrayBuffer())) }], entries: [{ url: "https://example.com/article" }] })
  })

  it("carries explicit playlist repetition through the modal submission contract", async () => {
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-repeat" })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [{ id: "repeat-talk", type: "video", status: "ok" }] })
    useQuickIngestSessionStore.getState().createDraftSession({ queueItems: [{ id: "repeat-talk", kind: "url", url: "https://youtube.com/watch?v=repeat", detectedType: "video", icon: "Film", fileSize: 0, validation: { valid: true }, playlist: { duplicateStatus: "duplicate_existing" }, conferenceOverride: { selected: true, duplicatePolicy: "skip" }, processAgain: true }] })
    render(<QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />)
    await waitFor(() => expect(mocks.submitQuickIngestBatch).toHaveBeenCalled())
    expect(mocks.submitQuickIngestBatch.mock.calls[0][0]).toMatchObject({ entries: [{ processAgain: true }], common: { overwrite_existing: false } })
  })

  it("exposes nonretryable correction from the mounted modal when Retry and Correct callbacks are both wired", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "partial_failure",
      queueItems: [{ id: "auth-failed", kind: "url", url: "https://source.test/source.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } }],
      results: [{ id: "auth-failed", status: "error", type: "pdf", error: "Unauthorized 401" }, { id: "saved", status: "ok", type: "pdf", mediaId: 7 }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Retry auth-failed" })).toBeNull()
    fireEvent.click(await screen.findByRole("button", { name: "Correct settings for auth-failed" }))
    expect(await screen.findByTestId("wizard-configure")).toBeVisible()
    expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([expect.objectContaining({ id: "saved", status: "ok", mediaId: 7 })]))
    expect(mocks.submitQuickIngestBatch).not.toHaveBeenCalled()
  })

  it("retains conference identity after failed retry upload and reload before the next fresh attempt", async () => {
    mocks.useActualResultsStep = true
    const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
    mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
    mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
    mocks.bgUpload.mockRejectedValueOnce(new Error("Network error")).mockResolvedValue({ batch_id: "second-retry", jobs: [{ id: 99 }] })
    mocks.bgRequest.mockResolvedValue({ ok: true, data: { status: "completed", result: { status: "Success", media_id: 99 } } })
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "partial_failure",
      conferenceBatchMetadata: { collectionName: "Original conference" },
      queueItems: [{ id: "talk-failed", kind: "url", url: "https://source.test/talk.mp4", detectedType: "video", icon: "Film", fileSize: 0, validation: { valid: true }, conferenceOverride: { title: "Original talk", selected: true } }],
      results: [{ id: "saved", status: "ok", type: "video", mediaId: 7 }, { id: "talk-failed", status: "error", type: "video", error: "Network error", collectionItemId: "81", retryAttempt: 2 }],
      tracking: { mode: "webui-direct", sessionId: "old-session", collectionId: "7", plannedItemIds: ["81"], jobIds: [77], durableMode: "durable_collection", startedAt: 1 },
    })
    const view = render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Retry talk-failed" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([expect.objectContaining({ id: "talk-failed", status: "error", collectionItemId: 81, retryAttempt: 3 })])))
    expect(useQuickIngestSessionStore.getState().session?.tracking).toMatchObject({ collectionId: "7", durableMode: "durable_collection", plannedItemIds: ["81"] })
    expect(useQuickIngestSessionStore.getState().session?.tracking?.jobIds).toBeUndefined()
    const reloaded = JSON.parse(JSON.stringify(useQuickIngestSessionStore.getState().session!))
    view.unmount()
    act(() => { useQuickIngestSessionStore.setState({ session: null }); useQuickIngestSessionStore.getState().createDraftSession(reloaded) })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Retry https://source.test/talk.mp4" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([expect.objectContaining({ id: "talk-failed", status: "ok", mediaId: 99, collectionItemId: 81, retryAttempt: 4 }), expect.objectContaining({ id: "saved", mediaId: 7 })])))
    expect(mocks.bgUpload.mock.calls.map(([request]) => request.fields.idempotency_key)).toEqual(["conference-retry-81-3", "conference-retry-81-4"])
    expect(mocks.bgUpload.mock.calls.map(([request]) => [request.fields.media_collection_id, request.fields.media_collection_item_id])).toEqual([[7,81],[7,81]])
    expect(mocks.bgRequest.mock.calls.some(([request]) => request.method === "POST" && request.path.startsWith("/api/v1/media/collections"))).toBe(false)
  })

  it.each(["mirror failure", "owner transition"])(
    "publishes one consistent owned snapshot with other-owner IDs already present during %s",
    async (mode) => {
    mocks.useActualResultsStep = true
    localStorage.setItem("media-review-selection", JSON.stringify([42]))
    localStorage.setItem("media-review-selection-snapshot", JSON.stringify({ version: 1, authorityKey: "other-owner", selectedIds: [42] }))
    const mirror = deferred<void>()
    mocks.reviewSelectionMirrorWait = mirror.promise
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed", results: [{ id: "saved", status: "ok", type: "pdf", mediaId: 18 }] })
    const authorityKey = useQuickIngestSessionStore.getState().authorityKey
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Review this 1 saved item" }))
    await waitFor(() => expect(mocks.reviewSelectionMirrorStarted).toHaveBeenCalledTimes(1))
    // Raw IDs still belong to the previous owner while the mirror write is suspended.
    expect(JSON.parse(localStorage.getItem("media-review-selection") || "null")).toEqual([42])
    expect(JSON.parse(localStorage.getItem("media-review-selection-snapshot") || "null")).toEqual({ version: 1, authorityKey, selectedIds: [18] })
    if (mode === "owner transition") act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    else mocks.failReviewSelectionMirror = true
    await act(async () => { mirror.resolve(); await mirror.promise })
    expect(JSON.parse(localStorage.getItem("media-review-selection-snapshot") || "null")).toEqual({ version: 1, authorityKey, selectedIds: [18] })
    if (mode === "owner transition") expect(mocks.navigate).not.toHaveBeenCalled()
    else await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/media-multi"))
  }
  )

  it("initializes a failed retry in the canonical queued state before executor acknowledgement", async () => {
    mocks.useActualResultsStep = true
    mocks.startQuickIngestSession.mockReturnValue(new Promise(() => {}))
    useQuickIngestSessionStore.getState().createDraftSession({
      currentStep: 5,
      lifecycle: "completed",
      queueItems: [
        {
          id: "failed",
          kind: "url",
          url: "https://source.test/retry.pdf",
          detectedType: "pdf",
          icon: "FileText",
          fileSize: 0,
          validation: { valid: true }
        }
      ],
      results: [{ id: "failed", status: "error", type: "pdf", error: "Network error" }]
    })
    render(<SessionBackedQuickIngestModal />)
    fireEvent.click(screen.getByRole("button", { name: "Retry failed" }))
    expect(screen.getByTestId("retry-item-status")).toHaveTextContent("queued")
  })

  it("retries a completed run with the real Modal and persists progress without a store feedback loop", async () => {
    mocks.useActualModal = true
    mocks.useActualResultsStep = true
    const response = deferred<any>()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-direct-live-retry"
    })
    mocks.submitQuickIngestBatch.mockReturnValue(response.promise)
    useQuickIngestSessionStore.getState().createDraftSession({
      currentStep: 5,
      lifecycle: "partial_failure",
      queueItems: [
        { id: "saved", kind: "url", url: "https://source.test/saved.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } },
        {
          id: "failed",
          kind: "url",
          url: "https://source.test/retry.pdf",
          detectedType: "pdf",
          icon: "FileText",
          fileSize: 0,
          validation: { valid: true }
        }
      ],
      results: [
        { id: "saved", status: "ok", type: "pdf", mediaId: 77 },
        { id: "failed", status: "error", type: "pdf", error: "Network error" }
      ],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-direct-previous",
        startedAt: Date.now() - 2000
      }
    })
    render(<SessionBackedQuickIngestModal />)
    const retryButton = await screen.findByRole("button", { name: "Retry failed" })
    flushSync(() => retryButton.click())
    await waitFor(() =>
      expect(
        useQuickIngestSessionStore.getState().session?.tracking?.sessionId
      ).toBe("qi-direct-live-retry")
    )
    expect(
      useQuickIngestSessionStore.getState().session?.processingState.status
    ).toBe("running")
    await act(async () =>
      response.resolve({
        ok: true,
        results: [{ id: "failed", status: "ok", type: "pdf", mediaId: 78 }]
      })
    )
    await waitFor(() =>
      expect(useQuickIngestSessionStore.getState().session?.results).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ id: "saved", mediaId: 77 }),
          expect.objectContaining({ id: "failed", mediaId: 78 })
        ])
      )
    )
    expect(
      screen.getByRole("dialog", { name: "Quick Ingest" })
    ).toBeInTheDocument()
  })

  it("retries only failed sources through the real executor and preserves saved successes and options", async () => {
    mocks.useActualResultsStep = true
    const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
    mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
    mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
    mocks.bgUpload.mockResolvedValue({ batch_id: "retry-batch", jobs: [{ id: 78 }] })
    mocks.bgRequest.mockResolvedValue({ ok: true, data: { status: "completed", result: { status: "Success", media_id: 78 } } })
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      presetConfig: { ...resolvePresetMap().quick, common: { perform_analysis: false, perform_chunking: false, overwrite_existing: false }, storeRemote: true, reviewBeforeStorage: false, typeDefaults: { audio: { language: "fr", diarize: true } } },
      queueItems: [
        { id: "saved", kind: "url", url: "https://source.test/saved.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } },
        { id: "failed", kind: "url", url: "https://source.test/failed.mp3", detectedType: "audio", icon: "Music", fileSize: 0, validation: { valid: true }, conferenceOverride: { title: "Original title", tags: ["original"] } },
      ],
      results: [{ id: "saved", status: "ok", type: "pdf", title: "Already saved", mediaId: 77, persisted: true }, { id: "failed", status: "error", type: "audio", url: "https://source.test/failed.mp3", error: "Network error" }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: /Retry https:\/\/source.test\/failed.mp3/ }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "saved", mediaId: 77, status: "ok" }), expect.objectContaining({ id: "failed", mediaId: 78, status: "ok" }),
    ])))
    expect(mocks.bgUpload).toHaveBeenCalledTimes(1)
    expect(mocks.bgUpload.mock.calls[0][0]).toMatchObject({ fields: { urls: ["https://source.test/failed.mp3"], transcription_language: "fr", diarize: true } })
    expect(mocks.submitQuickIngestBatch.mock.calls[0][0].entries[0].conferenceOverride).toEqual({ title: "Original title", tags: ["original"] })
    fireEvent.click(screen.getByRole("button", { name: "Open Already saved in Media" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/media?id=77")
  })

  it("retries conference failures in the original collection with fresh attempt identities", async () => {
    mocks.useActualResultsStep = true
    const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
    mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
    mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
    mocks.bgUpload.mockResolvedValue({ batch_id: "conference-retry", jobs: [{ id: 88 }] })
    let attempt = 0
    mocks.bgRequest.mockImplementation(async ({ path }: { path: string }) => path.includes("/ingest/jobs/")
      ? { ok: true, data: { status: "completed", result: ++attempt === 1 ? { status: "Error", error: "Network error" } : { status: "Success", media_id: 88 } } }
      : {})
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "partial_failure",
      presetConfig: { ...resolvePresetMap().quick, common: { perform_analysis: false, perform_chunking: false, overwrite_existing: false }, storeRemote: true },
      conferenceBatchMetadata: { collectionName: "Original conference" },
      queueItems: [{ id: "talk-failed", kind: "url", url: "https://source.test/talk.mp4", detectedType: "video", icon: "Film", fileSize: 0, validation: { valid: true }, conferenceOverride: { title: "Talk title", selected: true } }],
      results: [{ id: "talk-failed", status: "error", type: "video", url: "https://source.test/talk.mp4", error: "Network error", collectionItemId: "81", retryAttempt: 2 }],
      tracking: { mode: "webui-direct", collectionId: "7", durableMode: "durable_collection", startedAt: 1 },
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Retry https://source.test/talk.mp4" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual([expect.objectContaining({ id: "talk-failed", collectionItemId: 81, retryAttempt: 3, status: "error" })]))
    fireEvent.click(await screen.findByRole("button", { name: "Retry https://source.test/talk.mp4" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual([expect.objectContaining({ mediaId: 88, collectionItemId: 81, retryAttempt: 4 })]))
    expect(mocks.bgUpload.mock.calls.map(([request]) => request.fields.idempotency_key)).toEqual(["conference-retry-81-3", "conference-retry-81-4"])
    expect(mocks.bgUpload.mock.calls[0][0].fields).toMatchObject({ media_collection_id: 7, media_collection_item_id: 81 })
    expect(mocks.bgRequest.mock.calls.some(([request]) => request.method === "POST" && request.path.startsWith("/api/v1/media/collections"))).toBe(false)
  })

  it("offers reattachment after reload, then retries the original File source without repeating saved items", async () => {
    mocks.useActualResultsStep = true
    const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
    mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
    mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
    mocks.bgUpload.mockResolvedValue({ batch_id: "file-retry", jobs: [{ id: 79 }] })
    mocks.bgRequest.mockResolvedValue({ ok: true, data: { status: "completed", result: { status: "Success", media_id: 79 } } })
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      presetConfig: { ...resolvePresetMap().quick, common: { perform_analysis: false, perform_chunking: false, overwrite_existing: false }, storeRemote: true },
      queueItems: [{ id: "file-failed", kind: "file", fileName: "source.txt", size: 5, fileSize: 5, detectedType: "document", icon: "FileText", validation: { valid: true } }],
      results: [{ id: "file-failed", status: "error", type: "document", fileName: "source.txt", error: "Network error" }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Retry source.txt" })).toBeNull()
    fireEvent.change(await screen.findByLabelText("Reattach source.txt"), { target: { files: [new NodeFile(["hello"], "wrong.txt", { type: "text/plain" })] } })
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose the original file")
    expect(mocks.bgUpload).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Reattach source.txt"), { target: { files: [new NodeFile(["hello"], "source.txt", { type: "text/plain" })] } })
    fireEvent.click(await screen.findByRole("button", { name: "Retry source.txt" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual([expect.objectContaining({ id: "file-failed", mediaId: 79 })]))
    expect(mocks.bgUpload.mock.calls[0][0].file.name).toBe("source.txt")
  })

  it("abandons a retry when its owner changes before upload acknowledgement", async () => {
    mocks.useActualResultsStep = true
    const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
    mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
    mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
    const upload = deferred<any>()
    mocks.bgUpload.mockReturnValue(upload.promise)
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "partial_failure",
      queueItems: [{ id: "failed", kind: "url", url: "https://source.test/source.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } }],
      results: [{ id: "failed", status: "error", type: "pdf", error: "Network error" }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Retry failed" }))
    await waitFor(() => expect(mocks.bgUpload).toHaveBeenCalledTimes(1))
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    await act(async () => { upload.resolve({ batch_id: "old-owner", jobs: [{ id: 99 }] }); await upload.promise })
    expect(screen.queryByRole("button", { name: /Review.*saved items?/ })).toBeNull()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
  })

  it("accounts for partial terminal responses and preserves previous saved File IDs after reload", async () => {
    mocks.useActualResultsStep = true
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-partial" })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [{ id: "first", status: "ok", type: "pdf", mediaId: 8 }] })
    useQuickIngestSessionStore.getState().createDraftSession({ queueItems: [
      { id: "first", kind: "url", url: "https://source.test/a.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } },
      { id: "missing", kind: "url", url: "https://source.test/b.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } },
    ] })
    const view = render(<QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />)
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "missing", status: "error" }), expect.objectContaining({ id: "first", mediaId: 8 }),
    ])))
    view.unmount()
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      queueItems: [{ id: "saved-file", kind: "file", fileName: "saved.txt", detectedType: "document", icon: "FileText", fileSize: 4, validation: { valid: true } }],
      results: [{ id: "saved-file", status: "ok", type: "document", fileName: "saved.txt", mediaId: 9, persisted: true }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Review this 1 saved item" }))
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/media-multi"))
    expect(JSON.parse(localStorage.getItem("media-review-selection") || "null")).toEqual([9])
  })

  it.each([
    ["Review this 1 saved item", "/media-multi"],
    ["Open saved.pdf in Media", "/media?id=7"],
    ["Ask added items", "/knowledge?media_ids=7"],
  ])("opens sidebar result action %s in the full-page workspace", async (label, route) => {
    window.history.replaceState({}, "", "/sidepanel.html")
    mocks.createTab.mockClear()
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed", results: [
      { id: "saved", status: "ok", type: "pdf", fileName: "saved.pdf", mediaId: 7 },
    ] })
    try {
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      fireEvent.click(await screen.findByRole("button", { name: label }))
      await waitFor(() => expect(mocks.createTab).toHaveBeenCalledWith({ url: `chrome-extension://test/options.html#${route}` }))
      expect(mocks.navigate).not.toHaveBeenCalled()
      if (route === "/media-multi") expect(JSON.parse(localStorage.getItem("media-review-selection-snapshot") || "null")).toEqual({ version: 1, authorityKey: useQuickIngestSessionStore.getState().authorityKey, selectedIds: [7] })
    } finally {
      window.history.replaceState({}, "", "/")
    }
  })

  it.each(["success", "failure", "owner transition", "session replacement", "unmount"])("keeps the sidebar results until tab creation settles with %s", async (outcome) => {
    window.history.replaceState({}, "", "/sidepanel.html")
    mocks.useActualResultsStep = true
    const opening = deferred<object>();
    mocks.createTab.mockReturnValueOnce(opening.promise)
    const onClose = vi.fn()
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed", results: [{ id: "saved", status: "ok", type: "pdf", mediaId: 18 }] })
    const view = render(<QuickIngestWizardModal open onClose={onClose} />)
    fireEvent.click(await screen.findByRole("button", { name: "Review this 1 saved item" }))
    await waitFor(() => expect(mocks.createTab).toHaveBeenCalledTimes(1))
    expect(onClose).not.toHaveBeenCalled()
    if (outcome === "owner transition") act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    if (outcome === "session replacement") act(() => { useQuickIngestSessionStore.getState().replaceWithNewDraft() })
    if (outcome === "unmount") view.unmount()
    await act(async () => { if (outcome === "failure") opening.reject(new Error("Tab unavailable")); else opening.resolve({}); await opening.promise.catch(() => {}) })
    if (outcome === "success") expect(onClose).toHaveBeenCalledTimes(1)
    else expect(onClose).not.toHaveBeenCalled()
    if (outcome === "failure") expect(await screen.findByRole("alert")).toHaveTextContent("Could not open the full-page workspace")
  })

  it("does not open a sidebar saved workspace after owner changes during the selection write", async () => {
    window.history.replaceState({}, "", "/sidepanel.html")
    mocks.useActualResultsStep = true
    const mirror = deferred<void>()
    mocks.reviewSelectionMirrorWait = mirror.promise
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed", results: [{ id: "saved", status: "ok", type: "pdf", mediaId: 18 }] })
    const authorityKey = useQuickIngestSessionStore.getState().authorityKey
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Review this 1 saved item" }))
    await waitFor(() => expect(mocks.reviewSelectionMirrorStarted).toHaveBeenCalledTimes(1))
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    await act(async () => { mirror.resolve(); await mirror.promise })
    expect(mocks.createTab).not.toHaveBeenCalled()
    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(JSON.parse(localStorage.getItem("media-review-selection-snapshot") || "null")).toEqual({ version: 1, authorityKey, selectedIds: [18] })
  })

  it.each(["/", "/options.html"])("hands off unique authoritative saved IDs in place on %s and rejects stale owner actions", async (path) => {
    window.history.replaceState({}, "", path)
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed", results: [
      { id: "saved", status: "ok", type: "pdf", mediaId: 7 }, { id: "same", status: "ok", type: "pdf", mediaId: "7" },
      { id: "unsaved", status: "ok", type: "pdf", persisted: false, mediaId: null }, { id: "failed", status: "error", type: "pdf", mediaId: 10 },
    ] })
    const view = render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: /Review.*saved items?/ }))
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/media-multi"))
    expect(mocks.createTab).not.toHaveBeenCalled()
    expect(JSON.parse(localStorage.getItem("media-review-selection") || "null")).toEqual([7])
    expect(JSON.parse(localStorage.getItem("media-review-selection-snapshot") || "null")).toEqual({ version: 1, authorityKey: useQuickIngestSessionStore.getState().authorityKey, selectedIds: [7] })
    const staleButton = screen.getByRole("button", { name: /Review.*saved items?/ })
    mocks.navigate.mockClear()
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    fireEvent.click(staleButton)
    expect(mocks.navigate).not.toHaveBeenCalled()
    view.unmount()
  })

  it.each([false, true])(
    "passes durable retry identity with previous jobs, including pre-ack remount %s",
    async (remount) => {
      mocks.useActualResultsStep = true
      useQuickIngestSessionStore.getState().createDraftSession({
        ...createEmptyQuickIngestSession(),
        currentStep: 5,
        lifecycle: "partial_failure",
        queueItems: [
          {
            id: "failed-item",
            kind: "url",
            url: "https://example.com/source.pdf",
            detectedType: "pdf",
            icon: "File",
            fileSize: 0,
            validation: { valid: true }
          }
        ],
        results: [
          { id: "saved-item", type: "pdf", status: "ok", mediaId: 1 },
          {
            id: "failed-item",
            type: "pdf",
            status: "error",
            outcome: "failed",
            error: "Network timeout",
            collectionItemId: "11",
            retryAttempt: 1
          }
        ],
        tracking: {
          mode: "webui-direct",
          sessionId: "qi-direct-old",
          jobIds: [66],
          itemIds: ["failed-item"],
          jobIdToItemId: { "66": "failed-item" },
          collectionId: "7",
          durableMode: "durable_collection"
        },
        processingState: {
          status: "complete",
          perItemProgress: [],
          elapsed: 1,
          estimatedRemaining: 0
        }
      })
      mocks.startQuickIngestSession.mockResolvedValue({
        ok: true,
        sessionId: "qi-direct-durable-retry"
      })
      mocks.submitQuickIngestBatch.mockResolvedValue({
        ok: true,
        results: [
          {
            id: "failed-item",
            type: "pdf",
            status: "ok",
            mediaId: 3,
            collectionItemId: 11,
            retryAttempt: 2
          }
        ]
      })
      if (remount)
        mocks.startQuickIngestSession.mockImplementationOnce(
          () => new Promise(() => {})
        )
      const firstRender = render(
        <QuickIngestWizardModal open onClose={vi.fn()} />
      )
      fireEvent.click(await screen.findByRole("button", { name: /Retry all/ }))
      if (remount) {
        await waitFor(() =>
          expect(mocks.startQuickIngestSession).toHaveBeenCalledOnce()
        )
        expect(
          useQuickIngestSessionStore.getState().session?.tracking?.jobIds
        ).toBeUndefined()
        firstRender.unmount()
        render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      }
      await waitFor(() =>
        expect(mocks.submitQuickIngestBatch).toHaveBeenCalledOnce()
      )
      expect(
        mocks.submitQuickIngestBatch.mock.calls[0][0].conferenceRetry
      ).toEqual({
        collectionId: "7",
        items: [
          {
            resultId: "failed-item",
            collectionItemId: "11",
            retryAttempt: 2,
            idempotencyKey: "conference-retry-11-2"
          }
        ]
      })
      expect(
        useQuickIngestSessionStore.getState().session?.tracking
      ).toMatchObject({ collectionId: "7", durableMode: "durable_collection" })
      expect(
        useQuickIngestSessionStore
          .getState()
          .session?.results.filter((item) => item.status === "ok")
          .map((item) => item.mediaId)
      ).toEqual([1, 3])
    }
  )
  it.each([
    "remount",
    "remount-empty",
    "start-ack",
    "start-reject",
    "batch-reject"
  ])(
    "preserves durable lineage through %s failure and advances the next retry",
    async (failure) => {
      mocks.useActualResultsStep = true
      const remount = failure.startsWith("remount")
      const retryIdentity = {
        resultId: "failed-item",
        collectionItemId: "11",
        retryAttempt: 2,
        idempotencyKey: "conference-retry-11-2"
      }
      useQuickIngestSessionStore.getState().createDraftSession({
        ...createEmptyQuickIngestSession(),
        currentStep: remount ? 4 : 5,
        lifecycle: remount ? "processing" : "partial_failure",
        queueItems: [
          {
            id: "failed-item",
            kind: "url",
            url: "https://example.com/source.pdf",
            detectedType: "pdf",
            icon: "File",
            fileSize: 0,
            validation: { valid: true }
          }
        ],
        results: [
          { id: "saved-item", type: "pdf", status: "ok", mediaId: 1 },
          ...(remount
            ? []
            : [
                {
                  id: "failed-item",
                  type: "pdf",
                  status: "error" as const,
                  outcome: "failed" as const,
                  error: "Network timeout",
                  collectionItemId: "11",
                  retryAttempt: 1,
                  idempotencyKey: "conference-retry-11-1"
                }
              ])
        ],
        tracking: {
          mode: "webui-direct",
          sessionId: "qi-direct-old",
          collectionId: "7",
          durableMode: "durable_collection",
          ...(remount
            ? {
                jobIds: [77],
                itemIds: ["failed-item"],
                jobIdToItemId: { "77": "failed-item" },
                jobIdToCollectionItemId: { "77": "11" },
                retryItems: [retryIdentity]
              }
            : {})
        },
        processingState: {
          status: remount ? "running" : "complete",
          perItemProgress: [],
          elapsed: 1,
          estimatedRemaining: 0
        }
      })
      mocks.startQuickIngestSession.mockResolvedValue({
        ok: true,
        sessionId: "qi-direct-next-retry"
      })
      mocks.submitQuickIngestBatch.mockResolvedValue({
        ok: true,
        results: [
          {
            id: "failed-item",
            type: "pdf",
            status: "ok",
            mediaId: 3,
            collectionItemId: 11,
            retryAttempt: 3,
            idempotencyKey: "conference-retry-11-3"
          }
        ]
      })
      if (remount) {
        mocks.reattachQuickIngestSession.mockResolvedValue({
          lifecycle: "partial_failure",
          jobs:
            failure === "remount-empty"
              ? []
              : [
                  {
                    jobId: 77,
                    status: "failed",
                    error: "Network timeout"
                  }
                ],
          errorMessage: "Network timeout"
        })
      } else if (failure === "start-ack") {
        mocks.startQuickIngestSession.mockResolvedValueOnce({
          ok: false,
          error: "Network timeout"
        })
      } else if (failure === "start-reject") {
        mocks.startQuickIngestSession.mockRejectedValueOnce(
          new Error("Network timeout")
        )
      } else {
        mocks.submitQuickIngestBatch.mockRejectedValueOnce(
          new Error("Network timeout")
        )
      }
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      if (!remount) {
        fireEvent.click(await screen.findByRole("button", { name: /Retry all/ }))
        await waitFor(() =>
          expect(mocks.startQuickIngestSession).toHaveBeenCalledOnce()
        )
      }
      await waitFor(() => {
        expect(
          useQuickIngestSessionStore
            .getState()
            .session?.results.find((item) => item.id === "failed-item")
        ).toMatchObject({
          status: "error",
          collectionItemId: "11",
          retryAttempt: 2,
          idempotencyKey: "conference-retry-11-2"
        })
      })
      fireEvent.click(await screen.findByRole("button", { name: /Retry all/ }))
      await waitFor(() =>
        expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(
          failure === "batch-reject" ? 2 : 1
        )
      )
      const payload = mocks.submitQuickIngestBatch.mock.lastCall?.[0]
      expect(payload.conferenceRetry).toEqual({
        collectionId: "7",
        items: [
          {
            ...retryIdentity,
            retryAttempt: 3,
            idempotencyKey: "conference-retry-11-3"
          }
        ]
      })
      expect(payload.entries.map((item: { id: string }) => item.id)).toEqual([
        "failed-item",
      ]);
      await waitFor(() =>
        expect(
          useQuickIngestSessionStore
            .getState()
            .session?.results.filter((item) => item.status === "ok")
            .map((item) => item.mediaId)
        ).toEqual([1, 3])
      )
    }
  )

  it("keeps earlier successes when a retried direct job reattaches after remount", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore
      .getState()
      .createDraftSession({
        ...createEmptyQuickIngestSession(),
        currentStep: 4,
        lifecycle: "processing",
        queueItems: [
          {
            id: "failed-item",
            kind: "url",
            url: "https://example.com/source.pdf",
            detectedType: "pdf",
            icon: "File",
            fileSize: 0,
            validation: { valid: true }
          }
        ],
        results: [{ id: "saved-item", type: "pdf", status: "ok", mediaId: 1 }],
        tracking: {
          mode: "webui-direct",
          sessionId: "qi-direct-retry",
          jobIds: [77],
          itemIds: ["failed-item"],
          jobIdToItemId: { "77": "failed-item" }
        },
        processingState: {
          status: "running",
          perItemProgress: [],
          elapsed: 1,
          estimatedRemaining: 0
        }
      })
    mocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "completed",
      jobs: [{ id: 77, status: "completed", result: { media_id: 3 } }]
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await screen.findByRole("button", { name: "Ask added items" })
    expect(
      useQuickIngestSessionStore
        .getState()
        .session?.results.map((item) => item.mediaId)
    ).toEqual([1, 3])
    fireEvent.click(screen.getByRole("button", { name: "Ask added items" }))
    expect(mocks.navigate.mock.calls.at(-1)?.[0]).toBe("/knowledge?media_ids=1%2C3")
  })

  it.each(["all", "item"])(
    "retries only eligible actual results via %s and keeps the accumulated exact scope",
    async (mode) => {
      mocks.useActualResultsStep = true
      const ids = [
        "saved-item",
        "failed-item",
        "permanent-item",
        "cancelled-item"
      ]
      useQuickIngestSessionStore.getState().createDraftSession({
        queueItems: ids.map((id) => ({
          id,
          kind: "url",
          url: `https://example.com/${id}`,
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true }
        })),
        presetConfig: {
          ...resolvePresetMap().quick,
          common: {
            ...resolvePresetMap().quick.common,
            perform_chunking: true
          },
          advancedValues: { chunk_size: 456 }
        }
      })
      mocks.startQuickIngestSession
        .mockResolvedValueOnce({ ok: true, sessionId: "qi-mixed-first" })
        .mockResolvedValueOnce({ ok: true, sessionId: "qi-mixed-retry" })
      render(
        <QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />
      )
      await waitFor(() =>
        expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
      )
      act(() =>
        emitRuntimeMessage({
          type: "tldw:quick-ingest/completed",
          payload: {
            sessionId: "qi-mixed-first",
            results: [
              {
                id: "saved-item",
                type: "web",
                status: "ok",
                mediaId: 1,
                title: "Saved source"
              },
              {
                id: "failed-item",
                type: "web",
                status: "error",
                outcome: "failed",
                error: "Network timeout",
                title: "Retry source"
              },
              {
                id: "permanent-item",
                type: "web",
                status: "error",
                outcome: "failed",
                error: "Unsupported format",
                title: "Permanent source"
              },
              {
                id: "cancelled-item",
                type: "web",
                status: "error",
                outcome: "cancelled",
                error: "Network timeout",
                title: "Cancelled source"
              }
            ]
          }
        })
      )
      await screen.findByTestId("wizard-results-step")
      expect(
        screen.queryByRole("button", { name: "Retry Cancelled source" })
      ).toBeNull()
      expect(
        screen.queryByRole("button", { name: "Retry Permanent source" })
      ).toBeNull()
      fireEvent.click(
        screen.getByRole("button", {
          name: mode === "all" ? /Retry all/ : "Retry Retry source"
        })
      )
      await waitFor(() =>
        expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(2)
      )
      const retriedItemIds =
        mocks.startQuickIngestSession.mock.calls[1][0].entries.map(
          (item: { id: string }) => item.id
        )
      expect(retriedItemIds).toEqual(["failed-item"])
      expect(mocks.startQuickIngestSession.mock.calls[1][0].common).toEqual(
        mocks.startQuickIngestSession.mock.calls[0][0].common
      )
      expect(
        mocks.startQuickIngestSession.mock.calls[1][0].advancedValues
      ).toEqual({ chunk_size: 456 })
      act(() =>
        emitRuntimeMessage({
          type: "tldw:quick-ingest/completed",
          payload: {
            sessionId: "qi-mixed-retry",
            results: [
              {
                id: "failed-item",
                type: "web",
                status: "ok",
                mediaId: 3,
                title: "Retried source"
              }
            ]
          }
        })
      )
      await screen.findByTestId("wizard-results-step")
      const successfulMediaIdsAfterRetry = useQuickIngestSessionStore
        .getState()
        .session!.results.filter((item) => item.status === "ok")
        .map((item) => item.mediaId)
      expect(successfulMediaIdsAfterRetry).toEqual([1, 3])
      expect(
        useQuickIngestSessionStore
          .getState()
          .session!.results.filter((item) => item.status === "error")
          .map((item) => item.id)
      ).toEqual(["permanent-item", "cancelled-item"])
      fireEvent.click(screen.getByRole("button", { name: "Ask added items" }))
      const { parseKnowledgeMediaScope } =
        await import("@/utils/knowledge-scope-handoff")
      const reopenedScope = {
        include_media_ids: parseKnowledgeMediaScope(
          new URL(mocks.navigate.mock.calls.at(-1)![0], "https://local.test")
            .search
        )!.mediaIds
      }
      expect(reopenedScope.include_media_ids).toEqual([1, 3])
    }
  )

  it("confirms only the failed retry target through actual Configure and Review while keeping prior outcomes", async () => {
    mocks.useActualResultsStep = true
    mocks.useActualConfigureStep = true
    mocks.useActualReviewStep = true
    const file = new NodeFile(["text"], "broken.pdf", { type: "application/pdf" }) as unknown as File
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(), currentStep: 5, highestStep: 5, lifecycle: "partial_failure",
      queueItems: [
        { id: "saved-url", kind: "url", url: "https://example.com/", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } },
        { id: "duplicate-url", kind: "url", url: "https://example.com/", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } },
        { id: "saved-file", kind: "file", fileName: "saved.md", detectedType: "document", icon: "FileText", fileSize: 4, validation: { valid: false } },
        { id: "processed-source", kind: "url", url: "https://example.com/processed", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } },
        { id: "skipped-source", kind: "url", url: "https://example.com/skipped", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } },
        { id: "failed-file", kind: "file", fileName: "broken.pdf", detectedType: "pdf", icon: "FileText", fileSize: 4, validation: { valid: false } }
      ],
      results: [
        { id: "saved-url", type: "html", status: "ok", mediaId: 3 },
        { id: "saved-file", fileName: "saved.md", type: "document", status: "ok", mediaId: 4 },
        { id: "processed-source", type: "html", status: "ok", outcome: "processed", persisted: false, mediaId: null },
        { id: "skipped-source", type: "html", status: "ok", outcome: "skipped", persisted: false, mediaId: null },
        { id: "failed-file", fileName: "broken.pdf", type: "pdf", status: "error", outcome: "failed", error: "Invalid PDF 422" }
      ]
    })
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-correction" })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.change(await screen.findByLabelText("Reattach broken.pdf"), { target: { files: [file] } })
    fireEvent.click(await screen.findByRole("button", { name: "Correct settings for broken.pdf" }))
    expect(await screen.findByText("1 eligible item in this run")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    const list = screen.getByRole("list", { name: "Items to process" })
    expect(within(list).getAllByText(/Saved — excluded from this run/)).toHaveLength(2)
    expect(within(list).getByText(/Completed — excluded from this run/)).toBeVisible()
    expect(within(list).getByText(/Skipped — excluded from this run/)).toBeVisible()
    expect(within(list).getByText(/Already queued — excluded/)).toBeVisible()
    expect(within(list).queryByText(/Invalid — excluded/)).toBeNull()
    expect(within(list).getByText(/Extract/)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Start processing" }))
    await waitFor(() => expect(mocks.startQuickIngestSession).toHaveBeenCalledOnce())
    expect(mocks.startQuickIngestSession.mock.calls[0][0]).toMatchObject({ entries: [], files: [{ id: "failed-file", name: "broken.pdf" }] })
    expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "saved-url", status: "ok", mediaId: 3 }),
      expect.objectContaining({ id: "saved-file", status: "ok", mediaId: 4 })
    ]))
  })

  it("requires a missing queued file to be reattached and keeps its settings and result ID", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore
      .getState()
      .createDraftSession({
        ...createEmptyQuickIngestSession(),
        currentStep: 5,
        lifecycle: "partial_failure",
        queueItems: [
          {
            id: "failed-file",
            kind: "file",
            fileName: "source.txt",
            detectedType: "document",
            icon: "File",
            fileSize: 4,
            validation: {
              valid: false,
              warnings: ["Reattach this file after refresh to process it."]
            }
          }
        ],
        presetConfig: {
          ...resolvePresetMap().quick,
          advancedValues: { chunk_size: 123 }
        },
        results: [
          {
            id: "failed-file",
            fileName: "source.txt",
            type: "document",
            status: "error",
            outcome: "failed",
            error: "Network timeout"
          }
        ],
        processingState: {
          status: "complete",
          perItemProgress: [],
          elapsed: 0,
          estimatedRemaining: 0
        }
      })
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-file-retry"
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    expect(
      screen.queryByRole("button", { name: "Retry source.txt" })
    ).toBeNull()
    const file = new NodeFile(["text"], "source.txt", {
      type: "text/plain"
    }) as unknown as File
    fireEvent.change(await screen.findByLabelText("Reattach source.txt"), {
      target: { files: [file] }
    })
    fireEvent.click(
      await screen.findByRole("button", { name: "Retry source.txt" })
    )
    await waitFor(() =>
      expect(mocks.startQuickIngestSession).toHaveBeenCalledOnce()
    )
    const request = mocks.startQuickIngestSession.mock.calls[0][0]
    expect(request.files).toMatchObject([
      { id: "failed-file", name: "source.txt", data: [116, 101, 120, 116] }
    ])
    expect(request.advancedValues).toEqual({ chunk_size: 123 })
  })

  it.each([false, true])(
    "refreshes the actual catalogue for current wizard completion (before mount=%s)",
    async (beforeMount) => {
      let saved = false
      mocks.bgRequest.mockImplementation(async ({ path }: { path: string }) => path.startsWith('/api/v1/media/?')
      ? { items: saved ? [{ id: 7, title: 'New Cedar source', type: 'document' }] : [], pagination: { total_items: saved ? 1 : 0, total_pages: 1 } }
      : { keywords: [] })
      mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: 'qi-catalogue' })
      useQuickIngestSessionStore.getState().createDraftSession({ queueItems: [{ id: 'cedar', kind: 'url', url: 'https://example.com/cedar', detectedType: 'web', icon: 'Globe', fileSize: 0, validation: { valid: true } }] })
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      const view = render(
        <QueryClientProvider client={client}>
          {!beforeMount && <CatalogueProbe />}
          <QuickIngestWizardModal key="wizard" open autoProcessQueued onClose={vi.fn()} />
        </QueryClientProvider>
      )
      if (!beforeMount) await waitFor(() => expect(screen.getByTestId('actual-catalogue')).toHaveTextContent('/ 0'))
      await waitFor(() => expect(mocks.startQuickIngestSession).toHaveBeenCalled())
      await act(async () => {
      saved = true
      for (const listener of mocks.runtimeListeners) listener({ type: 'tldw:quick-ingest/completed', payload: { sessionId: 'qi-catalogue', results: [{ id: 'cedar', status: 'ok', type: 'document', mediaId: 7 }] } })
    })
      await screen.findByTestId('wizard-result-cedar')
      if (beforeMount)
        view.rerender(
          <QueryClientProvider client={client}>
            <CatalogueProbe />
            <QuickIngestWizardModal key="wizard" open autoProcessQueued onClose={vi.fn()} />
          </QueryClientProvider>
        )
      await waitFor(() => expect(screen.getByTestId('actual-catalogue')).toHaveTextContent('New Cedar source / 1'))
    }
  )

  it.each(["error", "process-only"])(
    "does not announce a saved catalogue change for %s results",
    async (kind) => {
    const listener = vi.fn()
    window.addEventListener('tldw:quick-ingest-complete', listener)
    try {
      useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: 'completed',
        results: [{ id: 'cedar', type: 'document', status: kind === 'error' ? 'error' : 'ok', ...(kind === 'error' ? { mediaId: 7 } : {}) }] })
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await screen.findByTestId('wizard-result-cedar')
      expect(listener).not.toHaveBeenCalled()
    } finally { window.removeEventListener('tldw:quick-ingest-complete', listener) }
  }
  )

  it("masks Bob's completed result on logout before a replacement account can act", async () => {
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(),
      currentStep: 5,
      lifecycle: "completed",
      processingState: { status: "complete", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
      results: [{ id: "Bob-private.pdf", status: "ok", type: "pdf", mediaId: 7 }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    expect(await screen.findByTestId("wizard-result-Bob-private.pdf")).toBeInTheDocument()
    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await waitFor(() => expect(screen.queryByTestId("wizard-result-Bob-private.pdf")).not.toBeInTheDocument())
    expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
  })

  beforeEach(async () => {
    mocks.useActualAddContentStep = false
    mocks.useActualModal = false
    mocks.useActualProcessingStep = false
    mocks.useActualResultsStep = false
    mocks.useActualReviewStep = false
    mocks.useActualConfigureStep = false
    mocks.queueSourceFile = false
    mocks.queuedReviewFile = null
    mocks.reviewBatches.clear()
    mocks.reviewDrafts.clear()
    mocks.reviewFiles.clear()
    mocks.failReviewWrite = false
    mocks.reviewWriteWait = null
    mocks.reviewSelectionMirrorWait = null
    mocks.failReviewSelectionMirror = false
    mocks.reviewSelectionMirrorStarted.mockClear()
    mocks.reviewWriteStarted.mockClear()
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
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
    mocks.createTab.mockClear()
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
    localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: "https://test.test", authMode: "single-user", apiKey: "synthetic" }))
    useQuickIngestSessionStore.getState().setAuthority(null)
    releaseAuthority = quickIngestAuthority.retain()
    await waitFor(() => expect(useQuickIngestSessionStore.getState().authorityKey).toBeTruthy())
    mocks.initialize.mockClear()
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
  })

  afterEach(() => {
    window.history.replaceState({}, "", "/")
    releaseAuthority?.()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it.each(["direct upload", "reattach", "StrictMode reattach"])(
    "keeps a saved-source Warning navigable through actual %s, session state, and results UI",
    async (mode) => {
      mocks.useActualResultsStep = true
      const result = { status: "Warning", media_id: 1, error: null, warnings: ["Analysis failed for chunk 1", "Analysis failed for chunk 1"] }
      mocks.bgRequest.mockResolvedValue({ ok: true, data: { status: "completed", result, error_message: null } })
      if (mode === "direct upload") {
      mocks.queueSourceFile = true
      const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
      mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
      mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
      mocks.bgUpload.mockResolvedValue({ batch_id: "warning-batch", jobs: [{ id: 77 }] })
    } else {
      const reattach = await vi.importActual<typeof import("@/services/tldw/quick-ingest-session-reattach")>("@/services/tldw/quick-ingest-session-reattach")
      mocks.reattachQuickIngestSession.mockImplementation(reattach.reattachQuickIngestSession)
      useQuickIngestSessionStore.getState().createDraftSession({
        ...createEmptyQuickIngestSession(), lifecycle: "processing", currentStep: 4,
        queueItems: [{ id: "queued-url-1", kind: "url", url: "https://source.test/source.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } }],
        processingState: { status: "running", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
        tracking: { mode: "webui-direct", batchId: "warning-batch", jobIds: [77], startedAt: Date.now() },
      })
    }
      const wizard = <QuickIngestWizardModal open onClose={vi.fn()} />
      render(
        mode === "StrictMode reattach" ? (
          <React.StrictMode>{wizard}</React.StrictMode>
        ) : (
          wizard
        )
      )
      if (mode === "direct upload") fireEvent.click(screen.getByText("Queue And Process"))
      expect(await screen.findByRole("region", { name: "Items saved with warnings" })).toBeVisible()
      expect(screen.getByText("Analysis failed for chunk 1")).toBeVisible()
      expect(screen.queryByText("Review failed items")).toBeNull()
      expect(useQuickIngestSessionStore.getState().session?.results).toEqual([
      expect.objectContaining({ status: "ok", mediaId: 1, warning: "Analysis failed for chunk 1", data: result }),
    ])
      expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/media/ingest/jobs/77", method: "GET" }))
      if (mode !== "direct upload") {
      expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
      expect(mocks.bgUpload).not.toHaveBeenCalled()
      expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
    }
      fireEvent.click(screen.getByRole("button", { name: /open .* media/i }))
      expect(mocks.navigate).toHaveBeenCalledWith(expect.stringContaining("media"))
    }
  )

  it("retains durable attempt identities when a partial retry response omits an item", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "partial_failure",
      queueItems: ["first", "second"].map(id => ({ id, kind: "url", url: `https://source.test/${id}.pdf`, detectedType: "pdf", icon: "File", fileSize: 0, validation: { valid: true } })),
      results: ["first", "second"].map((id, index) => ({ id, title: id, type: "pdf", status: "error", outcome: "failed", error: "Network error", collectionItemId: 11 + index, retryAttempt: 0 })),
      tracking: { mode: "webui-direct", collectionId: "7", durableMode: "durable_collection" }
    })
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-partial" })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [{ id: "first", type: "pdf", status: "ok", mediaId: 8, collectionItemId: 11, retryAttempt: 1 }] })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Retry all 2 retryable errors" }))
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "second", status: "error", collectionItemId: "12", retryAttempt: 1, idempotencyKey: "conference-retry-12-1" })
    ])))
  })

  it.each(["audio URL", "audio file", "PDF URL", "process-only audio URL", "process-only audio file"])(
    "offers scoped Ask for stored media without requiring original-file retention (%s)", async (mode) => {
      mocks.useActualResultsStep = true
      mocks.useActualAddContentStep = mode.includes("file")
      const batch = await vi.importActual<typeof import("@/services/tldw/quick-ingest-batch")>("@/services/tldw/quick-ingest-batch")
      mocks.startQuickIngestSession.mockImplementation(batch.startQuickIngestSession)
      mocks.submitQuickIngestBatch.mockImplementation(batch.submitQuickIngestBatch)
      const processOnly = mode.startsWith("process-only")
      const file = mode.includes("file")
      const type = mode.includes("PDF") ? "pdf" : "audio"
      const attached = new NodeFile(["audio"], "source.mp3", { type: "audio/mpeg" }) as unknown as File
      mocks.bgUpload.mockResolvedValue(processOnly ? { status: "Success", media_id: 7 } : { batch_id: "stored-source", jobs: [{ id: 77 }] })
      mocks.bgRequest.mockResolvedValue({ ok: true, data: { status: "completed", result: { status: "Success", media_id: 7 } } })
      useQuickIngestSessionStore.getState().createDraftSession({
        presetConfig: { ...resolvePresetMap().quick, storeRemote: !processOnly },
        queueItems: file ? [] : [{ id: "source", kind: file ? "file" : "url", ...(file ? { file: attached, fileName: attached.name } : { url: `https://source.test/source.${type === "pdf" ? "pdf" : "mp3"}` }), detectedType: type, icon: "File", fileSize: file ? attached.size : 0, validation: { valid: true } }]
      })
      render(<QuickIngestWizardModal open autoProcessQueued={!file} onClose={vi.fn()} />)
      if (file) {
        fireEvent.change(screen.getByTestId("qi-file-input"), { target: { files: [attached] } })
        fireEvent.click(await screen.findByRole("button", { name: "Use defaults & process" }))
      }
      await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.currentStep).toBe(5))
      if (processOnly) {
        expect(screen.queryByRole("button", { name: "Ask added items" })).toBeNull()
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Ask added items" }))
        expect(mocks.navigate.mock.calls.at(-1)?.[0]).toBe("/knowledge?media_ids=7")
      }
    }
  )

  it("continues only successfully added canonical media IDs into Knowledge", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(),
      currentStep: 5,
      lifecycle: "completed",
      processingState: {
        status: "complete",
        perItemProgress: [],
        elapsed: 1,
        estimatedRemaining: 0,
      },
      results: [
        {
          id: "added",
          title: "Added source",
          status: "ok",
          outcome: "ingested",
          type: "pdf",
          mediaId: 3,
        },
        {
          id: "warning",
          title: "Saved with warning",
          status: "ok",
          warning: "Analysis unavailable",
          type: "pdf",
          mediaId: "7",
        },
        {
          id: "duplicate",
          title: "Existing source",
          status: "ok",
          outcome: "skipped",
          type: "pdf",
          mediaId: 11,
        },
        {
          id: "failed",
          title: "Failed source",
          status: "error",
          type: "pdf",
          mediaId: 13,
        },
      ],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Ask added items" }),
    )
    expect(mocks.navigate.mock.calls.at(-1)?.[0]).toBe(
      "/knowledge?media_ids=3%2C7",
    )
  })

  it("does not offer a whole-library continuation for successful extraction without a canonical saved media ID", async () => {
    mocks.useActualResultsStep = true
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(),
      currentStep: 5,
      lifecycle: "completed",
      processingState: {
        status: "complete",
        perItemProgress: [],
        elapsed: 1,
        estimatedRemaining: 0,
      },
      results: [
        {
          id: "local",
          title: "Process only source",
          status: "ok",
          type: "pdf",
          mediaId: null,
        },
      ],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    expect(await screen.findByText("Process only source")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: /Ask added|Search your ingested/ }),
    ).toBeNull()
  })

  it("ignores the cancelled StrictMode poll after its replacement accepts terminal results", async () => {
    const staleRead = deferred<{ ok: boolean; data: { status: string; error_message: string } }>()
    const reattach = await vi.importActual<typeof import("@/services/tldw/quick-ingest-session-reattach")>("@/services/tldw/quick-ingest-session-reattach")
    mocks.reattachQuickIngestSession.mockImplementation(reattach.reattachQuickIngestSession)
    mocks.bgRequest.mockReturnValueOnce(staleRead.promise).mockResolvedValue({
      ok: true, data: { status: "completed", result: { status: "Success", media_id: 77 } },
    })
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(), lifecycle: "processing", currentStep: 4,
      queueItems: [{ id: "queued-url-1", kind: "url", url: "https://source.test/source.pdf", detectedType: "pdf", icon: "FileText", fileSize: 0, validation: { valid: true } }],
      processingState: { status: "running", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
      tracking: { mode: "webui-direct", batchId: "strict-batch", jobIds: [77], startedAt: Date.now() },
    })
    render(
      <React.StrictMode>
        <QuickIngestWizardModal open onClose={vi.fn()} />
      </React.StrictMode>
    )
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results).toEqual([
      expect.objectContaining({ mediaId: 77, status: "ok" }),
    ]))
    await act(async () => {
      staleRead.resolve({ ok: true, data: { status: "failed", error_message: "Stale first read" } })
      await staleRead.promise
    })
    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      lifecycle: "completed", currentStep: 5,
      results: [expect.objectContaining({ mediaId: 77, status: "ok" })],
    })
    expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
    expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
  })

  it.each(["processing button", "close confirmation"])(
    "minimizes the actual processing view through %s and resumes the same job",
    async (via) => {
      mocks.useActualProcessingStep = true
      useQuickIngestSessionStore.getState().createDraftSession({
        ...createEmptyQuickIngestSession(),
        lifecycle: "processing",
        currentStep: 4,
        queueItems: [{
          id: "minimize-item", kind: "url", url: "https://example.com/article",
          detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true },
        }],
        processingState: {
          status: "running", elapsed: 1, estimatedRemaining: 12,
          perItemProgress: [{
            id: "minimize-item", status: "processing", progressPercent: 40,
            currentStage: "Processing", estimatedRemaining: 12,
          }],
        },
        tracking: {
          mode: "extension-runtime", sessionId: "minimize-runtime",
          itemIds: ["minimize-item"], startedAt: Date.now(),
        },
      })
      const original = useQuickIngestSessionStore.getState().session!
      render(<SessionBackedQuickIngestModal />)
      const minimizeButton = await screen.findByRole("button", { name: "Minimize to Background" })
      if (via === "processing button") {
        await userEvent.click(minimizeButton)
      } else {
        await userEvent.click(screen.getByRole("button", { name: "Close", exact: true }))
        const { Modal } = await import("antd")
        const options = vi.mocked(Modal.confirm).mock.calls.at(-1)?.[0]
        await act(async () => { await options?.onOk?.() })
      }
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
      expect(useQuickIngestSessionStore.getState().session).toMatchObject({
        id: original.id, visibility: "hidden", lifecycle: "processing",
        tracking: original.tracking,
        processingState: {
          status: "running", perItemProgress: original.processingState.perItemProgress,
        },
      })
      expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
      expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
      await act(async () => { useQuickIngestSessionStore.getState().showSession() })
      expect(await screen.findByRole("button", { name: "Minimize to Background" })).toBeInTheDocument()
      expect(useQuickIngestSessionStore.getState().session?.id).toBe(original.id)
    }
  )

  it("leaves pending stages and percentages unchanged as elapsed time passes, then accepts a terminal result", async () => {
    vi.useFakeTimers()
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-confirmed-only" })
    useQuickIngestSessionStore.getState().createDraftSession({
      queueItems: [{ id: "cedar", kind: "url", url: "https://example.com/cedar", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } }],
    })
    await act(async () => {
      render(<QuickIngestWizardModal open autoProcessQueued onClose={vi.fn()} />)
      await vi.advanceTimersByTimeAsync(0)
    })
    const before = useQuickIngestSessionStore.getState().session!.processingState.perItemProgress[0]
    await act(async () => { await vi.advanceTimersByTimeAsync(60_000) })
    expect(useQuickIngestSessionStore.getState().session!.processingState.perItemProgress[0]).toEqual(before)
    expect(before.progressPercent).toBe(0)
    expect(useQuickIngestSessionStore.getState().session!.processingState.elapsed).toBe(60)
    act(() => {
      for (const listener of mocks.runtimeListeners) listener({ type: "tldw:quick-ingest/completed", payload: {
        sessionId: "qi-confirmed-only", results: [{ id: "cedar", status: "ok", type: "document", mediaId: 1 }],
      } })
    })
    expect(useQuickIngestSessionStore.getState().session!.processingState.perItemProgress[0]).toMatchObject({ status: "complete", progressPercent: 100 })
  })

  it.each([true, false])(
    "creates exact owned review drafts only when selected (review=%s)",
    async (review) => {
    const file = new NodeFile(["Original Cedar file"], "cedar.txt", { type: "text/plain" }) as unknown as File
    mocks.queueSourceFile = true; mocks.queuedReviewFile = file
    useQuickIngestSessionStore.getState().createDraftSession({
      presetConfig: { ...resolvePresetMap().quick, reviewBeforeStorage: review },

    })
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-review-399" })
    mocks.submitQuickIngestBatch.mockResolvedValue({ ok: true, results: [{ id: "queued-url-1", fileName: file.name, type: "document", status: "ok", data: { results: [{ status: "Success", input_ref: "cedar.txt", media_type: "document", title: "Cedar source", content: "Exact processed Cedar content", keywords: ["cedar"] }, { status: "Error", input_ref: "failed", content: "must not create a draft" }] } }] })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "Queue And Process" }))
    await screen.findByTestId("wizard-result-queued-url-1")
    if (!review) { expect(mocks.reviewDrafts.size).toBe(0); return }
    await waitFor(() => expect(mocks.reviewDrafts.size).toBe(1))
    const draft = [...mocks.reviewDrafts.values()][0]
    expect(draft).toMatchObject({ title: "Cedar source", content: "Exact processed Cedar content", originalContent: "Exact processed Cedar content", keywords: ["cedar"], source: { kind: "file", fileName: "cedar.txt" }, ownerScope: useQuickIngestSessionStore.getState().authorityKey })
    expect(mocks.reviewFiles.get(String(draft.id))).toBe(file)
    expect(mocks.navigate).toHaveBeenCalledWith("/content-review?batch=" + encodeURIComponent(String(draft.batchId)))
  }
  )

  it("reopens a completed batch after remount without overwriting reviewed edits", async () => {
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      presetConfig: { ...resolvePresetMap().quick, reviewBeforeStorage: true },
      processingState: { status: "complete", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
      results: [{ id: "cedar", status: "ok", type: "document", data: { content: "Original processed text", title: "Cedar" } }],
    })
    const view = render(<QuickIngestWizardModal open={false} onClose={vi.fn()} />)
    expect(mocks.reviewDrafts.size).toBe(0)
    expect(mocks.navigate).not.toHaveBeenCalled()
    view.rerender(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(1))
    const draft = [...mocks.reviewDrafts.values()][0]
    mocks.reviewDrafts.set(String(draft.id), { ...draft, content: "User reviewed edits" })
    view.unmount()
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(2))
    expect(mocks.reviewDrafts.size).toBe(1)
    expect(mocks.reviewDrafts.get(String(draft.id))?.content).toBe("User reviewed edits")
  })

  it.each(["close", "unmount", "rerender", "close-reopen"])(
    "settles pending review creation safely after %s",
    async (transition) => {
      let resolveWrite!: () => void
      mocks.reviewWriteWait = new Promise<void>((resolve) => {
        resolveWrite = resolve
      })
      useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      presetConfig: { ...resolvePresetMap().quick, reviewBeforeStorage: true },
      processingState: { status: "complete", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
      results: [{ id: "cedar", status: "ok", type: "document", data: { content: "Pending processed text", title: "Cedar" } }],
    })
      const onClose = vi.fn()
      const modal = (open: boolean) => (
        <React.StrictMode>
          <QuickIngestWizardModal open={open} onClose={onClose} />
        </React.StrictMode>
      )
      const view = render(modal(true))
      await waitFor(() => expect(mocks.reviewWriteStarted).toHaveBeenCalledTimes(1))
      if (transition === "unmount") view.unmount()
    else if (transition === "rerender") view.rerender(modal(true))
    else view.rerender(modal(false))
      if (transition === "close-reopen") view.rerender(modal(true))
      await act(async () => { resolveWrite() })
      await waitFor(() => expect(mocks.reviewDrafts.size).toBe(1))
      const shouldNavigate = transition === "rerender" || transition === "close-reopen"
      expect(mocks.navigate).toHaveBeenCalledTimes(shouldNavigate ? 1 : 0)
      expect(onClose).toHaveBeenCalledTimes(shouldNavigate ? 1 : 0)
      expect(mocks.reviewWriteStarted).toHaveBeenCalledTimes(1)
    }
  )

  it("uses the active locale for an untitled review draft", async () => {
    const translate = vi.spyOn(i18n, "t").mockReturnValue("Source sans titre")
    try {
      useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
        presetConfig: { ...resolvePresetMap().quick, reviewBeforeStorage: true },
        processingState: { status: "complete", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
        results: [{ id: "untitled", status: "ok", type: "document", data: { content: "Owned source without title" } }],
      })
      render(<QuickIngestWizardModal open onClose={vi.fn()} />)
      await waitFor(() => expect(mocks.reviewDrafts.size).toBe(1))
      expect([...mocks.reviewDrafts.values()][0].title).toBe("Source sans titre")
      expect(translate).toHaveBeenCalledWith("playground:sharedWorkspace.untitled", "Untitled source")
    } finally {
      translate.mockRestore()
    }
  })

  it("keeps failed draft creation recoverable and retries the actual saved results", async () => {
    mocks.failReviewWrite = true
    useQuickIngestSessionStore.getState().createDraftSession({ ...createEmptyQuickIngestSession(), currentStep: 5, lifecycle: "completed",
      presetConfig: { ...resolvePresetMap().quick, reviewBeforeStorage: true },
      processingState: { status: "complete", perItemProgress: [], elapsed: 1, estimatedRemaining: 0 },
      results: [{ id: "cedar", status: "ok", type: "document", data: { content: "Retained after failure", title: "Cedar" } }],
    })
    render(<QuickIngestWizardModal open onClose={vi.fn()} />)
    await screen.findByText("Review disk is full")
    expect(mocks.navigate).not.toHaveBeenCalled()
    mocks.failReviewWrite = false
    fireEvent.click(screen.getByRole("button", { name: "Retry saving review drafts" }))
    await waitFor(() => expect(mocks.reviewDrafts.size).toBe(1))
    expect([...mocks.reviewDrafts.values()][0].content).toBe("Retained after failure")
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

    useQuickIngestSessionStore.getState().createDraftSession({
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
      expect(screen.getByTestId("wizard-results")).toHaveTextContent("complete:2")
    })
  })

  it("settles native capture loading after the actual Add step persists its queued URL", async () => {
    mocks.useActualAddContentStep = true
    mocks.useActualModal = true
    const tabs = deferred<any>()
    vi.stubGlobal("browser", { runtime: { id: "extension" }, tabs: { query: vi.fn().mockReturnValue(tabs.promise) } })
    try {
      useQuickIngestSessionStore.getState().createDraftSession()
      render(<ReadyEventModalHost />)
      const capture = await screen.findByRole("button", { name: "Capture current tab" })
      flushSync(() => capture.click())
      await waitFor(() => expect(capture).toHaveAttribute("aria-busy", "true"))
      await act(async () => { tabs.resolve([{ url: "https://example.com/native-capture" }]); await tabs.promise })
      await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.queueItems).toEqual([expect.objectContaining({ url: "https://example.com/native-capture" })]))
      await waitFor(() => expect(screen.getByRole("button", { name: "Capture current tab" })).toHaveAttribute("aria-busy", "false"))
      expect(screen.getByRole("button", { name: "Capture current tab" })).toBeEnabled()
      vi.mocked((globalThis as any).browser.tabs.query).mockResolvedValue([{ url: "chrome://settings" }])
      await userEvent.click(screen.getByRole("button", { name: "Capture current tab" }))
      expect(await screen.findByRole("alert")).toHaveTextContent(/HTTP.*HTTPS/)
      expect(useQuickIngestSessionStore.getState().session?.queueItems).toHaveLength(1)
    } finally { vi.unstubAllGlobals() }
  })

  it("keeps a live mixed executor authoritative when durable tracking arrives and the wizard is minimized", async () => {
    mocks.useActualProcessingStep = true
    const response = deferred<any>()
    mocks.startQuickIngestSession.mockResolvedValue({ ok: true, sessionId: "qi-direct-live-mixed" })
    mocks.submitQuickIngestBatch.mockImplementation((payload: any) => {
      payload.onTrackingMetadata({ mode: "webui-direct", sessionId: "qi-direct-live-mixed", batchId: "batch-77", batchIds: ["batch-77"], jobIds: [77], itemIds: ["url", "file"], submittedItemIds: ["file"], jobIdToItemId: { "77": "file" }, startedAt: Date.now() })
      return response.promise
    })
    mocks.reattachQuickIngestSession.mockResolvedValue({ lifecycle: "completed", jobs: [{ jobId: 77, status: "completed", sourceItemId: "file", result: { media_id: 78, title: "File" } }], errorMessage: null })
    useQuickIngestSessionStore.getState().createDraftSession({ queueItems: [
      { id: "url", kind: "url", url: "https://example.com/article", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } },
      { id: "file", kind: "file", fileName: "source.txt", detectedType: "document", icon: "FileText", fileSize: 8, validation: { valid: true } }
    ] })
    render(<ReadyEventModalHost />)
    await userEvent.click(screen.getByRole("button", { name: "Submit current queue" }))
    await waitFor(() => expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1))
    expect(mocks.reattachQuickIngestSession).not.toHaveBeenCalled()
    await userEvent.click(screen.getByRole("button", { name: "Minimize to Background" }))
    act(() => useQuickIngestSessionStore.getState().showSession())
    await screen.findByRole("button", { name: "Minimize to Background" })
    await act(async () => {
      response.resolve({ ok: true, results: [{ id: "url", status: "ok", mediaId: 77, type: "web" }, { id: "file", status: "ok", mediaId: 78, type: "document" }] })
      await response.promise
    })
    await waitFor(() => expect(useQuickIngestSessionStore.getState().session?.results.map(item => item.mediaId)).toEqual([77, 78]))
    expect(useQuickIngestSessionStore.getState().session?.lifecycle).toBe("completed")
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

    useQuickIngestSessionStore.getState().createDraftSession({
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
    useQuickIngestSessionStore.getState().createDraftSession({
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
    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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
        }), expect.objectContaining({ requestScope: expect.any(Object), signal: expect.any(AbortSignal) })
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

    useQuickIngestSessionStore.getState().createDraftSession({
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
    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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

    expect(useQuickIngestSessionStore.getState().session?.processingState.perItemProgress).toEqual([
      expect.objectContaining({ id: "queued-url-1", status: "processing", progressPercent: 0 }),
    ])

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

    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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
    useQuickIngestSessionStore.getState().createDraftSession({
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
    useQuickIngestSessionStore.getState().createDraftSession({
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

    useQuickIngestSessionStore.getState().createDraftSession({
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

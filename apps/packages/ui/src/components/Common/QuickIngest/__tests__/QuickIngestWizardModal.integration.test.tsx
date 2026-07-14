import React from "react"
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { afterEach, describe, it, expect, vi, beforeEach } from "vitest"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

// ---------------------------------------------------------------------------
// Mocks — must be declared before any imports that reference them
// ---------------------------------------------------------------------------
const getTranscriptionModelsMock = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ all_models: [] })
)
const getProvidersStatusMock = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ providers: [], any_configured: false })
)

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn(),
}))

const lifecycleMocks = vi.hoisted(() => ({
  cancelQuickIngestSession: vi.fn(),
  queryQuickIngestSession: vi.fn(),
  reattachQuickIngestSession: vi.fn(),
  retryQuickIngestSession: vi.fn(),
  retryRunItems: vi.fn(),
  submitQuickIngestBatch: vi.fn(),
}))

const transportMocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
}))

const runtimeMocks = vi.hoisted(() => ({
  listeners: [] as Array<(message: any) => void>,
}))

const virtualizerMocks = vi.hoisted(() => ({
  latestOptions: null as any,
}))

// react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultOrOpts?: any) => {
      if (typeof defaultOrOpts === "string") return defaultOrOpts
      if (defaultOrOpts?.defaultValue) {
        return defaultOrOpts.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_: string, token: string) => String(defaultOrOpts[token] ?? "")
        )
      }
      return key
    },
  }),
}))

// antd — mock to simple HTML elements
vi.mock("antd", () => ({
  Modal: Object.assign(
    ({ children, open, onCancel, title, ...props }: any) =>
      open ? (
        <div data-testid="modal" role="dialog">
          <h2>{title}</h2>
          {children}
        </div>
      ) : null,
    { confirm: vi.fn(), destroyAll: vi.fn() }
  ),
  Button: ({
    children,
    onClick,
    disabled,
    type,
    danger,
    size,
    ...props
  }: any) => (
    <button
      onClick={onClick}
      disabled={disabled}
      data-type={type}
      data-danger={danger ? "true" : undefined}
      data-size={size}
      {...props}
    >
      {children}
    </button>
  ),
  Switch: ({ checked, onChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={checked}
      onChange={(e: any) => onChange?.(e.target.checked)}
      {...props}
    />
  ),
  Select: ({
    value,
    onChange,
    onClear,
    options,
    placeholder,
    allowClear,
    popupMatchSelectWidth,
    showSearch,
    loading,
    ...props
  }: any) => {
    const selectProps: any = { ...props }
    const clearAriaLabel = props["aria-label"]
      ? `Clear ${String(props["aria-label"])}`
      : "Clear"
    if (value !== undefined) {
      selectProps.value = value
    }
    if (popupMatchSelectWidth !== undefined) {
      selectProps["data-popup-match-select-width"] = String(
        popupMatchSelectWidth
      )
    }

    return (
      <React.Fragment>
        <select
          onChange={(e: any) => onChange?.(e.target.value)}
          {...selectProps}
        >
          {placeholder ? (
            <option value="" disabled hidden>
              {placeholder}
            </option>
          ) : null}
          {options?.map((o: any) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        {allowClear ? (
          <button
            type="button"
            aria-label={clearAriaLabel}
            onClick={() => onClear?.()}
          >
            Clear
          </button>
        ) : null}
      </React.Fragment>
    )
  },
  AutoComplete: React.forwardRef(function AutoComplete(
    {
      value,
      onChange,
      options,
      allowClear,
      onClear,
      children,
      loading,
      notFoundContent,
      ...props
    }: any,
    ref: React.ForwardedRef<{ focus: () => void }>
  ) {
    const inputRef = React.useRef<HTMLInputElement>(null)
    React.useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
    }))
    return (
      <React.Fragment>
        <input
          ref={inputRef}
          role="combobox"
          value={value || ""}
          onChange={(event) => onChange?.(event.target.value)}
          {...props}
        />
        <div role="listbox">
          {options?.map((option: any) => (
            <div key={option.value} role="option" aria-selected="false">
              {option.label}
            </div>
          ))}
        </div>
        {loading || options?.length === 0 ? (
          <span data-testid="analysis-provider-catalog-status">
            {notFoundContent}
          </span>
        ) : null}
        {allowClear ? (
          <button
            type="button"
            aria-label="Clear analysis provider"
            onClick={() => {
              onClear?.()
              onChange?.("")
            }}
          >
            Clear
          </button>
        ) : null}
        {children ? <span hidden>{children}</span> : null}
      </React.Fragment>
    )
  }),
  Segmented: ({ options, value, onChange, ...props }: any) => (
    <div role="group" {...props}>
      {options?.map((option: any) => {
        const optionValue =
          typeof option === "object" ? option.value : option
        const label = typeof option === "object" ? option.label : option
        return (
          <button
            key={optionValue}
            type="button"
            aria-pressed={value === optionValue}
            onClick={() => onChange?.(optionValue)}
          >
            {label}
          </button>
        )
      })}
    </div>
  ),
  Radio: Object.assign(
    ({ children, value, ...props }: any) => (
      <label>
        <input type="radio" value={value} {...props} />
        {children}
      </label>
    ),
    {
      Group: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    }
  ),
  Collapse: ({ items, ...props }: any) => (
    <div {...props}>
      {items?.map((i: any) => (
        <div key={i.key}>{i.children}</div>
      ))}
    </div>
  ),
  Tooltip: ({ children }: any) => <>{children}</>,
  Alert: ({
    message,
    description,
    action,
    children,
    icon,
    type,
    showIcon,
    ...props
  }: any) => (
    <div role="alert" data-alert-type={type} {...props}>
      {showIcon ? icon : null}
      {message ? <span>{message}</span> : null}
      {description ? <p>{description}</p> : null}
      {children}
      {action}
    </div>
  ),
  Input: Object.assign(
    (props: any) => <input {...props} />,
    {
      TextArea: ({
        value,
        onChange,
        onKeyDown,
        placeholder,
        autoSize,
        ...props
      }: any) => (
        <textarea
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          {...props}
        />
      ),
    }
  ),
  Tag: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Typography: {
    Title: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    Text: ({ children, strong, ...props }: any) => (
      <span data-strong={strong ? "true" : undefined} {...props}>
        {children}
      </span>
    ),
  },
  Progress: ({ percent, ...props }: any) => (
    <div data-testid="progress" data-percent={percent} {...props} />
  ),
}))

// lucide-react — render simple spans with the icon name for testability
vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>()
  const iconNames = [
    "ArrowLeft",
    "ArrowRight",
    "ChevronDown",
    "ChevronRight",
    "Minimize2",
    "XCircle",
    "Info",
    "FileText",
    "Film",
    "Globe",
    "Music",
    "Image",
    "BookOpen",
    "File",
    "X",
    "Plus",
    "Check",
    "CheckCircle",
    "Circle",
    "Loader2",
    "Video",
    "FileQuestion",
    "AlertTriangle",
    "Play",
    "ExternalLink",
    "MessageSquare",
    "RefreshCw",
    "Trash2",
    "Search",
    "Download",
  ]
  const mocks: Record<string, any> = { ...actual }
  for (const name of iconNames) {
    mocks[name] = (props: any) => (
      <span
        data-icon={name}
        aria-hidden={props?.["aria-hidden"]}
        className={props?.className}
      />
    )
  }
  return mocks
})

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      onMessage: {
        addListener: (listener: (message: any) => void) => {
          runtimeMocks.listeners.push(listener)
        },
        removeListener: (listener: (message: any) => void) => {
          const index = runtimeMocks.listeners.indexOf(listener)
          if (index >= 0) runtimeMocks.listeners.splice(index, 1)
        },
      },
    },
  },
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return { ...actual, useNavigate: () => vi.fn() }
})

// SSE hook — no-op
vi.mock("@/components/Common/QuickIngest/useIngestSSE", () => ({
  useIngestSSE: () => {},
  default: () => {},
}))

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: (options: any) => {
    virtualizerMocks.latestOptions = options
    const { count, getItemKey } = options
    const [start, setStart] = React.useState(0)
    const mountedCount = Math.min(count, 12)
    const boundedStart = Math.min(start, Math.max(0, count - mountedCount))
    return {
      getTotalSize: () => count * 72,
      getVirtualItems: () =>
        Array.from({ length: mountedCount }, (_, offset) => {
          const index = boundedStart + offset
          return {
          index,
          start: index * 72,
          size: 72,
          key: getItemKey?.(index) ?? index,
          }
        }),
      measureElement: vi.fn(),
      scrollToIndex: (index: number) =>
        setStart(Math.max(0, index - mountedCount + 1)),
    }
  },
}))

// FileDropZone — simple placeholder div
vi.mock(
  "@/components/Common/QuickIngest/QueueTab/FileDropZone",
  () => ({
    FileDropZone: ({ onFilesAdded, autoFocus }: any) => {
      const ref = React.useRef<HTMLDivElement>(null)
      React.useEffect(() => {
        if (autoFocus) ref.current?.focus()
      }, [autoFocus])
      return (
        <div data-testid="file-drop-zone" ref={ref} tabIndex={0}>
          FileDropZone
          <button
            type="button"
            onClick={() =>
              onFilesAdded?.([
                {
                  name: "large-audio.mp3",
                  size: 45 * 1024 * 1024,
                  type: "audio/mpeg",
                },
              ])
            }
          >
            Add large audio file
          </button>
        </div>
      )
    },
    validateQuickIngestFile: (file: File) =>
      file.name === "invalid-replacement.bin"
        ? "Unsupported replacement file"
        : null,
    default: function MockFileDropZone({ onFilesAdded, autoFocus }: any) {
      const ref = React.useRef<HTMLDivElement>(null)
      React.useEffect(() => {
        if (autoFocus) ref.current?.focus()
      }, [autoFocus])
      return (
        <div data-testid="file-drop-zone" ref={ref} tabIndex={0}>
          FileDropZone
          <button
            type="button"
            onClick={() =>
              onFilesAdded?.([
                {
                  name: "large-audio.mp3",
                  size: 45 * 1024 * 1024,
                  type: "audio/mpeg",
                },
              ])
            }
          >
            Add large audio file
          </button>
        </div>
      )
    },
  })
)

// background-proxy
vi.mock("@/services/background-proxy", () => ({
  bgRequest: transportMocks.bgRequest,
}))

vi.mock("@/services/tldw/quick-ingest-batch", () => ({
  cancelQuickIngestSession: lifecycleMocks.cancelQuickIngestSession,
  queryQuickIngestSession: lifecycleMocks.queryQuickIngestSession,
  retryQuickIngestSession: lifecycleMocks.retryQuickIngestSession,
  startQuickIngestSession: vi.fn().mockResolvedValue({ ok: true, sessionId: "qi-test" }),
  submitQuickIngestBatch: lifecycleMocks.submitQuickIngestBatch,
}))

vi.mock("@/services/tldw/quick-ingest-session-reattach", () => ({
  reattachQuickIngestSession: lifecycleMocks.reattachQuickIngestSession,
}))

vi.mock("@/services/tldw/playlist-ingest", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/tldw/playlist-ingest")>()
  return { ...actual, retryRunItems: lifecycleMocks.retryRunItems }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getTranscriptionModels: getTranscriptionModelsMock,
    getProvidersStatus: getProvidersStatusMock,
  },
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities(),
}))

vi.mock("@/components/Common/QuickIngest/FloatingProgressWidget", () => ({
  FloatingProgressWidget: () => null,
}))

// Deterministic UUIDs
let uuidCounter = 0
beforeEach(() => {
  uuidCounter = 0
  runtimeMocks.listeners.splice(0)
  virtualizerMocks.latestOptions = null
  getTranscriptionModelsMock.mockReset().mockResolvedValue({ all_models: [] })
  getProvidersStatusMock.mockReset().mockResolvedValue({
    providers: [],
    any_configured: false,
  })
  capabilityMocks.useServerCapabilities.mockReset()
  capabilityMocks.useServerCapabilities.mockReturnValue({
    capabilities: { ffmpegAvailable: true },
    loading: false,
  })
  lifecycleMocks.cancelQuickIngestSession.mockReset().mockResolvedValue({ ok: true })
  lifecycleMocks.queryQuickIngestSession.mockReset().mockResolvedValue({
    ok: true,
    active: true,
    event: null,
  })
  lifecycleMocks.retryQuickIngestSession.mockReset().mockResolvedValue({ ok: true })
  lifecycleMocks.submitQuickIngestBatch.mockReset().mockResolvedValue({
    ok: true,
    accepted: true,
    runId: "run-file-resume",
  })
  transportMocks.bgRequest.mockReset().mockResolvedValue({})
  lifecycleMocks.retryRunItems.mockReset().mockResolvedValue({
    contractVersion: 2,
    runId: "run-lifecycle",
    version: 2,
    processingOccurrences: [],
  })
  lifecycleMocks.reattachQuickIngestSession.mockReset().mockResolvedValue({
    lifecycle: "processing",
    jobs: [],
    errorMessage: null,
  })
})

afterEach(() => {
  vi.useRealTimers()
})
vi.stubGlobal(
  "crypto",
  Object.assign({}, globalThis.crypto, {
    randomUUID: () => `test-uuid-${uuidCounter++}`,
  })
)

// ---------------------------------------------------------------------------
// Import component under test (after mocks)
// ---------------------------------------------------------------------------
import {
  IngestWizardProvider,
  useIngestWizard,
  type IngestWizardState,
} from "@/components/Common/QuickIngest/IngestWizardContext"
import { AddContentStep } from "@/components/Common/QuickIngest/AddContentStep"
import { WizardConfigureStep } from "@/components/Common/QuickIngest/WizardConfigureStep"
import { ReviewStep } from "@/components/Common/QuickIngest/ReviewStep"
import { ProcessingStep } from "@/components/Common/QuickIngest/ProcessingStep"
import { WizardResultsStep } from "@/components/Common/QuickIngest/WizardResultsStep"
import { QuickIngestWizardModal } from "@/components/Common/QuickIngestWizardModal"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const emitRuntimeMessage = (message: any) => {
  for (const listener of [...runtimeMocks.listeners]) listener(message)
}

/**
 * ContextSpy renders nothing but captures the wizard context reference
 * so tests can directly manipulate state (e.g. simulate processing completion).
 */
let ctxRef: ReturnType<typeof useIngestWizard> | null = null

const ContextSpy: React.FC = () => {
  const ctx = useIngestWizard()

  React.useEffect(() => {
    ctxRef = ctx

    return () => {
      ctxRef = null
    }
  }, [ctx])

  return null
}

const InnerWizardContent: React.FC<{
  onClose: () => void
  isStepVisible?: boolean
  isOnlineForIngest?: boolean
  connectionRecoveryMessage?: string
  isCheckingConnection?: boolean
  onRetryConnection?: () => void
  onQuickProcess?: () => void
  analysisProviderWarning?: string | null
  focusAnalysisProvider?: boolean
}> = ({
  onClose,
  isStepVisible = true,
  isOnlineForIngest = true,
  connectionRecoveryMessage,
  isCheckingConnection,
  onRetryConnection,
  onQuickProcess,
  analysisProviderWarning,
  focusAnalysisProvider,
}) => {
  const ctx = useIngestWizard()
  const { currentStep } = ctx.state

  return (
    <div data-testid="modal" role="dialog">
      <h2>Quick Ingest</h2>
      {/* Stepper labels */}
      <nav aria-label="Ingest wizard progress">
        <span>Add</span>
        <span>Configure</span>
        <span>Review</span>
        <span>Processing</span>
        <span>Results</span>
      </nav>
      {currentStep === 1 && (
        <AddContentStep
          isOnlineForIngest={isOnlineForIngest}
          connectionRecoveryMessage={connectionRecoveryMessage}
          isCheckingConnection={isCheckingConnection}
          onRetryConnection={onRetryConnection}
          onQuickProcess={onQuickProcess}
        />
      )}
      {currentStep === 2 && (
        <WizardConfigureStep
          isStepVisible={isStepVisible}
          analysisProviderWarning={analysisProviderWarning}
          focusAnalysisProvider={focusAnalysisProvider}
        />
      )}
      {currentStep === 3 && (
        <ReviewStep
          isOnlineForIngest={isOnlineForIngest}
          connectionRecoveryMessage={connectionRecoveryMessage}
          isCheckingConnection={isCheckingConnection}
          onRetryConnection={onRetryConnection}
        />
      )}
      {currentStep === 4 && <ProcessingStep />}
      {currentStep === 5 && <WizardResultsStep onClose={onClose} />}
    </div>
  )
}

// Final testable wrapper
const WizardTestHarness: React.FC<{
  onClose: () => void
  isOnlineForIngest?: boolean
  connectionRecoveryMessage?: string
  isCheckingConnection?: boolean
  onRetryConnection?: () => void
  onQuickProcess?: () => void
  onCancelProcessing?: () => boolean
  onCancelItem?: (id: string) => boolean
  onCheckStatus?: (id: string) => void
  onReconnect?: () => void
  initialState?: Partial<IngestWizardState>
  analysisProviderWarning?: string | null
  focusAnalysisProvider?: boolean
}> = ({
  onClose,
  isOnlineForIngest,
  connectionRecoveryMessage,
  isCheckingConnection,
  onRetryConnection,
  onQuickProcess,
  onCancelProcessing,
  onCancelItem,
  onCheckStatus,
  onReconnect,
  initialState,
  analysisProviderWarning,
  focusAnalysisProvider,
}) => {
  return (
    <IngestWizardProvider
      initialState={initialState}
      onCancelProcessing={onCancelProcessing}
      onCancelItem={onCancelItem}
      onCheckStatus={onCheckStatus}
      onReconnect={onReconnect}
    >
      <ContextSpy />
      <InnerWizardContent
        onClose={onClose}
        isOnlineForIngest={isOnlineForIngest}
        connectionRecoveryMessage={connectionRecoveryMessage}
        isCheckingConnection={isCheckingConnection}
        onRetryConnection={onRetryConnection}
        onQuickProcess={onQuickProcess}
        analysisProviderWarning={analysisProviderWarning}
        focusAnalysisProvider={focusAnalysisProvider}
      />
    </IngestWizardProvider>
  )
}

const ReopenableConfigStepHarness: React.FC<{ onClose: () => void }> = ({
  onClose,
}) => {
  const [stepVisible, setStepVisible] = React.useState(true)

  return (
    <IngestWizardProvider>
      <ContextSpy />
      <button type="button" onClick={() => setStepVisible(false)}>
        Hide configure step
      </button>
      <button type="button" onClick={() => setStepVisible(true)}>
        Show configure step
      </button>
      <InnerWizardContent
        onClose={onClose}
        isStepVisible={stepVisible}
      />
    </IngestWizardProvider>
  )
}

/**
 * Expand the "Advanced options" collapsible in the configure step.
 * Must be called after navigating to step 2 (configure).
 * No-ops if already expanded (toggle shows "Hide advanced options").
 */
const expandAdvancedOptions = async (user: ReturnType<typeof userEvent.setup>) => {
  const collapsed = screen.queryByText("Advanced options")
  if (collapsed) {
    await user.click(collapsed)
  }
  // If already expanded ("Hide advanced options" is showing), nothing to do.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("QuickIngestWizardModal — full wizard flow integration", () => {
  it("hydrates playlist preflight seed from typed open detail", () => {
    const candidates = [
      path.resolve(__dirname, "../IngestWizardContext.tsx"),
      path.resolve(process.cwd(), "src/components/Common/QuickIngest/IngestWizardContext.tsx"),
      path.resolve(
        process.cwd(),
        "../packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx"
      ),
      path.resolve(
        process.cwd(),
        "apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx"
      )
    ]
    const contextPath = candidates.find((candidate) => existsSync(candidate))
    expect(contextPath).toBeTruthy()
    const source = readFileSync(contextPath!, "utf8")

    expect(source).toContain("playlistPreflightSeed")
    expect(source).toContain("SET_PLAYLIST_PREFLIGHT_SEED")
  })

  let onClose: () => void

  beforeEach(() => {
    onClose = vi.fn()
    ctxRef = null
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
  })

  // -------------------------------------------------------------------------
  // Step 1: Add Content
  // -------------------------------------------------------------------------
  it("Step 1 — explains what first-time users can add and where it appears", () => {
    render(<WizardTestHarness onClose={onClose} />)

    expect(screen.getByText(/Add URLs or files/i)).toBeInTheDocument()
    expect(screen.getByText(/Media/i)).toBeInTheDocument()
    expect(screen.getByText(/Knowledge/i)).toBeInTheDocument()
  })

  it("Step 1 — focuses file upload when first-source file choice opens quick ingest", async () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{ firstSourceAddMode: "file_upload" }}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("file-drop-zone")).toHaveFocus()
    })
  })

  it("Step 1 — focuses pasted text input when first-source paste choice opens quick ingest", async () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{ firstSourceAddMode: "paste_text" }}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByRole("textbox", { name: /pasted text input/i })
      ).toHaveFocus()
    })
  })

  it("Step 1 — queues pasted first-source text as a text file", async () => {
    const user = userEvent.setup()
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{ firstSourceAddMode: "paste_text" }}
      />
    )

    const pastedTextInput = await screen.findByRole("textbox", {
      name: /pasted text input/i
    })
    const pastedText = "  These are first-source notes.\nKeep spacing.  "
    await user.type(pastedTextInput, pastedText)
    await user.click(
      screen.getByRole("button", { name: /add pasted text to queue/i })
    )

    expect(await screen.findByText("pasted-text.txt")).toBeInTheDocument()
    await expect(ctxRef?.state.queueItems[0]?.file?.text()).resolves.toBe(
      pastedText
    )
    expect(pastedTextInput).toHaveValue("")
  })

  it("Step 1 — renders at step 1 and allows adding a URL", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // The modal should be rendered
    expect(screen.getByRole("dialog")).toBeTruthy()

    // Step 1 content: URL textarea and Add button should be present
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    expect(textarea).toBeTruthy()

    // The "Add" button should be disabled when no text is entered
    const addButton = screen.getByRole("button", { name: /Add URLs to queue/i })
    expect(addButton).toBeDisabled()

    // Type a URL into the textarea
    await user.type(textarea, "https://example.com/test-article")

    // Now the Add button should be enabled
    expect(addButton).not.toBeDisabled()

    // Click Add
    await user.click(addButton)

    // The URL should appear as a queued item
    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/test-article")
      ).toBeTruthy()
    })

    // The textarea should be cleared after adding
    expect(textarea).toHaveValue("")

    // "Configure 1 items >" button should appear
    const configureButton = screen.getByText(/Configure 1 items/i)
    expect(configureButton).toBeTruthy()
    expect(configureButton).not.toBeDisabled()
  })

  it("keeps a 500-item playlist queue bounded and filters visibility without removing rows", async () => {
    const queueItems = Array.from({ length: 500 }, (_, index) => ({
      id: `occ-queue-scale-${index + 1}`,
      sourceRef: {
        kind: "materialized_playlist_item" as const,
        materializationId: "queue-scale-materialization",
        occurrenceId: `occ-queue-scale-${index + 1}`,
      },
      detectedType: "video" as const,
      icon: "Film",
      fileSize: 0,
      validation: { valid: true },
      playlist: {
        title: `Queued video ${index + 1}`,
        playlistTitle: index < 250 ? "Queue playlist A" : "Queue playlist B",
        ordinal: index + 1,
        duplicateStatus: index % 2 === 0 ? ("new" as const) : ("duplicate_existing" as const),
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      },
      playlistReview: { selected: true },
    }))
    render(<WizardTestHarness onClose={onClose} initialState={{ queueItems }} />)

    const queueList = screen.getByRole("list", { name: "Queued ingest items" })
    expect(within(queueList).getAllByRole("listitem").length).toBeLessThan(30)

    const twelfth = queueList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-queue-scale-12"]'
    )
    twelfth?.focus()
    fireEvent.keyDown(twelfth as HTMLElement, { key: "ArrowDown" })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute(
        "data-occurrence-id",
        "occ-queue-scale-13"
      )
    )

    const thirteenth = queueList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-queue-scale-13"]'
    )
    const removeButton = within(thirteenth as HTMLElement).getByRole("button", {
      name: "Remove this item from queue",
    })
    removeButton.focus()
    fireEvent.keyDown(removeButton, { key: "ArrowDown" })
    expect(removeButton).toHaveFocus()
    fireEvent.keyDown(removeButton, { key: "Escape" })
    expect(thirteenth).toHaveFocus()

    fireEvent.change(screen.getByRole("combobox", { name: "Filter queued items by playlist" }), {
      target: { value: "Queue playlist B" },
    })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute(
        "data-occurrence-id",
        "occ-queue-scale-263"
      )
    )

    act(() => {
      ctxRef?.updateQueueItems((current) =>
        current.filter((item) => item.id !== "occ-queue-scale-263")
      )
    })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute(
        "data-occurrence-id",
        "occ-queue-scale-264"
      )
    )

    await userEvent.selectOptions(
      screen.getByRole("combobox", {
        name: "Filter queued items by duplicate state",
      }),
      "duplicates"
    )

    expect(screen.getByText("Showing 125 of 499 queued items")).toBeInTheDocument()
    expect(ctxRef?.state.queueItems).toHaveLength(499)
    for (const row of within(queueList).getAllByRole("listitem")) {
      const occurrenceNumber = Number(row.getAttribute("data-occurrence-id")?.split("-").at(-1))
      expect(occurrenceNumber % 2).toBe(0)
    }
  })

  it("resets stale queue filters when the last playlist row is removed", async () => {
    const ordinaryUrl = "https://example.com/ordinary-after-filter-reset"
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          queueItems: [
            {
              id: "filter-reset-playlist",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "filter-reset-materialization",
                occurrenceId: "filter-reset-playlist",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Playlist filter row",
                playlistTitle: "Temporary playlist",
                ordinal: 1,
                duplicateStatus: "new",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: { selected: true },
            },
            {
              id: "filter-reset-ordinary",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "filter-reset-ordinary",
                url: ordinaryUrl,
              },
              url: ordinaryUrl,
              detectedType: "web",
              icon: "Globe",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />
    )

    fireEvent.change(screen.getByRole("combobox", { name: "Filter queued items by playlist" }), {
      target: { value: "Temporary playlist" },
    })
    const queueList = screen.getByRole("list", { name: "Queued ingest items" })
    expect(within(queueList).queryByText(ordinaryUrl)).not.toBeInTheDocument()

    act(() => {
      ctxRef?.updateQueueItems((current) =>
        current.filter((item) => item.id !== "filter-reset-playlist")
      )
    })

    await waitFor(() => expect(within(queueList).getByText(ordinaryUrl)).toBeInTheDocument())
    expect(
      screen.queryByRole("combobox", { name: "Filter queued items by playlist" })
    ).not.toBeInTheDocument()
  })

  it("keeps conference metadata opt-in reachable for a materialized playlist", () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          queueItems: [
            {
              id: "occ-conference-opt-in",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "conference-opt-in-materialization",
                occurrenceId: "occ-conference-opt-in",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Opening keynote",
                playlistTitle: "Conference playlist",
                ordinal: 1,
                duplicateStatus: "new",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: { selected: true },
            },
          ],
        }}
      />
    )

    expect(screen.getByRole("region", { name: "Conference batch metadata" })).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Collection name" })).toHaveValue(
      "Conference playlist"
    )
  })

  it("keeps virtual conference rows keyboard reachable and re-homes focus after removal", async () => {
    const queueItems = Array.from({ length: 30 }, (_, index) => {
      const id = `conference-row-${index + 1}`
      const url = `https://example.com/talk-${index + 1}`
      return {
        id,
        sourceRef: { kind: "direct_url" as const, occurrenceId: id, url },
        url,
        detectedType: "video" as const,
        icon: "Film",
        fileSize: 0,
        validation: { valid: true },
      }
    })
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          queueItems,
          conferenceBatchMetadata: {
            collectionName: "Conference talks",
            conferenceName: "ExampleConf",
            eventDate: "",
            eventYear: "2026",
            sharedTags: [],
            sourcePlaylistUrl: "",
          },
        }}
      />
    )

    const list = screen.getByRole("list", { name: "Conference item metadata" })
    const twelfth = list.querySelector<HTMLElement>('[data-occurrence-id="conference-row-12"]')
    twelfth?.focus()
    fireEvent.keyDown(twelfth as HTMLElement, { key: "ArrowDown" })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "conference-row-13")
    )

    const titleInput = screen.getByRole("textbox", { name: "Title override for item 13" })
    titleInput.focus()
    fireEvent.keyDown(titleInput, { key: "ArrowDown" })
    expect(titleInput).toHaveFocus()
    fireEvent.keyDown(titleInput, { key: "Escape" })
    expect(document.activeElement).toHaveAttribute("data-occurrence-id", "conference-row-13")

    act(() => {
      ctxRef?.updateQueueItems((current) =>
        current.filter((item) => item.id !== "conference-row-13")
      )
    })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "conference-row-14")
    )
  })


  it("Step 1 — blocks quick processing while disconnected and shows recovery", async () => {
    const user = userEvent.setup()
    const retryConnection = vi.fn()
    const onQuickProcess = vi.fn()

    render(
      <WizardTestHarness
        onClose={onClose}
        isOnlineForIngest={false}
        connectionRecoveryMessage="Cannot reach your tldw server. Retry connection from this dialog or open Health & diagnostics."
        onRetryConnection={retryConnection}
        onQuickProcess={onQuickProcess}
      />
    )

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/offline-article")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    expect(screen.getByText(/server offline/i)).toBeInTheDocument()
    expect(screen.getByText(/cannot reach your tldw server/i)).toBeInTheDocument()

    const processButton = screen.getByRole("button", {
      name: /use defaults & process/i,
    })
    expect(processButton).toBeDisabled()

    const retryButton = screen.getByRole("button", { name: /retry connection/i })
    await user.click(retryButton)
    expect(retryConnection).toHaveBeenCalledTimes(1)

    const configureButton = screen.getByRole("button", {
      name: /configure 1 items/i,
    })
    expect(configureButton).not.toBeDisabled()
    await user.click(processButton)
    expect(onQuickProcess).not.toHaveBeenCalled()
  })

  it("Step 1 — renders large-file guidance with the design-system alert", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    await user.click(
      screen.getByRole("button", { name: /Add large audio file/i })
    )

    const warning = await screen.findByText(/Large file/i)
    const alert = warning.closest('[data-ds-component="Alert"]')
    expect(alert).toBeInTheDocument()
    expect(alert).toHaveAttribute("role", "status")
    expect(alert).toHaveAttribute("aria-live", "polite")
  })

  it("Step 1 — renders FFmpeg media warnings with design-system state primitives", async () => {
    capabilityMocks.useServerCapabilities.mockReturnValue({
      capabilities: { ffmpegAvailable: false },
      loading: false,
    })

    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://youtube.com/watch?v=test123")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    const warning = await screen.findByText(/FFmpeg is not installed/i)
    expect(warning.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    const badge = screen
      .getByText("Video")
      .closest('[data-ds-component="Badge"]')
    expect(badge).toBeInTheDocument()
    expect(badge).toHaveAttribute("data-ds-variant", "warning")
    expect(badge?.querySelector('[data-icon="AlertTriangle"]')).toHaveClass(
      "mr-0.5"
    )
  })

  it("Step 1 — warns when pasted URLs are duplicates after normalization", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(
      textarea,
      "https://EXAMPLE.com/article/?utm_source=newsletter#comments\nhttps://example.com/article"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    expect(
      screen.getByText("https://EXAMPLE.com/article/?utm_source=newsletter#comments")
    ).toBeTruthy()
    expect(screen.getByText("https://example.com/article")).toBeTruthy()
    expect(screen.getAllByText(/Already queued/i).length).toBeGreaterThanOrEqual(1)
  })

  it("Step 1 — renders detected media labels with the design-system badge", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/test-article")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    const badge = await screen.findByText("Web page")
    const badgeRoot = badge.closest('[data-ds-component="Badge"]')
    expect(badgeRoot).toBeInTheDocument()
    expect(badgeRoot).toHaveAttribute("data-ds-variant", "info")
  })

  it("Step 1 — summarizes mixed valid and invalid URL paste results", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/valid\nnot-a-url")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    expect(screen.getByText(/1 valid \/ 1 invalid/i)).toBeInTheDocument()
    expect(screen.getByText(/Invalid URL format/i)).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // Step 1 -> Step 2: Advance to Configure
  // -------------------------------------------------------------------------
  it("Step 1 -> Step 2 — clicking configure advances to preset selector", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // Add a URL
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/video")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/video")).toBeTruthy()
    })

    // Click configure button to go to step 2
    const configureButton = screen.getByText(/Configure 1 items/i)
    await user.click(configureButton)

    // Step 2: Preset cards should be visible (Quick, Standard, Deep)
    await waitFor(() => {
      // PresetSelector renders buttons with aria-pressed and preset labels
      const standardButton = screen.getByRole("button", {
        name: /standard preset/i,
      })
      expect(standardButton).toBeTruthy()
    })

    expect(
      screen.getByRole("button", { name: /quick preset/i })
    ).toBeTruthy()
    expect(
      screen.getByRole("button", { name: /deep preset/i })
    ).toBeTruthy()
  })

  it("Step 2 — offers configured providers and accepts session-scoped free text", async () => {
    getProvidersStatusMock.mockResolvedValue({
      providers: [
        { name: " openai ", configured: true, requires_api_key: true },
        { name: "openai", configured: true, requires_api_key: true },
        { name: "anthropic", configured: false, requires_api_key: true },
        { name: "ollama", configured: true, requires_api_key: false },
        { name: " ", configured: true, requires_api_key: false },
      ],
      any_configured: true,
    })
    const user = userEvent.setup()
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 2,
          highestStep: 2,
          selectedPreset: "standard",
          customBasePreset: "standard",
          presetConfig: resolvePresetMap().standard,
        }}
      />
    )

    const provider = await screen.findByRole("combobox", {
      name: "Analysis provider",
    })
    await waitFor(() => expect(getProvidersStatusMock).toHaveBeenCalledTimes(1))
    expect(screen.getAllByRole("option", { name: "openai" })).toHaveLength(1)
    expect(screen.getByRole("option", { name: "ollama" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "anthropic" })).toBeNull()

    await user.type(provider, "custom-local")
    await user.keyboard("{Enter}")

    expect(ctxRef?.state.presetConfig.advancedValues?.api_name).toBe(
      "custom-local"
    )

    await user.click(screen.getByRole("button", { name: "Clear analysis provider" }))
    expect(ctxRef?.state.presetConfig.advancedValues?.api_name).toBeUndefined()
  })

  it("Step 2 — focuses and describes the localized provider warning", async () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 2,
          highestStep: 2,
          selectedPreset: "standard",
          customBasePreset: "standard",
          presetConfig: resolvePresetMap().standard,
        }}
        analysisProviderWarning="Choose an analysis provider before running ingest analysis."
        focusAnalysisProvider
      />
    )

    const provider = await screen.findByRole("combobox", {
      name: "Analysis provider",
    })
    const warning = screen.getByRole("alert")

    expect(provider).toHaveFocus()
    expect(provider).toHaveAttribute("aria-describedby")
    expect(provider.getAttribute("aria-describedby")).toContain(warning.id)
    expect(warning).toHaveAttribute("aria-live", "assertive")
    expect(warning).toHaveTextContent(
      "Choose an analysis provider before running ingest analysis."
    )
  })

  it("Step 2 — remains editable when provider discovery fails", async () => {
    const catalogError = new Error("catalog unavailable")
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    getProvidersStatusMock.mockRejectedValue(catalogError)
    const user = userEvent.setup()
    try {
      render(
        <WizardTestHarness
          onClose={onClose}
          initialState={{
            currentStep: 2,
            highestStep: 2,
            selectedPreset: "standard",
            customBasePreset: "standard",
            presetConfig: resolvePresetMap().standard,
          }}
        />
      )

      const provider = await screen.findByRole("combobox", {
        name: "Analysis provider",
      })
      await waitFor(() => {
        expect(warn).toHaveBeenCalledWith(
          "[QuickIngest] Failed to load analysis providers",
          catalogError
        )
      })
      await user.type(provider, "local-provider")

      expect(ctxRef?.state.presetConfig.advancedValues?.api_name).toBe(
        "local-provider"
      )
    } finally {
      warn.mockRestore()
    }
  })

  // -------------------------------------------------------------------------
  // Step 2 -> Step 3: Advance to Review
  // -------------------------------------------------------------------------
  it("Step 2 -> Step 3 — clicking Next advances to review summary", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // Add a URL and advance to step 2
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://youtube.com/watch?v=test123")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://youtube.com/watch?v=test123")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    // Wait for step 2
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    // Click Next to go to step 3
    const nextButton = screen.getByText("Next")
    await user.click(nextButton)

    // Step 3: Review summary should show "Ready to Process"
    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })

    // The review should list the queued item
    expect(
      screen.getByText("https://youtube.com/watch?v=test123")
    ).toBeTruthy()

    // "Start Processing" button should be present
    const startButton = screen.getByText("Start Processing")
    expect(startButton).toBeTruthy()
  })

  it("Step 3 — renders estimate copy without duplicate approximation markers", () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "large-video",
              kind: "file",
              fileName: "large-video.mp4",
              detectedType: "video",
              icon: "Film",
              fileSize: 400 * 1024 * 1024,
              mimeType: "video/mp4",
              validation: { valid: true },
            },
          ],
        }}
      />
    )

    const summary = screen.getByText(/1 items \| Standard preset/i)
    expect(summary).toHaveTextContent(/~\d+ (sec|min|hr) estimated/)
    expect(summary).not.toHaveTextContent("~~")

    const longTimeWarning = screen.getByText(/Processing may take a while/i)
    expect(longTimeWarning).toHaveTextContent(/~\d+ (sec|min|hr)/)
    expect(longTimeWarning).not.toHaveTextContent("~~")
  })

  it("Step 3 — blocks final processing while disconnected and allows going back", async () => {
    const user = userEvent.setup()
    const retryConnection = vi.fn()
    render(
      <WizardTestHarness
        onClose={onClose}
        isOnlineForIngest={false}
        connectionRecoveryMessage="Cannot reach your tldw server. Retry connection from this dialog or open Health & diagnostics."
        onRetryConnection={retryConnection}
      />
    )

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/review-offline")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await user.click(screen.getByRole("button", { name: /Configure 1 items/i }))

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    await user.click(screen.getByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })
    const offlineTitle = screen.getByText(/server offline/i)
    expect(offlineTitle).toBeInTheDocument()
    expect(
      offlineTitle.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText(/cannot reach your tldw server/i)).toBeInTheDocument()

    const startButton = screen.getByRole("button", { name: /start processing/i })
    expect(startButton).toBeDisabled()

    await user.click(screen.getByRole("button", { name: /retry connection/i }))
    expect(retryConnection).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: /back to settings/i }))
    expect(screen.getByRole("button", { name: /standard preset/i })).toBeTruthy()
  })

  // -------------------------------------------------------------------------
  // Step 3 -> Step 4: Start Processing
  // -------------------------------------------------------------------------
  it("Step 3 -> Step 4 — clicking Start Processing shows processing view", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // Navigate: Add URL -> Configure -> Review
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/doc.pdf")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/doc.pdf")).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    await user.click(screen.getByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })

    // Click Start Processing
    await user.click(screen.getByText("Start Processing"))

    // Step 4: Processing view should appear — use role=list to confirm processing step
    await waitFor(() => {
      expect(screen.getByRole("list")).toBeTruthy()
    })

    // Per-item progress row should be present (role="listitem")
    const listItems = screen.getAllByRole("listitem")
    expect(listItems.length).toBeGreaterThanOrEqual(1)

    // The item name should appear in the processing row
    expect(screen.getByText("https://example.com/doc.pdf")).toBeTruthy()

    // Summary bar should show counts
    expect(screen.getAllByText(/Completed/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Queued/).length).toBeGreaterThan(0)

    // Cancel All button should be present
    expect(screen.getByText("Cancel All")).toBeTruthy()
  })

  it("Step 3 playlist review requires an explicit duplicate policy and serializes only edited allowlisted patches", async () => {
    const user = userEvent.setup()
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-review-4",
              kind: "url",
              url: "https://cached.example.invalid/watch?v=display-only",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "owner-bound-materialization",
                occurrenceId: "occ-review-4",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Playlist row",
                playlistTitle: "Research playlist",
                ordinal: 4,
                duplicateStatus: "duplicate_existing",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: { selected: true },
            },
          ],
        }}
      />
    )

    expect(screen.getAllByText("4. Playlist row")).toHaveLength(2)
    expect(screen.getByText("Research playlist")).toBeInTheDocument()
    expect(
      screen.getByText("https://cached.example.invalid/watch?v=display-only").closest("details")
    ).toBeInTheDocument()
    const start = screen.getByRole("button", { name: /start processing/i })
    expect(start).toBeDisabled()

    const duplicatePolicy = screen.getByRole("combobox", {
      name: "Duplicate policy for occurrence occ-review-4",
    })
    expect(
      within(duplicatePolicy).getByRole("option", {
        name: "update metadata only",
      })
    ).toBeDisabled()
    await user.type(
      screen.getByRole("textbox", {
        name: "Title override for occurrence occ-review-4",
      }),
      "Edited title"
    )
    await user.type(
      screen.getByRole("textbox", {
        name: "Author override for occurrence occ-review-4",
      }),
      "Display-only author"
    )
    await user.type(
      screen.getByRole("textbox", {
        name: "Keywords to add for occurrence occ-review-4",
      }),
      "research, video"
    )
    await user.selectOptions(duplicatePolicy, "update_metadata_only")
    expect(start).toBeEnabled()

    await user.click(start)

    expect(ctxRef?.state.pendingRunRequest).toEqual({
      inputs: [
        {
          inputKind: "materialized_playlist_item",
          occurrenceId: "occ-review-4",
          materializationId: "owner-bound-materialization",
        },
      ],
      reviewOverrides: {
        "occ-review-4": {
          duplicatePolicy: "update_metadata_only",
          metadataPatch: {
            title: "Edited title",
            author: "Display-only author",
            keywordsAdd: ["research", "video"],
          },
        },
      },
    })
    expect(ctxRef?.state.pendingRunRequest).not.toEqual(
      expect.objectContaining({
        url: expect.stringContaining("cached.example.invalid"),
      })
    )
    expect(screen.queryByRole("textbox", { name: /speaker for item/i })).not.toBeInTheDocument()
  })

  it("Step 3 offers only safe initial actions for an in-run duplicate", () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-in-run-actions",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "in-run-actions-materialization",
                occurrenceId: "occ-in-run-actions",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                duplicateStatus: "duplicate_in_batch",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: { selected: true },
            },
          ],
        }}
      />
    )

    const policy = screen.getByRole("combobox", {
      name: "Duplicate policy for occurrence occ-in-run-actions",
    }) as HTMLSelectElement
    expect(Array.from(policy.options).map((option) => option.value)).toEqual([
      "",
      "skip",
      "overwrite",
    ])
  })

  it("lets a user clear one playlist metadata edit and then choose skip", async () => {
    const user = userEvent.setup()
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-clear-edit",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "materialization-clear-edit",
                occurrenceId: "occ-clear-edit",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Original title",
                duplicateStatus: "duplicate_existing",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: {
                selected: true,
                duplicatePolicy: "update_metadata_only",
                metadataPatch: {
                  title: "Remove this edit",
                  author: "Keep this edit",
                },
                editedFields: ["title", "author"],
              },
            },
          ],
        }}
      />
    )

    await user.clear(
      screen.getByRole("textbox", {
        name: "Title override for occurrence occ-clear-edit",
      })
    )
    await user.clear(
      screen.getByRole("textbox", {
        name: "Author override for occurrence occ-clear-edit",
      })
    )
    expect(
      screen.getByRole("combobox", {
        name: "Duplicate policy for occurrence occ-clear-edit",
      })
    ).toHaveValue("")
    await user.selectOptions(
      screen.getByRole("combobox", {
        name: "Duplicate policy for occurrence occ-clear-edit",
      }),
      "skip"
    )

    const start = screen.getByRole("button", { name: /start processing/i })
    expect(start).toBeEnabled()
    await user.click(start)
    expect(ctxRef?.state.pendingRunRequest).toEqual({
      inputs: [
        {
          inputKind: "materialized_playlist_item",
          occurrenceId: "occ-clear-edit",
          materializationId: "materialization-clear-edit",
        },
      ],
      reviewOverrides: {
        "occ-clear-edit": { duplicatePolicy: "skip" },
      },
    })
  })

  it("does not offer duplicate actions or metadata patches for a known-new playlist row", () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-known-new",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "materialization-known-new",
                occurrenceId: "occ-known-new",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Known new row",
                duplicateStatus: "new",
                materializationExpiresAt: "2099-01-01T00:00:00Z",
              },
              playlistReview: {
                selected: true,
                duplicateEvidence: {
                  kind: "none",
                  existingMediaId: null,
                  duplicateOfOccurrenceId: null,
                },
              },
            },
          ],
        }}
      />
    )

    expect(screen.getByText("No duplicate action needed")).toBeInTheDocument()
    expect(
      screen.queryByRole("combobox", {
        name: "Duplicate policy for occurrence occ-known-new",
      })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("textbox", {
        name: "Title override for occurrence occ-known-new",
      })
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /start processing/i })).toBeEnabled()
  })

  it("Step 3 playlist review blocks expired materializations with reinspection guidance", () => {
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            {
              id: "occ-expired-review",
              url: "https://cached.example.invalid/watch?v=expired",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: "expired-materialization",
                occurrenceId: "occ-expired-review",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                title: "Expired playlist row",
                ordinal: 1,
                materializationExpiresAt: "2020-01-01T00:00:00Z",
              },
              playlistReview: { selected: true },
            },
          ],
        }}
      />
    )

    expect(
      screen.getByText("This staged playlist expired. Inspect it again before processing.")
    ).toBeInTheDocument()
    const sourceDetails = screen
      .getByText("https://cached.example.invalid/watch?v=expired")
      .closest("details")
    expect(sourceDetails).not.toBeNull()
    expect(sourceDetails?.closest(".truncate")).toBeNull()
    expect(screen.getByRole("button", { name: /start processing/i })).toBeDisabled()
  })

  it("Step 3 blocks a live 500-item playlist plus an ordinary URL without dropping either", () => {
    const playlistItems = Array.from({ length: 500 }, (_, index) => ({
      id: `occ-run-cap-${index + 1}`,
      sourceRef: {
        kind: "materialized_playlist_item" as const,
        materializationId: "materialization-run-cap",
        occurrenceId: `occ-run-cap-${index + 1}`,
      },
      detectedType: "video" as const,
      icon: "Film",
      fileSize: 0,
      validation: { valid: true },
      playlist: {
        title: `Capacity video ${index + 1}`,
        playlistTitle: "Capacity playlist",
        ordinal: index + 1,
        duplicateStatus: "new" as const,
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      },
      playlistReview: { selected: true },
    }))
    const directUrl = "https://example.com/ordinary-after-playlist"

    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 3,
          highestStep: 3,
          queueItems: [
            ...playlistItems,
            {
              id: "ordinary-after-playlist",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "ordinary-after-playlist",
                url: directUrl,
              },
              url: directUrl,
              detectedType: "web",
              icon: "Globe",
              fileSize: 0,
              validation: { valid: true },
            },
          ],
        }}
      />
    )

    expect(screen.getByText(/501 items \| Standard preset/i)).toBeInTheDocument()
    expect(
      screen.getByText("Too many items selected. Select no more than 500 items before processing.")
    ).toBeInTheDocument()
    expect(ctxRef?.state.queueItems).toHaveLength(501)
    expect(ctxRef?.state.queueItems.at(-1)?.url).toBe(directUrl)
    expect(screen.getByRole("button", { name: /start processing/i })).toBeDisabled()
  })

  it("revalidates materialization expiry atomically when Start Processing is clicked", () => {
    let now = Date.parse("2026-07-13T00:00:00Z")
    const nowSpy = vi.spyOn(Date, "now").mockImplementation(() => now)
    try {
      render(
        <WizardTestHarness
          onClose={onClose}
          initialState={{
            currentStep: 3,
            highestStep: 3,
            queueItems: [
              {
                id: "occ-expiry-boundary",
                sourceRef: {
                  kind: "materialized_playlist_item",
                  materializationId: "boundary-materialization",
                  occurrenceId: "occ-expiry-boundary",
                },
                detectedType: "video",
                icon: "Film",
                fileSize: 0,
                validation: { valid: true },
                playlist: {
                  title: "Boundary row",
                  duplicateStatus: "new",
                  materializationExpiresAt: "2026-07-13T00:00:01Z",
                },
                playlistReview: { selected: true },
              },
            ],
          }}
        />
      )

      const start = screen.getByRole("button", { name: /start processing/i })
      expect(start).toBeEnabled()
      now = Date.parse("2026-07-13T00:00:02Z")
      fireEvent.click(start)

      expect(ctxRef?.state.currentStep).toBe(3)
      expect(ctxRef?.state.pendingRunRequest).toBeNull()
      expect(ctxRef?.state.processingBlock).toEqual({
        code: "materialization_expired",
        occurrenceIds: ["occ-expiry-boundary"],
      })
      expect(screen.getByRole("button", { name: /start processing/i })).toBeInTheDocument()
    } finally {
      nowSpy.mockRestore()
    }
  })

  it("keeps a 500-item playlist review bounded and filters visibility without changing selection", async () => {
    const queueItems = Array.from({ length: 500 }, (_, index) => ({
      id: `occ-scale-${index + 1}`,
      sourceRef: {
        kind: "materialized_playlist_item" as const,
        materializationId: "materialization-scale",
        occurrenceId: `occ-scale-${index + 1}`,
      },
      detectedType: "video" as const,
      icon: "Film",
      fileSize: 0,
      validation: { valid: true },
      playlist: {
        title: `Video ${index + 1}`,
        playlistTitle: "Scale playlist",
        ordinal: index + 1,
        duplicateStatus: index % 2 === 0 ? ("new" as const) : ("duplicate_existing" as const),
        materializationExpiresAt: "2099-01-01T00:00:00Z",
      },
      playlistReview: { selected: true },
    }))
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{ currentStep: 3, highestStep: 3, queueItems }}
      />
    )

    const reviewList = screen.getByRole("list", { name: "Items to process" })
    expect(within(reviewList).getAllByRole("listitem").length).toBeLessThan(30)
    expect(screen.getByText(/500 items \| Standard preset/i)).toBeInTheDocument()

    const firstReviewRow = reviewList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-scale-1"]'
    )
    firstReviewRow?.focus()
    fireEvent.change(screen.getByRole("combobox", { name: "Filter review items" }), {
      target: { value: "duplicates" },
    })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "occ-scale-2")
    )

    fireEvent.change(screen.getByRole("combobox", { name: "Filter review items" }), {
      target: { value: "selected" },
    })
    const twelfthReviewRow = reviewList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-scale-12"]'
    )
    twelfthReviewRow?.focus()
    fireEvent.keyDown(twelfthReviewRow as HTMLElement, { key: "ArrowDown" })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "occ-scale-13")
    )

    const overrideList = screen.getByRole("list", { name: "Playlist review override items" })
    const twelfthOverrideRow = overrideList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-scale-12"]'
    )
    twelfthOverrideRow?.focus()
    fireEvent.keyDown(twelfthOverrideRow as HTMLElement, { key: "ArrowDown" })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "occ-scale-13")
    )

    const firstOverrideRow = overrideList.querySelector<HTMLElement>(
      '[data-occurrence-id="occ-scale-3"]'
    )
    firstOverrideRow?.focus()

    fireEvent.change(screen.getByRole("combobox", { name: "Filter review items" }), {
      target: { value: "duplicates" },
    })
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-occurrence-id", "occ-scale-6")
    )
    expect(screen.getByText("Showing 250 of 500 review items")).toBeInTheDocument()
    expect(ctxRef?.state.queueItems.filter((item) => item.playlistReview?.selected)).toHaveLength(
      500
    )
    for (const row of within(reviewList).getAllByRole("listitem")) {
      const occurrenceNumber = Number(row.getAttribute("data-occurrence-id")?.split("-").at(-1))
      expect(occurrenceNumber % 2).toBe(0)
    }
    for (const row of within(overrideList).getAllByRole("listitem")) {
      const occurrenceNumber = Number(row.getAttribute("data-occurrence-id")?.split("-").at(-1))
      expect(occurrenceNumber % 2).toBe(0)
    }
  })


  it("Step 4 — shows durable collection tracking and exports failed URLs", async () => {
    const user = userEvent.setup()
    const originalClipboard = navigator.clipboard
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    })

    try {
      useQuickIngestSessionStore.getState().createDraftSession({
        lifecycle: "processing",
        tracking: {
          mode: "webui-direct",
          sessionId: "qi-test",
          batchId: "batch-1",
          collectionId: "7",
          plannedItemIds: ["11", "12"],
          jobIds: [501, 502],
          jobIdToCollectionItemId: {
            "501": "11",
            "502": "12",
          },
          durableMode: "durable_collection",
          startedAt: 1234,
        },
      })

      render(
        <WizardTestHarness
          onClose={onClose}
          initialState={{
            currentStep: 4,
            highestStep: 4,
            queueItems: [
              {
                id: "talk-1",
                kind: "url",
                url: "https://example.com/fail",
                detectedType: "video",
                icon: "Video",
                fileSize: 0,
                validation: { valid: true },
              },
              {
                id: "talk-2",
                kind: "url",
                url: "https://example.com/processing",
                detectedType: "video",
                icon: "Video",
                fileSize: 0,
                validation: { valid: true },
              },
            ],
            processingState: {
              status: "running",
              elapsed: 42,
              estimatedRemaining: 120,
              perItemProgress: [
                {
                  id: "talk-1",
                  status: "failed",
                  progressPercent: 100,
                  currentStage: "processing",
                  estimatedRemaining: 0,
                  error: "Timed out while downloading",
                },
                {
                  id: "talk-2",
                  status: "processing",
                  progressPercent: 50,
                  currentStage: "processing",
                  estimatedRemaining: 120,
                },
              ],
            },
          }}
        />
      )

      const trackingPanel = screen.getByTestId("quick-ingest-run-tracking")
      expect(trackingPanel).toHaveTextContent("Durable collection tracking")
      expect(trackingPanel).toHaveTextContent("Collection 7")
      expect(trackingPanel).toHaveTextContent("2 planned items")
      expect(trackingPanel).toHaveTextContent("2 jobs")

      await user.click(
        screen.getByRole("button", { name: "Export failed items list" })
      )

      await waitFor(() => {
        expect(writeText).toHaveBeenCalledWith(
          expect.stringContaining("https://example.com/fail")
        )
      })
      expect(writeText.mock.calls[0]?.[0]).toContain("Timed out while downloading")
    } finally {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: originalClipboard,
      })
    }
  })

  it("Step 4 — renders authoritative lifecycle evidence without fabricated stages", async () => {
    const onCheckStatus = vi.fn()
    const onReconnect = vi.fn()
    const lifecycleStates = [
      ["upload-ready", "Upload ready", "awaiting_upload"],
      ["file-missing", "Missing file", "awaiting_upload"],
      ["submit-pending", "Pending submit", "submit_pending"],
      ["queued", "Queued talk", "queued"],
      ["running", "Running talk", "running"],
      ["cancel-requested", "Cancelling talk", "cancellation_requested"],
      ["status-unavailable", "Offline talk", "status_unavailable"],
      ["terminal", "Finished talk", "terminal"],
    ] as const
    const queueItems = lifecycleStates.map(([id, title], index) => ({
      id,
      kind: id.includes("file") || id === "upload-ready" ? "file" : "url",
      ...(id.includes("file") || id === "upload-ready"
        ? {
            fileName: `${id}.mp4`,
            sourceRef: { kind: "file_stub", occurrenceId: id },
            ...(id === "upload-ready"
              ? { file: new File(["video"], `${id}.mp4`, { type: "video/mp4" }) }
              : {}),
          }
        : {
            url: `https://example.com/${id}`,
            sourceRef: {
              kind: "direct_url",
              occurrenceId: id,
              url: `https://example.com/${id}`,
            },
          }),
      detectedType: "video",
      icon: "Film",
      fileSize: 5,
      validation: { valid: true },
      playlist: { title, ordinal: index + 1 },
    })) as any
    const perItemProgress = lifecycleStates.map(([id, , lifecycleState]) => ({
      id,
      status: "processing",
      lifecycleState,
      terminalOutcome: lifecycleState === "terminal" ? "completed" : null,
      progressPercent: lifecycleState === "running" ? 37 : 0,
      currentStage:
        lifecycleState === "running" ? "Downloading source" : lifecycleState,
      estimatedRemaining: 0,
      retryable: lifecycleState === "status_unavailable",
    })) as any

    render(
      <WizardTestHarness
        onClose={onClose}
        onCheckStatus={onCheckStatus}
        onReconnect={onReconnect}
        initialState={{
          currentStep: 4,
          highestStep: 4,
          queueItems,
          processingState: {
            status: "running",
            elapsed: 12,
            estimatedRemaining: 0,
            perItemProgress,
          },
        }}
      />
    )

    expect(screen.getByText("Awaiting upload")).toBeInTheDocument()
    expect(screen.getByText("File reattach required")).toBeInTheDocument()
    expect(screen.getByText("Submit pending")).toBeInTheDocument()
    expect(screen.getByText("Queued")).toBeInTheDocument()
    expect(screen.getByText("Running")).toBeInTheDocument()
    expect(screen.getByText("Cancellation requested")).toBeInTheDocument()
    expect(screen.getByText("Status unavailable")).toBeInTheDocument()
    expect(screen.getByText("Completed")).toBeInTheDocument()
    expect(screen.getByText("5. Running talk")).toBeInTheDocument()
    expect(screen.getByText("Downloading source")).toBeInTheDocument()
    expect(screen.queryByText("Analyze")).not.toBeInTheDocument()
    expect(screen.queryByText("Store")).not.toBeInTheDocument()
    expect(virtualizerMocks.latestOptions?.count).toBe(0)

    await userEvent.click(screen.getByRole("button", { name: "Check again" }))
    expect(onCheckStatus).toHaveBeenCalledWith("status-unavailable")
    await userEvent.click(screen.getByRole("button", { name: "Reconnect" }))
    expect(onReconnect).toHaveBeenCalledTimes(1)
  })

  it("Step 4 — reselects a missing local file without changing occurrence authority", async () => {
    const sourceRef = {
      kind: "file_stub" as const,
      occurrenceId: "occ-missing-file",
    }
    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 4,
          highestStep: 4,
          queueItems: [
            {
              id: "occ-missing-file",
              kind: "file",
              fileName: "missing-recording.mp4",
              sourceRef,
              detectedType: "video",
              icon: "Film",
              fileSize: 10,
              validation: { valid: true },
              playlist: { title: "Missing recording", ordinal: 4 },
            },
          ] as any,
          processingState: {
            status: "running",
            elapsed: 5,
            estimatedRemaining: 0,
            perItemProgress: [
              {
                id: "occ-missing-file",
                status: "queued",
                lifecycleState: "awaiting_upload",
                terminalOutcome: null,
                progressPercent: 0,
                currentStage: "File reattach required",
                estimatedRemaining: 0,
                retryable: true,
              },
            ],
          },
        }}
      />
    )

    const replacement = new File(["replacement"], "replacement.mp4", {
      type: "video/mp4",
    })
    expect(
      screen.getByRole("button", {
        name: "Reselect file for 4. Missing recording",
      })
    ).toBeInTheDocument()
    fireEvent.change(
      screen.getByLabelText("Replacement file for 4. Missing recording"),
      { target: { files: [replacement] } }
    )

    await waitFor(() => {
      expect(ctxRef?.state.queueItems[0]).toMatchObject({
        id: "occ-missing-file",
        fileName: "replacement.mp4",
        sourceRef,
        validation: { valid: true },
      })
      expect(ctxRef?.state.queueItems[0]?.file).toBe(replacement)
    })
  })

  it("Step 4 — resumes the same awaiting-upload occurrence in its existing run after file reselection", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-existing-run-file",
          kind: "file",
          fileName: "missing-existing-run.mp4",
          sourceRef: {
            kind: "file_stub",
            occurrenceId: "occ-existing-run-file",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 10,
          validation: { valid: false, warnings: ["File reattach required"] },
          playlist: { title: "Existing run recording", ordinal: 6 },
        } as any,
      ],
      processingState: {
        status: "running",
        elapsed: 5,
        estimatedRemaining: 0,
        perItemProgress: [
          {
            id: "occ-existing-run-file",
            status: "queued",
            lifecycleState: "awaiting_upload",
            terminalOutcome: null,
            progressPercent: 0,
            currentStage: "File reattach required",
            estimatedRemaining: 0,
            retryable: true,
          },
        ],
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-existing-run-file",
        runId: "run-existing-file-resume",
        submissionOccurrenceIds: ["occ-existing-run-file"],
        submittedItemIds: ["occ-existing-run-file"],
        startedAt: Date.now(),
      },
    })
    lifecycleMocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: null,
          status: "awaiting_upload",
          sourceItemId: "occ-existing-run-file",
          lifecycleState: "awaiting_upload",
          terminalOutcome: null,
          progressPercent: 0,
          progressMessage: "File reattach required",
          retryable: true,
          attempt: 4,
        },
      ],
      errorMessage: null,
    })

    render(<QuickIngestWizardModal open onClose={onClose} />)
    const replacement = new File(["replacement"], "replacement-existing.mp4", {
      type: "video/mp4",
    })
    Object.defineProperty(replacement, "arrayBuffer", {
      configurable: true,
      value: undefined,
    })
    fireEvent.change(
      await screen.findByLabelText(
        "Replacement file for 6. Existing run recording"
      ),
      { target: { files: [replacement] } }
    )

    await waitFor(() => {
      expect(lifecycleMocks.submitQuickIngestBatch).toHaveBeenCalledWith(
        expect.objectContaining({
          __quickIngestSessionId: "qi-existing-run-file",
          __quickIngestRunId: "run-existing-file-resume",
          pendingRunRequest: {
            inputs: [
              expect.objectContaining({
                inputKind: "file_stub",
                occurrenceId: "occ-existing-run-file",
                attempt: 4,
              }),
            ],
          },
          files: [
            expect.objectContaining({
              id: "occ-existing-run-file",
              name: "replacement-existing.mp4",
            }),
          ],
        })
      )
    })
    expect(
      useQuickIngestSessionStore.getState().session?.queueItems[0]
    ).toMatchObject({
      id: "occ-existing-run-file",
      fileName: "replacement-existing.mp4",
    })
  })

  it("Step 4 — preserves an extension retry attempt through reattach and a failed replacement upload retry", async () => {
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-existing-run-file-retry",
          kind: "file",
          fileName: "missing-existing-run-retry.mp4",
          sourceRef: {
            kind: "file_stub",
            occurrenceId: "occ-existing-run-file-retry",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 10,
          validation: { valid: false, warnings: ["File reattach required"] },
          playlist: { title: "Existing retry recording", ordinal: 7 },
        } as any,
      ],
      processingState: {
        status: "running",
        elapsed: 5,
        estimatedRemaining: 0,
        perItemProgress: [
          {
            id: "occ-existing-run-file-retry",
            status: "queued",
            lifecycleState: "awaiting_upload",
            terminalOutcome: null,
            progressPercent: 0,
            currentStage: "File reattach required",
            estimatedRemaining: 0,
            retryable: true,
          },
        ],
      },
      tracking: {
        mode: "extension-runtime",
        sessionId: "qi-existing-run-file-retry",
        runId: "run-existing-file-retry",
        generation: "generation-existing-file-retry",
        submissionOccurrenceIds: ["occ-existing-run-file-retry"],
        submittedItemIds: ["occ-existing-run-file-retry"],
        startedAt: Date.now(),
      },
    })
    lifecycleMocks.submitQuickIngestBatch
      .mockResolvedValueOnce({
        ok: false,
        accepted: false,
        error: "Upload temporarily unavailable",
      })
      .mockResolvedValueOnce({
        ok: true,
        accepted: true,
        runId: "run-existing-file-retry",
      })

    render(<QuickIngestWizardModal open onClose={onClose} />)
    await screen.findByLabelText(
      "Replacement file for 7. Existing retry recording"
    )
    act(() => {
      emitRuntimeMessage({
        type: "tldw:quick-ingest/progress",
        payload: {
          sessionId: "qi-existing-run-file-retry",
          runId: "run-existing-file-retry",
          generation: "generation-existing-file-retry",
          occurrenceId: "occ-existing-run-file-retry",
          jobId: null,
          attempt: 4,
          status: "awaiting_upload",
          lifecycleState: "awaiting_upload",
          progressPercentage: 0,
          progressMessage: "File reattach required",
          retryable: true,
          result: {
            id: "occ-existing-run-file-retry",
            status: "awaiting_upload",
            type: "video",
          },
        },
      })
    })
    await waitFor(() => {
      expect(
        useQuickIngestSessionStore.getState().session?.processingState
          .perItemProgress[0]
      ).toMatchObject({ attempt: 4, lifecycleState: "awaiting_upload" })
    })

    const invalidReplacement = new File(
      ["invalid"],
      "invalid-replacement.bin"
    )
    fireEvent.change(
      screen.getByLabelText("Replacement file for 7. Existing retry recording"),
      { target: { files: [invalidReplacement] } }
    )
    await waitFor(() => {
      expect(
        useQuickIngestSessionStore.getState().session?.processingState
          .perItemProgress[0]
      ).toMatchObject({
        attempt: 4,
        lifecycleState: "awaiting_upload",
        currentStage: "Unsupported replacement file",
      })
    })

    const replacement = new File(["replacement"], "replacement-retry.mp4", {
      type: "video/mp4",
    })
    let releaseFirstRead!: (bytes: ArrayBuffer) => void
    const firstReadGate = new Promise<ArrayBuffer>((resolve) => {
      releaseFirstRead = resolve
    })
    const readBytes = vi
      .fn()
      .mockImplementationOnce(() => firstReadGate)
      .mockResolvedValue(Uint8Array.from([9, 8, 7]).buffer)
    Object.defineProperty(replacement, "arrayBuffer", {
      configurable: true,
      value: readBytes,
    })
    fireEvent.change(
      screen.getByLabelText("Replacement file for 7. Existing retry recording"),
      { target: { files: [replacement] } }
    )

    await waitFor(() => {
      expect(
        useQuickIngestSessionStore.getState().session?.processingState
          .perItemProgress[0]
      ).toMatchObject({
        attempt: 4,
        lifecycleState: "awaiting_upload",
        currentStage: "File selected. Ready to upload.",
      })
    })
    act(() => {
      releaseFirstRead(Uint8Array.from([9, 8, 7]).buffer)
    })

    await waitFor(() => {
      expect(lifecycleMocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
      expect(
        useQuickIngestSessionStore.getState().session?.processingState
          .perItemProgress[0]
      ).toMatchObject({
        attempt: 4,
        lifecycleState: "awaiting_upload",
        currentStage: "Upload temporarily unavailable",
        retryable: true,
      })
    })

    await userEvent.click(
      screen.getByRole("button", {
        name: "Retry upload for 7. Existing retry recording",
      })
    )

    await waitFor(() =>
      expect(lifecycleMocks.submitQuickIngestBatch).toHaveBeenCalledTimes(2)
    )
    expect(readBytes).toHaveBeenCalledTimes(2)
    for (const [payload] of lifecycleMocks.submitQuickIngestBatch.mock.calls) {
      expect(payload).toMatchObject({
        __quickIngestSessionId: "qi-existing-run-file-retry",
        __quickIngestRunId: "run-existing-file-retry",
        pendingRunRequest: {
          inputs: [
            expect.objectContaining({
              occurrenceId: "occ-existing-run-file-retry",
              attempt: 4,
            }),
          ],
        },
        files: [
          {
            id: "occ-existing-run-file-retry",
            name: "replacement-retry.mp4",
            data: [9, 8, 7],
          },
        ],
      })
    }
    await waitFor(() => {
      expect(
        useQuickIngestSessionStore.getState().session?.processingState
          .perItemProgress[0]
      ).toMatchObject({
        attempt: 4,
        lifecycleState: "queued",
        currentStage: "Queued",
        retryable: false,
      })
    })
  })

  it.each([
    ["rate limited", 429],
    ["service unavailable", 503],
    ["network disconnected", undefined],
  ])(
    "maps %s status errors to recoverable status-unavailable evidence",
    async (message, status) => {
      const modalModule = await import(
        "@/components/Common/QuickIngestWizardModal"
      )
      const buildUnavailableProgress = (modalModule as any)
        .buildStatusUnavailableProgressFromReattachError
      expect(buildUnavailableProgress).toBeTypeOf("function")
      if (typeof buildUnavailableProgress !== "function") return
      const queueItems = [
        {
          id: "occ-status-recovery",
          kind: "url",
          url: "https://example.com/status-recovery",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
        },
      ] as any
      const perItemProgress = buildUnavailableProgress(
        Object.assign(new Error(message), status == null ? {} : { status }),
        queueItems,
        [
          {
            id: "occ-status-recovery",
            status: "processing",
            lifecycleState: "running",
            terminalOutcome: null,
            progressPercent: 31,
            currentStage: "Running",
            estimatedRemaining: 0,
          },
        ]
      )

      render(
        <WizardTestHarness
          onClose={onClose}
          initialState={{
            currentStep: 4,
            highestStep: 4,
            queueItems,
            processingState: {
              status: "running",
              elapsed: 1,
              estimatedRemaining: 0,
              perItemProgress,
            },
          }}
        />
      )

      expect(perItemProgress[0]).toMatchObject({
        lifecycleState: "status_unavailable",
        terminalOutcome: null,
        progressPercent: 31,
        retryable: true,
        currentStage: expect.stringMatching(new RegExp(message, "i")),
      })
      expect(
        screen.getByRole("button", { name: "Check again" })
      ).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Reconnect" })
      ).toBeInTheDocument()
    }
  )

  it.each([
    ["rate limited", 429, /too many|rate|try again/i],
    ["service unavailable", 503, /unavailable|try again/i],
    ["network disconnected", undefined, /network|connect|unavailable/i],
  ])(
    "propagates real service %s recovery evidence into the Modal",
    async (message, status, expectedMessage) => {
      const actualReattach = await vi.importActual<
        typeof import("@/services/tldw/quick-ingest-session-reattach")
      >("@/services/tldw/quick-ingest-session-reattach")
      lifecycleMocks.reattachQuickIngestSession.mockImplementation(
        actualReattach.reattachQuickIngestSession
      )
      transportMocks.bgRequest.mockRejectedValue(
        Object.assign(
          new Error(message),
          status == null ? {} : { status }
        )
      )
      useQuickIngestSessionStore.getState().upsertSession({
        lifecycle: "processing",
        currentStep: 4,
        queueItems: [
          {
            id: "occ-real-status-unavailable",
            kind: "url",
            url: "https://example.com/real-status-unavailable",
            detectedType: "video",
            icon: "Film",
            fileSize: 0,
            validation: { valid: true },
            playlist: { title: "Recovery target", ordinal: 2 },
          } as any,
        ],
        processingState: {
          status: "running",
          elapsed: 1,
          estimatedRemaining: 0,
          perItemProgress: [
            {
              id: "occ-real-status-unavailable",
              status: "processing",
              lifecycleState: "running",
              terminalOutcome: null,
              progressPercent: 31,
              currentStage: "Running",
              estimatedRemaining: 0,
            },
          ],
        },
        tracking: {
          mode: "webui-direct",
          sessionId: `qi-real-status-${status ?? "network"}`,
          runId: `run-real-status-${status ?? "network"}`,
          jobIds: [717],
          submissionOccurrenceIds: ["occ-real-status-unavailable"],
          submittedItemIds: ["occ-real-status-unavailable"],
          startedAt: Date.now(),
        },
      })

      render(<QuickIngestWizardModal open onClose={onClose} />)

      await waitFor(() => {
        expect(
          useQuickIngestSessionStore.getState().session?.processingState
            .perItemProgress[0]
        ).toMatchObject({
          id: "occ-real-status-unavailable",
          lifecycleState: "status_unavailable",
          retryable: true,
          currentStage: expect.stringMatching(expectedMessage),
        })
      })
      expect(
        screen.getByRole("button", { name: "Check again" })
      ).toBeInTheDocument()
    }
  )

  it("Step 4 — keeps authoritative row and run cancellation pending", async () => {
    const user = userEvent.setup()
    const onCancelItem = vi.fn().mockReturnValue(true)
    const onCancelProcessing = vi.fn().mockReturnValue(true)

    render(
      <WizardTestHarness
        onClose={onClose}
        onCancelItem={onCancelItem}
        onCancelProcessing={onCancelProcessing}
        initialState={{
          currentStep: 4,
          highestStep: 4,
          queueItems: [
            {
              id: "occ-cancel-row",
              kind: "url",
              url: "https://example.com/cancel-row",
              sourceRef: {
                kind: "direct_url",
                occurrenceId: "occ-cancel-row",
                url: "https://example.com/cancel-row",
              },
              detectedType: "video",
              icon: "Film",
              fileSize: 0,
              validation: { valid: true },
              playlist: { title: "Cancellation target", ordinal: 9 },
            },
          ],
          processingState: {
            status: "running",
            elapsed: 2,
            estimatedRemaining: 0,
            perItemProgress: [
              {
                id: "occ-cancel-row",
                status: "processing",
                lifecycleState: "running",
                terminalOutcome: null,
                progressPercent: 25,
                currentStage: "Downloading source",
                estimatedRemaining: 0,
              } as any,
            ],
          },
        }}
      />
    )

    await user.click(
      screen.getByRole("button", { name: "Cancel 9. Cancellation target" })
    )
    expect(onCancelItem).toHaveBeenCalledWith("occ-cancel-row")
    expect(screen.getByText("Cancellation requested")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Cancel All" }))
    expect(onCancelProcessing).toHaveBeenCalledTimes(1)
  })

  it("Step 4 — bounds 500 lifecycle rows and filters by actionable state", async () => {
    const queueItems = Array.from({ length: 500 }, (_, index) => {
      const occurrence = index + 1
      return {
        id: `occ-lifecycle-${occurrence}`,
        kind: "url",
        url: `https://example.com/watch/${occurrence}`,
        sourceRef: {
          kind: "direct_url",
          occurrenceId: `occ-lifecycle-${occurrence}`,
          url: `https://example.com/watch/${occurrence}`,
        },
        detectedType: "video",
        icon: "Film",
        fileSize: 0,
        validation: { valid: true },
        playlist: { title: `Lifecycle talk ${occurrence}`, ordinal: occurrence },
      }
    }) as any
    const perItemProgress = queueItems.map((item: any, index: number) => {
      const group = index % 3
      return {
        id: item.id,
        status: group === 2 ? "complete" : "processing",
        lifecycleState:
          group === 0
            ? "running"
            : group === 1
              ? "status_unavailable"
              : "terminal",
        terminalOutcome: group === 2 ? "completed" : null,
        progressPercent: group === 2 ? 100 : 20,
        currentStage: group === 0 ? "Downloading source" : "",
        estimatedRemaining: 0,
      }
    }) as any

    render(
      <WizardTestHarness
        onClose={onClose}
        initialState={{
          currentStep: 4,
          highestStep: 4,
          queueItems,
          processingState: {
            status: "running",
            elapsed: 10,
            estimatedRemaining: 0,
            perItemProgress,
          },
        }}
      />
    )

    const list = screen.getByRole("list", { name: "Processing items" })
    expect(within(list).getAllByRole("listitem").length).toBeLessThanOrEqual(12)
    expect(within(list).getAllByRole("listitem")[0]).toHaveAttribute(
      "aria-setsize",
      "500"
    )
    expect(within(list).getAllByRole("listitem")[0]).toHaveAttribute(
      "aria-posinset",
      "1"
    )
    expect(virtualizerMocks.latestOptions?.count).toBe(500)

    const initialRows = within(list).getAllByRole("listitem")
    initialRows[0].focus()
    fireEvent.keyDown(initialRows[0], { key: "ArrowDown" })
    expect(initialRows[1]).toHaveFocus()

    fireEvent.keyDown(initialRows[1], { key: "End" })
    await waitFor(() => {
      expect(screen.getByText("500. Lifecycle talk 500")).toBeInTheDocument()
      expect(within(list).getAllByRole("listitem").at(-1)).toHaveFocus()
    })

    fireEvent.keyDown(within(list).getAllByRole("listitem").at(-1)!, {
      key: "Home",
    })
    await waitFor(() => {
      expect(screen.getByText("1. Lifecycle talk 1")).toBeInTheDocument()
      expect(within(list).getAllByRole("listitem")[0]).toHaveFocus()
    })

    fireEvent.change(screen.getByRole("combobox", { name: "Filter processing items" }), {
      target: { value: "attention" },
    })
    expect(screen.getByText("Needs attention (167)")).toBeInTheDocument()
    expect(within(list).getAllByRole("listitem")[0]).toHaveFocus()
    for (const row of within(list).getAllByRole("listitem")) {
      expect(row).toHaveAttribute("data-lifecycle-group", "attention")
    }
  })

  it("dispatches occurrence and whole-run cancellation to durable run authority", async () => {
    const user = userEvent.setup()
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "processing",
      currentStep: 4,
      queueItems: [
        {
          id: "occ-real-cancel",
          kind: "url",
          url: "https://example.com/real-cancel",
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "occ-real-cancel",
            url: "https://example.com/real-cancel",
          },
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: { title: "Real cancellation", ordinal: 12 },
        } as any,
      ],
      processingState: {
        status: "running",
        elapsed: 4,
        estimatedRemaining: 0,
        perItemProgress: [
          {
            id: "occ-real-cancel",
            status: "processing",
            lifecycleState: "running",
            terminalOutcome: null,
            progressPercent: 25,
            currentStage: "Downloading source",
            estimatedRemaining: 0,
          } as any,
        ],
      },
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-real-cancel",
        runId: "run-real-cancel",
        submittedItemIds: ["occ-real-cancel"],
        startedAt: Date.now(),
      },
    })
    lifecycleMocks.reattachQuickIngestSession.mockResolvedValue({
      lifecycle: "processing",
      jobs: [
        {
          jobId: 77,
          status: "running",
          sourceItemId: "occ-real-cancel",
          lifecycleState: "running",
          terminalOutcome: null,
          progressPercent: 25,
          progressMessage: "Downloading source",
        },
      ],
      errorMessage: null,
    })

    render(<QuickIngestWizardModal open onClose={onClose} />)
    await user.click(
      await screen.findByRole("button", { name: "Cancel 12. Real cancellation" })
    )
    await waitFor(() => {
      expect(lifecycleMocks.cancelQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          tracking: expect.objectContaining({ runId: "run-real-cancel" }),
          occurrenceIds: ["occ-real-cancel"],
        })
      )
    })
    expect(screen.getByText("Cancellation requested")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Cancel All" }))
    await waitFor(() => {
      expect(lifecycleMocks.cancelQuickIngestSession).toHaveBeenCalledWith(
        expect.objectContaining({
          tracking: expect.objectContaining({ runId: "run-real-cancel" }),
          occurrenceIds: undefined,
        })
      )
    })
  })

  it("retries eligible run occurrences and reconciles the authoritative snapshot", async () => {
    vi.useFakeTimers()
    useQuickIngestSessionStore.getState().upsertSession({
      lifecycle: "partial_failure",
      currentStep: 5,
      queueItems: [
        {
          id: "occ-retry-run",
          kind: "url",
          url: "https://example.com/retry-target",
          detectedType: "video",
          icon: "Film",
          fileSize: 0,
          validation: { valid: true },
          playlist: { title: "Retry target", ordinal: 1 },
          sourceRef: {
            kind: "direct_url",
            occurrenceId: "occ-retry-run",
            url: "https://example.com/retry-target",
          },
        } as any,
      ],
      results: [
        {
          id: "occ-retry-run",
          status: "error",
          outcome: "failed",
          terminalOutcome: "processing_failed",
          type: "video",
          title: "Retry target",
          error: "Network timed out",
        } as any,
      ],
      tracking: {
        mode: "webui-direct",
        sessionId: "qi-retry-run",
        runId: "run-retry-run",
        submittedItemIds: ["occ-retry-run"],
        startedAt: Date.now(),
      },
    })
    lifecycleMocks.reattachQuickIngestSession
      .mockResolvedValueOnce({
        lifecycle: "processing",
        jobs: [
          {
            jobId: null,
            status: "queued",
            sourceItemId: "occ-retry-run",
            lifecycleState: "queued",
            terminalOutcome: null,
            progressPercent: 0,
            progressMessage: "Queued after retry reconciliation",
          },
        ],
        errorMessage: null,
      })
      .mockResolvedValue({
        lifecycle: "processing",
        jobs: [
          {
            jobId: 901,
            status: "failed",
            sourceItemId: "occ-retry-run",
            lifecycleState: "terminal",
            terminalOutcome: "processing_failed",
            progressPercent: 100,
            error: "Worker failed after retry",
          },
        ],
        errorMessage: null,
      })

    render(<QuickIngestWizardModal open onClose={onClose} />)
    fireEvent.click(
      screen.getByRole("button", { name: "Retry Retry target" })
    )
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(lifecycleMocks.retryQuickIngestSession).toHaveBeenCalledWith({
      sessionId: "qi-retry-run",
      tracking: expect.objectContaining({ runId: "run-retry-run" }),
      occurrenceIds: ["occ-retry-run"],
    })
    expect(lifecycleMocks.reattachQuickIngestSession).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-retry-run" })
    )
    expect(screen.getByText("Queued after retry reconciliation")).toBeInTheDocument()
    expect(lifecycleMocks.reattachQuickIngestSession).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })

    expect(lifecycleMocks.reattachQuickIngestSession).toHaveBeenCalledTimes(2)
    expect(screen.getAllByText("Worker failed after retry")).not.toHaveLength(0)
  })

  // -------------------------------------------------------------------------
  // Step 5: Results (via context manipulation)
  // -------------------------------------------------------------------------
  it("Step 5 — displays results after processing completes", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // Navigate: Add URL -> Configure -> Review -> Processing
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/article")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/article")).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    await user.click(screen.getByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })

    await user.click(screen.getByText("Start Processing"))

    await waitFor(() => {
      expect(screen.getByRole("list")).toBeTruthy()
    })

    // Directly manipulate context to simulate processing completion
    // and advance to results step
    expect(ctxRef).not.toBeNull()

    // Set results
    ctxRef!.setResults([
      {
        id: "test-uuid-0",
        status: "ok",
        outcome: "ingested",
        url: "https://example.com/article",
        type: "web",
        title: "Test Article",
        durationMs: 3500,
      },
    ])

    // Update processing state to complete
    ctxRef!.updateProcessingState({
      status: "complete",
      perItemProgress: [
        {
          id: "test-uuid-0",
          status: "complete",
          progressPercent: 100,
          currentStage: "done",
          estimatedRemaining: 0,
        },
      ],
      elapsed: 3.5,
    })

    // Advance from step 4 to step 5 (goToStep only allows backward nav,
    // so we use goNext which increments currentStep and highestStep)
    ctxRef!.goNext()

    // Step 5: Results should render
    await waitFor(() => {
      expect(screen.getByTestId("wizard-results-step")).toBeTruthy()
    })

    // The completed item should be listed
    expect(screen.getByText("Test Article")).toBeTruthy()

    // Summary line should show success count
    expect(screen.getByText(/1 succeeded/)).toBeTruthy()

    // "Done" button should be present
    const doneButton = screen.getByText("Done")
    expect(doneButton).toBeTruthy()

    // Clicking Done should call onClose
    await user.click(doneButton)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it("Step 5 — renders skipped duplicates separately and includes them in the summary", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/article")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/article")).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    await user.click(screen.getByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })

    await user.click(screen.getByText("Start Processing"))

    await waitFor(() => {
      expect(screen.getByRole("list")).toBeTruthy()
    })

    expect(ctxRef).not.toBeNull()

    ctxRef!.setResults([
      {
        id: "test-success-1",
        status: "ok",
        outcome: "ingested",
        url: "https://example.com/article",
        type: "web",
        title: "Fresh Article",
      },
      {
        id: "test-skipped-1",
        status: "ok",
        outcome: "skipped",
        url: "https://example.com/duplicate",
        type: "web",
        title: "Existing Article",
        message: "This item already exists in your library. Use the ‘Deep’ preset to overwrite.",
      },
      {
        id: "test-error-1",
        status: "error",
        outcome: "failed",
        url: "https://example.com/error",
        type: "web",
        title: "Broken Article",
        error: "Upload failed",
      },
    ])

    ctxRef!.updateProcessingState({
      status: "complete",
      perItemProgress: [
        {
          id: "test-success-1",
          status: "complete",
          progressPercent: 100,
          currentStage: "done",
          estimatedRemaining: 0,
        },
        {
          id: "test-skipped-1",
          status: "complete",
          progressPercent: 100,
          currentStage: "done",
          estimatedRemaining: 0,
        },
        {
          id: "test-error-1",
          status: "failed",
          progressPercent: 100,
          currentStage: "failed",
          estimatedRemaining: 0,
          error: "Upload failed",
        },
      ],
      elapsed: 2.5,
    })

    ctxRef!.goNext()

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results-step")).toBeTruthy()
    })

    expect(screen.getByText("Skipped existing (1)")).toBeTruthy()
    expect(screen.getByText("Existing Article")).toBeTruthy()
    expect(
      screen.getByText(/1 succeeded.*1 skipped.*1 failed/i)
    ).toBeTruthy()
  })

  // -------------------------------------------------------------------------
  // Full flow with multiple items
  // -------------------------------------------------------------------------
  it("supports adding multiple URLs in a single batch", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)

    // Type multiple URLs (separated by newlines via manual value)
    await user.type(
      textarea,
      "https://example.com/page1\nhttps://example.com/page2"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    // Both items should appear
    await waitFor(() => {
      expect(screen.getAllByText("https://example.com/page1").length).toBeGreaterThan(0)
      expect(screen.getAllByText("https://example.com/page2").length).toBeGreaterThan(0)
    })

    // The configure button should reference 2 items
    expect(screen.getByText(/Configure 2 items/i)).toBeTruthy()
  })

  it("captures shared conference metadata for a 34-talk batch and shows it in review", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    const urls = Array.from(
      { length: 34 },
      (_, index) => `https://youtube.com/watch?v=conference-talk-${index + 1}`
    ).join("\n")

    fireEvent.change(textarea, { target: { value: urls } })
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText(/Configure 34 items/i)).toBeTruthy()
    })

    await user.type(screen.getByLabelText("Collection name"), "Strange Loop 2012")
    await user.type(screen.getByLabelText("Conference name"), "Strange Loop")
    await user.type(screen.getByLabelText("Event year"), "2012")
    await user.type(screen.getByLabelText("Shared tags"), "conference, clojure")

    await user.click(screen.getByText(/Configure 34 items/i))
    await user.click(await screen.findByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })
    expect(screen.getByText(/34 selected/i)).toBeTruthy()
    expect(screen.getByText(/Strange Loop 2012/i)).toBeTruthy()
    expect(screen.getByText("Strange Loop")).toBeTruthy()
    expect(screen.getAllByText("2012").length).toBeGreaterThan(0)
    expect(screen.getByText(/conference, clojure/i)).toBeTruthy()
  })

  it("allows overriding one talk title and speaker inside a conference batch", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    fireEvent.change(textarea, {
      target: {
        value:
          "https://youtube.com/watch?v=talk-1\nhttps://youtube.com/watch?v=talk-2",
      },
    })
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText(/Configure 2 items/i)).toBeTruthy()
    })

    await user.type(screen.getByLabelText("Collection name"), "Strange Loop 2012")
    await user.type(screen.getByLabelText("Conference name"), "Strange Loop")
    await user.type(screen.getByLabelText("Title override for item 1"), "Simplicity Matters")
    await user.type(screen.getByLabelText("Speaker for item 1"), "Rich Hickey")

    await user.click(screen.getByText(/Configure 2 items/i))
    await user.click(await screen.findByText("Next"))

    await waitFor(() => {
      expect(screen.getByText("Ready to Process")).toBeTruthy()
    })
    expect(screen.getByText("Simplicity Matters")).toBeTruthy()
    expect(screen.getByText(/Rich Hickey/i)).toBeTruthy()
  })

  // -------------------------------------------------------------------------
  // Results step: Ingest More resets the wizard
  // -------------------------------------------------------------------------
  it("Step 5 — Ingest More resets the wizard back to step 1", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={onClose} />)

    // Fast-track: add an item, then use context to jump to results
    const textarea = screen.getByPlaceholderText(/https:\/\/example\.com/i)
    await user.type(textarea, "https://example.com/reset-test")
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/reset-test")).toBeTruthy()
    })

    // Navigate through steps to reach step 5 via context.
    // We need to advance highestStep by using goNext sequentially.
    expect(ctxRef).not.toBeNull()

    // Step 1 -> 2
    ctxRef!.goNext()
    // Step 2 -> 3
    ctxRef!.goNext()
    // Step 3 -> 4
    ctxRef!.goNext()
    // Step 4 -> 5
    ctxRef!.goNext()

    // Set results so the results step has content to display
    ctxRef!.setResults([
      {
        id: "test-uuid-0",
        status: "ok",
        outcome: "ingested",
        url: "https://example.com/reset-test",
        type: "web",
        title: "Reset Test",
      },
    ])

    await waitFor(() => {
      expect(screen.getByTestId("wizard-results-step")).toBeTruthy()
    })

    // Click "Ingest More"
    await user.click(screen.getByText("Ingest More"))

    // Wizard should reset back to step 1 (Add Content)
    await waitFor(() => {
      expect(
        screen.getByPlaceholderText(/https:\/\/example\.com/i)
      ).toBeTruthy()
    })

    // The file drop zone should be visible again
    expect(screen.getByTestId("file-drop-zone")).toBeTruthy()
  })
})

describe("QuickIngestWizardModal — real configure step", () => {
  beforeEach(() => {
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
    useQuickIngestSessionStore.getState().createDraftSession()
  })

  it("shows the full inline options surface without forcing the old full-modal placeholder", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /standard preset/i })
      ).toBeTruthy()
    })

    expect(
      screen.queryByText(/advanced options are available in the full ingest modal/i)
    ).not.toBeInTheDocument()
    const analysisToggle = screen.getByRole("checkbox", {
      name: /ingestion options – analysis/i,
    })
    expect(analysisToggle).toBeInTheDocument()
    expect(screen.getByText("Next")).toBeInTheDocument()

    // Advanced controls are hidden by default
    expect(screen.queryByTitle("Captions toggle")).not.toBeInTheDocument()
    expect(screen.queryByText("Review before saving")).not.toBeInTheDocument()

    // Expand advanced options to reveal them
    await expandAdvancedOptions(user)
    expect(screen.getByText("Review before saving")).toBeInTheDocument()
    expect(screen.getByTitle("Captions toggle")).toBeInTheDocument()

    await user.click(analysisToggle)

    expect(screen.getByText(/using custom settings/i)).toBeInTheDocument()
  })

  it("shows Auto chunking controls by default when chunking is enabled", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/article"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await user.click(screen.getByText(/Configure 1 items/i))

    await waitFor(() => {
      expect(
        screen.getByRole("group", { name: /chunking mode/i })
      ).toBeInTheDocument()
    })

    expect(screen.getByRole("button", { name: "Auto" })).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByLabelText("Auto chunking goal")).toHaveValue("balanced")
    expect(
      screen.getByLabelText("Use AI to improve chunk boundaries")
    ).not.toBeChecked()
  })

  it("reveals Manual chunking controls and hides them when switching back to Auto", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/article"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await user.click(screen.getByText(/Configure 1 items/i))

    await user.click(await screen.findByRole("button", { name: "Manual" }))

    expect(screen.getByLabelText("Chunk method")).toBeInTheDocument()
    expect(screen.getByLabelText("Chunk size")).toBeInTheDocument()
    expect(screen.getByLabelText("Chunk overlap")).toBeInTheDocument()

    await user.clear(screen.getByLabelText("Chunk size"))
    await user.type(screen.getByLabelText("Chunk size"), "900")

    expect(ctxRef).not.toBeNull()
    await waitFor(() => {
      expect(ctxRef!.state.presetConfig.common.chunking_mode).toBe("manual")
      expect(ctxRef!.state.presetConfig.advancedValues?.chunk_size).toBe(900)
    })

    await user.click(screen.getByRole("button", { name: "Auto" }))

    await waitFor(() => {
      expect(ctxRef!.state.presetConfig.common.chunking_mode).toBe("auto")
    })
    expect(screen.queryByLabelText("Chunk method")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Chunk size")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Chunk overlap")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Auto chunking goal")).toHaveValue("balanced")
    expect(ctxRef!.state.presetConfig.advancedValues?.chunk_size).toBeUndefined()
  })

  it("clamps Manual chunking numbers before storing advanced values", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/article"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await user.click(screen.getByText(/Configure 1 items/i))
    await user.click(await screen.findByRole("button", { name: "Manual" }))

    await user.clear(screen.getByLabelText("Chunk size"))
    await user.type(screen.getByLabelText("Chunk size"), "0")
    await user.clear(screen.getByLabelText("Chunk overlap"))
    await user.type(screen.getByLabelText("Chunk overlap"), "-5")

    await waitFor(() => {
      expect(ctxRef!.state.presetConfig.advancedValues?.chunk_size).toBe(1)
      expect(ctxRef!.state.presetConfig.advancedValues?.chunk_overlap).toBe(0)
    })
  })

  it("keeps review mode anchored to remote storage and leaves audio defaults available for video-only batches", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    const audioLanguageInput = await screen.findByTitle("Audio language")
    const diarizationToggle = screen.getByLabelText("Audio diarization toggle")
    const transcriptionModelSelect = screen.getByLabelText("Transcription model")

    expect(audioLanguageInput).not.toBeDisabled()
    expect(diarizationToggle).not.toBeDisabled()
    expect(transcriptionModelSelect).not.toBeDisabled()

    await user.click(
      screen.getByLabelText(/store ingest results on your tldw server/i)
    )

    const reviewToggle = screen.getByLabelText(/review before saving/i)
    await user.click(reviewToggle)

    expect(
      screen.getByLabelText(/store ingest results on your tldw server/i)
    ).toBeChecked()
  })

  it("disables audio transcription controls for document-only batches", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/research-paper.pdf"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(screen.getByText("https://example.com/research-paper.pdf")).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    const audioLanguageInput = await screen.findByTitle("Audio language")
    const diarizationToggle = screen.getByLabelText("Audio diarization toggle")
    const transcriptionModelSelect = screen.getByLabelText("Transcription model")

    expect(audioLanguageInput).toBeDisabled()
    expect(diarizationToggle).toBeDisabled()
    expect(transcriptionModelSelect).toBeDisabled()
  })

  it("stores a standard audio language option when selected", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await user.selectOptions(audioLanguageSelect, "en-US")

    expect(ctxRef).not.toBeNull()
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.typeDefaults.audio?.language
      ).toBe("en-US")
    })
  })

  it("clears a selected standard audio language back to unset", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await user.selectOptions(audioLanguageSelect, "en-US")

    expect(ctxRef).not.toBeNull()
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.typeDefaults.audio?.language
      ).toBe("en-US")
    })

    await user.click(
      screen.getByRole("button", { name: /clear audio language/i })
    )
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.typeDefaults.audio?.language
      ).toBeUndefined()
    })
    expect(audioLanguageSelect).not.toHaveAttribute("value")
    expect(screen.getByText("Select language")).toBeInTheDocument()
  })

  it("maps an unknown saved audio language to a custom entry field", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      typeDefaults: {
        audio: {
          language: "zz-Unknown",
        },
      },
    })

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await waitFor(() => {
      expect(audioLanguageSelect).toHaveValue("__custom__")
    })

    const customInput = await screen.findByLabelText("Custom audio language")
    await waitFor(() => {
      expect(customInput).toHaveValue("zz-Unknown")
    })
  })

  it("reopens custom audio language with current stored value after unknown-to-standard then custom", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      typeDefaults: {
        audio: {
          language: "zz-Unknown",
        },
      },
    })

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await waitFor(() => {
      expect(audioLanguageSelect).toHaveValue("__custom__")
    })

    let customInput = await screen.findByLabelText("Custom audio language")
    await waitFor(() => {
      expect(customInput).toHaveValue("zz-Unknown")
    })

    await user.selectOptions(audioLanguageSelect, "en-US")
    await waitFor(() => {
      expect(ctxRef!.state.presetConfig.typeDefaults.audio?.language).toBe("en-US")
    })

    await user.selectOptions(audioLanguageSelect, "__custom__")
    await waitFor(() => {
      customInput = screen.getByLabelText("Custom audio language")
      expect(customInput).toHaveValue("en-US")
    })
  })

  it("keeps audio language unselected when no value is saved", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      common: {
        ...ctxRef!.state.presetConfig.common,
        perform_analysis: false,
      },
      typeDefaults: {
        audio: {
          language: "",
        },
      },
    })

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await waitFor(() => {
      expect(ctxRef!.state.presetConfig.typeDefaults.audio?.language).toBe("")
    })
    expect(screen.getByText("Select language")).toBeInTheDocument()
    expect(audioLanguageSelect).not.toHaveAttribute("value")
    expect(screen.queryByLabelText("Custom audio language")).toBeNull()
  })

  it("selecting the custom option keeps stored language until custom input is edited", async () => {
    const user = userEvent.setup()
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      typeDefaults: {
        audio: {
          language: "en-US",
        },
      },
    })

    const audioLanguageSelect = await screen.findByLabelText("Audio language")
    await waitFor(() => {
      expect(audioLanguageSelect).toHaveValue("en-US")
      expect(screen.queryByLabelText("Custom audio language")).toBeNull()
    })

    await user.selectOptions(audioLanguageSelect, "__custom__")

    const customInput = await screen.findByLabelText("Custom audio language")
    expect(customInput).toBeInTheDocument()

    expect(ctxRef!.state.presetConfig.typeDefaults.audio?.language).toBe("en-US")
  })

  it("loads transcription models from the backend catalog", async () => {
    const user = userEvent.setup()
    getTranscriptionModelsMock.mockResolvedValue({
      all_models: ["whisper-large-v3", "parakeet-standard"],
    })
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    const transcriptionModelSelect = await screen.findByLabelText(
      "Transcription model"
    )
    await waitFor(() => {
      expect(screen.getByRole("option", { name: "whisper-large-v3" })).toBeTruthy()
      expect(screen.getByRole("option", { name: "parakeet-standard" })).toBeTruthy()
    })
    expect(transcriptionModelSelect).toHaveAttribute(
      "data-popup-match-select-width",
      "false"
    )

    await user.selectOptions(
      transcriptionModelSelect,
      "whisper-large-v3"
    )

    expect(ctxRef).not.toBeNull()
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.advancedValues?.transcription_model
      ).toBe("whisper-large-v3")
    })
  })

  it("preserves a current transcription model not returned by the backend catalog", async () => {
    const user = userEvent.setup()
    getTranscriptionModelsMock.mockResolvedValue({
      all_models: ["whisper-large-v3"],
    })
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      advancedValues: {
        transcription_model: "provider/custom-model",
      },
    })

    const transcriptionModelSelect = await screen.findByLabelText(
      "Transcription model"
    )
    await waitFor(() => {
      expect(transcriptionModelSelect).toHaveValue("provider/custom-model")
      expect(screen.getByRole("option", { name: "provider/custom-model" })).toBeTruthy()
    })
    expect(ctxRef).not.toBeNull()
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.advancedValues?.transcription_model
      ).toBe("provider/custom-model")
    })
  })

  it("clears transcription model selection via clear action", async () => {
    const user = userEvent.setup()
    getTranscriptionModelsMock.mockResolvedValue({
      all_models: ["whisper-large-v3"],
    })
    render(<WizardTestHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    expect(ctxRef).not.toBeNull()
    ctxRef!.setCustomOptions({
      advancedValues: {
        transcription_model: "provider/custom-model",
      },
    })

    await screen.findByLabelText("Transcription model")

    await user.click(await screen.findByRole("button", { name: /clear transcription model/i }))
    await waitFor(() => {
      expect(
        ctxRef!.state.presetConfig.advancedValues?.transcription_model
      ).toBeUndefined()
    })
  })

  it("refetches transcription model catalog when the configure step is hidden then shown again", async () => {
    const user = userEvent.setup()
    getTranscriptionModelsMock
      .mockResolvedValueOnce({
        all_models: ["whisper-large-v3"],
      })
      .mockResolvedValueOnce({
        all_models: ["parakeet-standard"],
      })
    render(<ReopenableConfigStepHarness onClose={vi.fn()} />)

    await user.type(
      screen.getByPlaceholderText(/https:\/\/example\.com/i),
      "https://example.com/library/video.mkv"
    )
    await user.click(screen.getByRole("button", { name: /Add URLs to queue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("https://example.com/library/video.mkv")
      ).toBeTruthy()
    })

    await user.click(screen.getByText(/Configure 1 items/i))

    await expandAdvancedOptions(user)

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "whisper-large-v3" })).toBeTruthy()
    })
    expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: /hide configure step/i }))
    await user.click(screen.getByRole("button", { name: /show configure step/i }))

    await expandAdvancedOptions(user)

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "parakeet-standard" })).toBeTruthy()
      expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(2)
    })
  })
})

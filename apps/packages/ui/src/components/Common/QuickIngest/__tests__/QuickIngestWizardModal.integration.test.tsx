import React from "react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

// ---------------------------------------------------------------------------
// Mocks — must be declared before any imports that reference them
// ---------------------------------------------------------------------------
const getTranscriptionModelsMock = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ all_models: [] })
)

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn(),
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
vi.mock("lucide-react", () => {
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
  ]
  const mocks: Record<string, any> = {}
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
        addListener: vi.fn(),
        removeListener: vi.fn(),
      },
    },
  },
}))

// SSE hook — no-op
vi.mock("@/components/Common/QuickIngest/useIngestSSE", () => ({
  useIngestSSE: () => {},
  default: () => {},
}))

// FileDropZone — simple placeholder div
vi.mock(
  "@/components/Common/QuickIngest/QueueTab/FileDropZone",
  () => ({
    FileDropZone: ({ onFilesAdded }: any) => (
      <div data-testid="file-drop-zone">
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
    ),
    default: ({ onFilesAdded }: any) => (
      <div data-testid="file-drop-zone">
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
    ),
  })
)

// background-proxy
vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue({}),
}))

vi.mock("@/services/tldw/quick-ingest-batch", () => ({
  cancelQuickIngestSession: vi.fn().mockResolvedValue({ ok: true }),
  startQuickIngestSession: vi.fn().mockResolvedValue({ ok: true, sessionId: "qi-test" }),
  submitQuickIngestBatch: vi.fn().mockResolvedValue({ ok: true, results: [] }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getTranscriptionModels: getTranscriptionModelsMock,
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
  getTranscriptionModelsMock.mockReset().mockResolvedValue({ all_models: [] })
  capabilityMocks.useServerCapabilities.mockReset()
  capabilityMocks.useServerCapabilities.mockReturnValue({
    capabilities: { ffmpegAvailable: true },
    loading: false,
  })
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
} from "@/components/Common/QuickIngest/IngestWizardContext"
import { AddContentStep } from "@/components/Common/QuickIngest/AddContentStep"
import { WizardConfigureStep } from "@/components/Common/QuickIngest/WizardConfigureStep"
import { ReviewStep } from "@/components/Common/QuickIngest/ReviewStep"
import { ProcessingStep } from "@/components/Common/QuickIngest/ProcessingStep"
import { WizardResultsStep } from "@/components/Common/QuickIngest/WizardResultsStep"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
}> = ({
  onClose,
  isStepVisible = true,
  isOnlineForIngest = true,
  connectionRecoveryMessage,
  isCheckingConnection,
  onRetryConnection,
  onQuickProcess,
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
        <WizardConfigureStep isStepVisible={isStepVisible} />
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
}> = ({
  onClose,
  isOnlineForIngest,
  connectionRecoveryMessage,
  isCheckingConnection,
  onRetryConnection,
  onQuickProcess,
}) => {
  return (
    <IngestWizardProvider>
      <ContextSpy />
      <InnerWizardContent
        onClose={onClose}
        isOnlineForIngest={isOnlineForIngest}
        connectionRecoveryMessage={connectionRecoveryMessage}
        isCheckingConnection={isCheckingConnection}
        onRetryConnection={onRetryConnection}
        onQuickProcess={onQuickProcess}
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
  let onClose: () => void

  beforeEach(() => {
    onClose = vi.fn()
    ctxRef = null
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

    expect(screen.getByText("Skipped (1)")).toBeTruthy()
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
      expect(screen.getByText("https://example.com/page1")).toBeTruthy()
      expect(screen.getByText("https://example.com/page2")).toBeTruthy()
    })

    // The configure button should reference 2 items
    expect(screen.getByText(/Configure 2 items/i)).toBeTruthy()
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

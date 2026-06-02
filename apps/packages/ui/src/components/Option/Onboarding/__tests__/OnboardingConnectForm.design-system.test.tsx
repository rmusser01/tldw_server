// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { getDesignSystemState } from "@/design-system"

const setupRouteMocks = vi.hoisted(() => ({
  optionLayout: vi.fn()
}))

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({
    children,
    hideHeader,
    hideSidebar
  }: {
    children: React.ReactNode
    hideHeader?: boolean
    hideSidebar?: boolean
  }) => {
    setupRouteMocks.optionLayout({ hideHeader, hideSidebar })
    return <main data-testid="setup-route-layout">{children}</main>
  }
}))

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: ({ onFinish }: { onFinish?: () => void }) => (
    <div>
      <input data-testid="onboarding-server-url" />
      <button type="button" onClick={onFinish}>
        Mock wizard
      </button>
    </div>
  )
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    loading: false,
    error: null
  })
}))

vi.mock("@/components/Option/Onboarding/UnifiedSetupWizard", () => ({
  UnifiedSetupWizard: () => (
    <div data-testid="unified-setup-shell" tabIndex={-1}>
      <button type="button">Mock unified setup</button>
    </div>
  )
}))

type MockConnectionState = {
  phase: "idle" | "connected"
  isConnected: boolean
  isChecking: boolean
  serverUrl: string
  knowledgeStatus: "ready" | "indexing" | "empty" | "error"
  lastStatusCode: number | null
  lastError: string | null
  errorKind: "auth" | null
}

type MockValidationResult = {
  success: boolean
  errorKind?: "auth_invalid" | "refused" | null
  error?: string | null
}

type RenderConnectionFormOptions = {
  validationResult?: MockValidationResult
  connectedState?: Partial<MockConnectionState>
  testConnectionRejects?: Error
}

type MockButtonProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "disabled" | "onClick" | "type"
> & {
  children?: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  loading?: boolean
  block?: boolean
  icon?: React.ReactNode
  size?: string
  type?: string
}

type MockInputChangeEvent = {
  target: {
    value: string
  }
}

type MockInputProps = Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  "disabled" | "onChange" | "size" | "value"
> & {
  value?: string
  onChange?: (event: MockInputChangeEvent) => void
  disabled?: boolean
  suffix?: React.ReactNode
  status?: string
  size?: string
}

type MockPasswordInputProps = Omit<MockInputProps, "suffix">

type MockSelectProps = Omit<
  React.SelectHTMLAttributes<HTMLSelectElement>,
  "disabled" | "onChange" | "value"
> & {
  value?: string
  onChange?: (value: string) => void
  options?: Array<{ value: string; label: string }>
  disabled?: boolean
}

const createIdleConnectionState = (): MockConnectionState => ({
  phase: "idle",
  isConnected: false,
  isChecking: true,
  serverUrl: "http://127.0.0.1:8000",
  knowledgeStatus: "ready",
  lastStatusCode: null,
  lastError: null,
  errorKind: null
})

const renderConnectionForm = async (options: RenderConnectionFormOptions = {}) => {
  vi.resetModules()
  let connectionState = createIdleConnectionState()
  const validationResult = options.validationResult ?? { success: true }
  const setConfigPartial = vi.fn().mockResolvedValue(undefined)
  const testConnectionFromOnboarding = vi.fn().mockImplementation(async () => {
    if (options.testConnectionRejects) {
      throw options.testConnectionRejects
    }
    connectionState = {
      ...connectionState,
      phase: "connected",
      isConnected: true,
      isChecking: false,
      knowledgeStatus: "ready",
      lastStatusCode: null,
      lastError: null,
      errorKind: null,
      ...options.connectedState
    }
  })
  const connectionActions = {
    beginOnboarding: vi.fn(),
    setConfigPartial,
    testConnectionFromOnboarding,
    setDemoMode: vi.fn(),
    markFirstRunComplete: vi.fn().mockResolvedValue(undefined),
    setUserPersona: vi.fn().mockResolvedValue(undefined)
  }

  vi.doMock("react-i18next", () => ({
    useTranslation: () => ({
      t: (
        key: string,
        defaultValueOrOptions?: string | { defaultValue?: string }
      ) => {
        if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
        if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
        return key
      }
    })
  }))

  vi.doMock("react-router-dom", async () => {
    const actual = await vi.importActual<typeof import("react-router-dom")>(
      "react-router-dom"
    )
    return {
      ...actual,
      useNavigate: () => vi.fn()
    }
  })

  vi.doMock("antd", () => ({
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      block: _block,
      icon: _icon,
      size: _size,
      type: _type,
      ...props
    }: MockButtonProps) => (
      <button
        type="button"
        onClick={onClick}
        disabled={disabled || loading}
        {...props}
      >
        {children}
      </button>
    ),
    Input: Object.assign(
      React.forwardRef<HTMLInputElement, MockInputProps>(
        (
          {
            value,
            onChange,
            disabled,
            suffix: _suffix,
            status: _status,
            size: _size,
            ...props
          },
          ref
        ) => (
          <input
            ref={ref}
            value={value}
            onChange={(event) =>
              onChange?.({ target: { value: event.currentTarget.value } })
            }
            disabled={disabled}
            {...props}
          />
        )
      ),
      {
        Password: React.forwardRef<HTMLInputElement, MockPasswordInputProps>(
          (
            {
              value,
              onChange,
              disabled,
              status: _status,
              size: _size,
              ...props
            },
            ref
          ) => (
            <input
              ref={ref}
              value={value}
              onChange={(event) =>
                onChange?.({ target: { value: event.currentTarget.value } })
              }
              disabled={disabled}
              {...props}
            />
          )
        )
      }
    ),
    Select: ({
      value,
      onChange,
      options = [],
      disabled,
      ...props
    }: MockSelectProps) => (
      <select
        value={value ?? ""}
        onChange={(event) => onChange?.(event.currentTarget.value)}
        disabled={disabled}
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ),
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn()
    }
  }))

  vi.doMock("@tanstack/react-query", () => ({
    useQuery: () => ({
      data: [],
      isLoading: false
    })
  }))

  vi.doMock("@plasmohq/storage/hook", () => ({
    useStorage: (_key: string, initialValue: string | null) =>
      [initialValue, vi.fn()] as const
  }))

  vi.doMock("@/services/tldw/TldwApiClient", () => ({
    tldwClient: {
      getConfig: vi.fn().mockResolvedValue({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-api-key"
      })
    }
  }))

  vi.doMock("@/services/tldw/TldwAuth", () => ({
    tldwAuth: {
      requestMagicLink: vi.fn()
    }
  }))

  vi.doMock("@/services/auth-errors", () => ({
    mapMultiUserLoginErrorMessage: () => "Login failed"
  }))

  vi.doMock("@/services/splash-auth", () => ({
    emitSplashAfterSingleUserAuthSuccess: vi.fn()
  }))

  vi.doMock("@/services/tldw-server", () => ({
    getTldwServerURL: vi.fn().mockResolvedValue(null),
    DEFAULT_TLDW_API_KEY: "test-api-key",
    fetchChatModels: vi.fn().mockResolvedValue([])
  }))

  vi.doMock("@/hooks/useConnectionState", () => ({
    useConnectionState: () => connectionState,
    useConnectionActions: () => connectionActions
  }))

  vi.doMock("@/hooks/useServerCapabilities", () => ({
    useServerCapabilities: () => ({
      capabilities: {
        hasGuardian: true
      }
    })
  }))

  vi.doMock("@/store/connection", () => ({
    useConnectionStore: Object.assign(vi.fn(), {
      getState: () => ({
        state: connectionState
      })
    })
  }))

  vi.doMock("@/context/demo-mode", () => ({
    useDemoMode: () => ({
      setDemoEnabled: vi.fn()
    })
  }))

  vi.doMock("@/store/quick-ingest", () => ({
    useQuickIngestStore: (
      selector: (state: {
        lastRunSummary: {
          status: "idle"
          successCount: number
          attemptedAt: null
          firstMediaId: null
          primarySourceLabel: null
        }
      }) => unknown
    ) =>
      selector({
        lastRunSummary: {
          status: "idle",
          successCount: 0,
          attemptedAt: null,
          firstMediaId: null,
          primarySourceLabel: null
        }
      })
  }))

  vi.doMock("@/utils/quick-ingest-open", () => ({
    requestQuickIngestIntro: vi.fn()
  }))

  vi.doMock("@/utils/sidepanel", () => ({
    openSidepanelForActiveTab: vi.fn()
  }))

  vi.doMock("@/utils/extension-permissions", () => ({
    requestOptionalHostPermission: vi.fn()
  }))

  vi.doMock("@/services/settings/registry", () => ({
    defineSetting: <T,>(
      key: string,
      defaultValue: T,
      coerceOrOptions?: ((value: unknown) => T) | Record<string, unknown>,
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof coerceOrOptions === "function") {
        return {
          key,
          defaultValue,
          coerce: coerceOrOptions,
          ...(maybeOptions || {})
        }
      }
      return {
        key,
        defaultValue,
        ...(coerceOrOptions || {})
      }
    },
    coerceBoolean: (value: unknown, fallback: boolean) => {
      if (typeof value === "boolean") return value
      if (typeof value === "string") return value === "true"
      return fallback
    },
    coerceOptionalString: (value: unknown) =>
      typeof value === "string" && value.length > 0 ? value : undefined,
    coerceNumber: (value: unknown, fallback: number) => {
      if (typeof value === "number" && Number.isFinite(value)) return value
      if (typeof value === "string") {
        const parsed = Number(value)
        if (Number.isFinite(parsed)) return parsed
      }
      return fallback
    },
    coerceString: (value: unknown, fallback: string) =>
      typeof value === "string" && value.length > 0 ? value : fallback,
    setSetting: vi.fn()
  }))

  vi.doMock("@/components/Layouts/header-shortcut-items", () => ({
    getDefaultShortcutsForPersona: (persona: string) => [`${persona}-shortcut`]
  }))

  vi.doMock("@/utils/browser-runtime", () => ({
    isExtensionRuntime: () => false
  }))

  vi.doMock("@/utils/provider-registry", () => ({
    getProviderDisplayName: (provider: string) => provider,
    normalizeProviderKey: (provider: string) => provider
  }))

  vi.doMock("@/utils/onboarding-ingestion-telemetry", () => ({
    trackOnboardingSuccessReached: vi.fn().mockResolvedValue(undefined),
    trackOnboardingFirstIngestSuccess: vi.fn().mockResolvedValue(undefined)
  }))

  vi.doMock("../validation", () => ({
    validateApiKey: vi.fn().mockResolvedValue(validationResult),
    validateMultiUserAuth: vi.fn(),
    validateMagicLinkAuth: vi.fn(),
    categorizeConnectionError: vi.fn().mockReturnValue(null)
  }))

  const { OnboardingConnectForm } = await import("../OnboardingConnectForm")

  render(
    <MemoryRouter>
      <OnboardingConnectForm />
    </MemoryRouter>
  )
}

const waitForConnectButton = async () => {
  const connectButton = await screen.findByTestId("onboarding-connect")
  await waitFor(() => {
    expect(connectButton).not.toBeDisabled()
  })
  return connectButton
}

describe("setup onboarding design-system state wiring", () => {
  afterEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    setupRouteMocks.optionLayout.mockClear()
  })

  it("frames setup in a setup-only shell with the canonical setup-required action", async () => {
    const { default: OptionSetup } = await import("@/routes/option-setup")

    render(
      <MemoryRouter>
        <OptionSetup />
      </MemoryRouter>
    )

    expect(setupRouteMocks.optionLayout).toHaveBeenCalledWith({
      hideHeader: true,
      hideSidebar: true
    })
    expect(
      screen.getByText(getDesignSystemState("setup_required").label)
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /continue setup/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Mock unified setup" })).toBeInTheDocument()

    const setupShell = screen.getByTestId("unified-setup-shell")
    fireEvent.click(screen.getByRole("button", { name: "Continue setup" }))
    expect(setupShell).toHaveFocus()
    expect(screen.queryByTestId("chat-header-theme-toggle")).not.toBeInTheDocument()
    expect(screen.queryByRole("navigation")).not.toBeInTheDocument()
  })

  it("announces retrying only while the connection test is busy", async () => {
    await renderConnectionForm()
    const connectButton = await waitForConnectButton()
    fireEvent.click(connectButton)

    expect(
      await screen.findByText(getDesignSystemState("retrying").label)
    ).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText(getDesignSystemState("ready").label)).toBeInTheDocument()
    })
  })

  it("uses auth-required labeling for failed auth progress instead of retrying", async () => {
    await renderConnectionForm({
      validationResult: {
        success: false,
        errorKind: "auth_invalid",
        error: "Invalid API key"
      }
    })
    const connectButton = await waitForConnectButton()
    fireEvent.click(connectButton)

    await waitFor(() => {
      expect(
        screen.getAllByText(getDesignSystemState("auth_required").label).length
      ).toBeGreaterThan(0)
    })
    expect(screen.queryByText(getDesignSystemState("retrying").label)).not.toBeInTheDocument()
  })

  it("uses the canonical ready label on successful connection", async () => {
    await renderConnectionForm()
    const connectButton = await waitForConnectButton()
    fireEvent.click(connectButton)

    await waitFor(() => {
      expect(screen.getByText(getDesignSystemState("ready").label)).toBeInTheDocument()
    })
  })
})

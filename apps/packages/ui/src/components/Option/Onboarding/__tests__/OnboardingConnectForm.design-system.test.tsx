// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { getDesignSystemState } from "@/design-system"

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => <main>{children}</main>
}))

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: ({ onFinish }: { onFinish?: () => void }) => (
    <button type="button" onClick={onFinish}>
      Mock wizard
    </button>
  )
}))

type FormUiState = {
  isConnecting: boolean
  progress: {
    serverReachable: "idle" | "checking" | "success" | "error" | "empty"
    authentication: "idle" | "checking" | "success" | "error" | "empty"
    knowledgeIndex: "idle" | "checking" | "success" | "error" | "empty"
  }
  errorKind: "auth_invalid" | "refused" | null
  errorMessage: string | null
  showSuccess: boolean
  hasRunConnectionTest: boolean
}

const defaultFormUiState: FormUiState = {
  isConnecting: false,
  progress: {
    serverReachable: "idle",
    authentication: "idle",
    knowledgeIndex: "idle"
  },
  errorKind: null,
  errorMessage: null,
  showSuccess: false,
  hasRunConnectionTest: false
}

const renderConnectionFormState = async (uiState: Partial<FormUiState>) => {
  vi.resetModules()
  const resolvedState: FormUiState = {
    ...defaultFormUiState,
    ...uiState,
    progress: {
      ...defaultFormUiState.progress,
      ...uiState.progress
    }
  }

  vi.doMock("react", async () => {
    const actual = await vi.importActual<typeof import("react")>("react")
    return {
      ...actual,
      useReducer: () => [resolvedState, vi.fn()]
    }
  })

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
    }: {
      children?: React.ReactNode
      onClick?: () => void
      disabled?: boolean
      loading?: boolean
      block?: boolean
      icon?: React.ReactNode
      size?: string
      type?: string
      [key: string]: unknown
    }) => (
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
      React.forwardRef<HTMLInputElement, {
        value?: string
        onChange?: (event: { target: { value: string } }) => void
        disabled?: boolean
        suffix?: React.ReactNode
        status?: string
        size?: string
        [key: string]: unknown
      }>(({ value, onChange, disabled, suffix: _suffix, status: _status, size: _size, ...props }, ref) => (
        <input
          ref={ref}
          value={value}
          onChange={(event) =>
            onChange?.({ target: { value: event.currentTarget.value } })
          }
          disabled={disabled}
          {...props}
        />
      )),
      {
        Password: React.forwardRef<HTMLInputElement, {
          value?: string
          onChange?: (event: { target: { value: string } }) => void
          disabled?: boolean
          status?: string
          size?: string
          [key: string]: unknown
        }>(({ value, onChange, disabled, status: _status, size: _size, ...props }, ref) => (
          <input
            ref={ref}
            value={value}
            onChange={(event) =>
              onChange?.({ target: { value: event.currentTarget.value } })
            }
            disabled={disabled}
            {...props}
          />
        ))
      }
    ),
    Select: ({
      value,
      onChange,
      options = [],
      disabled,
      ...props
    }: {
      value?: string
      onChange?: (value: string) => void
      options?: Array<{ value: string; label: string }>
      disabled?: boolean
      [key: string]: unknown
    }) => (
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
    useConnectionState: () => ({
      phase: resolvedState.showSuccess ? "connected" : "idle",
      isConnected: resolvedState.showSuccess,
      isChecking: resolvedState.isConnecting,
      serverUrl: "http://127.0.0.1:8000",
      knowledgeStatus: "ready",
      lastStatusCode: null,
      lastError: null,
      errorKind: null
    }),
    useConnectionActions: () => ({
      beginOnboarding: vi.fn(),
      setConfigPartial: vi.fn().mockResolvedValue(undefined),
      testConnectionFromOnboarding: vi.fn().mockResolvedValue(undefined),
      setDemoMode: vi.fn(),
      markFirstRunComplete: vi.fn().mockResolvedValue(undefined),
      setUserPersona: vi.fn().mockResolvedValue(undefined)
    })
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
        state: {
          isConnected: resolvedState.showSuccess
        }
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
    validateApiKey: vi.fn().mockResolvedValue({ success: true }),
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

describe("setup onboarding design-system state wiring", () => {
  afterEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
  })

  it("frames setup with the canonical setup-required label and primary action", async () => {
    const { default: OptionSetup } = await import("@/routes/option-setup")

    render(
      <MemoryRouter>
        <OptionSetup />
      </MemoryRouter>
    )

    expect(
      screen.getByText(getDesignSystemState("setup_required").label)
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /start setup|connect/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Mock wizard" })).toBeInTheDocument()
  })

  it("announces retrying only while the connection test is busy", async () => {
    await renderConnectionFormState({
      isConnecting: true,
      progress: {
        serverReachable: "checking",
        authentication: "idle",
        knowledgeIndex: "idle"
      }
    })

    expect(screen.getByText(getDesignSystemState("retrying").label)).toBeInTheDocument()
  })

  it("uses auth-required labeling for failed auth progress instead of retrying", async () => {
    await renderConnectionFormState({
      errorKind: "auth_invalid",
      errorMessage: "Invalid API key",
      progress: {
        serverReachable: "success",
        authentication: "error",
        knowledgeIndex: "idle"
      }
    })

    expect(
      screen.getAllByText(getDesignSystemState("auth_required").label).length
    ).toBeGreaterThan(0)
    expect(screen.queryByText(getDesignSystemState("retrying").label)).not.toBeInTheDocument()
  })

  it("uses the canonical ready label on successful connection", async () => {
    await renderConnectionFormState({
      showSuccess: true,
      hasRunConnectionTest: true,
      progress: {
        serverReachable: "success",
        authentication: "success",
        knowledgeIndex: "success"
      }
    })

    await waitFor(() => {
      expect(screen.getByText(getDesignSystemState("ready").label)).toBeInTheDocument()
    })
  })
})

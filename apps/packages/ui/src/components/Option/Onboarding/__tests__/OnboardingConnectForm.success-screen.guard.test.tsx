// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { HEADER_SHORTCUT_SELECTION_SETTING } from "@/services/settings/ui-settings"

type HarnessState = {
  navigate: ReturnType<typeof vi.fn>
  beginOnboarding: ReturnType<typeof vi.fn>
  setConfigPartial: ReturnType<typeof vi.fn>
  testConnectionFromOnboarding: ReturnType<typeof vi.fn>
  setDemoMode: ReturnType<typeof vi.fn>
  markFirstRunComplete: ReturnType<typeof vi.fn>
  setUserPersona: ReturnType<typeof vi.fn>
  setDemoEnabled: ReturnType<typeof vi.fn>
  openSidepanelForActiveTab: ReturnType<typeof vi.fn>
  setSetting: ReturnType<typeof vi.fn>
}

const buildHarness = (): HarnessState => ({
  navigate: vi.fn(),
  beginOnboarding: vi.fn(),
  setConfigPartial: vi.fn().mockResolvedValue(undefined),
  testConnectionFromOnboarding: vi.fn().mockResolvedValue(undefined),
  setDemoMode: vi.fn(),
  markFirstRunComplete: vi.fn().mockResolvedValue(undefined),
  setUserPersona: vi.fn().mockResolvedValue(undefined),
  setDemoEnabled: vi.fn(),
  openSidepanelForActiveTab: vi.fn().mockResolvedValue(undefined),
  setSetting: vi.fn().mockResolvedValue(undefined)
})

const renderSuccessScreen = async ({
  familyGuardrailsAvailable = true
}: {
  familyGuardrailsAvailable?: boolean
} = {}) => {
  const harness = buildHarness()

  vi.doMock("react", async () => {
    const actual = await vi.importActual<typeof import("react")>("react")
    return {
      ...actual,
      useReducer: () => [
        {
          isConnecting: false,
          progress: {
            serverReachable: "success",
            authentication: "success",
            knowledgeIndex: "success"
          },
          errorKind: null,
          errorMessage: null,
          showSuccess: true,
          hasRunConnectionTest: true
        },
        vi.fn()
      ]
    }
  })

  vi.doMock("react-i18next", () => ({
    useTranslation: () => ({
      t: (
        key: string,
        defaultValueOrOptions?:
          | string
          | {
              defaultValue?: string
            }
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
      useNavigate: () => harness.navigate
    }
  })

  vi.doMock("antd", () => ({
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      ...props
    }: {
      children?: React.ReactNode
      onClick?: () => void
      disabled?: boolean
      loading?: boolean
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
      ({
        value,
        onChange,
        disabled,
        ...props
      }: {
        value?: string
        onChange?: (event: { target: { value: string } }) => void
        disabled?: boolean
        [key: string]: unknown
      }) => (
        <input
          value={value}
          onChange={(event) =>
            onChange?.({ target: { value: event.currentTarget.value } })
          }
          disabled={disabled}
          {...props}
        />
      ),
      {
        Password: ({
          value,
          onChange,
          disabled,
          ...props
        }: {
          value?: string
          onChange?: (event: { target: { value: string } }) => void
          disabled?: boolean
          [key: string]: unknown
        }) => (
          <input
            value={value}
            onChange={(event) =>
              onChange?.({ target: { value: event.currentTarget.value } })
            }
            disabled={disabled}
            {...props}
          />
        )
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
      phase: "connected",
      isConnected: true,
      isChecking: false,
      serverUrl: "http://127.0.0.1:8000",
      knowledgeStatus: "ready",
      lastStatusCode: null,
      lastError: null,
      errorKind: null
    }),
    useConnectionActions: () => ({
      beginOnboarding: harness.beginOnboarding,
      setConfigPartial: harness.setConfigPartial,
      testConnectionFromOnboarding: harness.testConnectionFromOnboarding,
      setDemoMode: harness.setDemoMode,
      markFirstRunComplete: harness.markFirstRunComplete,
      setUserPersona: harness.setUserPersona
    })
  }))

  vi.doMock("@/hooks/useServerCapabilities", () => ({
    useServerCapabilities: () => ({
      capabilities: {
        hasGuardian: familyGuardrailsAvailable
      }
    })
  }))

  vi.doMock("@/store/connection", () => ({
    useConnectionStore: Object.assign(vi.fn(), {
      getState: () => ({
        state: {
          isConnected: true
        }
      })
    })
  }))

  vi.doMock("@/context/demo-mode", () => ({
    useDemoMode: () => ({
      setDemoEnabled: harness.setDemoEnabled
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
    openSidepanelForActiveTab: harness.openSidepanelForActiveTab
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
    setSetting: harness.setSetting
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
  render(<OnboardingConnectForm />)

  await waitFor(() => {
    expect(screen.getByTestId("onboarding-success-screen")).toBeInTheDocument()
  })

  return harness
}

describe("OnboardingConnectForm success screen guards", () => {
  afterEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
  })

  it("renders the success screen and intent choices", async () => {
    await renderSuccessScreen()

    expect(screen.getByTestId("intent-selector")).toBeInTheDocument()
    expect(
      screen.getByText("What would you like to do first?")
    ).toBeInTheDocument()
    expect(screen.getByText("Set your defaults")).toBeInTheDocument()
  })

  it("routes chat intent directly to chat after setting the explorer persona", async () => {
    const harness = await renderSuccessScreen()

    fireEvent.click(screen.getByText("Chat with AI").closest("button")!)

    await waitFor(() => {
      expect(harness.setUserPersona).toHaveBeenCalledWith("explorer")
      expect(harness.setSetting).toHaveBeenCalledWith(
        expect.objectContaining({ key: HEADER_SHORTCUT_SELECTION_SETTING.key }),
        ["explorer-shortcut"]
      )
      expect(harness.openSidepanelForActiveTab).toHaveBeenCalledTimes(1)
      expect(harness.markFirstRunComplete).toHaveBeenCalled()
      expect(harness.navigate).toHaveBeenCalledWith("/chat")
    })
  })

  it("keeps guided intent users on the same persona when they skip ahead to chat", async () => {
    const harness = await renderSuccessScreen()

    fireEvent.click(screen.getByText("Research my documents").closest("button")!)

    expect(screen.getByTestId("intent-steps-research")).toBeInTheDocument()

    harness.setUserPersona.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Skip, go to chat" }))

    await waitFor(() => {
      expect(harness.setUserPersona).not.toHaveBeenCalled()
      expect(harness.openSidepanelForActiveTab).toHaveBeenCalledTimes(1)
      expect(harness.navigate).toHaveBeenCalledWith("/chat")
    })
  })
})

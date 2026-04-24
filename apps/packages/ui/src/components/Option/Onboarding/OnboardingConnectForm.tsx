import { useState, useEffect, useMemo, useCallback, useRef, useReducer, type ReactNode } from "react"
import { Input, Button, Tooltip, message, Select } from "antd"
import type { InputRef } from "antd"
import {
  Check,
  X,
  Loader2,
  Server,
  Key,
  User,
  Lock,
  AlertCircle,
  Info,
  ExternalLink,
  Copy,
  ChevronDown,
  ChevronRight,
  Sparkles,
  ArrowRight,
  ArrowLeft,
  RefreshCw,
  MessageSquare,
  Shield,
  BookOpen,
} from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { useNavigate } from "react-router-dom"
import { DOCUMENTATION_URL } from "@/config/constants"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { mapMultiUserLoginErrorMessage } from "@/services/auth-errors"
import { emitSplashAfterSingleUserAuthSuccess } from "@/services/splash-auth"
import {
  getTldwServerURL,
  DEFAULT_TLDW_API_KEY,
  fetchChatModels,
} from "@/services/tldw-server"
import {
  useConnectionState,
  useConnectionActions,
} from "@/hooks/useConnectionState"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useConnectionStore } from "@/store/connection"
import { useDemoMode } from "@/context/demo-mode"
import { requestQuickIngestIntro } from "@/utils/quick-ingest-open"
import { openSidepanelForActiveTab } from "@/utils/sidepanel"
import { requestOptionalHostPermission } from "@/utils/extension-permissions"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { cn } from "@/libs/utils"
import { setSetting } from "@/services/settings/registry"
import { HEADER_SHORTCUT_SELECTION_SETTING } from "@/services/settings/ui-settings"
import { getDefaultShortcutsForPersona } from "@/components/Layouts/header-shortcut-items"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { getProviderDisplayName, normalizeProviderKey } from "@/utils/provider-registry"
import {
  trackOnboardingFirstIngestSuccess,
  trackOnboardingSuccessReached
} from "@/utils/onboarding-ingestion-telemetry"
import {
  validateApiKey,
  validateMultiUserAuth,
  validateMagicLinkAuth,
  categorizeConnectionError,
  type ConnectionErrorKind,
  type ValidationResult,
} from "./validation"
import { ProgressItem, type ProgressStatus } from "./ProgressItem"

type AuthMode = "single-user" | "multi-user"
type LoginMethod = "magic-link" | "password"

type ConnectionProgress = {
  serverReachable: ProgressStatus
  authentication: ProgressStatus
  knowledgeIndex: ProgressStatus
}

type ConnectionUiState = {
  isConnecting: boolean
  progress: ConnectionProgress
  errorKind: ConnectionErrorKind
  errorMessage: string | null
  showSuccess: boolean
  hasRunConnectionTest: boolean
}

type ConnectionUiAction =
  | { type: "START_CONNECT" }
  | { type: "FINISH_CONNECT" }
  | {
      type: "UPDATE_PROGRESS"
      updater: (prev: ConnectionProgress) => ConnectionProgress
    }
  | {
      type: "SET_ERROR"
      errorKind: ConnectionErrorKind
      errorMessage: string | null
    }
  | {
      type: "SET_SHOW_SUCCESS"
      showSuccess: boolean
    }
  | {
      type: "SET_HAS_RUN_TEST"
      hasRunConnectionTest: boolean
    }

const initialConnectionUiState: ConnectionUiState = {
  isConnecting: false,
  progress: {
    serverReachable: "idle",
    authentication: "idle",
    knowledgeIndex: "idle",
  },
  errorKind: null,
  errorMessage: null,
  showSuccess: false,
  hasRunConnectionTest: false,
}

function connectionUiReducer(
  state: ConnectionUiState,
  action: ConnectionUiAction
): ConnectionUiState {
  switch (action.type) {
    case "START_CONNECT":
      return {
        ...state,
        hasRunConnectionTest: true,
        isConnecting: true,
        errorKind: null,
        errorMessage: null,
        progress: {
          serverReachable: "checking",
          authentication: "idle",
          knowledgeIndex: "idle",
        },
        showSuccess: false,
      }
    case "FINISH_CONNECT":
      return {
        ...state,
        isConnecting: false,
      }
    case "UPDATE_PROGRESS":
      return {
        ...state,
        progress: action.updater(state.progress),
      }
    case "SET_ERROR":
      return {
        ...state,
        errorKind: action.errorKind,
        errorMessage: action.errorMessage,
      }
    case "SET_SHOW_SUCCESS":
      return {
        ...state,
        showSuccess: action.showSuccess,
      }
    case "SET_HAS_RUN_TEST":
      return {
        ...state,
        hasRunConnectionTest: action.hasRunConnectionTest,
      }
    default:
      return state
  }
}

interface Props {
  onFinish?: () => void
}

const QUICK_INGEST_OPEN_DELAY_MS = 120
const LOCALHOST_PROBE_URL = "http://localhost:8000/health"
const LOCALHOST_PROBE_TIMEOUT_MS = 2_000
const TROUBLESHOOTING_URL =
  "https://github.com/rmusser01/tldw/blob/main/Docs/Getting_Started/TROUBLESHOOTING.md"

type IntentStepsProps = {
  testId: string
  icon: ReactNode
  title: string
  steps: string[]
  primaryLabel: string
  onPrimaryClick: () => void
  secondaryLabel: string
  onSecondaryClick: () => void
  backLabel: string
  onBackClick: () => void
}

function IntentSteps({
  testId,
  icon,
  title,
  steps,
  primaryLabel,
  onPrimaryClick,
  secondaryLabel,
  onSecondaryClick,
  backLabel,
  onBackClick,
}: IntentStepsProps) {
  return (
    <div data-testid={testId} className="rounded-xl border border-border/60 bg-surface2/30 p-5">
      <div className="mb-4 flex items-center gap-2">
        {icon}
        <span className="text-sm font-semibold text-text">{title}</span>
      </div>
      <div className="space-y-2.5">
        {steps.map((step, index) => (
          <div key={step} className="flex items-start gap-2.5">
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              {index + 1}
            </span>
            <span className="text-sm text-text">{step}</span>
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center gap-2">
        <Button type="primary" onClick={onPrimaryClick}>
          {primaryLabel}
          <ArrowRight className="ml-1 h-4 w-4" />
        </Button>
        <Button onClick={onSecondaryClick}>{secondaryLabel}</Button>
        <button
          type="button"
          onClick={onBackClick}
          className="ml-auto flex items-center gap-1 text-xs text-text-muted hover:text-text"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          {backLabel}
        </button>
      </div>
    </div>
  )
}

/**
 * Single-step onboarding form for the new UX redesign.
 * Features:
 * - Progressive connection testing with real-time feedback
 * - Demo mode prominently displayed
 * - Granular error messages
 * - All fields on one page (no multi-step wizard)
 */
export function OnboardingConnectForm({ onFinish }: Props) {
  const { t } = useTranslation(["settings", "common"])
  const navigate = useNavigate()
  const { setDemoEnabled } = useDemoMode()
  const { capabilities } = useServerCapabilities()
  const familyGuardrailsAvailable = Boolean(capabilities?.hasGuardian)
  const connectionState = useConnectionState()
  const actions = useConnectionActions()
  const hostPermissionPromptKeyRef = useRef<string | null>(null)
  const hasTrackedOnboardingSuccessRef = useRef(false)
  const hasTrackedFirstIngestRef = useRef(false)

  // Form state
  const [serverUrl, setServerUrl] = useState("")
  const [authMode, setAuthMode] = useState<AuthMode>("single-user")
  const [apiKey, setApiKey] = useState(DEFAULT_TLDW_API_KEY)
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [loginMethod, setLoginMethod] = useState<LoginMethod>("magic-link")
  const [magicEmail, setMagicEmail] = useState("")
  const [magicToken, setMagicToken] = useState("")
  const [magicSent, setMagicSent] = useState(false)
  const [magicSending, setMagicSending] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [authTouched, setAuthTouched] = useState(false)
  const [selectedModel, setSelectedModel] = useStorage<string | null>(
    "selectedModel",
    null
  )
  const [defaultApiProvider, setDefaultApiProvider] = useStorage<
    string | null
  >("defaultApiProvider", null)

  // UI state (managed via reducer)
  const [uiState, dispatchUi] = useReducer(
    connectionUiReducer,
    initialConnectionUiState
  )
  const {
    isConnecting,
    progress,
    errorKind,
    errorMessage,
    showSuccess,
    hasRunConnectionTest,
  } = uiState

  // Post-connection guided flow: when user selects an intent, show persona-specific steps
  const [selectedIntent, setSelectedIntent] = useState<"chat" | "family" | "research" | null>(null)

  const {
    data: availableModels = [],
    isLoading: modelsLoading,
  } = useQuery({
    queryKey: ["onboarding-chat-models", serverUrl],
    queryFn: async () => fetchChatModels({ returnEmpty: true }),
    enabled: showSuccess,
    staleTime: 5 * 60 * 1000,
  })

  useEffect(() => {
    if (isConnecting) {
      hostPermissionPromptKeyRef.current = null
    }
  }, [isConnecting])

  const normalizedDefaultProvider = useMemo(() => {
    if (!defaultApiProvider) return ""
    const normalized = normalizeProviderKey(defaultApiProvider)
    return normalized === "unknown" ? "" : normalized
  }, [defaultApiProvider])
  const providerSelectValue = useMemo(
    () => normalizedDefaultProvider || "auto",
    [normalizedDefaultProvider]
  )

  const providerOptions = useMemo(() => {
    const providers = new Map<string, string>()
    for (const model of availableModels) {
      const rawProvider = model.details?.provider ?? model.provider
      if (!rawProvider) continue
      const key = normalizeProviderKey(rawProvider)
      if (!key || key === "unknown") continue
      if (!providers.has(key)) {
        providers.set(key, getProviderDisplayName(rawProvider))
      }
    }
    return Array.from(providers.entries())
      .map(([value, label]) => ({ value, label }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [availableModels])

  const modelOptions = useMemo(() => {
    return availableModels
      .filter((model) => {
        if (!normalizedDefaultProvider) return true
        const rawProvider = model.details?.provider ?? model.provider
        if (!rawProvider) return false
        return normalizeProviderKey(rawProvider) === normalizedDefaultProvider
      })
      .map((model) => {
        const rawProvider = model.details?.provider ?? model.provider
        const providerLabel = rawProvider
          ? getProviderDisplayName(rawProvider)
          : t("settings:onboarding.defaults.providerUnknown", "Provider")
        const modelLabel = model.nickname || model.model
        return {
          value: model.model,
          label: `${providerLabel} - ${modelLabel}`,
        }
      })
  }, [availableModels, normalizedDefaultProvider, t])

  const handleProviderChange = useCallback(
    (value: string) => {
      if (value === "auto") {
        setDefaultApiProvider(null)
        return
      }
      const normalized = normalizeProviderKey(value)
      setDefaultApiProvider(
        normalized && normalized !== "unknown" ? normalized : null
      )
    },
    [setDefaultApiProvider]
  )

  const handleModelChange = useCallback(
    (value: string) => {
      setSelectedModel(value || null)
    },
    [setSelectedModel]
  )

  useEffect(() => {
    if (!normalizedDefaultProvider || !selectedModel) return
    if (availableModels.length === 0) return
    const selectedEntry = availableModels.find(
      (model) => model.model === selectedModel
    )
    if (!selectedEntry) return
    const rawProvider = selectedEntry.details?.provider ?? selectedEntry.provider
    if (!rawProvider) return
    if (normalizeProviderKey(rawProvider) !== normalizedDefaultProvider) {
      setSelectedModel(null)
    }
  }, [
    availableModels,
    normalizedDefaultProvider,
    selectedModel,
    setSelectedModel,
  ])

  const urlInputRef = useRef<InputRef | null>(null)
  const hasLoadedInitialConfigRef = useRef(false)

  // Load initial config
  useEffect(() => {
    if (hasLoadedInitialConfigRef.current) return
    hasLoadedInitialConfigRef.current = true

    ;(async () => {
      try {
        actions.beginOnboarding()
        const cfg = await tldwClient.getConfig()
        if (cfg?.serverUrl) setServerUrl(cfg.serverUrl)
        if (cfg?.authMode) setAuthMode(cfg.authMode)
        if (cfg?.apiKey) setApiKey(cfg.apiKey)

        if (!cfg?.serverUrl) {
          const fallback = await getTldwServerURL()
          if (fallback) {
            setServerUrl(fallback)
          } else if (isExtensionRuntime()) {
            // B1: Auto-probe localhost when no URL is configured (extension only)
            // Use a normal CORS-mode fetch so we can distinguish a running server
            // (which may reject CORS with a TypeError) from a truly unreachable
            // host (which also throws TypeError but after an abort timeout).
            const probeController = new AbortController()
            const probeTimer = setTimeout(
              () => probeController.abort(),
              LOCALHOST_PROBE_TIMEOUT_MS
            )
            try {
              const resp = await fetch(LOCALHOST_PROBE_URL, {
                signal: probeController.signal,
              })
              clearTimeout(probeTimer)
              if (resp.ok) {
                setServerUrl("http://localhost:8000")
              }
            } catch (err) {
              clearTimeout(probeTimer)
              // A CORS rejection throws TypeError but the abort signal is NOT
              // triggered. An unreachable host times out and the signal IS aborted.
              if (err instanceof TypeError && !probeController.signal.aborted) {
                // Likely a CORS rejection — server is present but CORS not
                // configured for this origin. Still pre-fill the URL.
                setServerUrl("http://localhost:8000")
              }
              // AbortError or truly unreachable — leave URL empty for manual entry
            }
          }
        }
      } catch {
        // Ignore config load errors
      }
    })()
  }, [actions])
  // ^ Guarded by ref to effectively run only once while keeping actions in dependencies

  // URL validation
  const urlValidation = useMemo(() => {
    const trimmed = serverUrl.trim()
    if (!trimmed) return { valid: false, reason: "empty" as const }
    try {
      const parsed = new URL(trimmed)
      if (!["http:", "https:"].includes(parsed.protocol)) {
        return { valid: false, reason: "protocol" as const }
      }
      return { valid: true, reason: "ok" as const }
    } catch {
      return { valid: false, reason: "invalid" as const }
    }
  }, [serverUrl])

  const authValidation = useMemo(() => {
    if (authMode === "single-user") {
      const missingApiKey = apiKey.trim().length === 0
      return {
        valid: !missingApiKey,
        missingApiKey,
        missingUsername: false,
        missingPassword: false,
        missingMagicEmail: false,
        missingMagicToken: false,
      }
    }

    if (loginMethod === "magic-link") {
      const missingMagicEmail = magicEmail.trim().length === 0
      const missingMagicToken = magicToken.trim().length === 0
      return {
        valid: !missingMagicEmail && !missingMagicToken,
        missingApiKey: false,
        missingUsername: false,
        missingPassword: false,
        missingMagicEmail,
        missingMagicToken,
      }
    }

    const missingUsername = username.trim().length === 0
    const missingPassword = password.trim().length === 0
    return {
      valid: !missingUsername && !missingPassword,
      missingApiKey: false,
      missingUsername,
      missingPassword,
      missingMagicEmail: false,
      missingMagicToken: false,
    }
  }, [authMode, apiKey, username, password, loginMethod, magicEmail, magicToken])

  const showAuthErrors = authTouched && !authValidation.valid

  useEffect(() => {
    setAuthTouched(false)
  }, [authMode, loginMethod])

  // Derive a health-check URL from the user-entered server URL
  const healthCheckUrl = useMemo(() => {
    try {
      const parsed = new URL(serverUrl.trim())
      return `${parsed.origin}/health`
    } catch {
      return "http://localhost:8000/health"
    }
  }, [serverUrl])

  // Derive error messages from errorKind
  const errorHint = useMemo(() => {
    switch (errorKind) {
      case "dns_failed":
        return t(
          "settings:onboarding.errors.dns",
          "Could not find server. Check the URL for typos and make sure the hostname is correct."
        )
      case "refused":
        return t(
          "settings:onboarding.errors.refused",
          `The server is not accepting connections. Is it running? Try: curl ${healthCheckUrl}`
        )
      case "timeout":
        return t(
          "settings:onboarding.errors.timeout",
          "The server did not respond in time. If using Docker, check containers are running: docker ps"
        )
      case "cors_blocked":
        return t(
          "settings:onboarding.errors.cors",
          "Your browser can't reach the server due to security settings. If you manage the server, add your browser's origin to ALLOWED_ORIGINS in the server's .env file. Otherwise, ask your server administrator for help."
        )
      case "ssl_error":
        return t(
          "settings:onboarding.errors.ssl",
          "SSL certificate error. For local development, try http:// instead of https://"
        )
      case "auth_invalid":
        return t(
          "settings:onboarding.errors.auth",
          "API key not accepted. Check your key. Docker users: run make show-api-key in terminal"
        )
      case "server_error":
        return t(
          "settings:onboarding.errors.server",
          "The server returned an error. Check server logs for details: docker compose logs --tail=50"
        )
      default:
        return null
    }
  }, [errorKind, healthCheckUrl, t])

  const handleSendMagicLink = useCallback(async () => {
    if (!magicEmail.trim()) {
      dispatchUi({
        type: "SET_ERROR",
        errorKind: "auth_invalid",
        errorMessage: t(
          "settings:onboarding.magicLink.missingEmail",
          "Enter your email to receive a magic link."
        ),
      })
      return
    }
    setMagicSending(true)
    try {
      await tldwAuth.requestMagicLink(magicEmail.trim())
      setMagicSent(true)
      message.success(
        t(
          "settings:onboarding.magicLink.sent",
          "Magic link sent. Check your inbox."
        )
      )
    } catch (error: unknown) {
      const { status, message: msg } = (() => {
        const err = error as { status?: number; message?: string }
        return { status: err?.status ?? null, message: err?.message ?? null }
      })()
      dispatchUi({
        type: "SET_ERROR",
        errorKind: categorizeConnectionError(status, msg) ?? "auth_invalid",
        errorMessage: mapMultiUserLoginErrorMessage(t, error, "onboarding"),
      })
    } finally {
      setMagicSending(false)
    }
  }, [magicEmail, t, dispatchUi])

  // Handle progressive connection test
  const handleConnect = useCallback(async () => {
    if (!urlValidation.valid) return
    setAuthTouched(true)
    if (!authValidation.valid) {
      dispatchUi({ type: "SET_ERROR", errorKind: null, errorMessage: null })
      return
    }

    dispatchUi({ type: "START_CONNECT" })

    try {
      // Phase 1: Set server URL and check reachability
      requestOptionalHostPermission(serverUrl, (granted, origin) => {
        if (!granted) {
          message.warning(
            t(
              "settings:siteAccessDenied",
              "Permission not granted for {{origin}}",
              { origin }
            )
          )
        }
      })
      await actions.setConfigPartial({ serverUrl })

      // Give a moment for the UI to show "checking"
      await new Promise((r) => setTimeout(r, 300))

      // Phase 2: Test auth
      dispatchUi({
        type: "UPDATE_PROGRESS",
        updater: (p) => ({
          ...p,
          serverReachable: "success",
          authentication: "checking",
        }),
      })

      // Validate auth credentials
      let authResult: ValidationResult | null = null
      if (authMode === "multi-user" && loginMethod === "password" && username && password) {
        authResult = await validateMultiUserAuth(username, password, t)
      } else if (authMode === "multi-user" && loginMethod === "magic-link" && magicToken) {
        authResult = await validateMagicLinkAuth(magicToken, t)
      } else if (authMode === "single-user" && apiKey) {
        // Validate API key before saving
        authResult = await validateApiKey(serverUrl, apiKey, t)
      }

      if (authResult && !authResult.success) {
        dispatchUi({
          type: "UPDATE_PROGRESS",
          updater: (p) => ({
            ...p,
            authentication: "error",
          }),
        })
        if (authResult.errorKind || authResult.error) {
          dispatchUi({
            type: "SET_ERROR",
            errorKind: authResult.errorKind ?? null,
            errorMessage: authResult.error ?? null,
          })
        }

        dispatchUi({ type: "FINISH_CONNECT" })
        return
      }

      await actions.setConfigPartial({
        authMode,
        apiKey: authMode === "single-user" ? apiKey : undefined,
      })

      // Phase 3: Run full connection test (authentication is verified here)
      dispatchUi({
        type: "UPDATE_PROGRESS",
        updater: (p) => ({
          ...p,
          authentication: "checking",
          knowledgeIndex: "idle",
        }),
      })

      try {
        await actions.testConnectionFromOnboarding()
        const latestConnection = useConnectionStore.getState().state
        emitSplashAfterSingleUserAuthSuccess(authMode, latestConnection.isConnected)
      } catch (error) {
        // If full connection test fails, reflect auth error if we're still in that phase
        dispatchUi({
          type: "UPDATE_PROGRESS",
          updater: (p) => ({
            ...p,
            authentication:
              p.authentication === "checking" ? "error" : p.authentication,
          }),
        })
        throw error
      }
    } catch (error) {
      dispatchUi({
        type: "UPDATE_PROGRESS",
        updater: (p) => ({
          ...p,
          // Only set serverReachable to error if we never marked it successful
          serverReachable:
            p.serverReachable === "checking" ? "error" : p.serverReachable,
        }),
      })
      const errorMessage = (error as Error)?.message || null
      const status =
        (error as any)?.status ??
        (error as any)?.response?.status ??
        (error as any)?.statusCode ??
        null
      const kind =
        categorizeConnectionError(status, errorMessage) ??
        ("refused" as ConnectionErrorKind)
      if (kind === "refused" || kind === "timeout" || kind === "dns_failed") {
        requestOptionalHostPermission(serverUrl, (granted, origin) => {
          if (!granted) {
            message.warning(
              t(
                "settings:siteAccessDenied",
                "Permission not granted for {{origin}}",
                { origin }
              )
            )
          }
        })
      }
      dispatchUi({
        type: "SET_ERROR",
        errorKind: kind,
        errorMessage: errorMessage || "Connection failed",
      })
    } finally {
      dispatchUi({ type: "FINISH_CONNECT" })
    }
  }, [
    urlValidation.valid,
    authValidation.valid,
    serverUrl,
    authMode,
    apiKey,
    username,
    password,
    loginMethod,
    magicToken,
    t,
    actions,
    dispatchUi,
  ])

  // React to connection test results using hook state
  useEffect(() => {
    if (!hasRunConnectionTest) return

    const state = connectionState
    const isConnected = state.isConnected

    if (isConnected) {
      const knowledgeOk =
        state.knowledgeStatus === "ready" ||
        state.knowledgeStatus === "indexing"
      const knowledgeEmpty = state.knowledgeStatus === "empty"

      dispatchUi({
        type: "UPDATE_PROGRESS",
        updater: (p) => ({
          ...p,
          authentication: "success",
          knowledgeIndex: knowledgeEmpty
            ? "empty"
            : knowledgeOk
              ? "success"
              : "error",
        }),
      })

      // Show success state
      dispatchUi({ type: "SET_SHOW_SUCCESS", showSuccess: true })
    } else if (!state.isChecking) {
      // Connection failed
      const kind = categorizeConnectionError(
        state.lastStatusCode,
        state.lastError
      )
      if (kind === "refused" || kind === "timeout" || kind === "dns_failed") {
        const promptKey = `${serverUrl}|${state.lastStatusCode ?? ""}|${state.lastError ?? ""}`
        if (hostPermissionPromptKeyRef.current !== promptKey) {
          hostPermissionPromptKeyRef.current = promptKey
          requestOptionalHostPermission(serverUrl, (granted, origin) => {
            if (!granted) {
              message.warning(
                t(
                  "settings:siteAccessDenied",
                  "Permission not granted for {{origin}}",
                  { origin }
                )
              )
            }
          })
        }
      }
      dispatchUi({
        type: "SET_ERROR",
        errorKind: kind,
        errorMessage: state.lastError ?? null,
      })

      if (state.errorKind === "auth") {
        dispatchUi({
          type: "UPDATE_PROGRESS",
          updater: (p) => ({
            ...p,
            serverReachable: "success",
            authentication: "error",
            knowledgeIndex: "idle",
          }),
        })
      } else {
        dispatchUi({
          type: "UPDATE_PROGRESS",
          updater: (p) => ({
            ...p,
            serverReachable: "error",
            authentication: "idle",
            knowledgeIndex: "idle",
          }),
        })
      }
    }
  }, [hasRunConnectionTest, connectionState, dispatchUi, serverUrl, t])

  // Handle demo mode
  const handleDemoMode = useCallback(async () => {
    setDemoEnabled(true)
    actions.setDemoMode()
    try {
      await actions.markFirstRunComplete()
    } catch {
      // ignore persistence errors; demo mode remains active in memory
    }
    onFinish?.()
  }, [setDemoEnabled, actions, onFinish])

  const finalizeOnboarding = useCallback(async (invokeOnFinish: boolean) => {
    try {
      await actions.markFirstRunComplete()
    } catch {
      // ignore persistence errors; UI has already completed onboarding
    }
    if (invokeOnFinish) {
      onFinish?.()
    }
  }, [actions, onFinish])

  const completeOnboarding = useCallback(async () => {
    await finalizeOnboarding(true)
  }, [finalizeOnboarding])

  const finishAndNavigate = useCallback(
    async (path: string, options?: { openQuickIngestIntro?: boolean }) => {
      await finalizeOnboarding(false)
      navigate(path)
      if (options?.openQuickIngestIntro && typeof window !== "undefined") {
        window.setTimeout(() => {
          requestQuickIngestIntro()
        }, QUICK_INGEST_OPEN_DELAY_MS)
      }
    },
    [finalizeOnboarding, navigate]
  )

  const handleOpenIngestFlow = useCallback(async () => {
    try {
      await actions.setUserPersona("researcher")
    } catch (err) {
      console.debug("[OnboardingConnectForm] setUserPersona failed", err)
    }
    try {
      const researcherShortcuts = getDefaultShortcutsForPersona("researcher")
      await setSetting(HEADER_SHORTCUT_SELECTION_SETTING, researcherShortcuts)
    } catch (err) {
      console.debug("[OnboardingConnectForm] Failed to persist researcher shortcuts", err)
    }
    await finishAndNavigate("/media", { openQuickIngestIntro: true })
  }, [actions, finishAndNavigate])

  const handleResearchGetStarted = handleOpenIngestFlow

  const handleOpenMediaFlow = useCallback(async () => {
    await finishAndNavigate("/media")
  }, [finishAndNavigate])

  const handleGoToChat = useCallback(async () => {
    try {
      await openSidepanelForActiveTab()
    } catch (err) {
      console.debug("[OnboardingConnectForm] Failed to open sidepanel", err)
    }
    await finishAndNavigate("/chat")
  }, [finishAndNavigate])

  const handleOpenChatFlow = useCallback(async () => {
    try {
      await actions.setUserPersona("explorer")
    } catch (err) {
      console.debug("[OnboardingConnectForm] setUserPersona failed", err)
    }
    // Explorer persona sees all features — no shortcut filtering needed
    await handleGoToChat()
  }, [actions, handleGoToChat])

  const handleOpenSettingsFlow = useCallback(async () => {
    await finishAndNavigate("/settings/tldw")
  }, [finishAndNavigate])

  const handleOpenFamilyFlow = useCallback(async () => {
    try {
      await actions.setUserPersona("family")
    } catch (err) {
      console.debug("[OnboardingConnectForm] setUserPersona failed", err)
    }
    try {
      const familyShortcuts = getDefaultShortcutsForPersona("family")
      await setSetting(HEADER_SHORTCUT_SELECTION_SETTING, familyShortcuts)
    } catch (err) {
      console.debug("[OnboardingConnectForm] Failed to persist family shortcuts", err)
    }
    await finishAndNavigate("/settings/family-guardrails")
  }, [actions, finishAndNavigate])

  // Copy server command
  const handleCopyCommand = useCallback(
    (cmd: string) => {
      if (!navigator.clipboard?.writeText) {
        message.error(t("common:copyFailed", "Copy failed"))
        return
      }

      navigator.clipboard.writeText(cmd).then(
        () => message.success(t("common:copied", "Copied!")),
        () => message.error(t("common:copyFailed", "Copy failed"))
      )
    },
    [t]
  )

  // Open docs
  const openDocs = useCallback(() => {
    window.open(DOCUMENTATION_URL, "_blank", "noopener,noreferrer")
  }, [])

  const quickIngestLastRun = useQuickIngestStore((s) => s.lastRunSummary)
  const hasSuccessfulIngest =
    quickIngestLastRun.status === "success" && quickIngestLastRun.successCount > 0
  const hasFailedIngest = quickIngestLastRun.status === "error"
  const shouldPrioritizeMedia = hasSuccessfulIngest
  const primarySourcePreview = useMemo(() => {
    const label = quickIngestLastRun.primarySourceLabel
    if (!label) return null
    return label.length > 68 ? `${label.slice(0, 65)}...` : label
  }, [quickIngestLastRun.primarySourceLabel])

  useEffect(() => {
    if (!showSuccess) {
      hasTrackedOnboardingSuccessRef.current = false
      hasTrackedFirstIngestRef.current = false
      return
    }
    if (hasTrackedOnboardingSuccessRef.current) return
    hasTrackedOnboardingSuccessRef.current = true
    void trackOnboardingSuccessReached("setup")
  }, [showSuccess])

  useEffect(() => {
    if (!showSuccess || !hasSuccessfulIngest) return
    if (hasTrackedFirstIngestRef.current) return
    hasTrackedFirstIngestRef.current = true
    void trackOnboardingFirstIngestSuccess({
      successCount: quickIngestLastRun.successCount,
      attemptedAt: quickIngestLastRun.attemptedAt,
      firstMediaId: quickIngestLastRun.firstMediaId,
      primarySourceLabel: quickIngestLastRun.primarySourceLabel
    })
  }, [
    quickIngestLastRun.attemptedAt,
    hasSuccessfulIngest,
    quickIngestLastRun.firstMediaId,
    quickIngestLastRun.primarySourceLabel,
    quickIngestLastRun.successCount,
    showSuccess
  ])

  // Success screen
  if (showSuccess) {
    return (
      <div
        className="mx-auto w-full max-w-2xl rounded-3xl border border-border/70 bg-surface/95 p-8 shadow-lg shadow-black/5 backdrop-blur"
        data-testid="onboarding-success-screen"
        data-ingest-status={quickIngestLastRun.status}
      >
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-success/10">
            <Check className="size-7 text-success" />
          </div>
          <h2 className="text-2xl font-semibold text-text tracking-tight">
            {hasSuccessfulIngest
              ? t(
                  "settings:onboarding.success.titlePostIngest",
                  "Connected and ingest is working. Continue to verification."
                )
              : t(
                  "settings:onboarding.success.title",
                  "You're connected. Start by ingesting one source."
                )}
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            {hasSuccessfulIngest
              ? t(
                  "settings:onboarding.success.subtitlePostIngest",
                  "Great start. Next, verify the result in Media, then ask Chat for a summary."
                )
              : t(
                  "settings:onboarding.success.subtitle",
                  "Follow this sequence to complete your first-value loop: ingest -> verify -> ask."
                )}
          </p>
        </div>

        {/* Intent selector — route to persona-appropriate next step */}
        <div data-testid="intent-selector" className="mb-6">
          {selectedIntent == null ? (
            <>
              <p className="mb-3 text-sm font-medium text-text-muted">
                {t("settings:onboarding.success.intentTitle", "What would you like to do first?")}
              </p>
              <div className="grid gap-3 sm:grid-cols-3">
                <button
                  type="button"
                  onClick={handleOpenChatFlow}
                  className="flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors hover:border-primary/50 hover:bg-surface2"
                >
                  <MessageSquare className="h-5 w-5 text-primary" />
                  <span className="text-sm font-medium text-text">
                    {t("settings:onboarding.success.intentChat", "Chat with AI")}
                  </span>
                  <span className="text-xs text-text-muted">
                    {t("settings:onboarding.success.intentChatDesc", "Start a conversation with your configured models.")}
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => familyGuardrailsAvailable && setSelectedIntent("family")}
                  disabled={!familyGuardrailsAvailable}
                  className={cn(
                    "flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors",
                    familyGuardrailsAvailable
                      ? "hover:border-primary/50 hover:bg-surface2"
                      : "opacity-50 cursor-not-allowed"
                  )}
                >
                  <Shield className={cn("h-5 w-5", familyGuardrailsAvailable ? "text-primary" : "text-text-subtle")} />
                  <span className="text-sm font-medium text-text">
                    {t("settings:onboarding.success.intentFamily", "Set up family safety")}
                  </span>
                  <span className="text-xs text-text-muted">
                    {familyGuardrailsAvailable
                      ? t("settings:onboarding.success.intentFamilyDesc", "Create family profiles and content safety rules.")
                      : t("settings:onboarding.success.intentFamilyUnavailable", "Not available on this server.")}
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setSelectedIntent("research")}
                  className="flex flex-col items-start gap-2 rounded-xl border border-border/60 bg-surface2/30 p-4 text-left transition-colors hover:border-primary/50 hover:bg-surface2"
                >
                  <BookOpen className="h-5 w-5 text-primary" />
                  <span className="text-sm font-medium text-text">
                    {t("settings:onboarding.success.intentResearch", "Research my documents")}
                  </span>
                  <span className="text-xs text-text-muted">
                    {t("settings:onboarding.success.intentResearchDesc", "Import documents and ask questions about them.")}
                  </span>
                </button>
              </div>
            </>
          ) : selectedIntent === "family" ? (
            <IntentSteps
              testId="intent-steps-family"
              icon={<Shield className="h-5 w-5 text-primary" />}
              title={t("settings:onboarding.success.familyStepsTitle", "Your next steps")}
              steps={[
                t("settings:onboarding.success.familyStep1", "Set up family profiles in the Family Guardrails wizard"),
                t("settings:onboarding.success.familyStep2", "Review content safety rules in Content Controls"),
                t("settings:onboarding.success.familyStep3", "Test your rules with a sample message"),
              ]}
              primaryLabel={t("settings:onboarding.success.getStarted", "Get Started")}
              onPrimaryClick={handleOpenFamilyFlow}
              secondaryLabel={t("settings:onboarding.success.skipToChat", "Skip, go to chat")}
              onSecondaryClick={handleGoToChat}
              backLabel={t("settings:onboarding.success.backToChoices", "Back")}
              onBackClick={() => setSelectedIntent(null)}
            />
          ) : (
            <IntentSteps
              testId="intent-steps-research"
              icon={<BookOpen className="h-5 w-5 text-primary" />}
              title={t("settings:onboarding.success.researchStepsTitle", "Your next steps")}
              steps={[
                t("settings:onboarding.success.researchStep1", "Import your first document via Quick Ingest"),
                t("settings:onboarding.success.researchStep2", "Browse your library in Media"),
                t("settings:onboarding.success.researchStep3", "Ask questions about it in Chat"),
              ]}
              primaryLabel={t("settings:onboarding.success.getStarted", "Get Started")}
              onPrimaryClick={handleOpenIngestFlow}
              secondaryLabel={t("settings:onboarding.success.skipToChat", "Skip, go to chat")}
              onSecondaryClick={handleGoToChat}
              backLabel={t("settings:onboarding.success.backToChoices", "Back")}
              onBackClick={() => setSelectedIntent(null)}
            />
          )}
        </div>

        <div className="mb-6 rounded-2xl border border-border/70 bg-surface p-4">
          <div className="mb-3">
            <div className="text-sm font-semibold text-text">
              {t("settings:onboarding.defaults.title", "Set your defaults")}
            </div>
            <p className="text-xs text-text-subtle">
              {t(
                "settings:onboarding.defaults.subtitle",
                "Pick a default provider and model for new chats."
              )}
            </p>
          </div>
          {modelsLoading ? (
            <div className="flex items-center gap-2 text-xs text-text-subtle">
              <Loader2 className="size-3 animate-spin" />
              {t(
                "settings:onboarding.defaults.loading",
                "Loading models..."
              )}
            </div>
          ) : availableModels.length === 0 ? (
            <p className="text-xs text-text-subtle">
              {t(
                "settings:onboarding.defaults.empty",
                "No models are available yet. You can set this later in Settings > Models."
              )}
            </p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-xs font-medium text-text">
                  {t(
                    "settings:onboarding.defaults.providerLabel",
                    "Default API provider"
                  )}
                </label>
                <Select
                  size="large"
                  value={providerSelectValue}
                  onChange={handleProviderChange}
                  options={[
                    {
                      value: "auto",
                      label: t(
                        "settings:onboarding.defaults.providerAuto",
                        "Auto (from model)"
                      ),
                    },
                    ...providerOptions,
                  ]}
                  className="w-full"
                />
                <p className="mt-1 text-[11px] text-text-subtle">
                  {t(
                    "settings:onboarding.defaults.providerHelp",
                    "Leave on Auto to use the provider attached to each model."
                  )}
                </p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-text">
                  {t(
                    "settings:onboarding.defaults.modelLabel",
                    "Default model"
                  )}
                </label>
                <Select
                  showSearch
                  size="large"
                  value={selectedModel || undefined}
                  onChange={handleModelChange}
                  placeholder={t(
                    "settings:onboarding.defaults.modelPlaceholder",
                    "Select a model"
                  )}
                  options={modelOptions}
                  optionFilterProp="label"
                  className="w-full"
                  allowClear
                />
                <p className="mt-1 text-[11px] text-text-subtle">
                  {t(
                    "settings:onboarding.defaults.modelHelp",
                    "This becomes the starting model when you open a new chat."
                  )}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="mb-4 rounded-2xl border border-primary/20 bg-primary/5 px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-primary">
            {t("settings:onboarding.success.guidedFlow", "Recommended first run")}
          </div>
          <ol className="mt-1 space-y-1 text-xs text-text-muted">
            <li>
              {hasSuccessfulIngest
                ? t(
                    "settings:onboarding.success.guidedFlowIngestDone",
                    "1. Ingest one URL, document, or recording. Completed."
                  )
                : t(
                    "settings:onboarding.success.guidedFlowIngest",
                    "1. Ingest one URL, document, or recording."
                  )}
            </li>
            <li>
              {t(
                "settings:onboarding.success.guidedFlowVerify",
                "2. Verify it appears in Media."
              )}
            </li>
            <li>
              {t(
                "settings:onboarding.success.guidedFlowChat",
                "3. Ask Chat to summarize or analyze it."
              )}
            </li>
          </ol>
        </div>

        <div className="grid gap-4">
          <button
            onClick={handleOpenIngestFlow}
            className={cn(
              "flex items-center gap-3 rounded-2xl border p-4 text-left transition-colors",
              shouldPrioritizeMedia
                ? "border-border/70 bg-surface hover:bg-surface2"
                : "border-primary/40 bg-primary/5 hover:bg-primary/10"
            )}
            data-testid="onboarding-success-ingest"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              <Server className="size-5 text-primary" />
            </div>
            <div className="flex-1">
              <div className="font-medium text-text">
                {hasSuccessfulIngest
                  ? t(
                      "settings:onboarding.success.ingestAgain",
                      "Ingest another source"
                    )
                  : hasFailedIngest
                    ? t(
                        "settings:onboarding.success.ingestRetry",
                        "Retry ingest"
                      )
                    : t(
                        "settings:onboarding.success.ingest",
                        "Ingest first source"
                      )}
              </div>
              <div className="text-xs text-text-subtle">
                {hasSuccessfulIngest && primarySourcePreview
                  ? t(
                      "settings:onboarding.success.ingestDescWithSource",
                      "Last successful source: {{source}}",
                      { source: primarySourcePreview }
                    )
                  : hasSuccessfulIngest
                    ? t(
                        "settings:onboarding.success.ingestDescAfterSuccess",
                        "Your latest run succeeded. Add more sources any time."
                      )
                    : hasFailedIngest
                      ? t(
                          "settings:onboarding.success.ingestDescRetry",
                          "Latest ingest run failed. Reopen Quick Ingest and try again."
                        )
                      : t(
                          "settings:onboarding.success.ingestDesc",
                          "Open Quick Ingest and add your first URL, file, or recording."
                        )}
              </div>
            </div>
            <span
              className={cn(
                "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                hasSuccessfulIngest
                  ? "bg-success/15 text-success"
                  : hasFailedIngest
                    ? "bg-warning/20 text-warning"
                    : "bg-primary/15 text-primary"
              )}
              data-testid="onboarding-ingest-status"
            >
              {hasSuccessfulIngest
                ? t("settings:onboarding.success.stateCompleted", "Completed")
                : hasFailedIngest
                  ? t("settings:onboarding.success.stateRetry", "Retry")
                  : t("settings:onboarding.success.stateStart", "Start")}
            </span>
            <ArrowRight className="size-4 text-text-subtle" />
          </button>

          <button
            onClick={handleOpenMediaFlow}
            className={cn(
              "flex items-center gap-3 rounded-2xl border p-4 text-left transition-colors",
              shouldPrioritizeMedia
                ? "border-primary/40 bg-primary/5 hover:bg-primary/10"
                : "border-border/70 bg-surface hover:bg-surface2"
            )}
            data-testid="onboarding-success-media"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              <Server className="size-5 text-primary" />
            </div>
            <div className="flex-1">
              <div className="font-medium text-text">
                {t("settings:onboarding.success.media", "Verify in Media")}
              </div>
              <div className="text-xs text-text-subtle">
                {hasSuccessfulIngest
                  ? t(
                      "settings:onboarding.success.mediaDescReady",
                      "You have {{count}} successful item(s). Confirm they are ready to review.",
                      { count: quickIngestLastRun.successCount }
                    )
                  : t(
                      "settings:onboarding.success.mediaDesc",
                      "Confirm your ingested source appears and is ready to review."
                    )}
              </div>
            </div>
            <ArrowRight className="size-4 text-text-subtle" />
          </button>

          <button
            onClick={handleOpenChatFlow}
            className={cn(
              "flex items-center gap-3 rounded-2xl border border-border/70 bg-surface p-4 text-left transition-colors hover:bg-surface2",
              hasSuccessfulIngest ? "ring-1 ring-transparent" : ""
            )}
            data-testid="onboarding-success-chat"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/10">
              <Sparkles className="size-5 text-accent" />
            </div>
            <div className="flex-1">
              <div className="font-medium text-text">
                {t("settings:onboarding.success.chat", "Ask in Chat")}
              </div>
              <div className="text-xs text-text-subtle">
                {hasSuccessfulIngest
                  ? t(
                      "settings:onboarding.success.chatDescAfterIngest",
                      "Use Chat to summarize or analyze what you just ingested."
                    )
                  : t(
                      "settings:onboarding.success.chatDesc",
                      "Use chat to summarize, extract action items, or query your ingested source."
                    )}
              </div>
            </div>
            <ArrowRight className="size-4 text-text-subtle" />
          </button>

          <button
            onClick={handleOpenSettingsFlow}
            className="flex items-center gap-3 rounded-2xl border border-border/70 bg-surface p-4 text-left transition-colors hover:bg-surface2"
            data-testid="onboarding-success-settings"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              <Server className="size-5 text-primary" />
            </div>
            <div className="flex-1">
              <div className="font-medium text-text">
                {t("settings:onboarding.success.explore", "Explore settings")}
              </div>
              <div className="text-xs text-text-subtle">
                {t(
                  "settings:onboarding.success.exploreDesc",
                  "Adjust models, prompts, and workspace defaults."
                )}
              </div>
            </div>
            <ArrowRight className="size-4 text-text-subtle" />
          </button>
        </div>

        <div className="mt-6 text-center">
          <button
            onClick={completeOnboarding}
            className="text-sm text-text-subtle hover:text-text"
          >
            {t("common:done", "Done")}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto w-full max-w-2xl rounded-3xl border border-border/70 bg-surface/95 p-8 shadow-lg shadow-black/5 backdrop-blur">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-text tracking-tight">
          {t("settings:onboarding.title", "Welcome to tldw Assistant")}
        </h2>
        <p className="mt-2 text-sm text-text-muted">
          {t(
            "settings:onboarding.valueProp",
            "Chat with AI, save web content, and build your personal knowledge base."
          )}
        </p>
      </div>

      {/* Demo Mode - Prominent placement for users without a server */}
      <div className="mb-6 rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-accent/10 to-surface p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary text-white shadow-sm shadow-primary/20">
            <Sparkles className="size-5" />
          </div>
          <div className="min-w-0 sm:flex-1">
            <h3 className="font-medium text-text">
              {t("settings:onboarding.demo.titleNoServer", "No server? Try Demo Mode")}
            </h3>
            <p className="text-xs text-text-muted">
              {t(
                "settings:onboarding.demo.descriptionShort",
                "Explore the extension with sample data - no setup required."
              )}
            </p>
          </div>
          <Button
            type="primary"
            onClick={handleDemoMode}
            className="w-full rounded-full border-0 bg-primary px-4 font-medium text-white hover:bg-primaryStrong sm:w-auto"
          >
            {t("settings:onboarding.demo.buttonTry", "Try Demo")}
          </Button>
        </div>
      </div>

      {/* Divider with "or connect to your server" */}
      <div className="relative mb-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border" />
        </div>
        <div className="relative flex justify-center text-xs uppercase">
          <span className="bg-surface px-2 text-text-subtle">
            {t("settings:onboarding.orConnectServer", "or connect to your server")}
          </span>
        </div>
      </div>

      {/* Server URL */}
      <div className="space-y-4">
        <div>
          <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
            <Server className="size-4" />
            {t("settings:onboarding.serverUrl.label", "Server URL")}
          </label>
          <div className="relative">
            <Input
              ref={urlInputRef}
              data-testid="onboarding-server-url"
              placeholder={t(
                "settings:onboarding.serverUrl.placeholder",
                "http://127.0.0.1:8000"
              )}
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              status={
                serverUrl && !urlValidation.valid ? "error" : undefined
              }
              disabled={isConnecting}
              size="large"
              className="rounded-2xl"
              suffix={
                <span
                  className="inline-flex h-4 w-4 items-center justify-center"
                  aria-hidden={!serverUrl}
                  style={{ visibility: serverUrl ? "visible" : "hidden" }}
                >
                  {serverUrl && urlValidation.valid ? (
                    <Check
                      className="size-4 text-success"
                      aria-label={t("common:valid", "Valid")}
                    />
                  ) : serverUrl && !urlValidation.valid ? (
                    <X
                      className="size-4 text-danger"
                      aria-label={t("common:invalid", "Invalid")}
                    />
                  ) : null}
                </span>
              }
              aria-describedby={serverUrl && !urlValidation.valid ? "url-error" : undefined}
            />
          </div>
          {serverUrl && !urlValidation.valid && (
            <p id="url-error" role="alert" className="mt-1 text-xs text-danger">
              {urlValidation.reason === "protocol"
                ? t(
                    "settings:onboarding.serverUrl.protocolError",
                    "URL must start with http:// or https://"
                  )
                : t(
                    "settings:onboarding.serverUrl.invalidError",
                    "Please enter a valid URL"
                  )}
            </p>
          )}
          {serverUrl && urlValidation.valid && !isConnecting && progress.serverReachable === "idle" && (
            <p className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-text">
              <Check className="size-3.5 text-success" aria-hidden="true" />
              {t("settings:onboarding.serverUrl.validUrl", "URL format is valid. Click Connect to test the connection.")}
            </p>
          )}
        </div>

        {/* Auth Mode Toggle */}
        <div>
          <label className="mb-1.5 block text-sm font-medium text-text">
            {t("settings:onboarding.authMode.label", "Authentication")}
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setAuthMode("single-user")}
              disabled={isConnecting}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                authMode === "single-user"
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border/70 text-text-muted hover:bg-surface2"
              )}
            >
              <Key className="size-4" />
              {t("settings:onboarding.authMode.single", "API Key")}
            </button>
            <button
              type="button"
              onClick={() => setAuthMode("multi-user")}
              disabled={isConnecting || isExtensionRuntime()}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                authMode === "multi-user" && !isExtensionRuntime()
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : isExtensionRuntime()
                    ? "border-border/40 text-text-subtle cursor-not-allowed opacity-60"
                    : "border-border/70 text-text-muted hover:bg-surface2"
              )}
              title={isExtensionRuntime() ? t("settings:onboarding.authMode.extensionApiKeyOnly", "The extension only supports API key authentication") : undefined}
            >
              <User className="size-4" />
              {t("settings:onboarding.authMode.multi", "Login")}
            </button>
          </div>
          {/* Auth-mode-aware contextual hint */}
          <p className="mt-1.5 text-xs text-text-muted" data-testid="onboarding-auth-mode-hint">
            {authMode === "single-user"
              ? t(
                  "settings:onboarding.authMode.singleHint",
                  "Single-user mode: paste your API key to connect. Best for personal or local setups."
                )
              : t(
                  "settings:onboarding.authMode.multiHint",
                  "Multi-user mode: log in with the credentials your administrator provided."
                )}
          </p>
          {/* B2: Multi-user mode notice for extension context */}
          {authMode === "multi-user" && isExtensionRuntime() && (
            <div
              className="mt-2 flex items-start gap-2 rounded-xl border border-amber-300/40 bg-amber-50 px-3 py-2 dark:border-amber-500/30 dark:bg-amber-950/30"
              data-testid="onboarding-multi-user-extension-notice"
              role="note"
            >
              <Info className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
              <p className="text-xs text-amber-800 dark:text-amber-300">
                {t(
                  "settings:onboarding.authMode.multiUserExtensionNotice",
                  "Multi-user mode detected. The browser extension currently supports API key authentication only. Ask your admin for an API key."
                )}
              </p>
            </div>
          )}
        </div>

        {/* Auth Fields — extension always uses API key, even if server is multi-user */}
        {authMode === "single-user" || (authMode === "multi-user" && isExtensionRuntime()) ? (
          <div>
            <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
              <Key className="size-4" />
              {t("settings:onboarding.apiKey.label", "Paste your API key")}
            </label>
            <Input.Password
              data-testid="onboarding-api-key"
              placeholder={t(
                "settings:onboarding.apiKey.placeholder",
                "Enter your API key"
              )}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              disabled={isConnecting}
              size="large"
              className="rounded-2xl"
              status={
                showAuthErrors && authValidation.missingApiKey ? "error" : undefined
              }
              aria-describedby={
                showAuthErrors && authValidation.missingApiKey
                  ? "api-key-error"
                  : undefined
              }
            />
            {showAuthErrors && authValidation.missingApiKey ? (
              <p
                id="api-key-error"
                role="alert"
                className="mt-1 text-xs text-danger"
              >
                {t(
                  "settings:onboarding.apiKeyRequired",
                  "Enter your API key to continue."
                )}
              </p>
            ) : (
              <p className="mt-1 text-xs text-text-subtle">
                {t(
                  "settings:onboarding.apiKeyHelp",
                  "Find your API key by running `make show-api-key` or checking your .env file for SINGLE_USER_API_KEY. Docker quickstart users connect automatically."
                )}
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setLoginMethod("magic-link")}
                disabled={isConnecting}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                  loginMethod === "magic-link"
                    ? "border-primary/40 bg-primary/10 text-primary"
                    : "border-border/70 text-text-muted hover:bg-surface2"
                )}
              >
                <Sparkles className="size-4" />
                {t("settings:onboarding.loginMethod.magic", "Magic link")}
              </button>
              <button
                type="button"
                onClick={() => setLoginMethod("password")}
                disabled={isConnecting}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                  loginMethod === "password"
                    ? "border-primary/40 bg-primary/10 text-primary"
                    : "border-border/70 text-text-muted hover:bg-surface2"
                )}
              >
                <Lock className="size-4" />
                {t("settings:onboarding.loginMethod.password", "Password")}
              </button>
            </div>

            {loginMethod === "magic-link" ? (
              <div className="space-y-3">
                <div>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
                    <User className="size-4" />
                    {t("settings:onboarding.magicLink.email.label", "Email")}
                  </label>
                  <Input
                    placeholder={t(
                      "settings:onboarding.magicLink.email.placeholder",
                      "you@company.com"
                    )}
                    value={magicEmail}
                    onChange={(e) => setMagicEmail(e.target.value)}
                    disabled={isConnecting}
                    size="large"
                    className="rounded-2xl"
                    status={
                      showAuthErrors && authValidation.missingMagicEmail
                        ? "error"
                        : undefined
                    }
                    aria-describedby={
                      showAuthErrors && authValidation.missingMagicEmail
                        ? "magic-email-error"
                        : undefined
                    }
                  />
                  {showAuthErrors && authValidation.missingMagicEmail && (
                    <p
                      id="magic-email-error"
                      role="alert"
                      className="mt-1 text-xs text-danger"
                    >
                      {t(
                        "settings:onboarding.magicLink.emailRequired",
                        "Enter your email to continue."
                      )}
                    </p>
                  )}
                </div>
                <div>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
                    <Sparkles className="size-4" />
                    {t("settings:onboarding.magicLink.token.label", "Magic link token")}
                  </label>
                  <Input
                    placeholder={t(
                      "settings:onboarding.magicLink.token.placeholder",
                      "Paste the token from your email"
                    )}
                    value={magicToken}
                    onChange={(e) => setMagicToken(e.target.value)}
                    disabled={isConnecting}
                    size="large"
                    className="rounded-2xl"
                    status={
                      showAuthErrors && authValidation.missingMagicToken
                        ? "error"
                        : undefined
                    }
                    aria-describedby={
                      showAuthErrors && authValidation.missingMagicToken
                        ? "magic-token-error"
                        : undefined
                    }
                  />
                  {showAuthErrors && authValidation.missingMagicToken && (
                    <p
                      id="magic-token-error"
                      role="alert"
                      className="mt-1 text-xs text-danger"
                    >
                      {t(
                        "settings:onboarding.magicLink.tokenRequired",
                        "Paste the magic link token to continue."
                      )}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handleSendMagicLink}
                    loading={magicSending}
                    disabled={isConnecting}
                  >
                    {magicSent
                      ? t("settings:onboarding.magicLink.resend", "Resend magic link")
                      : t("settings:onboarding.magicLink.send", "Send magic link")}
                  </Button>
                  <p className="text-xs text-text-muted">
                    {t(
                      "settings:onboarding.magicLink.help",
                      "Send a sign-in link, then paste the token from your email."
                    )}
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
                    <User className="size-4" />
                    {t("settings:onboarding.username.label", "Username")}
                  </label>
                  <Input
                    placeholder={t(
                      "settings:onboarding.username.placeholder",
                      "Enter username"
                    )}
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    disabled={isConnecting}
                    size="large"
                    className="rounded-2xl"
                    status={
                      showAuthErrors && authValidation.missingUsername ? "error" : undefined
                    }
                    aria-describedby={
                      showAuthErrors && authValidation.missingUsername
                        ? "username-error"
                        : undefined
                    }
                  />
                  {showAuthErrors && authValidation.missingUsername && (
                    <p
                      id="username-error"
                      role="alert"
                      className="mt-1 text-xs text-danger"
                    >
                      {t(
                        "settings:onboarding.usernameRequired",
                        "Enter your username to continue."
                      )}
                    </p>
                  )}
                </div>
                <div>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-text">
                    <Lock className="size-4" />
                    {t("settings:onboarding.password.label", "Password")}
                  </label>
                  <Input.Password
                    placeholder={t(
                      "settings:onboarding.password.placeholder",
                      "Enter password"
                    )}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={isConnecting}
                    size="large"
                    className="rounded-2xl"
                    status={
                      showAuthErrors && authValidation.missingPassword ? "error" : undefined
                    }
                    aria-describedby={
                      showAuthErrors && authValidation.missingPassword
                        ? "password-error"
                        : undefined
                    }
                  />
                  {showAuthErrors && authValidation.missingPassword && (
                    <p
                      id="password-error"
                      role="alert"
                      className="mt-1 text-xs text-danger"
                    >
                      {t(
                        "settings:onboarding.passwordRequired",
                        "Enter your password to continue."
                      )}
                    </p>
                  )}
                </div>
                <p className="text-xs text-text-subtle" data-testid="onboarding-multi-user-hint">
                  {t(
                    "settings:onboarding.multiUserHelp",
                    "Ask your administrator for your username and password. If you don't have an account yet, contact your server admin."
                  )}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Connection Progress */}
        {(progress.serverReachable !== "idle" || isConnecting) && (
          <div
            className="space-y-2 rounded-2xl border border-primary/20 bg-primary/5 p-4"
            role="status"
            aria-live="polite"
            aria-busy={isConnecting}
          >
            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-primary">
              {isConnecting && <Loader2 className="size-3 animate-spin" />}
              {t("settings:onboarding.progress.title", "Connection Status")}
            </div>
            <ProgressItem
              label={t("settings:onboarding.progress.server", "Server reachable")}
              status={progress.serverReachable}
              statusText={
                progress.serverReachable === "checking"
                  ? t("settings:onboarding.progress.serverChecking", "Checking server...")
                  : progress.serverReachable === "success"
                    ? t("settings:onboarding.progress.serverOk", "Reachable")
                    : undefined
              }
            />
            <ProgressItem
              label={t("settings:onboarding.progress.auth", "Authentication")}
              status={progress.authentication}
              statusText={
                progress.authentication === "checking"
                  ? t("settings:onboarding.progress.authChecking", "Validating credentials...")
                  : progress.authentication === "success"
                    ? t("settings:onboarding.progress.authOk", "Connected!")
                    : undefined
              }
            />
            <ProgressItem
              label={t("settings:onboarding.progress.knowledge", "Knowledge index")}
              status={progress.knowledgeIndex}
            />
          </div>
        )}

        {/* Error display */}
        {errorKind && (
          <div className="rounded-2xl border border-danger/30 bg-danger/10 p-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 size-4 shrink-0 text-danger" />
              <div>
                <div className="text-sm font-medium text-danger">
                  {t("settings:onboarding.connectionFailed", "Connection failed")}
                </div>
                {errorHint && (
                  <p className="mt-1 text-xs text-danger">
                    {errorHint}
                  </p>
                )}
                {errorMessage && errorMessage !== errorHint && (
                  <p className="mt-1 font-mono text-xs text-danger">
                    {errorMessage}
                  </p>
                )}
              </div>
            </div>
            {/* B4: Troubleshooting docs link */}
            <a
              href={TROUBLESHOOTING_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-flex items-center gap-1 text-xs text-danger/80 underline decoration-danger/40 underline-offset-2 hover:text-danger"
              data-testid="onboarding-troubleshooting-link"
            >
              {t(
                "settings:onboarding.errors.troubleshootingLink",
                "Having trouble? See setup guide"
              )}
              <ExternalLink className="size-3" />
            </a>
          </div>
        )}

        {/* Connect Button */}
        <Button
          type="primary"
          size="large"
          block
          onClick={handleConnect}
          data-testid="onboarding-connect"
          disabled={!urlValidation.valid || isConnecting}
          loading={isConnecting}
          icon={isConnecting ? undefined : <ArrowRight className="size-4" />}
          className="!h-12 rounded-full font-medium"
        >
          {isConnecting
            ? t("settings:onboarding.buttons.connecting", "Connecting...")
            : t("settings:onboarding.buttons.connect", "Connect")}
        </Button>

        {/* Retry if error */}
        {errorKind && !isConnecting && (
          <Button
            type="default"
            block
            onClick={handleConnect}
            icon={<RefreshCw className="size-4" />}
            className="!h-11 rounded-full"
          >
            {t("common:retry", "Retry")}
          </Button>
        )}
      </div>

      {/* Advanced: Server commands */}
      <div className="mt-4">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 rounded-full px-2 py-1 text-sm text-text-subtle transition-colors hover:bg-surface2 hover:text-text"
        >
          {showAdvanced ? (
            <ChevronDown className="size-4" />
          ) : (
            <ChevronRight className="size-4" />
          )}
          {t(
            "settings:onboarding.advanced.title",
            "Need to start your server?"
          )}
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-3 text-xs">
            {[
              {
                label: t(
                  "settings:onboarding.startServer.optionLocal",
                  "Run locally with Python"
                ),
                command:
                  "python -m uvicorn tldw_Server_API.app.main:app --reload",
              },
              {
                label: t(
                  "settings:onboarding.startServer.optionDocker",
                  "Run with Docker"
                ),
                command:
                  "docker compose -f Dockerfiles/docker-compose.yml up -d --build",
              },
            ].map((cmd) => (
              <div
                key={cmd.command}
                className="rounded-2xl border border-border/70 bg-surface p-3"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="font-medium text-text">{cmd.label}</span>
                  <Tooltip title={t("common:copy", "Copy")}>
                    <button
                      onClick={() => handleCopyCommand(cmd.command)}
                      className="rounded-full p-1 hover:bg-surface2"
                    >
                      <Copy className="size-3 text-text-subtle" />
                    </button>
                  </Tooltip>
                </div>
                <pre className="overflow-x-auto rounded-xl bg-surface2 px-2 py-2 text-text">
                  <code>{cmd.command}</code>
                </pre>
              </div>
            ))}

            <button
              onClick={openDocs}
              className="inline-flex items-center gap-1 text-primary hover:text-primaryStrong"
            >
              {t("settings:onboarding.serverDocsCta", "View full setup guide")}
              <ExternalLink className="size-3" />
            </button>
          </div>
        )}
      </div>

      {/* Skip link */}
      <div className="mt-6 text-center">
        <button
          onClick={completeOnboarding}
          className="text-sm text-text-muted underline decoration-text-muted/80 underline-offset-4 transition-colors hover:text-text"
        >
          {t("settings:onboarding.buttons.skip", "Skip for now")}
        </button>
      </div>
    </div>
  )
}

export default OnboardingConnectForm

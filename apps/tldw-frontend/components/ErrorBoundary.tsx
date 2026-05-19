import { Component, Fragment, type ReactNode, type ErrorInfo } from "react"
import * as SentryModule from "@sentry/nextjs"
import BackendUnavailableRecovery, {
  type BackendUnavailableRecoveryDetails
} from "@/components/Common/BackendUnavailableRecovery"
import {
  classifyBackendUnreachableError,
  type BackendUnreachableClassification
} from "@/services/backend-unreachable"
import { createSafeStorage } from "@/utils/safe-storage"

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
  backendRecovery?: BackendUnavailableRecoveryDetails | null
  resetKey: number
}

type StoredTldwConfig = {
  serverUrl?: unknown
}

const BACKEND_RECOVERY_FALLBACK_TITLE = "Can't reach your tldw server"
const BACKEND_RECOVERY_FALLBACK_MESSAGE =
  "The web app could not reach your configured tldw server. Check that the server is running and reachable from this browser."

const getStoredServerUrl = (value: unknown): string | undefined => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined
  const { serverUrl } = value as StoredTldwConfig
  if (typeof serverUrl !== "string") return undefined
  const trimmed = serverUrl.trim()
  return trimmed || undefined
}

const toError = (value: unknown, fallbackMessage: string): Error => {
  if (value instanceof Error) return value
  return new Error(fallbackMessage || "Unexpected error")
}

const toBackendRecoveryDetails = (
  classification: BackendUnreachableClassification,
  serverUrl?: string
): BackendUnavailableRecoveryDetails => ({
  title: classification.title || BACKEND_RECOVERY_FALLBACK_TITLE,
  message: classification.message || BACKEND_RECOVERY_FALLBACK_MESSAGE,
  fixHint: classification.fixHint,
  subtype: classification.subtype,
  method: classification.method,
  path: classification.path,
  serverUrl,
  status: classification.status,
  rawMessage: classification.rawMessage,
  source: classification.source,
  recentRequestError: classification.recentRequestError,
  diagnostics: classification.diagnostics
})

export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    hasError: false,
    backendRecovery: null,
    resetKey: 0
  }

  private isUnmounted = false

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    const classification = classifyBackendUnreachableError(error)
    if (classification.kind === "backend_unreachable") {
      return {
        hasError: true,
        error,
        errorInfo: undefined,
        backendRecovery: toBackendRecoveryDetails(classification)
      }
    }

    return {
      hasError: true,
      error,
      errorInfo: undefined,
      backendRecovery: null
    }
  }

  componentDidMount(): void {
    if (typeof window === "undefined") {
      return
    }

    window.addEventListener(
      "unhandledrejection",
      this.handleUnhandledRejection as EventListener
    )
  }

  componentWillUnmount(): void {
    this.isUnmounted = true
    if (typeof window === "undefined") {
      return
    }

    window.removeEventListener(
      "unhandledrejection",
      this.handleUnhandledRejection as EventListener
    )
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught error:", error, info.componentStack)
    this.setState({ errorInfo: info })

    void this.enrichBackendRecovery(error)

    if (typeof window === "undefined") {
      return
    }

    // Use direct @sentry/nextjs import when available, fall back to window.Sentry
    if (SentryModule?.captureException) {
      SentryModule.captureException(error, {
        extra: { componentStack: info.componentStack }
      })
    } else {
      const sentry = (window as unknown as {
        Sentry?: {
          captureException?: (
            err: Error,
            context?: { extra?: Record<string, unknown> }
          ) => void
        }
      }).Sentry
      if (sentry?.captureException) {
        sentry.captureException(error, {
          extra: { componentStack: info.componentStack }
        })
      }
    }

    const analytics = (window as unknown as {
      analytics?: {
        track?: (event: string, properties?: Record<string, unknown>) => void
      }
    }).analytics
    if (analytics?.track) {
      analytics.track("error_boundary", {
        message: error.message,
        componentStack: info.componentStack
      })
    }
  }

  private handleUnhandledRejection = (event: PromiseRejectionEvent): void => {
    const classification = classifyBackendUnreachableError(event.reason)
    if (classification.kind !== "backend_unreachable") {
      return
    }

    event.preventDefault()
    this.setState({
      hasError: true,
      error: toError(event.reason, classification.rawMessage),
      errorInfo: undefined,
      backendRecovery: toBackendRecoveryDetails(classification)
    })

    void this.enrichBackendRecovery(event.reason)
  }

  private async enrichBackendRecovery(reason: unknown): Promise<void> {
    const initialClassification = classifyBackendUnreachableError(reason)
    if (initialClassification.kind !== "backend_unreachable") {
      return
    }

    try {
      const storage = createSafeStorage({ area: "local" })
      const [recentRequestError, config] = await Promise.all([
        storage.get("__tldwLastRequestError").catch(() => null),
        storage.get("tldwConfig").catch(() => null)
      ])

      if (this.isUnmounted) {
        return
      }

      const classification = classifyBackendUnreachableError(reason, {
        recentRequestError
      })
      if (classification.kind !== "backend_unreachable") {
        return
      }

      this.setState((prev) => {
        if (!prev.backendRecovery) {
          return prev
        }

        return {
          ...prev,
          backendRecovery: toBackendRecoveryDetails(
            classification,
            getStoredServerUrl(config)
          )
        }
      })
    } catch {
      // Keep the initial recovery screen even if storage enrichment fails.
    }
  }

  handleReset = () => {
    this.setState((prev) => ({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
      backendRecovery: null,
      resetKey: prev.resetKey + 1
    }))
  }

  private handleReload = () => {
    window.location.reload()
  }

  private openDiagnostics = () => {
    window.location.assign("/settings/health")
  }

  private openSettings = () => {
    window.location.assign("/settings/tldw")
  }

  render() {
    if (!this.state.hasError) {
      return <Fragment key={this.state.resetKey}>{this.props.children}</Fragment>
    }

    if (this.state.backendRecovery) {
      return (
        <BackendUnavailableRecovery
          details={this.state.backendRecovery}
          onRetry={this.handleReset}
          onReload={this.handleReload}
          onOpenDiagnostics={this.openDiagnostics}
          onOpenSettings={this.openSettings}
        />
      )
    }

    const showErrorDetails = process.env.NODE_ENV !== "production"

    return (
      <div className="min-h-screen bg-bg px-4 py-12" data-testid="error-boundary">
        <div className="mx-auto max-w-lg rounded-lg bg-surface p-6 text-center shadow">
          <h1 className="text-2xl font-semibold text-text">Something went wrong</h1>
          <p className="mt-2 text-sm text-text-muted">
            An unexpected error occurred. Try again or reload the page.
          </p>
          {showErrorDetails && this.state.error?.message && (
            <p className="mt-4 text-sm text-danger">{this.state.error.message}</p>
          )}
          {!showErrorDetails && (
            <p className="mt-4 text-sm text-text-muted">Something went wrong.</p>
          )}
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button
              type="button"
              onClick={this.handleReset}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primaryStrong"
            >
              Try again
            </button>
            <button
              type="button"
              onClick={this.handleReload}
              className="rounded-md border border-border px-4 py-2 text-sm font-medium text-text hover:bg-surface2"
            >
              Reload page
            </button>
          </div>
        </div>
      </div>
    )
  }
}

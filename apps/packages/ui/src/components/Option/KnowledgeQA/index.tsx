/**
 * KnowledgeQA - Research-grade question-answering interface
 *
 * A Perplexity-style Q&A experience combining discoverability with
 * the full power of the RAG pipeline.
 */

import React, { useEffect, useMemo, useRef, useState } from "react"
import { KnowledgeQAProvider, useKnowledgeQA } from "./KnowledgeQAProvider"
import { KnowledgeQALayout } from "./layout/KnowledgeQALayout"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { WifiOff, AlertCircle } from "lucide-react"
import {
  getRetryCountdownSeconds,
  KNOWLEDGE_QA_RETRY_INTERVAL_MS,
  KNOWLEDGE_QA_RETRY_TICK_MS,
} from "./retryScheduler"
import { useLocation, useNavigate } from "react-router-dom"

const LazySettingsPanel = React.lazy(() =>
  import("./SettingsPanel").then((module) => ({ default: module.SettingsPanel })),
)
const LazyExportDialog = React.lazy(() =>
  import("./ExportDialog").then((module) => ({ default: module.ExportDialog })),
)

const ROUTE_HYDRATION_RETRY_DELAY_MS = 1500
const ROUTE_HYDRATION_MAX_RETRIES = 2

function normalizeThreadId(rawValue: string | null | undefined): string | null {
  if (typeof rawValue !== "string") return null
  const candidate = rawValue.trim()
  if (candidate.length === 0) return null
  try {
    const decoded = decodeURIComponent(candidate).trim()
    return decoded.length > 0 ? decoded : null
  } catch {
    return candidate
  }
}

function normalizeShareToken(rawValue: string | null | undefined): string | null {
  if (typeof rawValue !== "string") return null
  const candidate = rawValue.trim()
  if (candidate.length === 0) return null
  try {
    const decoded = decodeURIComponent(candidate).trim()
    return decoded.length > 0 ? decoded : null
  } catch {
    return candidate
  }
}

// Main page component (inner, uses context)
function KnowledgeQAContent() {
  const {
    settingsPanelOpen,
    setSettingsPanelOpen,
    currentThreadId,
    selectThread,
    selectSharedThread,
  } = useKnowledgeQA()
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [retryNowMs, setRetryNowMs] = useState(() => Date.now())
  const [routeHydrationRetryVersion, setRouteHydrationRetryVersion] = useState(0)
  const routeHydratedThreadRef = useRef<string | null>(null)
  const routeHydratedShareRef = useRef<string | null>(null)
  const routeRetryThreadTargetRef = useRef<string | null>(null)
  const routeRetryShareTargetRef = useRef<string | null>(null)
  const routeHydrationThreadRetriesRef = useRef(0)
  const routeHydrationShareRetriesRef = useRef(0)
  const routeHydrationThreadTimeoutRef = useRef<number | null>(null)
  const routeHydrationShareTimeoutRef = useRef<number | null>(null)
  const location = useLocation()
  const navigate = useNavigate()
  const online = useServerOnline(KNOWLEDGE_QA_RETRY_INTERVAL_MS)
  const { capabilities, loading: capabilitiesLoading, refresh: refreshCapabilities } =
    useServerCapabilities()
  const { checkOnce } = useConnectionActions()
  const { isChecking, lastCheckedAt } = useConnectionState()
  const { uxState } = useConnectionUxState()

  // Check if RAG is supported
  const hasRag = capabilities?.hasRag ?? true
  const retryCountdownSeconds = useMemo(
    () =>
      getRetryCountdownSeconds({
        lastAttemptAt: lastCheckedAt,
        now: retryNowMs,
        retryIntervalMs: KNOWLEDGE_QA_RETRY_INTERVAL_MS,
      }),
    [lastCheckedAt, retryNowMs]
  )
  const routeThreadId = useMemo(() => {
    const pathMatch = location.pathname.match(/\/knowledge\/thread\/([^/?#]+)/i)
    if (pathMatch?.[1]) {
      return normalizeThreadId(pathMatch[1])
    }
    const searchParams = new URLSearchParams(location.search)
    return normalizeThreadId(searchParams.get("thread"))
  }, [location.pathname, location.search])
  const routeShareToken = useMemo(() => {
    const pathMatch = location.pathname.match(/\/knowledge\/shared\/([^/?#]+)/i)
    if (pathMatch?.[1]) {
      return normalizeShareToken(pathMatch[1])
    }
    const searchParams = new URLSearchParams(location.search)
    return normalizeShareToken(searchParams.get("share"))
  }, [location.pathname, location.search])

  useEffect(() => {
    if (online) {
      return
    }
    setRetryNowMs(Date.now())
    const interval = window.setInterval(
      () => setRetryNowMs(Date.now()),
      KNOWLEDGE_QA_RETRY_TICK_MS
    )
    return () => window.clearInterval(interval)
  }, [online])

  useEffect(
    () => () => {
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
      }
      if (routeHydrationShareTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationShareTimeoutRef.current)
      }
    },
    []
  )

  useEffect(() => {
    if (!routeShareToken) {
      routeHydratedShareRef.current = null
      routeRetryShareTargetRef.current = null
      routeHydrationShareRetriesRef.current = 0
      if (routeHydrationShareTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationShareTimeoutRef.current)
        routeHydrationShareTimeoutRef.current = null
      }
      return
    }
    if (routeRetryShareTargetRef.current !== routeShareToken) {
      routeRetryShareTargetRef.current = routeShareToken
      routeHydrationShareRetriesRef.current = 0
      if (routeHydrationShareTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationShareTimeoutRef.current)
        routeHydrationShareTimeoutRef.current = null
      }
    }
    if (routeHydratedShareRef.current === routeShareToken) {
      return
    }
    routeHydratedShareRef.current = routeShareToken
    let cancelled = false
    void (async () => {
      const loaded = await selectSharedThread(routeShareToken)
      if (cancelled || routeHydratedShareRef.current !== routeShareToken) {
        return
      }
      if (loaded) {
        routeHydrationShareRetriesRef.current = 0
        if (routeHydrationShareTimeoutRef.current != null) {
          window.clearTimeout(routeHydrationShareTimeoutRef.current)
          routeHydrationShareTimeoutRef.current = null
        }
        return
      }
      routeHydratedShareRef.current = null
      if (routeHydrationShareRetriesRef.current >= ROUTE_HYDRATION_MAX_RETRIES) {
        return
      }
      routeHydrationShareRetriesRef.current += 1
      if (routeHydrationShareTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationShareTimeoutRef.current)
      }
      routeHydrationShareTimeoutRef.current = window.setTimeout(() => {
        routeHydrationShareTimeoutRef.current = null
        setRouteHydrationRetryVersion((version) => version + 1)
      }, ROUTE_HYDRATION_RETRY_DELAY_MS)
    })()
    return () => {
      cancelled = true
    }
  }, [routeShareToken, routeHydrationRetryVersion, selectSharedThread])

  useEffect(() => {
    if (routeShareToken) {
      routeHydratedThreadRef.current = null
      routeRetryThreadTargetRef.current = null
      routeHydrationThreadRetriesRef.current = 0
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
        routeHydrationThreadTimeoutRef.current = null
      }
      return
    }
    if (!routeThreadId) {
      routeHydratedThreadRef.current = null
      routeRetryThreadTargetRef.current = null
      routeHydrationThreadRetriesRef.current = 0
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
        routeHydrationThreadTimeoutRef.current = null
      }
      return
    }
    if (routeRetryThreadTargetRef.current !== routeThreadId) {
      routeRetryThreadTargetRef.current = routeThreadId
      routeHydrationThreadRetriesRef.current = 0
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
        routeHydrationThreadTimeoutRef.current = null
      }
    }
    if (currentThreadId === routeThreadId) {
      routeHydratedThreadRef.current = routeThreadId
      routeHydrationThreadRetriesRef.current = 0
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
        routeHydrationThreadTimeoutRef.current = null
      }
      return
    }
    if (routeHydratedThreadRef.current === routeThreadId) {
      return
    }
    routeHydratedThreadRef.current = routeThreadId
    let cancelled = false
    void (async () => {
      const loaded = await selectThread(routeThreadId)
      if (cancelled || routeHydratedThreadRef.current !== routeThreadId) {
        return
      }
      if (loaded) {
        routeHydrationThreadRetriesRef.current = 0
        if (routeHydrationThreadTimeoutRef.current != null) {
          window.clearTimeout(routeHydrationThreadTimeoutRef.current)
          routeHydrationThreadTimeoutRef.current = null
        }
        return
      }
      routeHydratedThreadRef.current = null
      if (routeHydrationThreadRetriesRef.current >= ROUTE_HYDRATION_MAX_RETRIES) {
        return
      }
      routeHydrationThreadRetriesRef.current += 1
      if (routeHydrationThreadTimeoutRef.current != null) {
        window.clearTimeout(routeHydrationThreadTimeoutRef.current)
      }
      routeHydrationThreadTimeoutRef.current = window.setTimeout(() => {
        routeHydrationThreadTimeoutRef.current = null
        setRouteHydrationRetryVersion((version) => version + 1)
      }, ROUTE_HYDRATION_RETRY_DELAY_MS)
    })()
    return () => {
      cancelled = true
    }
  }, [currentThreadId, routeHydrationRetryVersion, routeShareToken, routeThreadId, selectThread])

  const handleRetryConnection = () => {
    void checkOnce()
  }

  const handleRetryCapabilities = () => {
    void refreshCapabilities()
  }

  // -----------------------------------------------------------------------
  // Offline-state priority chain (checked in order of severity):
  //   1. Unconfigured  -- needs first-time setup before anything works
  //   2. Auth error    -- server reachable but credentials missing/invalid
  //   3. Unreachable   -- server configured but not responding
  //   4. Generic offline (fallback for any other !online state)
  //   5. No RAG        -- server online but embedding model not configured
  // -----------------------------------------------------------------------
  if (!online && uxState !== "testing") {
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <WifiOff className="w-16 h-16 mx-auto mb-4 text-text-muted" />
            <h2 className="text-xl font-semibold mb-2">
              Setup Required
            </h2>
            <p className="text-text-muted mb-4">
              Complete the server setup to start searching your documents.
            </p>
            <button
              type="button"
              onClick={() => navigate("/")}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Finish Setup
            </button>
          </div>
        </div>
      )
    }

    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <WifiOff className="w-16 h-16 mx-auto mb-4 text-text-muted" />
            <h2 className="text-xl font-semibold mb-2">
              Add your credentials to use Knowledge QA
            </h2>
            <p className="text-text-muted mb-4">
              Your server is reachable, but Knowledge QA needs valid credentials before it can load.
            </p>
            <button
              type="button"
              onClick={() => navigate("/settings/tldw")}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Open Settings
            </button>
          </div>
        </div>
      )
    }

    if (uxState === "error_unreachable") {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <WifiOff className="w-16 h-16 mx-auto mb-4 text-text-muted" />
            <h2 className="text-xl font-semibold mb-2">Can't reach your tldw server right now</h2>
            <p className="text-text-muted mb-3">
              Your server settings are saved, but Knowledge QA cannot reach the tldw server right now.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={handleRetryConnection}
                disabled={isChecking}
                className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {isChecking ? "Checking connection..." : "Retry connection"}
              </button>
              <button
                type="button"
                onClick={() => navigate("/settings/health")}
                className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
              >
                Health & diagnostics
              </button>
            </div>
            <p className="mt-2 text-xs text-text-muted">
              Retrying automatically in {retryCountdownSeconds}s...
            </p>
          </div>
        </div>
      )
    }
  }

  // Offline state
  if (!online) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md">
          <WifiOff className="w-16 h-16 mx-auto mb-4 text-text-muted" />
          <h2 className="text-xl font-semibold mb-2">Server Offline</h2>
          <p className="text-text-muted mb-3">
            Cannot connect to the server. Please ensure the tldw server is running
            and try again.
          </p>
          <button
            type="button"
            onClick={handleRetryConnection}
            disabled={isChecking}
            className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isChecking ? "Checking connection..." : "Retry connection"}
          </button>
          <p className="mt-2 text-xs text-text-muted">
            Retrying automatically in {retryCountdownSeconds}s...
          </p>
        </div>
      </div>
    )
  }

  // No RAG support
  if (!capabilitiesLoading && !hasRag) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md">
          <AlertCircle className="w-16 h-16 mx-auto mb-4 text-text-muted" />
          <h2 className="text-xl font-semibold mb-2">Document Search Not Set Up</h2>
          <p className="text-text-muted mb-2">
            Your server needs an embedding model configured before you can search
            your documents here.
          </p>
          <p className="text-text-muted mb-4">
            Set up document search in your server configuration, then restart the
            server and check again.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            <button
              type="button"
              onClick={handleRetryCapabilities}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Check again
            </button>
            <a
              href="https://github.com/rmusser01/tldw_server2#readme"
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-md border border-primary/40 bg-primary/10 text-primary hover:bg-primary/15 transition-colors"
            >
              Setup guide
            </a>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      data-testid="knowledge-page-root"
      className="relative flex h-full w-full min-w-0 flex-1"
    >
      <KnowledgeQALayout onExportClick={() => setExportDialogOpen(true)} />

      {/* Settings panel (drawer) */}
      {settingsPanelOpen ? (
        <React.Suspense fallback={null}>
          <LazySettingsPanel
            open={settingsPanelOpen}
            onClose={() => setSettingsPanelOpen(false)}
          />
        </React.Suspense>
      ) : null}

      {/* Export dialog */}
      {exportDialogOpen ? (
        <React.Suspense fallback={null}>
          <LazyExportDialog
            open={exportDialogOpen}
            onClose={() => setExportDialogOpen(false)}
          />
        </React.Suspense>
      ) : null}
    </div>
  )
}

// Export the wrapped component
export function KnowledgeQA() {
  return (
    <KnowledgeQAProvider>
      <KnowledgeQAContent />
    </KnowledgeQAProvider>
  )
}

// Also export individual components for flexibility
export { KnowledgeQAProvider, useKnowledgeQA } from "./KnowledgeQAProvider"
export { SearchBar } from "./SearchBar"
export { AnswerPanel } from "./AnswerPanel"
export { SearchDetailsPanel } from "./SearchDetailsPanel"
export { ConversationThread } from "./ConversationThread"
export { SourceCard } from "./SourceCard"
export { SourceList } from "./SourceList"
export { FollowUpInput } from "./FollowUpInput"
export { HistorySidebar } from "./HistorySidebar"
export { SettingsPanel } from "./SettingsPanel"
export { ExportDialog } from "./ExportDialog"
export { KnowledgeQALayout } from "./layout/KnowledgeQALayout"
export type * from "./types"

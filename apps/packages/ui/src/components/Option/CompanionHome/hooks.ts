import React from "react"
import { useCallerCapabilities } from "@/hooks/useCallerCapabilities"
import { callerCapabilityErrorStatus } from "@/services/caller-capabilities"
import { isServicePromptScopeUnresolvedError, loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { createSafeStorage } from "@/utils/safe-storage"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"

import {
  fetchPersonalizationProfile,
  type PersonalizationProfile,
  updatePersonalizationOptIn
} from "@/services/companion"
import {
  fetchCompanionHomeSnapshot,
  type CompanionHomeSnapshot,
  type CompanionHomeSurface
} from "@/services/companion-home"
import { listNotifications } from "@/services/notifications"
import {
  listScheduledTaskResults,
  listScheduledTasks,
  type ScheduledTaskReadOptions,
  type ScheduledTaskResultResponse
} from "@/services/scheduled-tasks-control-plane"
import {
  DEFAULT_COMPANION_HOME_LAYOUT,
  loadCompanionHomeLayout,
  saveCompanionHomeLayout,
  type CompanionHomeLayoutCard
} from "@/store/companion-home-layout"
import {
  buildScheduledTaskAutomationHomeItems,
  buildScheduledTaskAutomationHomeItemsFromNotifications,
  mapScheduledTaskApiResults,
  mergeScheduledTaskAutomationHomeItems,
  projectScheduledTaskResults,
  type ScheduledTaskAutomationHomeItem
} from "../ScheduledTasks/scheduled-task-results"

export const createEmptySnapshot = (
  surface: CompanionHomeSurface
): CompanionHomeSnapshot => ({
  surface,
  inbox: [],
  needsAttention: [],
  resumeWork: [],
  goalsFocus: [],
  recentActivity: [],
  readingQueue: [],
  degradedSources: ["workspace", "reading", "notes"],
  summary: {
    activityCount: 0,
    inboxCount: 0,
    needsAttentionCount: 0,
    resumeWorkCount: 0
  }
})

type UseCompanionHomeDataArgs = {
  surface: CompanionHomeSurface
  capsLoading: boolean
  hasPersonalization: boolean
  onPersonalizationEnabled?: () => void
}

type UseCompanionHomeDataResult = {
  snapshot: CompanionHomeSnapshot | null
  profile: PersonalizationProfile | null
  profileLoaded: boolean
  loading: boolean
  error: string | null
  enablingCompanion: boolean
  refresh: () => void
  enableCompanion: () => Promise<void>
}

export const useCompanionHomeData = ({
  surface,
  capsLoading,
  hasPersonalization,
  onPersonalizationEnabled
}: UseCompanionHomeDataArgs): UseCompanionHomeDataResult => {
  const [snapshot, setSnapshot] = React.useState<CompanionHomeSnapshot | null>(null)
  const [profile, setProfile] = React.useState<PersonalizationProfile | null>(null)
  const [profileLoaded, setProfileLoaded] = React.useState(false)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [enablingCompanion, setEnablingCompanion] = React.useState(false)
  const [refreshToken, setRefreshToken] = React.useState(0)

  React.useEffect(() => {
    if (capsLoading) return

    let cancelled = false
    setLoading(true)
    setError(null)
    setProfileLoaded(false)

    if (!hasPersonalization) {
      setSnapshot(createEmptySnapshot(surface))
      setProfile(null)
      setProfileLoaded(true)
      setLoading(false)
      return () => {
        cancelled = true
      }
    }

    const load = async () => {
      const [snapshotResult, profileResult] = await Promise.allSettled([
        fetchCompanionHomeSnapshot(surface),
        hasPersonalization ? fetchPersonalizationProfile() : Promise.resolve(null)
      ])

      if (cancelled) {
        return
      }

      if (snapshotResult.status === "fulfilled") {
        setSnapshot(snapshotResult.value)
      } else {
        setSnapshot(createEmptySnapshot(surface))
        setError("Companion Home is partially unavailable right now.")
      }

      if (profileResult.status === "fulfilled") {
        setProfile(profileResult.value)
        setProfileLoaded(true)
      } else {
        setProfile(null)
        setError((current) => current || "Companion setup status could not be loaded.")
      }

      setLoading(false)
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [capsLoading, hasPersonalization, refreshToken, surface])

  const refresh = React.useCallback(() => {
    setRefreshToken((value) => value + 1)
  }, [])

  const enableCompanion = React.useCallback(async () => {
    setEnablingCompanion(true)
    setError(null)
    try {
      const nextProfile = await updatePersonalizationOptIn(true)
      setProfile(nextProfile)
      onPersonalizationEnabled?.()
      refresh()
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to enable companion personalization."
      )
    } finally {
      setEnablingCompanion(false)
    }
  }, [onPersonalizationEnabled, refresh])

  return {
    snapshot,
    profile,
    profileLoaded,
    loading,
    error,
    enablingCompanion,
    refresh,
    enableCompanion
  }
}

type UseScheduledTaskHomeSignalsArgs = {
  enabled: boolean
}

type UseScheduledTaskHomeSignalsResult = {
  items: ScheduledTaskAutomationHomeItem[]
  loading: boolean
  partial: boolean
  error: string | null
  sourceStates: AutomationHomeSourceStates
  refresh: () => void
}

export type AutomationHomeSourceState = "ready" | "denied" | "unsupported" | "unknown" | "error"
export type AutomationHomeSourceStates = Record<"tasks" | "results" | "notifications", AutomationHomeSourceState>

const unknownAutomationSources: AutomationHomeSourceStates = {
  tasks: "unknown", results: "unknown", notifications: "unknown"
}

const automationFailureState = (error: unknown): AutomationHomeSourceState => {
  const status = callerCapabilityErrorStatus(error)
  if (status === 403) return "denied"
  if (status === 404 || status === 405 || status === 501) return "unsupported"
  if (status === 401 || status === 410 || status === 412) return "unknown"
  return "error"
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isHomeVisibleScheduledTaskResult = (
  result: ScheduledTaskResultResponse
): boolean => {
  const visibility = result.visibility_destination
  if (!isRecord(visibility)) return true
  return visibility.home !== false
}

export const useScheduledTaskHomeSignals = ({
  enabled
}: UseScheduledTaskHomeSignalsArgs): UseScheduledTaskHomeSignalsResult => {
  const caller = useCallerCapabilities()
  // The callback carries discovery's authority generation, including A→B→A.
  const owner = caller.refreshAfterForbidden
  const refreshCaller = caller.refresh
  const [result, setResult] = React.useState<{
    owner: typeof owner
    items: ScheduledTaskAutomationHomeItem[]
    sourceStates: AutomationHomeSourceStates
    partial: boolean
    error: string | null
  } | null>(null)
  const controllerRef = React.useRef<AbortController | null>(null)
  const [refreshToken, setRefreshToken] = React.useState(0)

  React.useEffect(() => {
    const invalidate = () => {
      controllerRef.current?.abort()
      setResult(null)
    }
    const configChanged = (event: Event) => {
      const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
      if (detail?.authorityChanged !== false || detail.refreshSessionInvalidated) invalidate()
    }
    const storage = createSafeStorage({ area: "local" })
    const watchers = { tldwConfig: invalidate, tldwCookieSessionConfig: invalidate, tldwRefreshRotation: invalidate }
    storage.watch(watchers)
    window.addEventListener("tldw:auth-credentials-changed", invalidate)
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    window.addEventListener("tldw:config-updated", configChanged)
    return () => {
      controllerRef.current?.abort()
      storage.unwatch(watchers)
      window.removeEventListener("tldw:auth-credentials-changed", invalidate)
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }, [])

  React.useEffect(() => {
    setResult(null)
    if (!enabled || caller.loading) return
    const controller = new AbortController()
    controllerRef.current = controller
    let lease: ServicePromptSnapshot | undefined
    const sourceStates: AutomationHomeSourceStates = {
      tasks: caller.scheduledTasks === "denied" ? "denied" : "unknown",
      results: caller.scheduledTasks === "denied" ? "denied" : "unknown",
      notifications: caller.notifications === "denied" ? "denied" : "unknown"
    }

    const load = async () => {
      let options: ScheduledTaskReadOptions
      try {
        if (Object.values(sourceStates).every(state => state === "denied")) {
          setResult({ owner, items: [], sourceStates, partial: true, error: null })
          return
        }
        lease = await loadServicePromptSnapshot([], { signal: controller.signal })
        controller.signal.throwIfAborted()
        const scope = lease.requestScope
        const currentConfig = await tldwClient.ensureConfigForRequest(true)
        controller.signal.throwIfAborted()
        const currentScopeKey = `${buildChatSurfaceScopeKeyFromConfig(currentConfig, { userId: caller.userId })}:${currentConfig.authSource ?? "manual"}`
        if ((caller.scopeKey !== null && caller.scopeKey !== currentScopeKey) ||
          (scope.userId !== null && caller.userId !== null && String(scope.userId) !== String(caller.userId))) {
          throw Object.assign(new Error("Automation account changed"), { status: 412 })
        }
        const expectedUserId = scope.userId ?? caller.userId
        options = {
          abortSignal: lease.scopeSignal,
          servicePromptConfig: { ...scope.config, expectedUserId },
          ...(expectedUserId !== null ? { headers: { "X-TLDW-Expected-User-ID": String(expectedUserId) } } : {}),
          suppressBackendUnavailableEvent: true,
          expectedStatuses: [401, 403, 404, 405, 410, 501]
        }
      } catch (error) {
        if (controller.signal.aborted) return
        // Failure to verify the owner says nothing about that owner's entitlement.
        const state = !isServicePromptScopeUnresolvedError(error) && automationFailureState(error) === "error" ? "error" : "unknown"
        for (const key of ["tasks", "results", "notifications"] as const) {
          if (sourceStates[key] !== "denied") sourceStates[key] = state
        }
        setResult({ owner, items: [], sourceStates, partial: true,
          error: state === "error" ? "Automation signals unavailable" : null })
        return
      }
      const [tasksResult, notificationsResult, resultsResult] = await Promise.allSettled([
        caller.scheduledTasks === "denied" ? Promise.resolve(null) : listScheduledTasks(options),
        caller.notifications === "denied" ? Promise.resolve(null) : listNotifications({ limit: 50 }, options),
        caller.scheduledTasks === "denied" ? Promise.resolve(null) : listScheduledTaskResults({ limit: 50 }, options)
      ])

      if (controller.signal.aborted || lease.scopeSignal.aborted) {
        return
      }
      const outcomes = { tasks: tasksResult, results: resultsResult, notifications: notificationsResult }
      for (const key of ["tasks", "results", "notifications"] as const) {
        const outcome = outcomes[key]
        if (sourceStates[key] !== "denied") {
          sourceStates[key] = outcome.status === "fulfilled" ? "ready" : automationFailureState(outcome.reason)
        }
      }

      const normalizedResultItems =
        resultsResult.status === "fulfilled" && resultsResult.value
          ? buildScheduledTaskAutomationHomeItems(
              mapScheduledTaskApiResults(
                resultsResult.value.items.filter(isHomeVisibleScheduledTaskResult),
                { capabilityMode: "normalized_results_read" }
              )
            )
          : []
      const projectedResults =
        tasksResult.status === "fulfilled" && tasksResult.value
          ? projectScheduledTaskResults(tasksResult.value?.items ?? [])
          : []
      const projectedItems =
        tasksResult.status === "fulfilled" && tasksResult.value
          ? buildScheduledTaskAutomationHomeItems(
              resultsResult.status === "fulfilled" && resultsResult.value
                ? projectedResults.filter((result) => result.owner !== "scheduled_tasks")
                : projectedResults
            )
          : []
      const notificationItems =
        notificationsResult.status === "fulfilled" && notificationsResult.value
          ? buildScheduledTaskAutomationHomeItemsFromNotifications(
              notificationsResult.value.items
            )
          : []
      const nextPartial =
        Object.values(sourceStates).some(state => state !== "ready") ||
        (tasksResult.status === "fulfilled" && Boolean(tasksResult.value?.partial))

      let nextError: string | null = null
      if (sourceStates.tasks === "error") {
        nextError =
          normalizedResultItems.length > 0 || notificationItems.length > 0
            ? "Some scheduled-task signals could not be loaded."
            : "Automation signals unavailable"
      } else if (sourceStates.notifications === "error") {
        nextError = "Recent automation notifications could not be loaded."
      } else if (sourceStates.results === "error") {
        nextError = "Scheduled-task results could not be loaded."
      } else if (tasksResult.status === "fulfilled" && tasksResult.value?.partial) {
        nextError = "Some scheduled-task sources are temporarily unavailable."
      }

      setResult({ owner, sourceStates, partial: nextPartial, error: nextError,
        items: mergeScheduledTaskAutomationHomeItems([
          normalizedResultItems,
          projectedItems,
          notificationItems
        ])
      })
    }

    void load().finally(() => {
      if (controller.signal.aborted) lease?.release()
    })

    return () => {
      controller.abort()
      lease?.release()
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }, [enabled, refreshToken, caller.loading, caller.scheduledTasks, caller.notifications, caller.userId, caller.scopeKey, owner])

  const refresh = React.useCallback(() => {
    void refreshCaller()
    setRefreshToken((value) => value + 1)
  }, [refreshCaller])

  const current = enabled && !caller.loading && result?.owner === owner ? result : null
  return {
    items: current?.items ?? [],
    loading: enabled && !current,
    partial: current?.partial ?? false,
    error: current?.error ?? null,
    sourceStates: current?.sourceStates ?? unknownAutomationSources,
    refresh
  }
}

type UseCompanionHomeLayoutResult = {
  layout: CompanionHomeLayoutCard[] | null
  updateLayout: (nextLayout: CompanionHomeLayoutCard[]) => void
}

export const useCompanionHomeLayout = (
  surface: CompanionHomeSurface
): UseCompanionHomeLayoutResult => {
  const [layout, setLayout] = React.useState<CompanionHomeLayoutCard[] | null>(null)
  const layoutLoadRequestRef = React.useRef(0)
  const layoutMutationVersionRef = React.useRef(0)

  React.useEffect(() => {
    let cancelled = false
    const requestId = layoutLoadRequestRef.current + 1
    const mutationVersionAtRequest = layoutMutationVersionRef.current

    layoutLoadRequestRef.current = requestId
    setLayout(null)

    const loadLayout = async () => {
      try {
        const nextLayout = await loadCompanionHomeLayout(surface)
        if (
          !cancelled &&
          layoutLoadRequestRef.current === requestId &&
          layoutMutationVersionRef.current === mutationVersionAtRequest
        ) {
          setLayout(nextLayout)
        }
      } catch {
        if (
          !cancelled &&
          layoutLoadRequestRef.current === requestId &&
          layoutMutationVersionRef.current === mutationVersionAtRequest
        ) {
          setLayout(DEFAULT_COMPANION_HOME_LAYOUT)
        }
      }
    }

    void loadLayout()

    return () => {
      cancelled = true
    }
  }, [surface])

  const updateLayout = React.useCallback(
    (nextLayout: CompanionHomeLayoutCard[]) => {
      layoutMutationVersionRef.current += 1
      setLayout(nextLayout)
      void saveCompanionHomeLayout(surface, nextLayout)
    },
    [surface]
  )

  return {
    layout,
    updateLayout
  }
}

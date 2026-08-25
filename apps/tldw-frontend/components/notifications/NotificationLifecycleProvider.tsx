import React from "react"

import {
  buildNotificationScopeKey,
  classifyNotificationError,
  reduceNotificationLifecycle,
  type NotificationLifecycleState
} from "@/services/notification-lifecycle"
import { getApiBearer, getApiKey } from "@web/lib/authStorage"
import { getApiBaseUrl } from "@web/lib/api"
import {
  AUTH_CREDENTIALS_CHANGED_EVENT,
  type AuthCredentialsChangedDetail
} from "@web/lib/auth-events"
import {
  getUnreadCount,
  listNotifications,
  subscribeNotificationsStream,
  type NotificationStreamEvent
} from "@web/lib/api/notifications"

type ExposedNotificationState = Exclude<NotificationLifecycleState, "idle">
const MAX_BUFFERED_EVENTS = 256

export type SequencedNotificationEvent = {
  sequence: number
  event: NotificationStreamEvent
}

export type NotificationLifecycleContextValue = {
  scopeKey: string
  lifecycleEpoch: number
  state: ExposedNotificationState
  unreadCount: number
  updatedAt: number
  latestEvent: NotificationStreamEvent | null
  eventSequence: number
  events: SequencedNotificationEvent[]
  mutationError: unknown | null
  tryAgain: () => Promise<void>
  refreshPermissions: () => Promise<void>
  reportRequestError: (error: unknown) => void
  reportMutationError: (error: unknown) => void
}

type NotificationRuntimeSnapshot = Omit<
  NotificationLifecycleContextValue,
  "tryAgain" | "refreshPermissions" | "reportRequestError" | "reportMutationError"
>

type NotificationLifecycleProviderProps = {
  children: React.ReactNode
  enabled?: boolean
  pollIntervalMs?: number
  scopeKey?: string
}

const DEFAULT_POLL_INTERVAL_MS = 30_000
const NotificationLifecycleContext =
  React.createContext<NotificationLifecycleContextValue | null>(null)

const initialSnapshot = (scopeKey: string, lifecycleEpoch: number): NotificationRuntimeSnapshot => ({
  scopeKey,
  lifecycleEpoch,
  state: "connecting",
  unreadCount: 0,
  updatedAt: Date.now(),
  latestEvent: null,
  eventSequence: 0,
  events: [],
  mutationError: null
})

const unreadIncrementForEvent = (event: NotificationStreamEvent): number => {
  if (event.event === "notification") return 1
  if (event.event !== "notifications_coalesced" || !event.payload || typeof event.payload !== "object") {
    return 0
  }
  const count = Number((event.payload as Record<string, unknown>).count ?? 0)
  return Number.isFinite(count) && count > 0 ? count : 0
}

const readStoredOrgId = (): string | number | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem("tldwConfig")
    if (!raw) return null
    const config = JSON.parse(raw) as { orgId?: unknown }
    return typeof config.orgId === "string" || typeof config.orgId === "number"
      ? config.orgId
      : null
  } catch {
    return null
  }
}

export const buildWebNotificationScopeKey = (): string => {
  const jwt =
    typeof window !== "undefined" ? window.localStorage.getItem("access_token") : null
  const bearer = jwt || getApiBearer()
  const apiKey = getApiKey()
  return buildNotificationScopeKey({
    serverUrl: getApiBaseUrl(),
    authMode: bearer ? "multi-user" : "single-user",
    orgId: readStoredOrgId(),
    userId: null,
    accessToken: bearer,
    apiKey
  })
}

export const useOptionalNotificationLifecycle = (): NotificationLifecycleContextValue | null =>
  React.useContext(NotificationLifecycleContext)

export const useNotificationLifecycle = (): NotificationLifecycleContextValue => {
  const value = useOptionalNotificationLifecycle()
  if (!value) {
    throw new Error("useNotificationLifecycle must be used within NotificationLifecycleProvider")
  }
  return value
}

export function NotificationLifecycleProvider({
  children,
  enabled = true,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  scopeKey: suppliedScopeKey
}: NotificationLifecycleProviderProps) {
  const [liveScopeKey, setLiveScopeKey] = React.useState(() =>
    suppliedScopeKey ?? buildWebNotificationScopeKey()
  )
  const scopeKey = suppliedScopeKey ?? liveScopeKey
  const [snapshot, setSnapshot] = React.useState<NotificationRuntimeSnapshot>(() =>
    initialSnapshot(scopeKey, 0)
  )
  const lifecycleEpochRef = React.useRef(0)
  const generationRef = React.useRef(0)
  const streamOpenRef = React.useRef(false)
  const unreadCurrentRef = React.useRef(false)
  const cursorCurrentRef = React.useRef(false)
  const terminalGenerationRef = React.useRef<number | null>(null)
  const terminalStateRef = React.useRef<"auth-required" | "unavailable" | null>(null)
  const unsubscribeRef = React.useRef<(() => void) | null>(null)
  const pollTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(null)
  const requestAbortRef = React.useRef<AbortController | null>(null)
  const effectSetupSeenRef = React.useRef(false)

  const stopWork = React.useCallback(() => {
    streamOpenRef.current = false
    requestAbortRef.current?.abort()
    requestAbortRef.current = null
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
    const unsubscribe = unsubscribeRef.current
    unsubscribeRef.current = null
    unsubscribe?.()
  }, [])

  const updateCurrent = React.useCallback(
    (
      generation: number,
      update: (current: NotificationRuntimeSnapshot) => NotificationRuntimeSnapshot
    ) => {
      if (generation !== generationRef.current) return
      setSnapshot((current) =>
        current.scopeKey === scopeKey ? update(current) : current
      )
    },
    [scopeKey]
  )

  const applyFailure = React.useCallback(
    (error: unknown, generation: number): "idle" | "retry" | "terminal" => {
      const classification = classifyNotificationError(error)
      if (classification.kind === "idle") return "idle"
      if (classification.kind === "retry") {
        updateCurrent(generation, (current) => ({
          ...current,
          state: reduceNotificationLifecycle(current.state, { type: "retry" }) as ExposedNotificationState,
          updatedAt: Date.now()
        }))
        return "retry"
      }

      terminalGenerationRef.current = generation
      stopWork()
      const action = classification.kind === "auth-required" ? "auth-required" : "unavailable"
      terminalStateRef.current = action
      updateCurrent(generation, (current) => ({
        ...current,
        state: reduceNotificationLifecycle(current.state, {
          type: action
        }) as ExposedNotificationState,
        updatedAt: Date.now()
      }))
      return "terminal"
    },
    [stopWork, updateCurrent]
  )

  const startWork = React.useCallback(async (): Promise<void> => {
    const generation = ++generationRef.current
    const lifecycleEpoch = ++lifecycleEpochRef.current
    stopWork()
    unreadCurrentRef.current = false
    cursorCurrentRef.current = false
    terminalGenerationRef.current = null
    terminalStateRef.current = null
    setSnapshot(initialSnapshot(scopeKey, lifecycleEpoch))
    if (!enabled) return
    const requestAbort = new AbortController()
    requestAbortRef.current = requestAbort

    const isCurrent = () => generation === generationRef.current
    const canBeActive = () =>
      streamOpenRef.current && unreadCurrentRef.current && cursorCurrentRef.current

    const openStream = (cursor: number) => {
      if (!isCurrent() || terminalGenerationRef.current === generation || unsubscribeRef.current) {
        return
      }
      const unsubscribe = subscribeNotificationsStream({
        after: cursor,
        onOpen: () => {
          if (!isCurrent()) return
          streamOpenRef.current = true
          updateCurrent(generation, (current) => ({
            ...current,
            state: canBeActive()
              ? (reduceNotificationLifecycle(current.state, {
                  type: "open"
                }) as ExposedNotificationState)
              : current.state,
            updatedAt: Date.now()
          }))
        },
        onError: (error) => {
          if (!isCurrent()) return
          streamOpenRef.current = false
          applyFailure(error, generation)
        },
        onEvent: (event) => {
          if (!isCurrent()) return
          updateCurrent(generation, (current) => {
            const sequence = current.eventSequence + 1
            return {
              ...current,
              latestEvent: event,
              eventSequence: sequence,
              events: [...current.events, { sequence, event }].slice(-MAX_BUFFERED_EVENTS),
              unreadCount: current.unreadCount + unreadIncrementForEvent(event),
              updatedAt: Date.now()
            }
          })
        }
      })
      if (!isCurrent() || terminalGenerationRef.current === generation) {
        unsubscribe()
        return
      }
      unsubscribeRef.current = unsubscribe
    }

    try {
      const result = await getUnreadCount({ signal: requestAbort.signal })
      if (!isCurrent()) return
      unreadCurrentRef.current = true
      updateCurrent(generation, (current) => ({
        ...current,
        unreadCount: Math.max(0, Number(result?.unread_count) || 0),
        updatedAt: Date.now()
      }))
    } catch (error) {
      if (!isCurrent()) return
      if (applyFailure(error, generation) === "terminal") return
    }

    let cursor = 0
    try {
      const latest = await listNotifications({
        limit: 1,
        offset: 0,
        include_archived: false,
        signal: requestAbort.signal
      })
      if (!isCurrent()) return
      cursor = latest.items.reduce(
        (maximum, item) => Math.max(maximum, Number(item.id) || 0),
        0
      )
      cursorCurrentRef.current = true
    } catch (error) {
      if (!isCurrent()) return
      if (applyFailure(error, generation) === "terminal") return
    }

    if (!isCurrent()) return
    if (cursorCurrentRef.current) openStream(cursor)
    if (terminalGenerationRef.current === generation) return

    let pollInFlight = false
    const pollNotificationState = async () => {
      if (pollInFlight) return
      pollInFlight = true
      try {
        try {
          const result = await getUnreadCount({ signal: requestAbort.signal })
          if (!isCurrent()) return
          unreadCurrentRef.current = true
          updateCurrent(generation, (current) => ({
            ...current,
            state:
              canBeActive() && current.state === "degraded"
                ? (reduceNotificationLifecycle(current.state, {
                    type: "open"
                  }) as ExposedNotificationState)
                : current.state,
            unreadCount: Math.max(0, Number(result?.unread_count) || 0),
            updatedAt: Date.now()
          }))
        } catch (error) {
          if (!isCurrent()) return
          unreadCurrentRef.current = false
          if (applyFailure(error, generation) === "terminal") return
        }

        if (!cursorCurrentRef.current && terminalGenerationRef.current !== generation) {
          try {
            const latest = await listNotifications({
              limit: 1,
              offset: 0,
              include_archived: false,
              signal: requestAbort.signal
            })
            if (!isCurrent()) return
            const nextCursor = latest.items.reduce(
              (maximum, item) => Math.max(maximum, Number(item.id) || 0),
              0
            )
            cursorCurrentRef.current = true
            openStream(nextCursor)
          } catch (error) {
            if (!isCurrent()) return
            if (applyFailure(error, generation) === "terminal") return
          }
        }
      } finally {
        pollInFlight = false
      }
    }

    if (terminalGenerationRef.current !== generation) {
      pollTimerRef.current = setInterval(() => void pollNotificationState(), pollIntervalMs)
    }
  }, [applyFailure, enabled, pollIntervalMs, scopeKey, stopWork, updateCurrent])

  React.useEffect(() => {
    let cancelled = false
    if (effectSetupSeenRef.current) {
      void startWork()
    } else {
      effectSetupSeenRef.current = true
      queueMicrotask(() => {
        if (!cancelled) void startWork()
      })
    }
    return () => {
      cancelled = true
      generationRef.current += 1
      stopWork()
    }
  }, [startWork, stopWork])

  React.useEffect(() => {
    if (typeof window === "undefined") return

    const stopForRemovedCredentials = () => {
      generationRef.current += 1
      stopWork()
      terminalGenerationRef.current = generationRef.current
      terminalStateRef.current = "auth-required"
      unreadCurrentRef.current = false
      cursorCurrentRef.current = false
      setSnapshot({
        ...initialSnapshot(scopeKey, ++lifecycleEpochRef.current),
        state: "auth-required"
      })
    }
    const resetForChangedScope = (): boolean => {
      if (suppliedScopeKey !== undefined) return false
      const nextScopeKey = buildWebNotificationScopeKey()
      if (nextScopeKey === scopeKey) return false
      generationRef.current += 1
      stopWork()
      terminalGenerationRef.current = null
      terminalStateRef.current = null
      unreadCurrentRef.current = false
      cursorCurrentRef.current = false
      setSnapshot(initialSnapshot(nextScopeKey, ++lifecycleEpochRef.current))
      setLiveScopeKey(nextScopeKey)
      return true
    }
    const onCredentialsChanged = (event: Event) => {
      const detail = (event as CustomEvent<AuthCredentialsChangedDetail>).detail
      if (detail?.authenticated) {
        if (resetForChangedScope()) return
        if (terminalStateRef.current === "unavailable") return
        void startWork()
      } else {
        stopForRemovedCredentials()
      }
    }
    const onStorage = (event: StorageEvent) => {
      if (event.key === "tldwConfig") {
        resetForChangedScope()
        return
      }
      if (event.key !== "access_token") return
      if (event.newValue) {
        if (resetForChangedScope()) return
        if (terminalStateRef.current === "unavailable") return
        void startWork()
      } else {
        stopForRemovedCredentials()
      }
    }
    const onConfigUpdated = () => {
      resetForChangedScope()
    }

    window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, onCredentialsChanged)
    window.addEventListener("tldw:config-updated", onConfigUpdated)
    window.addEventListener("storage", onStorage)
    return () => {
      window.removeEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, onCredentialsChanged)
      window.removeEventListener("tldw:config-updated", onConfigUpdated)
      window.removeEventListener("storage", onStorage)
    }
  }, [scopeKey, startWork, stopWork, suppliedScopeKey])

  const reportRequestError = React.useCallback(
    (error: unknown) => {
      applyFailure(error, generationRef.current)
    },
    [applyFailure]
  )

  const reportMutationError = React.useCallback(
    (error: unknown) => {
      const generation = generationRef.current
      const result = applyFailure(error, generation)
      if (result === "idle") return
      updateCurrent(generation, (current) => ({
        ...current,
        mutationError: error,
        updatedAt: Date.now()
      }))
    },
    [applyFailure, updateCurrent]
  )

  const projected =
    snapshot.scopeKey === scopeKey
      ? snapshot
      : initialSnapshot(scopeKey, lifecycleEpochRef.current)
  const value = React.useMemo<NotificationLifecycleContextValue>(
    () => ({
      scopeKey: projected.scopeKey,
      lifecycleEpoch: projected.lifecycleEpoch,
      state: projected.state,
      unreadCount: projected.unreadCount,
      updatedAt: projected.updatedAt,
      latestEvent: projected.latestEvent,
      eventSequence: projected.eventSequence,
      events: projected.events,
      mutationError: projected.mutationError,
      tryAgain: startWork,
      refreshPermissions: startWork,
      reportRequestError,
      reportMutationError
    }),
    [projected, reportMutationError, reportRequestError, startWork]
  )

  return (
    <NotificationLifecycleContext.Provider value={value}>
      {children}
    </NotificationLifecycleContext.Provider>
  )
}

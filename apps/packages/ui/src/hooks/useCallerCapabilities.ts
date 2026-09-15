import { useCallback, useEffect, useRef, useState } from "react"
import { useConnectionStore } from "@/store/connection"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  callerCapabilityErrorStatus,
  loadCallerCapabilities,
  type CallerCapabilityDecision
} from "@/services/caller-capabilities"

type Discovery = {
  generation: number
  result: Awaited<ReturnType<typeof loadCallerCapabilities>> | null
  unavailable: "unknown" | "unsupported"
  loading: boolean
}

/** Ephemeral caller permissions, masked at every connection/authority boundary. */
export function useCallerCapabilities() {
  const online = useConnectionStore(
    (s) => s.state.isConnected && s.state.mode !== "demo"
  )
  const generation = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const mounted = useRef(false)
  const [discovery, setDiscovery] = useState<Discovery>({
    generation: 0,
    result: null,
    unavailable: "unknown",
    loading: online
  })

  const invalidate = useCallback(() => {
    const next = ++generation.current
    controller.current?.abort()
    controller.current = null
    setDiscovery({
      generation: next,
      result: null,
      unavailable: "unknown",
      loading: false
    })
    return next
  }, [])

  const refresh = useCallback(async () => {
    if (!mounted.current) return
    const request = invalidate()
    const connection = useConnectionStore.getState().state
    if (!connection.isConnected || connection.mode === "demo") return
    const pending = new AbortController()
    controller.current = pending
    const isCurrent = () =>
      mounted.current &&
      request === generation.current &&
      !pending.signal.aborted
    setDiscovery({
      generation: request,
      result: null,
      unavailable: "unknown",
      loading: true
    })
    try {
      const result = await loadCallerCapabilities(pending.signal)
      if (isCurrent())
        setDiscovery({
          generation: request,
          result,
          unavailable: "unknown",
          loading: false
        })
    } catch (error) {
      if (isCurrent()) {
        const status = callerCapabilityErrorStatus(error)
        setDiscovery({
          generation: request,
          result: null,
          loading: false,
          unavailable:
            status === 404 || status === 405 || status === 501
              ? "unsupported"
              : "unknown"
        })
      }
    } finally {
      if (controller.current === pending) controller.current = null
    }
  }, [invalidate])

  useEffect(() => {
    mounted.current = true
    const storage = createSafeStorage({ area: "local" })
    const changed = () => {
      void refresh()
    }
    const principalChanged = (event: Event) => {
      if ((event as CustomEvent<{ kind?: string }>).detail?.kind === "logout")
        invalidate()
      else changed()
    }
    const configChanged = (event: Event) => {
      const detail = (
        event as CustomEvent<{
          authorityChanged?: boolean
          refreshSessionInvalidated?: boolean
        }>
      ).detail
      if (
        detail?.authorityChanged !== false ||
        detail.refreshSessionInvalidated
      )
        changed()
    }
    const watchers = {
      tldwConfig: changed,
      tldwCookieSessionConfig: changed,
      tldwRefreshRotation: changed
    }
    storage.watch(watchers)
    window.addEventListener("tldw:auth-credentials-changed", changed)
    window.addEventListener("tldw:auth-principal-changed", principalChanged)
    window.addEventListener("tldw:config-updated", configChanged)
    return () => {
      mounted.current = false
      generation.current += 1
      controller.current?.abort()
      storage.unwatch(watchers)
      window.removeEventListener("tldw:auth-credentials-changed", changed)
      window.removeEventListener(
        "tldw:auth-principal-changed",
        principalChanged
      )
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }, [invalidate, refresh])

  useEffect(() => {
    void refresh()
  }, [online, refresh])

  const result =
    online && discovery.generation === generation.current
      ? discovery.result
      : null
  const unavailable = online ? discovery.unavailable : "unknown"
  const decision = (allowed: boolean | undefined): CallerCapabilityDecision =>
    allowed === undefined ? unavailable : allowed ? "allowed" : "denied"
  const refreshAfterForbidden = useCallback(
    async (error: unknown) => {
      // A protected read retains the callback from its own authority generation.
      if (
        callerCapabilityErrorStatus(error) === 403 &&
        discovery.generation === generation.current
      ) {
        await refresh()
      }
    },
    [discovery.generation, refresh]
  )
  return {
    scheduledTasks: decision(result?.capabilities.can_read_scheduled_tasks),
    notifications: decision(result?.capabilities.can_read_notifications),
    monitoringAlerts: decision(result?.capabilities.can_read_monitoring_alerts),
    scopeKey: result?.scopeKey ?? null,
    loading: online && discovery.loading,
    refresh,
    refreshAfterForbidden
  }
}

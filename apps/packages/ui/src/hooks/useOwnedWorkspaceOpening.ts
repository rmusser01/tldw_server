import { useCallback, useEffect, useRef, useState } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import { OwnedWorkspaceLoadError } from "@/store/workspace-api"
import type { OwnedWorkspaceScope } from "@/store/owned-workspace-state"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import { getStructuredApiErrorDetail } from "@/services/tldw/api-error"
import {
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential"
import {
  createOwnedWorkspaceReadContext,
  OwnedWorkspaceOpeningError
} from "@/services/owned-workspace-opening"

type OpeningFailure =
  | "denied"
  | "unavailable"
  | "connection"
  | "timeout"
  | "invalid-response"
  | "draft-conflict"
type OpeningState =
  | { status: "loading" }
  | { status: "ready"; scope: OwnedWorkspaceScope }
  | { status: "error"; reason: OpeningFailure }
export type OwnedWorkspaceOpening =
  | { status: "loading" }
  | { status: "ready" }
  | { status: "error"; reason: OpeningFailure; retry: () => void }

function failureReason(error: unknown): OpeningFailure {
  if (
    error instanceof OwnedWorkspaceLoadError ||
    error instanceof OwnedWorkspaceOpeningError
  )
    return error.reason
  const status =
    error && typeof error === "object" && "status" in error
      ? error.status
      : null
  if (status === 401 || status === 403) return "denied"
  if (
    status === 412 &&
    getStructuredApiErrorDetail(error)?.code === "request_config_scope_changed"
  ) return "denied"
  if (status === 404 || status === 410) return "unavailable"
  return "connection"
}

function waitForHydration(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted()
  if (useWorkspaceStore.getState().storeHydrated) return Promise.resolve()
  return new Promise((resolve, reject) => {
    let unsubscribe = () => undefined as void
    const cleanup = () => {
      unsubscribe()
      signal.removeEventListener("abort", abort)
    }
    const abort = () => {
      cleanup()
      reject(signal.reason)
    }
    const ready = () => {
      if (!useWorkspaceStore.getState().storeHydrated) return
      cleanup()
      resolve()
    }
    unsubscribe = useWorkspaceStore.subscribe(ready)
    signal.addEventListener("abort", abort, { once: true })
    if (signal.aborted) abort()
    else ready()
  })
}

/** Owns one route opening. The editable subtree must still guard its own writes. */
export function useOwnedWorkspaceOpening(
  workspaceId: string
): OwnedWorkspaceOpening {
  const hydrated = useWorkspaceStore((state) => state.storeHydrated)
  const activeId = useWorkspaceStore((state) => state.workspaceId)
  const origin = useWorkspaceStore((state) => state.activeWorkspaceOrigin)
  const attempt = useWorkspaceStore((state) => state.ownedWorkspaceAttempt)
  const [opening, setOpening] = useState<
    OpeningState & { workspaceId: string }
  >({ status: "loading", workspaceId })
  const restart = useRef<() => void>(() => undefined)
  const retry = useCallback(() => restart.current(), [])

  useEffect(() => {
    let generation = 0
    let disposed = false
    let controller: AbortController | null = null
    let deadline: ReturnType<typeof setTimeout> | undefined
    const stop = () => {
      generation++
      controller?.abort()
      clearTimeout(deadline)
      useWorkspaceStore.getState().invalidateOwnedWorkspace()
    }
    const run = (retryHydration: boolean) => {
      stop()
      setOpening({ status: "loading", workspaceId })
      if (disposed) return
      const currentGeneration = generation
      const request = new AbortController()
      controller = request
      const current = () =>
        !disposed && generation === currentGeneration && !request.signal.aborted
      deadline = setTimeout(() => {
        if (!current()) return
        stop()
        setOpening({ status: "error", reason: "timeout", workspaceId })
      }, 30_000)
      void (async () => {
        try {
          if (retryHydration && !useWorkspaceStore.getState().storeHydrated)
            await useWorkspaceStore.persist.rehydrate()
          await waitForHydration(request.signal)
          if (!current()) return
          const context = await createOwnedWorkspaceReadContext(
            workspaceId,
            request.signal
          )
          if (!current()) return
          const ticket = useWorkspaceStore
            .getState()
            .beginOwnedWorkspace(context.scope, workspaceId)
          const bundle = await context.load()
          if (!current()) return
          const result = useWorkspaceStore
            .getState()
            .activateOwnedWorkspace(ticket, bundle)
          if (result === "activated") {
            setOpening({ status: "ready", scope: context.scope, workspaceId })
          } else {
            setOpening({
              status: "error",
              reason:
                result === "draft-conflict" ? "draft-conflict" : "unavailable",
              workspaceId
            })
          }
        } catch (error) {
          if (current()) {
            clearTimeout(deadline)
            request.abort()
            useWorkspaceStore.getState().invalidateOwnedWorkspace()
            setOpening({
              status: "error",
              reason: failureReason(error),
              workspaceId
            })
          }
        } finally {
          if (current()) clearTimeout(deadline)
        }
      })()
    }
    const start = () => run(false)
    restart.current = () => run(true)
    const auth = (event: Event) => {
      if ((event as CustomEvent<{ kind?: string }>).detail?.kind === "logout") {
        stop()
        setOpening({ status: "error", reason: "denied", workspaceId })
      } else start()
    }
    const suspend = () => {
      stop()
      setOpening({ status: "loading", workspaceId })
    }
    const visibility = () => {
      if (document.visibilityState === "hidden") suspend()
      else start()
    }
    const storage = (event: StorageEvent) => {
      if (
        event.key === null ||
        [
          "tldwConfig",
          MANUAL_SESSION_KEY,
          REFRESH_ROTATION_KEY,
          COOKIE_SESSION_CONFIG_KEY
        ].includes(event.key)
      )
        start()
    }
    window.addEventListener("tldw:config-updated", start)
    window.addEventListener("tldw:auth-principal-changed", auth)
    window.addEventListener("focus", start)
    window.addEventListener("pageshow", start)
    window.addEventListener("pagehide", suspend)
    window.addEventListener("storage", storage)
    document.addEventListener("visibilitychange", visibility)
    start()
    return () => {
      disposed = true
      stop()
      restart.current = () => undefined
      window.removeEventListener("tldw:config-updated", start)
      window.removeEventListener("tldw:auth-principal-changed", auth)
      window.removeEventListener("focus", start)
      window.removeEventListener("pageshow", start)
      window.removeEventListener("pagehide", suspend)
      window.removeEventListener("storage", storage)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [workspaceId])

  // Do not expose the previous effect's ready state during a new route render.
  if (opening.workspaceId !== workspaceId) return { status: "loading" }
  if (opening.status === "error")
    return { status: "error", reason: opening.reason, retry }
  if (!hydrated) return { status: "loading" }
  if (opening.status === "ready") {
    if (
      activeId !== workspaceId ||
      attempt ||
      origin.kind !== "server-owned" ||
      origin.scope.serverBase !== opening.scope.serverBase ||
      origin.scope.principalId !== opening.scope.principalId ||
      origin.scope.organizationId !== opening.scope.organizationId
    )
      return { status: "loading" }
    return { status: "ready" }
  }
  return { status: "loading" }
}

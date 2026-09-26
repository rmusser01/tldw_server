import { useCallback, useEffect, useRef, useState } from "react"
import { browser } from "wxt/browser"
import { createSharedWorkspaceCloneContext } from "@/services/tldw/domains/shared-workspaces"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import {
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential"
import {
  SharedCloneManager,
  type CloneRowState
} from "@/services/shared-clone-manager"
import {
  CLONE_RECOVERY_KEY,
  clearCloneRecovery,
  type CloneStorage
} from "@/services/shared-clone-recovery"

const browserStorage: CloneStorage = {
  getItem: (key) => window.localStorage.getItem(key),
  setItem: (key, value) => window.localStorage.setItem(key, value),
  removeItem: (key) => window.localStorage.removeItem(key)
}

const contextStorageKeys = new Set([
  "tldwConfig",
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY,
  COOKIE_SESSION_CONFIG_KEY
])

export function useSharedWorkspaceClones() {
  const [rows, setRows] = useState<CloneRowState[]>([])
  const [scope, setScope] = useState<string | null>(null)
  const [status, setStatus] = useState<
    "loading" | "ready" | "auth_required" | "recovery_conflict"
  >("loading")
  const current = useRef<{ scope: string; manager: SharedCloneManager } | null>(
    null
  )
  const ready = useRef(false)
  const refreshScope = useRef<() => void>(() => undefined)

  useEffect(() => {
    let disposed = false
    let generation = 0
    let verification: AbortController | undefined
    const reset = () => {
      generation++
      verification?.abort()
      ready.current = false
      current.current?.manager.suspend()
      setScope(null)
      setRows([])
    }
    const verify = async () => {
      reset()
      setStatus("loading")
      const expected = generation
      const controller = new AbortController()
      verification = controller
      const deadline = setTimeout(() => controller.abort(), 30_000)
      try {
        const context = await createSharedWorkspaceCloneContext(
          controller.signal
        )
        if (disposed || generation !== expected) return
        const nextScope = context.scope
        if (current.current?.scope !== nextScope) {
          current.current?.manager.dispose()
          current.current = null
          const manager = new SharedCloneManager(
            nextScope,
            browserStorage,
            context.api,
            () => {
              if (ready.current && current.current?.manager === manager)
                setRows(manager.rows())
            },
            (reason) => {
              if (disposed || current.current?.manager !== manager) return
              reset()
              setStatus(reason)
            }
          )
          current.current = { scope: nextScope, manager }
        }
        current.current.manager.setVisible(
          document.visibilityState !== "hidden"
        )
        await current.current.manager.resume(context.api)
        if (disposed || generation !== expected) return
        ready.current = true
        setScope(nextScope)
        setStatus("ready")
        setRows(current.current.manager.rows())
      } catch {
        if (disposed || generation !== expected) return
        current.current?.manager.dispose()
        current.current = null
        setStatus("auth_required")
      } finally {
        clearTimeout(deadline)
      }
    }
    refreshScope.current = () => {
      void verify()
    }
    const auth = (event: Event) => {
      if ((event as CustomEvent<{ kind?: string }>).detail?.kind === "logout") {
        reset()
        current.current?.manager.dispose()
        current.current = null
        void clearCloneRecovery()
        setStatus("auth_required")
      } else void verify()
    }
    const restore = () => {
      void verify()
    }
    const storage = (event: StorageEvent) => {
      if (event.key !== null && contextStorageKeys.has(event.key)) {
        void verify()
        return
      }
      if (event.key !== CLONE_RECOVERY_KEY && event.key !== null) return
      if (
        event.key === CLONE_RECOVERY_KEY &&
        event.newValue &&
        event.newValue.length <= 32 * 1024 &&
        ready.current &&
        current.current
      ) {
        try {
          const value = JSON.parse(event.newValue)
          if (value?.scope === current.current.scope) {
            // Another tab's progress must not cancel a queued user admission.
            void current.current.manager.sync()
            return
          }
        } catch {
          // Malformed records still take the verified scope/cleanup path.
        }
      }
      void verify()
    }
    const extensionStorage = (
      changes: Record<string, unknown>,
      area: string
    ) => {
      if (
        (area === "local" || area === "session" || area === "sync") &&
        Object.keys(changes).some((key) => contextStorageKeys.has(key))
      )
        void verify()
    }
    const visibility = () => {
      if (document.visibilityState === "hidden")
        current.current?.manager.setVisible(false)
      else void verify()
    }
    const pagehide = () => {
      reset()
      setStatus("loading")
    }
    window.addEventListener("tldw:config-updated", restore)
    window.addEventListener("tldw:auth-principal-changed", auth)
    window.addEventListener("focus", restore)
    window.addEventListener("pageshow", restore)
    window.addEventListener("pagehide", pagehide)
    window.addEventListener("storage", storage)
    browser.storage.onChanged.addListener(extensionStorage)
    document.addEventListener("visibilitychange", visibility)
    void verify()
    return () => {
      disposed = true
      generation++
      verification?.abort()
      ready.current = false
      current.current?.manager.dispose()
      current.current = null
      window.removeEventListener("tldw:config-updated", restore)
      window.removeEventListener("tldw:auth-principal-changed", auth)
      window.removeEventListener("focus", restore)
      window.removeEventListener("pageshow", restore)
      window.removeEventListener("pagehide", pagehide)
      window.removeEventListener("storage", storage)
      browser.storage.onChanged.removeListener(extensionStorage)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [])

  const begin = useCallback((shareId: number, name: string) => {
    if (ready.current) void current.current?.manager.begin(shareId, name)
  }, [])
  const refresh = useCallback((shareId?: number) => {
    if (ready.current) current.current?.manager.refresh(shareId)
    else refreshScope.current()
  }, [])

  return { rows, scope, status, begin, refresh }
}

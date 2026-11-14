import { useCallback, useEffect, useRef, useState } from "react"
import { authConnectionTargetsMatch, type AuthConnectionTarget } from "@/services/tldw/auth-connection-target"
import type { FormInstance } from "antd"
import { createSafeStorage } from "@/utils/safe-storage"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import {
  REFRESH_ROTATION_KEY,
  REFRESH_SESSION_INVALIDATION_PREFIX,
  refreshSessionInvalidationKey
} from "@/services/tldw/single-user-credential"

/** Keep authentication presentation current without replacing the Settings draft. */
export const useSettingsLoginStatus = (form: FormInstance, enabled: boolean) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loggedInUsername, setLoggedInUsername] = useState<string | null>(null)
  const [signInTargetSaved, setSignInTargetSaved] = useState(false)
  const savedTarget = useRef<AuthConnectionTarget | null>(null)
  const refreshRef = useRef<() => Promise<void>>(async () => {})
  const refreshLoginStatus = useCallback(() => refreshRef.current(), [])

  useEffect(() => {
    if (!enabled) return
    const storage = createSafeStorage({ area: "local" })
    let disposed = false
    let generation = 0
    let markerKey: string | null = null
    let markerWatch: Record<string, () => void> | undefined
    const refresh = async () => {
      setLoggedInUsername(null)
      setSignInTargetSaved(authConnectionTargetsMatch(savedTarget.current, form.getFieldsValue()))
      const currentGeneration = ++generation
      try {
        const config = await resolveDirectBrowserConfig(storage)
        if (disposed || generation !== currentGeneration) return
        const nextMarker = config ? refreshSessionInvalidationKey(config) : null
        if (nextMarker !== markerKey) {
          if (markerWatch) storage.unwatch(markerWatch)
          markerKey = nextMarker
          markerWatch = nextMarker ? { [nextMarker]: () => { void refresh() } } : undefined
          if (markerWatch) {
            storage.watch(markerWatch)
            // Re-read after subscribing so a concurrent invalidation cannot be missed.
            await refresh()
            return
          }
        }
        savedTarget.current = config
        const displayed = form.getFieldsValue()
        setSignInTargetSaved(authConnectionTargetsMatch(config, displayed))
        const normalizeUrl = (value: unknown) => String(value || "").trim().replace(/\/+$/, "")
        const loggedIn = Boolean(
          config?.authMode === "multi-user" && config.accessToken &&
          displayed.authMode === "multi-user" &&
          normalizeUrl(config.serverUrl) === normalizeUrl(displayed.serverUrl)
        )
        setIsLoggedIn(loggedIn)
        if (loggedIn) {
          try {
            const user = await tldwAuth.getCurrentUser()
            if (disposed || generation !== currentGeneration) return
            if (authConnectionTargetsMatch(config, form.getFieldsValue()) && user.is_active) {
              setLoggedInUsername(user.username.trim() || null)
            }
          } catch {
            // A failed identity read must not invent an account or discard a working connection.
          }
        }
      } catch {
        if (!disposed && generation === currentGeneration) {
          savedTarget.current = null
          setSignInTargetSaved(false)
          setIsLoggedIn(false)
        }
      }
    }
    const revalidate = () => { void refresh() }
    const onStorage = (event: StorageEvent) => {
      if (event.key === null || event.key === "tldwConfig" ||
        event.key === REFRESH_ROTATION_KEY || event.key === COOKIE_SESSION_CONFIG_KEY ||
        event.key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)) revalidate()
    }
    const watchers = {
      tldwConfig: revalidate,
      [REFRESH_ROTATION_KEY]: revalidate,
      [COOKIE_SESSION_CONFIG_KEY]: revalidate
    }
    storage.watch(watchers)
    refreshRef.current = refresh
    window.addEventListener("storage", onStorage)
    window.addEventListener("tldw:config-updated", revalidate)
    window.addEventListener("tldw:auth-principal-changed", revalidate)
    window.addEventListener("tldw:auth-credentials-changed", revalidate)
    void refresh()
    return () => {
      disposed = true
      generation += 1
      refreshRef.current = async () => {}
      storage.unwatch(watchers)
      if (markerWatch) storage.unwatch(markerWatch)
      window.removeEventListener("storage", onStorage)
      window.removeEventListener("tldw:config-updated", revalidate)
      window.removeEventListener("tldw:auth-principal-changed", revalidate)
      window.removeEventListener("tldw:auth-credentials-changed", revalidate)
    }
  }, [enabled, form])

  return { isLoggedIn, setIsLoggedIn, loggedInUsername: isLoggedIn ? loggedInUsername : null, signInTargetSaved, refreshLoginStatus }
}

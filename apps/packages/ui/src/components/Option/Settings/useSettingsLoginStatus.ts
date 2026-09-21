import { useCallback, useEffect, useRef, useState } from "react"
import type { FormInstance } from "antd"
import { createSafeStorage } from "@/utils/safe-storage"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import {
  REFRESH_ROTATION_KEY,
  REFRESH_SESSION_INVALIDATION_PREFIX,
  refreshSessionInvalidationKey
} from "@/services/tldw/single-user-credential"

/** Keep authentication presentation current without replacing the Settings draft. */
export const useSettingsLoginStatus = (form: FormInstance, enabled: boolean) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
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
        const displayed = form.getFieldsValue()
        const normalizeUrl = (value: unknown) => String(value || "").trim().replace(/\/+$/, "")
        setIsLoggedIn(Boolean(
          config?.authMode === "multi-user" && config.accessToken &&
          displayed.authMode === "multi-user" &&
          normalizeUrl(config.serverUrl) === normalizeUrl(displayed.serverUrl)
        ))
      } catch {
        if (!disposed && generation === currentGeneration) setIsLoggedIn(false)
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
    void refresh()
    return () => {
      disposed = true
      generation += 1
      refreshRef.current = async () => {}
      storage.unwatch(watchers)
      if (markerWatch) storage.unwatch(markerWatch)
      window.removeEventListener("storage", onStorage)
      window.removeEventListener("tldw:config-updated", revalidate)
    }
  }, [enabled, form])

  return { isLoggedIn, setIsLoggedIn, refreshLoginStatus }
}

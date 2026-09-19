import { browser } from "wxt/browser"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { safeStorageSerde } from "@/utils/safe-storage"

/** Observe account boundaries even when the Chat route is not mounted. */
export const watchChatAccountChanges = (
  changed: (invalidated: boolean) => void
): (() => void) => {
  if (typeof window === "undefined") return () => {}
  const principalChanged = () => changed(true)
  const configChanged = (event: Event) => {
    changed(Boolean((event as CustomEvent<{ authorityChanged?: boolean }>).detail?.authorityChanged))
  }
  const compare = (previous: unknown, current: unknown) => {
    changed(!current || !connectionAuthoritiesMatch(
      safeStorageSerde.deserializer(current), safeStorageSerde.deserializer(previous)
    ))
  }
  const configKey = (key: string) => key === "tldwConfig" || key === "tldwCookieSessionConfig"
  const storageChanged = (event: StorageEvent) => {
    if (event.key === null) changed(true)
    else if (configKey(event.key)) compare(event.oldValue, event.newValue)
  }
  const extensionChanged = (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => {
    if (area !== "local") return
    for (const [key, value] of Object.entries(changes)) {
      if (configKey(key)) compare(value.oldValue, value.newValue)
    }
  }
  window.addEventListener("tldw:auth-principal-changed", principalChanged)
  window.addEventListener("tldw:auth-credentials-changed", principalChanged)
  window.addEventListener("tldw:config-updated", configChanged)
  window.addEventListener("storage", storageChanged)
  browser?.storage?.onChanged?.addListener(extensionChanged)
  return () => {
    window.removeEventListener("tldw:auth-principal-changed", principalChanged)
    window.removeEventListener("tldw:auth-credentials-changed", principalChanged)
    window.removeEventListener("tldw:config-updated", configChanged)
    window.removeEventListener("storage", storageChanged)
    browser?.storage?.onChanged?.removeListener(extensionChanged)
  }
}

import { consumePendingQuickIngestOpen } from "@/utils/quick-ingest-open"
import { useEffect } from "react"
import { browser } from "wxt/browser"
import { createSafeStorage } from "@/utils/safe-storage"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { useQuickIngestSessionStore, type createQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { tldwClient, type TldwConfig } from "./TldwApiClient"
import { REFRESH_ROTATION_KEY, REFRESH_SESSION_INVALIDATION_PREFIX } from "./single-user-credential"
import { createServicePromptScopeChangedError, servicePromptTargetsMatch, servicePromptPrincipalMatches, servicePromptSingleUserApiKeyScopeMatches } from "./service-prompt-scope-error"
import { isHostedTldwDeployment } from "./deployment-mode"
import type { ServicePromptRequestScope } from "./domains/service-prompts"

/** Metadata only: never persist the access token or API key with ingest results. */
export const quickIngestAuthorityKey = ({ config, userId }: ServicePromptRequestScope): string => JSON.stringify([
  config.serverUrl.trim().replace(/\/+$/, ""), config.authMode,
  config.authSource || "manual", config.orgId ?? null, userId,
  config.expectedSingleUserApiKeyScope ?? null
])

export type QuickIngestOperation = {
  authorityKey: string
  authorityRevision: number
  requestScope: ServicePromptRequestScope
  signal: AbortSignal
  isCurrent: () => boolean
  assertCurrent: () => void
}

/** The mounted ingest surfaces share one verified lease. Closing the last UI
 * abandons its callbacks while retaining owned data for the next verification.
 * The extension worker owns its own lifetime, independent of this UI lease.
 */
export const createQuickIngestAuthority = (store: ReturnType<typeof createQuickIngestSessionStore>) => {
  const storage = createSafeStorage({ area: "local" })
  let owners = 0
  let revision = 0
  let validationRevision = 0
  let controller: AbortController | null = null
  let snapshot: ServicePromptSnapshot | null = null
  let detach: (() => void) | null = null

  const stop = (retainSession: boolean) => {
    revision += 1
    validationRevision += 1
    controller?.abort()
    controller = null
    snapshot?.release()
    snapshot = null
    store.getState().setAuthority(null, retainSession)
  }
  const resolve = () => {
    const currentRevision = revision
    const currentController = new AbortController()
    controller = currentController
    void loadServicePromptSnapshot([], { signal: currentController.signal }).then(value => {
      if (!owners || currentRevision !== revision || currentController.signal.aborted || value.scopeSignal.aborted) {
        value.release()
        return
      }
      snapshot = value
      value.scopeInvalidatedSignal.addEventListener("abort", invalidate, { once: true })
      store.getState().setAuthority(quickIngestAuthorityKey(value.requestScope))
    }).catch(() => {
      if (currentRevision === revision) controller = null
      // An unresolved owner remains masked. An ordinary login/config event can
      // resolve it again; connection failures do not invalidate a verified lease.
    })
  }
  const invalidate = () => {
    consumePendingQuickIngestOpen()
    stop(false)
    const currentRevision = revision
    queueMicrotask(() => { if (owners && revision === currentRevision) resolve() })
  }
  const configChanged = (event: Event) => {
    const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
    if (detail?.authorityChanged === true) invalidate()
    else if (detail?.refreshSessionInvalidated || detail?.authorityChanged === false) validateMarker()
    else invalidate()
  }
  const validateMarker = () => {
    const check = ++validationRevision
    const currentRevision = revision
    const captured = snapshot
    if (!captured) {
      if (owners && !controller) resolve()
      return
    }
    const isCurrent = () => owners > 0 && currentRevision === revision &&
      check === validationRevision && captured === snapshot
    // Expiry notifications describe the failed credential, which may already
    // have been superseded by a valid same-owner rotation.
    void tldwClient.ensureConfigForRequest(true).then(current => {
      if (!isCurrent()) return
      const { config, userId } = captured.requestScope
      if (!current || !servicePromptTargetsMatch(current, config) ||
        (!isHostedTldwDeployment() && current.authSource !== "cookie-session" &&
          (!servicePromptSingleUserApiKeyScopeMatches(current, config.expectedSingleUserApiKeyScope) ||
            !servicePromptPrincipalMatches(current, userId)))) invalidate()
    }).catch(() => {
      if (isCurrent()) invalidate()
    })
  }
  const nativeStorageChanged = (event: StorageEvent) => {
    if (event.key === REFRESH_ROTATION_KEY || event.key?.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)) validateMarker()
  }
  const extensionChanges = (changes: Record<string, unknown>) => {
    if (Object.keys(changes).some(key => key === REFRESH_ROTATION_KEY || key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX))) validateMarker()
  }
  const watch = { [REFRESH_ROTATION_KEY]: validateMarker, tldwConfig: (change: { oldValue?: unknown; newValue?: unknown }) => {
    if (!connectionAuthoritiesMatch(change.oldValue as Partial<TldwConfig> | null, change.newValue as Partial<TldwConfig> | null)) invalidate()
    else validateMarker()
  } }

  return {
    retain: () => {
      if (owners++ === 0) {
        // Subscribe before the resolver's first await, including A → B → A.
        storage.watch(watch)
        window.addEventListener("tldw:config-updated", configChanged)
        window.addEventListener("tldw:auth-principal-changed", invalidate)
        window.addEventListener("tldw:auth-credentials-changed", invalidate)
        window.addEventListener("storage", nativeStorageChanged)
        const extensionEvents = isExtensionRuntime() ? (browser as unknown as {
          storage?: { onChanged?: { addListener: (listener: typeof extensionChanges) => void; removeListener: (listener: typeof extensionChanges) => void } }
        }).storage?.onChanged : undefined
        extensionEvents?.addListener(extensionChanges)
        detach = () => {
          storage.unwatch(watch)
          window.removeEventListener("tldw:config-updated", configChanged)
          window.removeEventListener("tldw:auth-principal-changed", invalidate)
          window.removeEventListener("tldw:auth-credentials-changed", invalidate)
          window.removeEventListener("storage", nativeStorageChanged)
          extensionEvents?.removeListener(extensionChanges)
        }
        resolve()
      }
      let released = false
      return () => {
        if (released) return
        released = true
        if (--owners === 0) { detach?.(); detach = null; stop(true) }
      }
    },
    capture: ({ sessionBound = true }: { sessionBound?: boolean } = {}): QuickIngestOperation => {
      const state = store.getState()
      const captured = snapshot
      if (!captured || !state.authorityKey || captured.scopeSignal.aborted) throw createServicePromptScopeChangedError()
      const currentRevision = revision
      const { generation, authorityKey } = state
      const isCurrent = () => currentRevision === revision && !captured.scopeSignal.aborted &&
        (!sessionBound || store.getState().generation === generation) && store.getState().authorityKey === authorityKey
      return {
        authorityKey, authorityRevision: currentRevision, requestScope: captured.requestScope, signal: captured.scopeSignal,
        isCurrent,
        assertCurrent: () => { if (!isCurrent()) throw createServicePromptScopeChangedError() }
      }
    }
  }
}

export const quickIngestAuthority = createQuickIngestAuthority(useQuickIngestSessionStore)
export const useQuickIngestAuthority = () => {
  useEffect(() => quickIngestAuthority.retain(), [])
  return useQuickIngestSessionStore(state => state.authorityKey)
}

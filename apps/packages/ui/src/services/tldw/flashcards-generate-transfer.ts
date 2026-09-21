import { browser } from "wxt/browser"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { createSafeStorage } from "@/utils/safe-storage"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { tldwClient, type TldwConfig } from "./TldwApiClient"
import { REFRESH_SESSION_INVALIDATION_PREFIX } from "./single-user-credential"
import {
  buildFlashcardsGenerateRoute,
  createFlashcardsGenerateHandoff,
  createStudyPackHandoff,
  removeFlashcardsGenerateHandoff,
  type FlashcardsGenerateIntent
} from "./flashcards-generate-handoff"

import { buildStudyPackRoute, type StudyPackIntent } from "./study-pack-handoff"

export const flashcardsHandoffAuthority = (snapshot: ServicePromptSnapshot): string => {
  const { config, userId } = snapshot.requestScope
  return JSON.stringify([
    config.serverUrl.trim().replace(/\/+$/, ""), config.authMode,
    config.authSource || "manual", config.orgId ?? null,
    userId, config.expectedSingleUserApiKeyScope ?? null
  ])
}

export type FlashcardsTransferTarget =
  | { newTab: true }
  | { navigate: (route: string) => void | Promise<void> }

/** Watch the initiating context before the shared resolver's first await.
 * A capture callback with a parent signal keeps its failure state invalidatable
 * until that parent cancels it; a rejected resolution never authorizes transfer.
 */
export const loadFlashcardsTransferSnapshot = async (
  signal?: AbortSignal,
  onInvalidated?: () => void
): Promise<ServicePromptSnapshot> => {
  const controller = new AbortController()
  const abort = () => {
    if (controller.signal.aborted) return
    controller.abort()
    release()
    onInvalidated?.()
  }
  const storage = createSafeStorage({ area: "local" })
  const watch = { tldwConfig: (change: { oldValue?: unknown; newValue?: unknown }) => {
    if (!connectionAuthoritiesMatch(change.oldValue as Partial<TldwConfig> | null, change.newValue as Partial<TldwConfig> | null)) abort()
  } }
  const configChanged = (event: Event) => {
    const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
    if (detail?.authorityChanged !== false || detail.refreshSessionInvalidated) abort()
  }
  const validateInvalidation = () => {
    // A different account's old marker is harmless; canonical credentials decide.
    void tldwClient.ensureConfigForRequest(true).catch(abort)
  }
  const nativeStorageChanged = (event: StorageEvent) => {
    if (event.key?.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)) validateInvalidation()
  }
  const extensionChanges = (changes: Record<string, unknown>) => {
    if (Object.keys(changes).some(key => key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX))) validateInvalidation()
  }
  const extensionEvents = isExtensionRuntime() ? (browser as unknown as {
    storage?: { onChanged?: { addListener: (listener: typeof extensionChanges) => void; removeListener: (listener: typeof extensionChanges) => void } }
  }).storage?.onChanged : undefined
  let released = false
  const release = () => {
    if (released) return
    released = true
    signal?.removeEventListener("abort", abort)
    storage.unwatch(watch)
    window.removeEventListener("tldw:config-updated", configChanged)
    window.removeEventListener("tldw:auth-principal-changed", abort)
    window.removeEventListener("tldw:auth-credentials-changed", abort)
    window.removeEventListener("storage", nativeStorageChanged)
    extensionEvents?.removeListener(extensionChanges)
  }
  try {
    signal?.throwIfAborted()
    signal?.addEventListener("abort", abort, { once: true })
    storage.watch(watch)
    window.addEventListener("tldw:config-updated", configChanged)
    window.addEventListener("tldw:auth-principal-changed", abort)
    window.addEventListener("tldw:auth-credentials-changed", abort)
    window.addEventListener("storage", nativeStorageChanged)
    extensionEvents?.addListener(extensionChanges)
    const snapshot = await loadServicePromptSnapshot([], { signal: controller.signal })
    if (controller.signal.aborted) { snapshot.release(); controller.signal.throwIfAborted() }
    snapshot.scopeSignal.addEventListener("abort", abort, { once: true })
    return { ...snapshot, release: () => {
      snapshot.scopeSignal.removeEventListener("abort", abort)
      release()
      snapshot.release()
    } }
  } catch (error) {
    if (!signal || !onInvalidated || controller.signal.aborted) release()
    throw error
  }
}

const reserveTarget = (target: FlashcardsTransferTarget) => {
  if ("navigate" in target) return { navigate: target.navigate, close: async () => {} }
  if (isExtensionRuntime()) {
    // The WebUI shim intentionally has a different API shape. This branch is
    // reached only with a native extension runtime, and verifies the returned ID.
    const tabs = browser.tabs as unknown as {
      create: (options: { url: string }) => Promise<{ id?: number } | undefined>
      remove: (id: number) => Promise<void>
    }
    let tabId: number | undefined
    return {
      navigate: async (route: string) => {
        const tab = await tabs.create({
          url: browser.runtime.getURL(`options.html#${route}`)
        })
        if (tab?.id == null) throw new Error("The Flashcards tab could not be opened. Keep the source open and try again.")
        tabId = tab.id
      },
      close: async () => { if (tabId != null) await tabs.remove(tabId) }
    }
  }
  // Reserve during the initiating click, before any asynchronous authority or
  // source reads can lose popup activation. Never trust the WebUI tabs shim.
  const popup = window.open("about:blank", "_blank")
  if (!popup) throw new Error("The Flashcards popup was blocked. Allow a new tab and try again; the source is still available.")
  try { popup.opener = null } catch (error) { popup.close(); throw error }
  return {
    navigate: (route: string) => {
      if (popup.closed) throw new Error("The Flashcards tab was closed. Reopen it from the source.")
      popup.location.replace(new URL(route, window.location.origin).toString())
    },
    close: async () => popup.close()
  }
}

/** Capture the owner before source acquisition, and retain source state on failure. */
type PrivateTransferIntent =
  | { kind: "generate"; intent: FlashcardsGenerateIntent }
  | { kind: "study-pack"; intent: StudyPackIntent }

const transferPrivateFlashcardsSource = async (
  acquire: (snapshot: ServicePromptSnapshot) => Promise<PrivateTransferIntent>,
  target: FlashcardsTransferTarget,
  signal?: AbortSignal
): Promise<void> => {
  signal?.throwIfAborted()
  const destination = reserveTarget(target)
  let snapshot: ServicePromptSnapshot | undefined
  let token: string | undefined
  try {
    snapshot = await loadFlashcardsTransferSnapshot(signal)
    snapshot.scopeSignal.throwIfAborted()
    const source = await acquire(snapshot)
    snapshot.scopeSignal.throwIfAborted()
    token = source.kind === "study-pack"
      ? await createStudyPackHandoff(source.intent, flashcardsHandoffAuthority(snapshot), snapshot.scopeSignal)
      : await createFlashcardsGenerateHandoff(source.intent, flashcardsHandoffAuthority(snapshot), snapshot.scopeSignal)
    snapshot.scopeSignal.throwIfAborted()
    await destination.navigate(source.kind === "study-pack" ? buildStudyPackRoute(token) : buildFlashcardsGenerateRoute(token))
    // Same-tab navigation unmounts the source and aborts its hook. Delivery has
    // succeeded; the destination now validates and consumes the owned record.
    if ("newTab" in target) snapshot.scopeSignal.throwIfAborted()
  } catch (error) {
    await destination.close().catch(() => {
      console.warn("Could not close an abandoned Flashcards transfer tab.")
    })
    if (token) await removeFlashcardsGenerateHandoff(token).catch(() => {
      console.warn("Could not remove an abandoned Flashcards transfer.")
    })
    throw error
  } finally {
    snapshot?.release()
  }
}

export const transferFlashcardsSource = (
  acquire: (snapshot: ServicePromptSnapshot) => FlashcardsGenerateIntent | Promise<FlashcardsGenerateIntent>,
  target: FlashcardsTransferTarget, signal?: AbortSignal
) => transferPrivateFlashcardsSource(async snapshot => ({ kind: "generate", intent: await acquire(snapshot) }), target, signal)

export const transferStudyPackSource = (
  acquire: (snapshot: ServicePromptSnapshot) => StudyPackIntent | Promise<StudyPackIntent>,
  target: FlashcardsTransferTarget, signal?: AbortSignal
) => transferPrivateFlashcardsSource(async snapshot => ({ kind: "study-pack", intent: await acquire(snapshot) }), target, signal)

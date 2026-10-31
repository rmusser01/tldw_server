import type { Storage } from "@plasmohq/storage"

/** Certify the backend of this instance, never an unrelated global storage probe. */
export const requirePersistentStorage = (storage: Storage): void => {
  const backend = storage as unknown as {
    readonly hasPersistentBackend?: boolean
    readonly primaryClient?: { get?: unknown; set?: unknown }
    readonly hasExtensionApi?: boolean
    readonly area?: string
  }
  // WebUI explicitly identifies the backend selected at construction.
  if (backend.hasPersistentBackend !== undefined) {
    if (backend.hasPersistentBackend === true) return
  } else if (
    // Installed Plasmo routes get/set to this captured client only while the
    // extension API is active. Its no-client path can silently return/no-op.
    backend.hasExtensionApi === true &&
    (backend.area === "local" || backend.area === "sync") &&
    typeof backend.primaryClient?.get === "function" &&
    typeof backend.primaryClient?.set === "function"
  ) {
    return
  }
  throw new Error("fork_chat_settings_unavailable")
}

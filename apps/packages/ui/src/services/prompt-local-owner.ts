import type { Prompt } from "@/db/dexie/types"
import { resolveRecipePersistenceOwnerView } from "@/services/recipe-persistence-uncertainty"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import { createSafeStorage } from "@/utils/safe-storage"

/** Ownership refusal is distinct from a transient local-storage read failure. */
export class PromptOwnershipError extends Error {}

/** Reuse actual transport ownership; local rows never contain credentials. */
export async function resolvePromptOwnerId(expectedOwnerId?: string | null): Promise<string | null> {
  const ownerId = (await resolveRecipePersistenceOwnerView())?.ownerId ?? null
  if (ownerId === null) {
    const storage = createSafeStorage({ area: "local" })
    let unreadable = false
    const config = await resolveDirectBrowserConfig({
      get: async <T>(key: string): Promise<T | null | undefined> => {
        try { return await storage.get<T>(key) }
        catch (error) { unreadable = true; throw error }
      },
      set: (key, value) => storage.set(key, value),
      remove: key => storage.remove(key)
    })
    if (unreadable || config?.authMode === "multi-user" || config?.authSource === "cookie-session" || config?.credentialSource === "manual" || config?.apiKey || config?.accessToken)
      throw new PromptOwnershipError("Prompt account is unavailable")
  }
  if (expectedOwnerId !== undefined && ownerId !== expectedOwnerId)
    throw new PromptOwnershipError("Prompt account changed")
  return ownerId
}

export type CapturedPromptOwnerScope = {
  ownerId: string | null
  assertCurrent: () => void
  release: () => void
}

/** Capture before file reads; transaction consumers never perform auth I/O. */
export async function capturePromptOwnerScope(expectedOwnerId?: string | null): Promise<CapturedPromptOwnerScope> {
  let current = true
  const retire = () => { current = false }
  const configChanged = (event: Event) => {
    if ((event as CustomEvent<{ authorityChanged?: boolean }>).detail?.authorityChanged !== false) retire()
  }
  const assertCurrent = () => { if (!current) throw new PromptOwnershipError("Prompt account changed") }
  const release = () => {
    current = false
    if (typeof window !== "undefined") {
      window.removeEventListener("tldw:auth-principal-changed", retire)
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }
  if (typeof window !== "undefined") {
    window.addEventListener("tldw:auth-principal-changed", retire)
    window.addEventListener("tldw:config-updated", configChanged)
  }
  try {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId)
    assertCurrent()
    return { ownerId, assertCurrent, release }
  } catch (error) { release(); throw error }
}

export function isPromptOwnedBy(prompt: Prompt, ownerId: string | null): boolean {
  if (prompt.ownerId != null) return prompt.ownerId === ownerId
  // Existing linked rows cannot be assigned to whichever account logs in next.
  return prompt.serverId == null && prompt.studioPromptId == null && prompt.studioProjectId == null &&
    prompt.serverLibraryId == null && prompt.serverLibraryUuid == null &&
    prompt.serverParentVersionId == null && prompt.sourceSystem !== "studio"
}

export function assertPromptOwnedBy(prompt: Prompt | undefined, ownerId: string | null): asserts prompt is Prompt {
  if (!prompt || !isPromptOwnedBy(prompt, ownerId))
    throw new PromptOwnershipError("Prompt owner is unavailable or belongs to another account")
}

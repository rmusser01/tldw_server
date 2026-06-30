import React from "react"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"
import { createLocalRegistryBucket } from "@/services/settings/local-bucket"

const DRAFT_BUCKET_PREFIX = "registry:draft:"
const DRAFT_TTL_MS = 30 * 24 * 60 * 60 * 1000

export type DraftMetadataPrimitive = string | number | boolean | null

export type DraftMetadataValue =
  | DraftMetadataPrimitive
  | DraftMetadataObject
  | DraftMetadataArray

export interface DraftMetadataObject {
  [key: string]: DraftMetadataValue
}

export type DraftMetadataArray = DraftMetadataValue[]

export type DraftMetadata = DraftMetadataObject

type DraftPayload = {
  content: string
  metadata?: DraftMetadata
}

type DraftValue = string | DraftPayload

const draftBucket = createLocalRegistryBucket<DraftValue>({
  prefix: DRAFT_BUCKET_PREFIX,
  ttlMs: DRAFT_TTL_MS
})

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (!value || typeof value !== "object") return false
  const proto = Object.getPrototypeOf(value)
  return proto === Object.prototype || proto === null
}

const isJsonSafe = (value: unknown): value is DraftMetadataValue => {
  if (value === null) return true
  const valueType = typeof value
  if (valueType === "string" || valueType === "number" || valueType === "boolean") {
    return true
  }
  if (Array.isArray(value)) {
    return value.every((item) => isJsonSafe(item))
  }
  if (isPlainObject(value)) {
    return Object.values(value).every((item) => isJsonSafe(item))
  }
  return false
}

const isDraftPayload = (value: DraftValue | null): value is DraftPayload => {
  if (!isPlainObject(value)) return false
  if (!Object.prototype.hasOwnProperty.call(value, "content")) return false
  const payload = value as { content?: unknown }
  return typeof payload.content === "string"
}

const hasDraftContent = (draft: DraftPayload | null) =>
  typeof draft?.content === "string" && draft.content.trim().length > 0

const buildDraftSignature = (
  content: string,
  metadata?: DraftMetadata
): string => {
  if (metadata === undefined) return content
  try {
    return `${content}\u001f${JSON.stringify(metadata)}`
  } catch {
    return `${content}\u001f[metadata-unserializable]`
  }
}

const normalizeDraftValue = (value: DraftValue | null): DraftPayload | null => {
  if (typeof value === "string") {
    return { content: value }
  }
  if (!isDraftPayload(value)) return null
  const metadata =
    isPlainObject(value.metadata) && isJsonSafe(value.metadata)
      ? (value.metadata as DraftMetadata)
      : undefined
  return { content: value.content, metadata }
}

const readLegacyDraft = (storageKey: string): string | null => {
  if (typeof window === "undefined") return null
  try {
    const draft = window.localStorage.getItem(storageKey)
    if (!draft || draft.length === 0) return null
    return draft
  } catch {
    return null
  }
}

const clearLegacyDraft = (storageKey: string) => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(storageKey)
  } catch {
    // ignore legacy storage errors
  }
}

interface DraftPersistenceOptions {
  storageKey: string
  getValue: () => string
  setValue: (value: string) => void
  getMetadata?: () => DraftMetadata | undefined
  setValueWithMetadata?: (value: string, metadata?: DraftMetadata) => void
  enabled?: boolean
}

interface DraftPersistenceResult {
  draftSaved: boolean
  clearDraft: () => void
}

/**
 * Hook for persisting draft messages to a local-only registry bucket.
 *
 * - Restores draft on mount
 * - Persists draft whenever value changes
 * - Shows "Draft saved" indicator briefly after save
 * - Cleans up timeouts on unmount
 */
export const useDraftPersistence = ({
  storageKey,
  getValue,
  setValue,
  getMetadata,
  setValueWithMetadata,
  enabled = true
}: DraftPersistenceOptions): DraftPersistenceResult => {
  const [draftSaved, setDraftSaved] = React.useState(false)
  const [hydrated, setHydrated] = React.useState(!enabled)
  const draftSavedTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const setValueRef = React.useRef(setValue)
  const setValueWithMetadataRef = React.useRef(setValueWithMetadata)
  const getMetadataRef = React.useRef(getMetadata)
  const lastPersistedSignatureRef = React.useRef<string | null>(null)

  React.useEffect(() => {
    setValueRef.current = setValue
  }, [setValue])

  React.useEffect(() => {
    setValueWithMetadataRef.current = setValueWithMetadata
  }, [setValueWithMetadata])

  React.useEffect(() => {
    getMetadataRef.current = getMetadata
  }, [getMetadata])

  // Restore unsent draft on mount
  React.useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setHydrated(false)
    lastPersistedSignatureRef.current = null
    const restoreDraft = async () => {
      const record = await draftBucket.get(storageKey)
      const storedValue = record?.value ?? null
      let draftValue = normalizeDraftValue(storedValue)
      const hasInvalidRecord = storedValue != null && draftValue === null

      if (!hasDraftContent(draftValue)) {
        const legacyDraft = readLegacyDraft(storageKey)
        if (legacyDraft && legacyDraft.trim().length > 0) {
          await draftBucket.set(storageKey, legacyDraft)
          clearLegacyDraft(storageKey)
          draftValue = { content: legacyDraft }
        } else if (legacyDraft) {
          clearLegacyDraft(storageKey)
        } else if (hasInvalidRecord) {
          await draftBucket.remove(storageKey)
        }
      } else {
        clearLegacyDraft(storageKey)
      }

      if (hasDraftContent(draftValue)) {
        lastPersistedSignatureRef.current = buildDraftSignature(
          draftValue.content,
          draftValue.metadata
        )
      } else {
        lastPersistedSignatureRef.current = null
      }

      if (!cancelled && hasDraftContent(draftValue)) {
        const setValueWithMetadata = setValueWithMetadataRef.current
        if (setValueWithMetadata) {
          setValueWithMetadata(draftValue.content, draftValue.metadata)
        } else {
          setValueRef.current(draftValue.content)
        }
      }

      if (!cancelled) {
        setHydrated(true)
      }
    }

    void restoreDraft()
    return () => {
      cancelled = true
    }
  }, [storageKey, enabled])

  React.useEffect(() => {
    if (!enabled) return
    void draftBucket.cleanup()
  }, [enabled])

  // Get current value for effect dependency
  const currentValue = getValue()

  // Persist draft whenever the message changes
  React.useEffect(() => {
    if (!enabled) return
    if (!hydrated) return
    let cancelled = false
    const value = currentValue
    if (persistTimeoutRef.current) {
      clearTimeout(persistTimeoutRef.current)
      persistTimeoutRef.current = null
    }
    if (draftSavedTimeoutRef.current) {
      clearTimeout(draftSavedTimeoutRef.current)
      draftSavedTimeoutRef.current = null
    }
    if (typeof value !== "string") return

    if (value.trim().length === 0) {
      void draftBucket.remove(storageKey)
      clearLegacyDraft(storageKey)
      lastPersistedSignatureRef.current = null
      if (!cancelled) {
        setDraftSaved(false)
      }
      return
    }
    persistTimeoutRef.current = setTimeout(() => {
      void (async () => {
        let metadata: DraftMetadata | undefined
        try {
          metadata = getMetadataRef.current?.() ?? undefined
        } catch {
          metadata = undefined
        }
        if (metadata && (!isPlainObject(metadata) || !isJsonSafe(metadata))) {
          metadata = undefined
        }
        const nextSignature = buildDraftSignature(value, metadata)
        if (nextSignature === lastPersistedSignatureRef.current) {
          return
        }
        setDraftSaved(false)
        const nextValue: DraftValue =
          metadata === undefined ? value : { content: value, metadata }
        await draftBucket.set(storageKey, nextValue)
        clearLegacyDraft(storageKey)
        lastPersistedSignatureRef.current = nextSignature
        if (cancelled) return

        setDraftSaved(true)
        draftSavedTimeoutRef.current = setTimeout(() => {
          setDraftSaved(false)
        }, COMPOSER_CONSTANTS.DRAFT_SAVED_DISPLAY_MS)
      })()
    }, COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS)

    return () => {
      cancelled = true
      if (persistTimeoutRef.current) {
        clearTimeout(persistTimeoutRef.current)
        persistTimeoutRef.current = null
      }
      if (draftSavedTimeoutRef.current) {
        clearTimeout(draftSavedTimeoutRef.current)
        draftSavedTimeoutRef.current = null
      }
    }
  }, [currentValue, storageKey, enabled, hydrated])

  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (draftSavedTimeoutRef.current) {
        clearTimeout(draftSavedTimeoutRef.current)
      }
      if (persistTimeoutRef.current) {
        clearTimeout(persistTimeoutRef.current)
      }
    }
  }, [])

  const clearDraft = React.useCallback(() => {
    void draftBucket.remove(storageKey)
    clearLegacyDraft(storageKey)
    lastPersistedSignatureRef.current = null
    setDraftSaved(false)
  }, [storageKey])

  return {
    draftSaved,
    clearDraft
  }
}

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

const durableDraftBucket = createLocalRegistryBucket<DraftValue>({
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
  tabScoped?: boolean
  /** Discard this older, unowned key instead of migrating its private content. */
  legacyStorageKey?: string
  /** Captures the owner of this render, including synchronous logout invalidation. */
  isCurrent?: () => boolean
  getValue: () => string
  setValue: (value: string) => void
  getMetadata?: () => DraftMetadata | undefined
  setValueWithMetadata?: (value: string, metadata?: DraftMetadata) => void
  enabled?: boolean
}

interface DraftPersistenceResult {
  draftSaved: boolean
  /** True only after this owner/key has finished restoring its saved draft. */
  draftReady: boolean
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
  tabScoped = false,
  legacyStorageKey,
  isCurrent,
  getValue,
  setValue,
  getMetadata,
  setValueWithMetadata,
  enabled = true
}: DraftPersistenceOptions): DraftPersistenceResult => {
  const draftBucket = React.useMemo(() => tabScoped
    ? createLocalRegistryBucket<DraftValue>({ prefix: DRAFT_BUCKET_PREFIX, ttlMs: DRAFT_TTL_MS, tabScoped: true })
    : durableDraftBucket, [tabScoped])
  const [draftSaved, setDraftSaved] = React.useState(false)
  const identity = React.useMemo(() => ({ storageKey, enabled, isCurrent }), [storageKey, enabled, isCurrent])
  const [hydrated, setHydrated] = React.useState<object | null>(null)
  const draftSavedTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingPersistRef = React.useRef<(() => void) | null>(null)
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

  React.useLayoutEffect(() => {
    getMetadataRef.current = getMetadata
  }, [getMetadata])

  // Restore unsent draft on mount
  React.useEffect(() => {
    if (!enabled) return
    let cancelled = false
    const current = () => !cancelled && (!isCurrent || isCurrent())
    setHydrated(null)
    setDraftSaved(false)
    lastPersistedSignatureRef.current = null
    const restoreDraft = async () => {
      if (legacyStorageKey) {
        clearLegacyDraft(legacyStorageKey)
        await durableDraftBucket.remove(legacyStorageKey)
        if (!current()) return
      }
      const record = await draftBucket.get(storageKey)
      if (!current()) return
      const storedValue = record?.value ?? null
      let draftValue = normalizeDraftValue(storedValue)
      const hasInvalidRecord = storedValue != null && draftValue === null

      if (!hasDraftContent(draftValue)) {
        const legacyDraft = legacyStorageKey ? null : readLegacyDraft(storageKey)
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

      if (!current()) return

      if (hasDraftContent(draftValue)) {
        lastPersistedSignatureRef.current = buildDraftSignature(
          draftValue.content,
          draftValue.metadata
        )
      } else {
        lastPersistedSignatureRef.current = null
      }

      if (current() && hasDraftContent(draftValue)) {
        const setValueWithMetadata = setValueWithMetadataRef.current
        if (setValueWithMetadata) {
          setValueWithMetadata(draftValue.content, draftValue.metadata)
        } else {
          setValueRef.current(draftValue.content)
        }
      }

      if (current()) {
        setHydrated(identity)
      }
    }

    void restoreDraft()
    return () => {
      cancelled = true
    }
  }, [storageKey, legacyStorageKey, enabled, identity, isCurrent, draftBucket])

  React.useEffect(() => {
    if (!enabled) return
    void draftBucket.cleanup()
  }, [enabled, draftBucket])

  // Get current value for effect dependency
  const currentValue = getValue()

  // Persist draft whenever the message changes
  React.useLayoutEffect(() => {
    if (!enabled) return
    if (hydrated !== identity || (isCurrent && !isCurrent())) return
    let cancelled = false
    const current = () => !cancelled && (!isCurrent || isCurrent())
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
    const persist = () => {
      if (pendingPersistRef.current === persist) pendingPersistRef.current = null
      if (persistTimeoutRef.current) {
        clearTimeout(persistTimeoutRef.current)
        persistTimeoutRef.current = null
      }
      void (async () => {
        if (!current()) return
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
        if (!current()) return
        clearLegacyDraft(storageKey)
        lastPersistedSignatureRef.current = nextSignature

        setDraftSaved(true)
        draftSavedTimeoutRef.current = setTimeout(() => {
          setDraftSaved(false)
        }, COMPOSER_CONSTANTS.DRAFT_SAVED_DISPLAY_MS)
      })()
    }
    pendingPersistRef.current = persist
    persistTimeoutRef.current = setTimeout(persist, COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS)

    return () => {
      cancelled = true
      if (pendingPersistRef.current === persist) pendingPersistRef.current = null
      if (persistTimeoutRef.current) {
        clearTimeout(persistTimeoutRef.current)
        persistTimeoutRef.current = null
      }
      if (draftSavedTimeoutRef.current) {
        clearTimeout(draftSavedTimeoutRef.current)
        draftSavedTimeoutRef.current = null
      }
    }
  }, [currentValue, storageKey, enabled, hydrated, identity, isCurrent, draftBucket])

  React.useEffect(() => {
    // The tab-scoped bucket pins the record synchronously before its durable
    // write awaits a lock, so a reload cannot drop the pending debounce edit.
    const flush = () => pendingPersistRef.current?.()
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") flush()
    }
    window.addEventListener("pagehide", flush)
    document.addEventListener("visibilitychange", onVisibilityChange)
    return () => {
      window.removeEventListener("pagehide", flush)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [])

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
    if (isCurrent && !isCurrent()) return
    pendingPersistRef.current = null
    if (persistTimeoutRef.current) {
      clearTimeout(persistTimeoutRef.current)
      persistTimeoutRef.current = null
    }
    void draftBucket.remove(storageKey)
    clearLegacyDraft(storageKey)
    lastPersistedSignatureRef.current = null
    setDraftSaved(false)
  }, [storageKey, isCurrent, draftBucket])

  return {
    draftSaved,
    draftReady: enabled && hydrated === identity && (!isCurrent || isCurrent()),
    clearDraft
  }
}

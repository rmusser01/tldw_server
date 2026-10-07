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
  /** A first-ID acknowledgement may rekey this current, hydrated source. */
  bindFromStorageKey?: string
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
  /** Keep an exact in-flight fresh submission until its owner acknowledges it. */
  preserveEmptyDraft?: boolean
  /** Fresh locally held input, such as an image, also prevents first-owner hydration. */
  hasUnpersistedInput?: boolean
}

export type RetainedComposerDraft = DraftPayload & { storageKey: string; updatedAt: number }

interface DraftPersistenceResult {
  /** Flush this currently owned committed edit before resetting its input. */
  flushDraft: () => void
  /** Retire only the exact source of this view's accepted first-ID bind. */
  acknowledgeBoundDraft: (sourceStorageKey: string) => Promise<void>
  listDrafts: () => Promise<{ drafts: RetainedComposerDraft[]; unavailable: boolean }>
  retainDraft: (key: string, draft: DraftPayload) => Promise<boolean>
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
  bindFromStorageKey,
  tabScoped = false,
  legacyStorageKey,
  isCurrent,
  getValue,
  setValue,
  getMetadata,
  setValueWithMetadata,
  enabled = true,
  preserveEmptyDraft = false,
  hasUnpersistedInput = false
}: DraftPersistenceOptions): DraftPersistenceResult => {
  const draftBucket = React.useMemo(() => tabScoped
    ? createLocalRegistryBucket<DraftValue>({ prefix: DRAFT_BUCKET_PREFIX, ttlMs: DRAFT_TTL_MS, tabScoped: true })
    : durableDraftBucket, [tabScoped])
  const [draftSaved, setDraftSaved] = React.useState(false)
  const identity = React.useMemo(() => ({ storageKey, enabled, isCurrent }), [storageKey, enabled, isCurrent])
  const currentIdentityRef = React.useRef(identity)
  const [hydrated, setHydrated] = React.useState<object | null>(null)
  const draftReady = enabled && hydrated === identity && (!isCurrent || isCurrent())
  const boundIdentityRef = React.useRef<{ identity: object; sourceStorageKey: string } | null>(null)
  React.useLayoutEffect(() => {
    const previous = currentIdentityRef.current
    if (previous !== identity) {
      boundIdentityRef.current = enabled && previous.enabled &&
        hydrated === previous && previous.storageKey === bindFromStorageKey &&
        (!previous.isCurrent || previous.isCurrent())
        ? { identity, sourceStorageKey: previous.storageKey } : null
    }
    currentIdentityRef.current = identity
    // A save receipt cannot outlive readable authority or reappear on re-enable.
    if (!draftReady) setDraftSaved(false)
  }, [identity, enabled, hydrated, bindFromStorageKey, draftReady])
  const draftSavedTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingPersistRef = React.useRef<(() => void) | null>(null)
  const setValueRef = React.useRef(setValue)
  const setValueWithMetadataRef = React.useRef(setValueWithMetadata)
  const currentValue = getValue()
  const committedDraftRef = React.useRef<DraftPayload>({ content: currentValue })
  const draftRevisionRef = React.useRef(0)
  const hydrationEditsRef = React.useRef({ identity, changed: false })
  const lastPersistedSignatureRef = React.useRef<string | null>(null)

  React.useEffect(() => {
    setValueRef.current = setValue
  }, [setValue])

  React.useEffect(() => {
    setValueWithMetadataRef.current = setValueWithMetadata
  }, [setValueWithMetadata])

  React.useLayoutEffect(() => {
    let metadata: DraftMetadata | undefined
    try {
      const value = getMetadata?.()
      if (value && isPlainObject(value) && isJsonSafe(value)) {
        metadata = JSON.parse(JSON.stringify(value)) as DraftMetadata
      }
    } catch {
      // Invalid or unavailable metadata must not prevent retaining the text.
    }
    const previous = committedDraftRef.current
    if (hydrationEditsRef.current.identity !== identity) {
      hydrationEditsRef.current = {
        identity,
        changed: (!hydrationEditsRef.current.identity.enabled && enabled && (currentValue.length > 0 || hasUnpersistedInput)) ||
          (hydrationEditsRef.current.identity.enabled && enabled && previous.content !== currentValue &&
            (!hydrationEditsRef.current.identity.isCurrent || hydrationEditsRef.current.identity.isCurrent()))
      }
    } else if (previous.content !== currentValue) {
      hydrationEditsRef.current.changed = true
    }
    if (previous.content !== currentValue) {
      draftRevisionRef.current++
    }
    committedDraftRef.current = { content: currentValue, metadata }
  }, [currentValue, getMetadata, identity, enabled, hasUnpersistedInput])

  // Restore unsent draft on mount
  React.useEffect(() => {
    if (!enabled) return
    // This view already owns the input; assigning its first ID is not a load.
    // Mounts and skipped source transitions have no matching prior identity.
    if (boundIdentityRef.current?.identity === identity) {
      lastPersistedSignatureRef.current = null
      setDraftSaved(false)
      setHydrated(identity)
      return
    }
    let cancelled = false
    const current = () => !cancelled && (!isCurrent || isCurrent())
    const restoreRevision = draftRevisionRef.current
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

      if (current() && draftRevisionRef.current === restoreRevision && !hydrationEditsRef.current.changed) {
        const content = hasDraftContent(draftValue) ? draftValue.content : ""
        const metadata = hasDraftContent(draftValue) ? draftValue.metadata : undefined
        const setValueWithMetadata = setValueWithMetadataRef.current
        if (setValueWithMetadata) {
          setValueWithMetadata(content, metadata)
        } else {
          setValueRef.current(content)
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

  React.useLayoutEffect(() => () => {
    // In-app navigation has no pagehide event. Flush before the persistence
    // cleanup cancels this write and passive owner teardown revokes its lease.
    pendingPersistRef.current?.()
  }, [identity])

  const retireBoundSource = React.useCallback(async (sourceStorageKey?: string) => {
    const bound = boundIdentityRef.current
    const destination = currentIdentityRef.current
    const current = () => boundIdentityRef.current === bound && bound?.identity === destination &&
      currentIdentityRef.current === destination && destination.enabled && (!destination.isCurrent || destination.isCurrent())
    if (!bound || (sourceStorageKey && bound.sourceStorageKey !== sourceStorageKey) || !current()) return
    const record = await draftBucket.get(bound.sourceStorageKey)
    if (!record || !current()) return
    const payload = normalizeDraftValue(record.value)
    if (!payload?.metadata?.chatDraftHandle) return
    const metadata = { ...payload.metadata }
    delete metadata.chatDraftHandle
    // Source text remains reachable until destination durability is proven.
    // Conditional retirement cannot replace another physical tab's newer text.
    await draftBucket.set(bound.sourceStorageKey, { content: payload.content, metadata }, record.updatedAt, record)
    if (current()) boundIdentityRef.current = null
  }, [draftBucket])

  // Persist draft whenever the message changes
  React.useLayoutEffect(() => {
    if (!enabled) return
    // Only genuine edits belong to an unhydrated destination; carried source
    // text must remain untouched until that destination's read completes.
    if ((hydrated !== identity && !hydrationEditsRef.current.changed) || (isCurrent && !isCurrent())) return
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
      if (preserveEmptyDraft) { setDraftSaved(false); return }
      void draftBucket.remove(storageKey)
      clearLegacyDraft(storageKey)
      lastPersistedSignatureRef.current = null
      if (!cancelled) {
        setDraftSaved(false)
      }
      return
    }
    const persist = () => {
      if (persistTimeoutRef.current) {
        clearTimeout(persistTimeoutRef.current)
        persistTimeoutRef.current = null
      }
      void (async () => {
        if (!current()) return
        // Key-change cleanup runs before the destination layout updates this
        // snapshot, retaining source text and metadata as one committed draft.
        const metadata = committedDraftRef.current.metadata
        const nextSignature = buildDraftSignature(value, metadata)
        if (nextSignature === lastPersistedSignatureRef.current) {
          return
        }
        if (draftSavedTimeoutRef.current) clearTimeout(draftSavedTimeoutRef.current)
        setDraftSaved(false)
        const nextValue: DraftValue =
          metadata === undefined ? value : { content: value, metadata }
        const acknowledged = await draftBucket.set(storageKey, nextValue)
        if (!current() || acknowledged === false) return
        clearLegacyDraft(storageKey)
        lastPersistedSignatureRef.current = nextSignature

        setDraftSaved(true)
        draftSavedTimeoutRef.current = setTimeout(() => {
          setDraftSaved(false)
        }, COMPOSER_CONSTANTS.DRAFT_SAVED_DISPLAY_MS)
        await retireBoundSource()
      })()
    }
    // Retain the callback after debounce so departure also flushes later
    // collapse-only changes without autosaving on metadata callback changes.
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
  }, [currentValue, storageKey, enabled, hydrated, identity, isCurrent, draftBucket, retireBoundSource, preserveEmptyDraft])

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

  const flushDraft = React.useCallback(() => {
    if (currentIdentityRef.current !== identity || (isCurrent && !isCurrent())) return
    pendingPersistRef.current?.()
  }, [identity, isCurrent])

  const clearDraft = React.useCallback(() => {
    if (currentIdentityRef.current !== identity || (isCurrent && !isCurrent())) return
    draftRevisionRef.current++
    pendingPersistRef.current = null
    if (persistTimeoutRef.current) {
      clearTimeout(persistTimeoutRef.current)
      persistTimeoutRef.current = null
    }
    void draftBucket.remove(storageKey)
    clearLegacyDraft(storageKey)
    lastPersistedSignatureRef.current = null
    setDraftSaved(false)
  }, [storageKey, isCurrent, draftBucket, identity])

  const canRead = React.useCallback(() => enabled && hydrated === identity && currentIdentityRef.current === identity && (!isCurrent || isCurrent()), [enabled, hydrated, identity, isCurrent])
  const listDrafts = React.useCallback(async () => {
    if (!canRead()) return { drafts: [], unavailable: false }
    const result = await draftBucket.list()
    if (!canRead()) return { drafts: [], unavailable: false }
    const drafts = Object.entries(result.records).flatMap(([key, record]) => {
      const payload = normalizeDraftValue(record.value)
      return hasDraftContent(payload) ? [{ ...payload!, storageKey: key, updatedAt: record.updatedAt }] : []
    })
    return { drafts, unavailable: result.unavailable }
  }, [draftBucket, canRead])
  const retainDraft = React.useCallback(async (key: string, draft: DraftPayload) => {
    if (!canRead()) return false
    const acknowledged = await draftBucket.set(key, draft)
    return acknowledged && canRead()
  }, [draftBucket, canRead])

  return {
    flushDraft,
    acknowledgeBoundDraft: retireBoundSource,
    listDrafts,
    retainDraft,
    draftSaved: draftSaved && draftReady,
    draftReady,
    clearDraft
  }
}

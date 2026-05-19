/**
 * Hook for auto-saving form drafts to localStorage
 * Provides form persistence across browser sessions with recovery prompt
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { checkStorageBeforeWrite, notifyStorageWrite } from '@/utils/storage-guard'
import { estimateStorageCost } from '@/utils/storage-budget'

export interface FormDraft<T = Record<string, any>> {
  formData: T
  formType: 'create' | 'edit'
  editId?: string
  savedAt: number
}

const DRAFT_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000 // 7 days

interface UseFormDraftOptions<T> {
  /** Storage key for this form's draft */
  storageKey: string
  /** Interval in ms for auto-save (default: 30000 = 30 seconds) */
  autoSaveInterval?: number
  /** Called when a recoverable draft is found */
  onDraftFound?: (draft: FormDraft<T>) => void
  /** Form type for this instance */
  formType: 'create' | 'edit'
  /** Edit ID if editing an existing item */
  editId?: string
}

interface UseFormDraftReturn<T> {
  /** Whether there's a recoverable draft available */
  hasDraft: boolean
  /** The recovered draft data (null if none) */
  draftData: FormDraft<T> | null
  /** Save current form data to draft */
  saveDraft: (formData: T) => void
  /** Clear the saved draft */
  clearDraft: () => void
  /** Apply the recovered draft (clears hasDraft state) */
  applyDraft: () => T | null
  /** Dismiss the draft without applying */
  dismissDraft: () => void
  /** Last save timestamp */
  lastSaved: number | null
  /** Advisory storage warning from the last save attempt (null if OK) */
  storageWarning: string | null
}

export function useFormDraft<T = Record<string, any>>(
  options: UseFormDraftOptions<T>
): UseFormDraftReturn<T> {
  const {
    storageKey,
    autoSaveInterval = 30000,
    onDraftFound,
    formType,
    editId
  } = options

  const [hasDraft, setHasDraft] = useState(false)
  const [draftData, setDraftData] = useState<FormDraft<T> | null>(null)
  const [lastSaved, setLastSaved] = useState<number | null>(null)
  const [storageWarning, setStorageWarning] = useState<string | null>(null)
  const autoSaveRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pendingDataRef = useRef<T | null>(null)
  const onDraftFoundRef = useRef(onDraftFound)

  // Keep callback ref updated
  useEffect(() => {
    onDraftFoundRef.current = onDraftFound
  }, [onDraftFound])

  // Check for existing draft on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(storageKey)
      if (!stored) return

      const draft: FormDraft<T> = JSON.parse(stored)

      // Check if draft has expired
      if (Date.now() - draft.savedAt > DRAFT_EXPIRY_MS) {
        localStorage.removeItem(storageKey)
        return
      }

      // For edit forms, only recover if it's the same item
      if (formType === 'edit' && draft.formType === 'edit') {
        if (draft.editId !== editId) {
          // Different item being edited, don't recover
          return
        }
      }

      // For create forms, recover any create draft
      if (formType === 'create' && draft.formType !== 'create') {
        return
      }

      setDraftData(draft)
      setHasDraft(true)
      onDraftFoundRef.current?.(draft)
    } catch (e) {
      // Invalid draft data, remove it
      localStorage.removeItem(storageKey)
    }
  }, [storageKey, formType, editId])

  const saveDraft = useCallback(
    (formData: T) => {
      pendingDataRef.current = formData

      const draft: FormDraft<T> = {
        formData,
        formType,
        editId,
        savedAt: Date.now()
      }

      try {
        const serialized = JSON.stringify(draft)
        const guard = checkStorageBeforeWrite(estimateStorageCost(serialized), storageKey)
        setStorageWarning(guard.recommendation)
        if (guard.recommendation) {
          console.warn('[useFormDraft]', guard.recommendation)
        }
        // Advisory only — always attempt the write
        localStorage.setItem(storageKey, serialized)
        notifyStorageWrite()
        setLastSaved(draft.savedAt)
      } catch (e) {
        // localStorage might be full or disabled
        console.warn('Failed to save form draft:', e)
      }
    },
    [storageKey, formType, editId]
  )

  const clearDraft = useCallback(() => {
    localStorage.removeItem(storageKey)
    setHasDraft(false)
    setDraftData(null)
    setLastSaved(null)
    pendingDataRef.current = null
  }, [storageKey])

  const applyDraft = useCallback(() => {
    if (!draftData) return null
    setHasDraft(false)
    return draftData.formData
  }, [draftData])

  const dismissDraft = useCallback(() => {
    clearDraft()
  }, [clearDraft])

  // Set up auto-save interval
  useEffect(() => {
    if (autoSaveInterval <= 0) return

    autoSaveRef.current = setInterval(() => {
      if (pendingDataRef.current) {
        saveDraft(pendingDataRef.current)
      }
    }, autoSaveInterval)

    return () => {
      if (autoSaveRef.current) {
        clearInterval(autoSaveRef.current)
      }
    }
  }, [autoSaveInterval, saveDraft])

  return {
    hasDraft,
    draftData,
    saveDraft,
    clearDraft,
    applyDraft,
    dismissDraft,
    lastSaved,
    storageWarning
  }
}

/**
 * Format a timestamp for display (e.g., "2 minutes ago")
 */
export function formatDraftAge(savedAt: number): string {
  const now = Date.now()
  const diffMs = now - savedAt
  const diffMinutes = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMinutes < 1) return 'just now'
  if (diffMinutes === 1) return '1 minute ago'
  if (diffMinutes < 60) return `${diffMinutes} minutes ago`
  if (diffHours === 1) return '1 hour ago'
  if (diffHours < 24) return `${diffHours} hours ago`
  if (diffDays === 1) return '1 day ago'
  return `${diffDays} days ago`
}

import React from "react"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import {
  useDraftPersistence,
  type DraftMetadataObject
} from "@/hooks/useDraftPersistence"

/**
 * Shared composer text primitive consumed by both Playground and Sidepanel.
 *
 * Bundles three pieces every composer needs:
 *   - message/image form state (`useSimpleForm`)
 *   - draft persistence keyed per surface (`useDraftPersistence`)
 *   - a focus helper that no-ops on mobile to avoid unwanted keyboard pop-ups
 *
 * Auto-resize (`useDynamicTextareaSize`) is intentionally NOT bundled here —
 * Playground sizes against a derived display value (paste-collapse overlay),
 * while Sidepanel sizes against the raw form value. Callers wire it in one
 * line using `textareaMaxHeight` from this hook. Passing ownership to the
 * caller is simpler than gating auto-resize behind an option flag.
 *
 * This is the foundation of the Phase 2 hook extraction. Playground's
 * `useComposerInput` wraps this and adds collapse/perf overlays; Sidepanel
 * adopts it directly, replacing its inline equivalents in `form.tsx`.
 */

export interface UseComposerTextOptions {
  /** Persistence key for draft messages. Each surface uses its own key. */
  draftKey: string
  /** Textarea ref owned by the caller (usually the composer component). */
  textareaRef: React.RefObject<HTMLTextAreaElement>
  /** Pro mode gets a taller textarea (160px) vs casual (120px). */
  isProMode?: boolean
  /** Explicit max height override; wins over `isProMode`. */
  maxHeight?: number
  /**
   * Optional hook into draft-restore to carry surface-specific metadata
   * (e.g., Playground's `collapsedRange` + `wasExpanded`).
   */
  getDraftMetadata?: () => DraftMetadataObject | undefined
  /**
   * Called by `useDraftPersistence` when a stored draft is restored with
   * metadata. Callers override when they need to restore collapse state etc.
   */
  restoreWithMetadata?: (value: string, metadata?: DraftMetadataObject) => void
  /** Disable draft persistence entirely. Default true. */
  draftEnabled?: boolean
}

export interface UseComposerTextResult {
  /** The underlying `useSimpleForm` instance. Callers can use `getInputProps("message")`. */
  form: ReturnType<typeof useSimpleForm<{ message: string; image: string }>>
  /** Convenience setter for the message field. */
  setMessageValue: (value: string) => void
  /** Focuses the textarea (no-op on mobile to avoid unwanted keyboard pop-up). */
  textAreaFocus: () => void
  /** Whether the draft was recently persisted — for the "Draft saved" indicator. */
  draftSaved: boolean
  /** Imperative draft clear (call after a successful send). */
  clearDraft: () => void
  /** Computed max height actually applied to the textarea. Surfaced for tests. */
  textareaMaxHeight: number
}

const DEFAULT_MAX_HEIGHT_CASUAL = 120
const DEFAULT_MAX_HEIGHT_PRO = 160

export function useComposerText(
  options: UseComposerTextOptions
): UseComposerTextResult {
  const {
    draftKey,
    textareaRef,
    isProMode = false,
    maxHeight: explicitMaxHeight,
    getDraftMetadata,
    restoreWithMetadata,
    draftEnabled = true,
  } = options

  const form = useSimpleForm<{ message: string; image: string }>({
    initialValues: { message: "", image: "" },
  })

  const setMessageValue = React.useCallback(
    (value: string) => {
      form.setFieldValue("message", value)
    },
    [form]
  )

  const restoreMessage = React.useCallback(
    (value: string, metadata?: DraftMetadataObject) => {
      form.setFieldValue("message", value)
      restoreWithMetadata?.(value, metadata)
    },
    [form, restoreWithMetadata]
  )

  const { draftSaved, clearDraft } = useDraftPersistence({
    storageKey: draftKey,
    getValue: () => form.values.message,
    getMetadata: getDraftMetadata,
    setValue: (value) => setMessageValue(value),
    setValueWithMetadata: restoreMessage,
    enabled: draftEnabled,
  })

  const textareaMaxHeight =
    explicitMaxHeight ??
    (isProMode ? DEFAULT_MAX_HEIGHT_PRO : DEFAULT_MAX_HEIGHT_CASUAL)

  const textAreaFocus = React.useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    if (el.selectionStart === el.selectionEnd) {
      const ua = typeof navigator !== "undefined" ? navigator.userAgent : ""
      const isMobile =
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua)
      if (!isMobile) {
        el.focus()
      } else {
        el.blur()
      }
    }
  }, [textareaRef])

  return {
    form,
    setMessageValue,
    textAreaFocus,
    draftSaved,
    clearDraft,
    textareaMaxHeight,
  }
}

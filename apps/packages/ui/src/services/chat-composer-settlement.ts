export const CAPTURE_SAVED_COMPOSER_INPUT_EVENT =
  "tldw:chat:capture-saved-input"

export type SavedComposerInputCapture = {
  inputText: string
  capture: (acknowledge: () => boolean) => void
}

/** Keep Retry's original composer receipt through the server wait. */
export const captureSavedComposerInput = (
  inputText: string
): (() => boolean) | undefined => {
  if (typeof window === "undefined") return
  const candidates: Array<() => boolean> = []
  window.dispatchEvent(
    new CustomEvent<SavedComposerInputCapture>(
      CAPTURE_SAVED_COMPOSER_INPUT_EVENT,
      {
        detail: {
          inputText,
          capture: (acknowledge) => candidates.push(acknowledge)
        }
      }
    )
  )
  return candidates.length === 1 ? candidates[0] : undefined
}

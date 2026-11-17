import { useEffect, useRef } from "react"
import { captureChatDraftSource } from "./chat-draft-transition"
import type { HistorySelectionController } from "./useHistorySelection"

export const FOCUS_CHECK_DEBOUNCE_MS = 400
export const FOCUS_CHECK_MIN_INTERVAL_MS = 5_000

/** Only active conversation surfaces mount this listener; preparation controllers remain passive. */
export const useChatHistoryFreshness = (
  selection: HistorySelectionController,
  isEligible: () => boolean = () => true
) => {
  const live = useRef({ selection, isEligible })
  live.current = { selection, isEligible }
  useEffect(() => {
    let mounted = true
    let timer: ReturnType<typeof setTimeout> | null = null
    let last: {
      view: HistorySelectionController["view"]
      time: number
    } | null = null
    const schedule = () => {
      if (timer) clearTimeout(timer)
      const controller = live.current.selection
      const view = controller.getCurrent().view
      const draft = captureChatDraftSource(controller.fence())
      const current = () =>
        mounted &&
        live.current.selection.getCurrent().view === view &&
        draft.isCurrent() &&
        live.current.isEligible()
      timer = setTimeout(() => {
        timer = null
        if (!current() || !controller.checkFreshness) return
        if (
          last?.view === view &&
          Date.now() - last.time < FOCUS_CHECK_MIN_INTERVAL_MS
        )
          return
        last = { view, time: Date.now() }
        void controller.checkFreshness(current)
      }, FOCUS_CHECK_DEBOUNCE_MS)
    }
    const visible = () => {
      if (document.visibilityState !== "hidden") schedule()
    }
    window.addEventListener("focus", schedule)
    document.addEventListener("visibilitychange", visible)
    return () => {
      mounted = false
      window.removeEventListener("focus", schedule)
      document.removeEventListener("visibilitychange", visible)
      if (timer) clearTimeout(timer)
    }
  }, [])
}

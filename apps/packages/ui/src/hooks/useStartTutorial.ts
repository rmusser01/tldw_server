import { useShallow } from "zustand/react/shallow"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useCallback, useLayoutEffect, useRef } from "react"
import { useLocation } from "react-router-dom"
import { useTutorialStore, type TutorialFocusReturn } from "@/store/tutorials"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

type StartTutorialOptions = {
  /**
   * What makes a launcher stale. "address" (the default): any change of the
   * address since it rendered. "page": only a change of page, so a same-page
   * address change (opening a note writes /notes?note=<id>) keeps it.
   */
  fence?: "address" | "page"
}

/** Transfer explicit launcher intent before its Help/Quick modal is removed. */
export const useStartTutorial = ({ fence = "address" }: StartTutorialOptions = {}) => {
  const location = useLocation()
  const pathname = location.pathname
  const selection = useHistorySelectionContext()
  const start = useTutorialStore((state) => state.startTutorial)
  const route = JSON.stringify([
    location.key,
    location.pathname,
    location.search,
    location.hash
  ])
  useStoreMessageOption(
    useShallow((state) => [
      state.chatDraftRevision,
      state.chatDraftAddress,
      state.historyId,
      state.serverChatId
    ])
  )
  usePlaygroundSessionStore(
    useShallow((state) => [
      state.restoreRevision,
      state.sourceSelectionRevision
    ])
  )
  const source = captureChatDraftSource(selection?.fence())
  const owner = selection?.owner,
    account = selection?.draftAccount
  const currentRoute = useRef(route)
  const currentPathname = useRef(pathname)
  useLayoutEffect(() => {
    currentRoute.current = route
    currentPathname.current = pathname
  }, [route, pathname])
  const mounted = useRef(true)
  useLayoutEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  return useCallback(
    (tutorialId: string, target?: HTMLElement | null) => {
      const sourceIsCurrent = () =>
        (fence === "page"
          ? pathname === currentPathname.current
          : route === currentRoute.current) &&
        source.isCurrent() &&
        (!owner ||
          owner.kind === "unavailable" ||
          owner.validate_lease?.() !== false) &&
        (!account?.requestScope || account.isCurrent())
      if (!mounted.current || !sourceIsCurrent()) return false
      // The run belongs to the address it starts on (the launcher's own,
      // unless a page-fenced launcher survived a same-page address change).
      const startRoute = currentRoute.current
      const launcher = document.activeElement
      let owned = true,
        enteredTour = false
      let lastFocusedControl: HTMLElement | null = null
      const inTour = (target: EventTarget | null) =>
        target instanceof Element &&
        !!target.closest(
          "#react-joyride-portal, .react-joyride__tooltip, .react-joyride__beacon"
        )
      const onIntent = (event: Event) => {
        if (event.type === "input") owned = false
        else if (inTour(event.target)) {
          enteredTour = true
          if (event.type === "focusin" && event.target instanceof HTMLElement)
            lastFocusedControl = event.target
        } else if (
          event.type === "pointerdown" ||
          enteredTour ||
          event.target !== launcher
        )
          owned = false
      }
      const onVisibility = () => {
        if (document.visibilityState !== "visible") owned = false
      }
      const stopAccount = watchChatAccountChanges((invalidated) => {
        if (invalidated) owned = false
      })
      const intent: TutorialFocusReturn = {
        target:
          target === undefined
            ? launcher instanceof HTMLElement && launcher !== document.body
              ? launcher
              : null
            : target,
        route: startRoute,
        isCurrent: () => owned && sourceIsCurrent(),
        lastFocusedControl: () => lastFocusedControl,
        dispose: () => {
          document.removeEventListener("focusin", onIntent)
          document.removeEventListener("pointerdown", onIntent)
          document.removeEventListener("input", onIntent)
          document.removeEventListener("visibilitychange", onVisibility)
          stopAccount()
        }
      }
      document.addEventListener("focusin", onIntent)
      document.addEventListener("pointerdown", onIntent)
      document.addEventListener("input", onIntent)
      document.addEventListener("visibilitychange", onVisibility)
      start(tutorialId, intent)
      const run = useTutorialStore.getState().activeRun
      if (run?.focusReturn !== intent || !sourceIsCurrent()) {
        if (run?.focusReturn === intent)
          useTutorialStore.getState().endTutorial(run)
        return false
      }
      return true
    },
    [start, source, owner, account, route, pathname, fence]
  )
}

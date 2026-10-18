import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

export type PaletteFocusRequest = {
  target: HTMLElement | null
  isCurrent: () => boolean
  isSourceCurrent: () => boolean
  isOpen: () => boolean
  setRoot: (root: HTMLElement | null) => void
  dispose: () => void
}

/** Capture the explicit opener before a demand-loaded palette can mount. */
export const capturePaletteFocusRequest = (
  selection: HistorySelectionController | null
): PaletteFocusRequest => {
  const source = captureChatDraftSource(selection?.fence())
  const owner = selection?.owner,
    account = selection?.draftAccount
  const opener = document.activeElement
  const target =
    opener instanceof HTMLElement && opener !== document.body ? opener : null
  let active = true,
    owned = true,
    entered = false,
    root: HTMLElement | null = null
  const onIntent = (event: Event) => {
    if (event.target instanceof Node && root?.contains(event.target)) {
      entered = true
      return
    }
    if (event.type !== "focusin" || entered || event.target !== target)
      owned = false
  }
  const onVisibility = () => {
    if (document.visibilityState !== "visible") owned = false
  }
  const stopAccount = watchChatAccountChanges((invalidated) => {
    if (invalidated) {
      active = false
      owned = false
    }
  })
  document.addEventListener("focusin", onIntent)
  document.addEventListener("pointerdown", onIntent)
  document.addEventListener("input", onIntent)
  document.addEventListener("visibilitychange", onVisibility)
  const isSourceCurrent = () =>
    active &&
    source.isCurrent() &&
    document.visibilityState === "visible" &&
    (!owner ||
      owner.kind === "unavailable" ||
      owner.validate_lease?.() !== false) &&
    (!account?.requestScope || account.isCurrent())
  return {
    target,
    isCurrent: () => owned && isSourceCurrent(),
    isSourceCurrent,
    isOpen: () => root?.isConnected === true,
    setRoot: (element) => {
      root = element
      if (root?.contains(document.activeElement)) entered = true
    },
    dispose: () => {
      active = false
      owned = false
      document.removeEventListener("focusin", onIntent)
      document.removeEventListener("pointerdown", onIntent)
      document.removeEventListener("input", onIntent)
      document.removeEventListener("visibilitychange", onVisibility)
      stopAccount()
    }
  }
}

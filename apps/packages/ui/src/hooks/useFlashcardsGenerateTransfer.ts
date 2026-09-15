import React from "react"
import { transferFlashcardsSource } from "@/services/tldw/flashcards-generate-transfer"

/** A newer click or source unmount cancels a handoff before delivery. */
export const useFlashcardsGenerateTransfer = () => {
  const current = React.useRef<AbortController | null>(null)
  React.useEffect(() => () => current.current?.abort(), [])
  const transfer = React.useCallback((...args: [Parameters<typeof transferFlashcardsSource>[0], Parameters<typeof transferFlashcardsSource>[1]]) => {
    current.current?.abort()
    const controller = new AbortController()
    current.current = controller
    return transferFlashcardsSource(args[0], args[1], controller.signal).finally(() => {
      if (current.current === controller) current.current = null
    })
  }, [])
  return React.useMemo(() => Object.assign(transfer, { cancel: () => current.current?.abort() }), [transfer])
}

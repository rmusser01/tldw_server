import React from "react"
import { transferFlashcardsSource, transferStudyPackSource, type FlashcardsTransferTarget } from "@/services/tldw/flashcards-generate-transfer"

/** A newer click or source unmount cancels a handoff before delivery. */
const usePrivateFlashcardsTransfer = <T,>(send: (acquire: T, target: FlashcardsTransferTarget, signal?: AbortSignal) => Promise<void>) => {
  const current = React.useRef<AbortController | null>(null)
  React.useEffect(() => () => current.current?.abort(), [])
  const transfer = React.useCallback((...args: [T, FlashcardsTransferTarget]) => {
    current.current?.abort()
    const controller = new AbortController()
    current.current = controller
    return send(args[0], args[1], controller.signal).finally(() => {
      if (current.current === controller) current.current = null
    })
  }, [send])
  return React.useMemo(() => Object.assign(transfer, { cancel: () => current.current?.abort() }), [transfer])
}

export const useFlashcardsGenerateTransfer = () => usePrivateFlashcardsTransfer(transferFlashcardsSource)
export const useStudyPackTransfer = () => usePrivateFlashcardsTransfer(transferStudyPackSource)

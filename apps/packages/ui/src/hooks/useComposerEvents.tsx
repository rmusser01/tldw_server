import React from "react"
import type { QuickIngestOpenDetail } from "@/utils/quick-ingest-open"

export interface ComposerEventHandlers {
  /** Called when tldw:focus-composer event is fired */
  onFocusComposer?: () => void
  /** Called when tldw:open-quick-ingest event is fired */
  onOpenQuickIngest?: (detail?: QuickIngestOpenDetail) => void
}

export interface DocumentGeneratorSeed {
  conversationId?: string | null
  message?: string | null
  messageId?: string | null
}

export interface UseComposerEventsResult {
  // Modal open states
  openActorSettings: boolean
  setOpenActorSettings: React.Dispatch<React.SetStateAction<boolean>>
  openModelSettings: boolean
  setOpenModelSettings: React.Dispatch<React.SetStateAction<boolean>>
  documentGeneratorOpen: boolean
  setDocumentGeneratorOpen: React.Dispatch<React.SetStateAction<boolean>>
  documentGeneratorSeed: DocumentGeneratorSeed
  setDocumentGeneratorSeed: React.Dispatch<React.SetStateAction<DocumentGeneratorSeed>>
}

/**
 * Hook that consolidates window event listeners for the composer.
 * Handles events like:
 * - tldw:open-actor-settings
 * - tldw:open-model-settings
 * - tldw:focus-composer
 * - tldw:open-document-generator
 * - tldw:open-quick-ingest
 */
export const useComposerEvents = (
  options: ComposerEventHandlers & { serverChatId?: string | null }
): UseComposerEventsResult => {
  const { onFocusComposer, onOpenQuickIngest, serverChatId } = options

  const [openActorSettings, setOpenActorSettings] = React.useState(false)
  const [openModelSettings, setOpenModelSettings] = React.useState(false)
  const [documentGeneratorOpen, setDocumentGeneratorOpen] = React.useState(false)
  const [documentGeneratorSeed, setDocumentGeneratorSeed] = React.useState<DocumentGeneratorSeed>({})

  // tldw:open-actor-settings
  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = () => setOpenActorSettings(true)
    window.addEventListener("tldw:open-actor-settings", handler)
    return () => {
      window.removeEventListener("tldw:open-actor-settings", handler)
    }
  }, [])

  // tldw:open-model-settings
  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = () => setOpenModelSettings(true)
    window.addEventListener("tldw:open-model-settings", handler)
    return () => {
      window.removeEventListener("tldw:open-model-settings", handler)
    }
  }, [])

  // tldw:focus-composer
  React.useEffect(() => {
    if (!onFocusComposer) return
    const handler = () => {
      if (document.visibilityState === "visible") {
        onFocusComposer()
      }
    }
    window.addEventListener("tldw:focus-composer", handler)
    return () => window.removeEventListener("tldw:focus-composer", handler)
  }, [onFocusComposer])

  // tldw:open-document-generator
  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = (event: Event) => {
      const detail = (event as CustomEvent)?.detail || {}
      setDocumentGeneratorSeed({
        conversationId: detail?.conversationId ?? serverChatId ?? null,
        message: detail?.message ?? null,
        messageId: detail?.messageId ?? null
      })
      setDocumentGeneratorOpen(true)
    }
    window.addEventListener("tldw:open-document-generator", handler)
    return () => {
      window.removeEventListener("tldw:open-document-generator", handler)
    }
  }, [serverChatId])

  // tldw:open-quick-ingest
  React.useEffect(() => {
    if (!onOpenQuickIngest) return
    const handler = (event: Event) => {
      onOpenQuickIngest((event as CustomEvent<QuickIngestOpenDetail>).detail)
    }
    window.addEventListener("tldw:open-quick-ingest", handler)
    return () => {
      window.removeEventListener("tldw:open-quick-ingest", handler)
    }
  }, [onOpenQuickIngest])

  return {
    openActorSettings,
    setOpenActorSettings,
    openModelSettings,
    setOpenModelSettings,
    documentGeneratorOpen,
    setDocumentGeneratorOpen,
    documentGeneratorSeed,
    setDocumentGeneratorSeed
  }
}

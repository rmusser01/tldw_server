import React, { useCallback, useState, useRef, useEffect, useLayoutEffect } from "react"
import { useTranslation } from "react-i18next"
import { Button, Dropdown, message, Modal, Spin, Select, Tooltip } from "antd"
import type { MenuProps } from "antd"
import {
  Copy,
  Highlighter,
  Languages,
  MessageSquare,
  ChevronDown,
  Volume2,
  VolumeX
} from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { useTranslate } from "@/hooks/document-workspace/useTranslate"
import { useDocumentTTS } from "@/hooks/document-workspace/useDocumentTTS"
import { useMediaQuery, useMobile } from "@/hooks/useMediaQuery"
import type { AnnotationColor } from "../types"
import { HIGHLIGHT_COLORS, TARGET_LANGUAGES } from "../config"

interface TextSelectionPopoverProps {
  documentId: number
  text: string
  position: { x: number; y: number }
  onClose: () => void
  /** Optional CFI location for EPUB annotations */
  epubCfi?: string
  pdfPage?: number
}

type SelectionMenu = "highlight" | "ask-ai"

const selectionMenuClass = "document-selection-menu"

export const TextSelectionPopover: React.FC<TextSelectionPopoverProps> = ({
  documentId,
  text,
  position,
  onClose,
  epubCfi,
  pdfPage
}) => {
  const { t } = useTranslation(["option", "common"])
  const popoverRef = useRef<HTMLDivElement>(null)
  const focusOrigin = useRef<HTMLElement | null>(null)
  const translationRequest = useRef<AbortController | null>(null)
  const [openMenu, setOpenMenu] = useState<SelectionMenu | null>(null)
  const [translateModalOpen, setTranslateModalOpen] = useState(false)
  const [translatedText, setTranslatedText] = useState<string | null>(null)
  const [translationFailed, setTranslationFailed] = useState(false)
  const [targetLanguage, setTargetLanguage] = useState("English")
  const [adjustedPosition, setAdjustedPosition] = useState(position)
  const [positionReady, setPositionReady] = useState(false)

  const { mutateAsync: translate, isPending: translating } = useTranslate()
  const { speak, state: ttsState, stop: stopTTS } = useDocumentTTS()

  const [workspaceAuthority] = useState(() => useDocumentWorkspaceStore.getState().workspaceAuthority)
  const [{ currentPage, currentPercentage, currentChapterTitle }] = useState(() => {
    const state = useDocumentWorkspaceStore.getState()
    return {
      currentPage: state.currentPage,
      currentPercentage: state.currentPercentage,
      currentChapterTitle: state.currentChapterTitle
    }
  })
  const addAnnotation = useDocumentWorkspaceStore((s) => s.addAnnotation)
  const queueSelectionPrompt = useDocumentWorkspaceStore(s => s.queueSelectionPrompt)

  const isMobileView = useMobile()
  const hasCoarsePointer = useMediaQuery("(any-pointer: coarse)")
  const showLabels = isMobileView || hasCoarsePointer
  const isCurrentDocument = useCallback(() => {
    const current = useDocumentWorkspaceStore.getState()
    return Boolean(workspaceAuthority) && current.activeDocumentId === documentId &&
      current.workspaceAuthority === workspaceAuthority
  }, [documentId, workspaceAuthority])

  useLayoutEffect(() => {
    focusOrigin.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    return () => translationRequest.current?.abort()
  }, [])

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      if (translateModalOpen || (event.target instanceof Element &&
          event.target.closest(`[data-selection-popover], .${selectionMenuClass}`))) return
      onClose()
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented || translateModalOpen) return
      if (openMenu) {
        setOpenMenu(null)
        popoverRef.current?.querySelector<HTMLButtonElement>('button[aria-expanded="true"]')?.focus()
      } else {
        onClose()
        focusOrigin.current?.focus({ preventScroll: true })
      }
      e.preventDefault()
      e.stopPropagation()
    }
    document.addEventListener("pointerdown", handlePointerDown)
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown)
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [onClose, openMenu, translateModalOpen])

  // Adjust position to keep popover in viewport
  useLayoutEffect(() => {
    if (isMobileView) {
      setAdjustedPosition(position)
      setPositionReady(true)
      return
    }

    if (!popoverRef.current) return

    setPositionReady(false)

    const rect = popoverRef.current.getBoundingClientRect()
    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight

    let x = position.x
    let y = position.y

    // Adjust horizontal position if going off-screen
    if (x + rect.width > viewportWidth - 16) {
      x = viewportWidth - rect.width - 16
    }
    if (x < 16) {
      x = 16
    }

    // Adjust vertical position if going off-screen
    if (y + rect.height > viewportHeight - 16) {
      y = position.y - rect.height - 8 // Show above selection
    }
    if (y < 16) {
      y = 16
    }

    setAdjustedPosition({ x, y })
    const animationFrame = window.requestAnimationFrame(() => {
      setPositionReady(true)
    })

    return () => window.cancelAnimationFrame(animationFrame)
  }, [isMobileView, position])

  const copyText = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      message.success(t("option:documentWorkspace.copiedToClipboard", "Copied to clipboard"))
      return true
    } catch (err) {
      message.error(t("option:documentWorkspace.copyFailed", "Failed to copy"))
      return false
    }
  }, [t])

  // Highlight with color
  const handleHighlight = useCallback(
    (color: AnnotationColor) => {
      if (!isCurrentDocument()) return

      // Use CFI for EPUB, page number for PDF
      const location = epubCfi || pdfPage || currentPage
      const isEpub = !!epubCfi

      addAnnotation({
        documentId,
        location,
        text,
        color,
        annotationType: "highlight",
        // Include chapter context for EPUB annotations
        ...(isEpub && currentChapterTitle ? { chapterTitle: currentChapterTitle } : {}),
        ...(isEpub ? { percentage: currentPercentage } : {})
      })
      message.success(t("option:documentWorkspace.highlightAdded", "Highlight added"))
      onClose()
    },
    [documentId, currentPage, currentPercentage, currentChapterTitle, epubCfi, pdfPage, text, addAnnotation, isCurrentDocument, onClose, t]
  )

  // Ask AI - prefill chat with a prompt template about selected text
  const handleAskAI = useCallback((promptTemplate: string) => {
    if (queueSelectionPrompt(documentId, workspaceAuthority, promptTemplate)) onClose()
  }, [documentId, onClose, queueSelectionPrompt, workspaceAuthority])

  // Ask AI prompt templates
  const askAIMenuItems: MenuProps["items"] = [
    {
      key: "explain",
      label: t("option:documentWorkspace.aiExplain", "Explain"),
      onClick: () => handleAskAI(`Explain this passage: "${text}"`)
    },
    {
      key: "summarize",
      label: t("option:documentWorkspace.aiSummarize", "Summarize"),
      onClick: () => handleAskAI(`Summarize this passage: "${text}"`)
    },
    {
      key: "define",
      label: t("option:documentWorkspace.aiDefine", "Define terms"),
      onClick: () => handleAskAI(`Define the key terms in this passage: "${text}"`)
    },
    {
      key: "simplify",
      label: t("option:documentWorkspace.aiSimplify", "Simplify"),
      onClick: () => handleAskAI(`Simplify this passage in plain language: "${text}"`)
    }
  ]

  const listening = ttsState.isPlaying && ttsState.currentText === text
  const handleListen = () => {
    if (!isCurrentDocument()) return
    if (listening) stopTTS()
    else void speak(text)
  }

  const translateFor = useCallback(async (language: string) => {
    if (!isCurrentDocument()) return
    translationRequest.current?.abort()
    const controller = new AbortController()
    translationRequest.current = controller
    setTranslatedText(null)
    setTranslationFailed(false)
    try {
      const result = await translate({ text, targetLanguage: language, signal: controller.signal })
      if (!controller.signal.aborted && isCurrentDocument()) {
        if (result.translated_text?.trim()) setTranslatedText(result.translated_text)
        else setTranslationFailed(true)
      }
    } catch {
      if (!controller.signal.aborted && isCurrentDocument()) setTranslationFailed(true)
    }
  }, [isCurrentDocument, text, translate])

  const closeTranslation = () => {
    translationRequest.current?.abort()
    setTranslateModalOpen(false)
  }

  const highlightMenuItems: MenuProps["items"] = HIGHLIGHT_COLORS.map((color) => ({
    key: color.key,
    label: (
      <span className="flex items-center gap-2">
        <span className="h-4 w-4 rounded border border-border" style={{ backgroundColor: color.color }} />
        {color.label}
      </span>
    ),
    onClick: () => handleHighlight(color.key)
  }))

  const withLabel = (label: string, control: React.ReactElement) => (
    <Tooltip title={showLabels ? null : label} trigger={["hover", "focus"]} placement="top" mouseEnterDelay={0.3}>
      {control}
    </Tooltip>
  )

  const actionButton = (label: string, icon: React.ReactNode, props: React.ComponentProps<typeof Button> = {}) => (
    <Button type="text" size={showLabels ? "middle" : "small"} aria-label={label} icon={icon} {...props}>
      {showLabels && <span className="text-xs">{label}</span>}
      {props.children}
    </Button>
  )
  const action = (label: string, icon: React.ReactNode, props: React.ComponentProps<typeof Button>) =>
    withLabel(label, actionButton(label, icon, props))

  const menuAction = (key: SelectionMenu, label: string, icon: React.ReactNode, items: MenuProps["items"]) => (
    withLabel(label,
      <span className="inline-flex">
        <Dropdown
          menu={{ items }}
          trigger={["click"]}
          autoFocus
          open={openMenu === key}
          onOpenChange={open => setOpenMenu(current => open ? key : current === key ? null : current)}
          classNames={{ root: selectionMenuClass }}
        >
          {actionButton(label, icon, {
            "aria-haspopup": "menu",
            "aria-expanded": openMenu === key,
            children: <ChevronDown className="h-3 w-3" aria-hidden="true" />
          })}
        </Dropdown>
      </span>
    )
  )

  const actionsLabel = t("option:documentWorkspace.textSelectionActions", "Selected text actions")
  const actionButtons = (
    <>
      {action(t("common:copy", "Copy"), <Copy className="h-4 w-4" />, {
        onClick: async () => { if (isCurrentDocument() && await copyText(text)) onClose() }
      })}
      {menuAction("highlight", t("option:documentWorkspace.highlight", "Highlight"),
        <Highlighter className="h-4 w-4" />, highlightMenuItems)}
      {action(t("option:documentWorkspace.translate", "Translate"), <Languages className="h-4 w-4" />, {
        onClick: () => {
          setTranslateModalOpen(true)
          void translateFor(targetLanguage)
        }
      })}
      {menuAction("ask-ai", t("option:documentWorkspace.askAI", "Ask AI"),
        <MessageSquare className="h-4 w-4" />, askAIMenuItems)}
      {action(
        listening ? t("option:documentWorkspace.stopListening", "Stop listening") : t("option:documentWorkspace.listen", "Listen"),
        listening ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />,
        { onClick: handleListen, loading: ttsState.isLoading }
      )}
    </>
  )

  useEffect(() => {
    if (ttsState.error) message.error(t("option:documentWorkspace.listenFailed", "Could not play this passage."))
  }, [t, ttsState.error])

  return (
    <>
      {isMobileView ? (
        // Mobile: bottom sheet
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-[99] bg-black/20"
            aria-hidden="true"
          />
          <div
            ref={popoverRef}
            data-selection-popover
            role="group"
            aria-label={actionsLabel}
            className="fixed bottom-0 left-0 right-0 z-[100] rounded-t-xl border-t border-border bg-surface px-4 pb-[env(safe-area-inset-bottom,8px)] pt-3 shadow-2xl animate-in slide-in-from-bottom duration-200"
          >
            {/* Selected text preview */}
            <p className="mb-3 line-clamp-2 text-xs text-text-muted italic">"{text}"</p>
            {/* Actions */}
            <div className="flex flex-wrap items-center gap-2">
              {actionButtons}
            </div>
          </div>
        </>
      ) : (
        // Desktop: floating popover
        <div
          ref={popoverRef}
          data-selection-popover
          role="group"
          aria-label={actionsLabel}
          className="fixed z-[100] flex max-w-[calc(100vw-2rem)] flex-wrap items-center gap-1 rounded-lg border border-border bg-surface p-1.5 shadow-lg"
          style={{
            left: adjustedPosition.x,
            top: adjustedPosition.y,
            opacity: positionReady ? 1 : 0,
            pointerEvents: positionReady ? "auto" : "none",
            transition: "opacity 100ms"
          }}
        >
          {actionButtons}
        </div>
      )}

      {/* Translation Modal */}
      <Modal
        title={t("option:documentWorkspace.translation", "Translation")}
        open={translateModalOpen}
        onCancel={closeTranslation}
        footer={[
          <Button key="close" onClick={closeTranslation}>
            {t("common:close", "Close")}
          </Button>,
          <Button
            key="copy"
            type="primary"
            disabled={!translatedText || translating}
            onClick={() => { if (translatedText) void copyText(translatedText) }}
          >
            {t("option:documentWorkspace.copyTranslation", "Copy translation")}
          </Button>
        ]}
        width={500}
      >
        <div className="space-y-4">
          {/* Original text */}
          <div>
            <div className="text-xs font-medium text-muted mb-1">
              {t("option:documentWorkspace.originalText", "Original")}
            </div>
            <div className="rounded border border-border bg-surface2 p-3 text-sm max-h-32 overflow-auto">
              {text}
            </div>
          </div>

          {/* Target language selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted">
              {t("option:documentWorkspace.translateTo", "Translate to")}
            </span>
            <Select
              value={targetLanguage}
              onChange={(language) => {
                setTargetLanguage(language)
                void translateFor(language)
              }}
              aria-label={t("option:documentWorkspace.translateTo", "Translate to")}
              options={TARGET_LANGUAGES}
              size="small"
              className="w-32"
            />
          </div>

          {/* Translated text */}
          <div>
            <div className="text-xs font-medium text-muted mb-1">
              {t("option:documentWorkspace.translatedText", "Translation")}
            </div>
            <div className="rounded border border-border bg-surface2 p-3 text-sm min-h-[80px] max-h-48 overflow-auto">
              {translating ? (
                <div role="status" className="flex items-center justify-center py-4">
                  <Spin size="small" />
                  <span className="ml-2 text-muted">
                    {t("option:documentWorkspace.translating", "Translating...")}
                  </span>
                </div>
              ) : translationFailed ? (
                <>
                  <p role="alert" className="text-error">
                    {t("option:documentWorkspace.translationFailed", "Translation failed. Please try again.")}
                  </p>
                  <Button onClick={() => void translateFor(targetLanguage)}>
                    {t("common:retry", "Retry")}
                  </Button>
                </>
              ) : (
                translatedText || (
                  <span className="text-muted">
                    {t("option:documentWorkspace.clickTranslate", "Click translate to see result")}
                  </span>
                )
              )}
            </div>
          </div>
        </div>
      </Modal>
    </>
  )
}

export default TextSelectionPopover

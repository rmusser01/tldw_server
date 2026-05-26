import React from "react"
import { useTranslation } from "react-i18next"
import { ExternalLink, Layers, MessageSquareText } from "lucide-react"
import { Button, Typography } from "antd"
import { browser } from "wxt/browser"
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"

const { Text, Title } = Typography

export default function SidepanelFlashcards() {
  const { t } = useTranslation()
  const [captureError, setCaptureError] = React.useState<string | null>(null)
  const [captureLoading, setCaptureLoading] = React.useState(false)

  const openOptionsHashRoute = React.useCallback((route: string) => {
    const normalizedRoute = route.startsWith("/") ? route : `/${route}`
    const url = browser.runtime.getURL(`/options.html#${normalizedRoute}`)
    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch(() => {
        window.open(url, "_blank")
      })
      return
    }
    window.open(url, "_blank")
  }, [])

  const openFlashcards = React.useCallback(() => {
    openOptionsHashRoute("/flashcards")
  }, [openOptionsHashRoute])

  const handleGenerateFromPageSelection = React.useCallback(async () => {
    setCaptureError(null)
    setCaptureLoading(true)
    const captureUnavailableMessage = t(
      "sidepanel:flashcards.selectionCaptureUnavailable",
      "Page selection capture is unavailable in this browser."
    )
    const activeTabUnavailableMessage = t(
      "sidepanel:flashcards.activeTabUnavailable",
      "Open a page tab before generating flashcards."
    )
    const noSelectionMessage = t(
      "sidepanel:flashcards.noPageSelection",
      "Select text on the page first."
    )
    const captureFailedMessage = t(
      "sidepanel:flashcards.selectionCaptureFailed",
      "Could not read the current page selection. Use the Save to Notes context menu or paste text into Create & Import."
    )
    try {
      if (!browser.tabs?.query || !browser.scripting?.executeScript) {
        throw new Error(captureUnavailableMessage)
      }

      const activeTabs = await browser.tabs.query({
        active: true,
        currentWindow: true
      })
      const activeTab = activeTabs[0]
      const tabId = activeTab?.id
      if (typeof tabId !== "number") {
        throw new Error(activeTabUnavailableMessage)
      }

      const results = await browser.scripting.executeScript({
        target: { tabId },
        func: () => window.getSelection()?.toString() ?? ""
      })
      const selectedText = String(results?.[0]?.result ?? "").trim()
      if (!selectedText) {
        throw new Error(noSelectionMessage)
      }

      openOptionsHashRoute(
        buildFlashcardsGenerateRoute({
          text: selectedText,
          sourceId: activeTab.url || String(tabId),
          sourceTitle: activeTab.title || activeTab.url || undefined
        })
      )
    } catch (error) {
      const message =
        error instanceof Error && error.message.trim() ? error.message : ""
      const validationMessages = [
        captureUnavailableMessage,
        activeTabUnavailableMessage,
        noSelectionMessage
      ]
      setCaptureError(
        validationMessages.includes(message) ? message : captureFailedMessage
      )
    } finally {
      setCaptureLoading(false)
    }
  }, [openOptionsHashRoute, t])

  return (
    <main className="flex min-h-full flex-col items-center justify-center gap-4 p-6 text-center">
      <div className="rounded-full border border-border bg-surface2 p-3">
        <Layers className="size-8 text-text-muted" aria-hidden="true" />
      </div>
      <Title level={4} className="!mb-0">
        {t("sidepanel:flashcards.title", "Flashcards")}
      </Title>
      <Text type="secondary">
        {t(
          "sidepanel:flashcards.workspaceDescription",
          "Study, manage, and create cards in the full Flashcards workspace."
        )}
      </Text>
      <div className="flex w-full max-w-xs flex-col gap-2">
        <Button
          type="primary"
          block
          icon={<ExternalLink className="size-4" aria-hidden="true" />}
          onClick={openFlashcards}
        >
          {t("sidepanel:flashcards.openFull", "Open full Flashcards")}
        </Button>
        <Button
          block
          loading={captureLoading}
          icon={<MessageSquareText className="size-4" aria-hidden="true" />}
          onClick={handleGenerateFromPageSelection}
        >
          {t(
            "sidepanel:flashcards.generateFromPageSelection",
            "Generate from page selection"
          )}
        </Button>
      </div>
      <Text type="secondary" className="max-w-xs text-xs">
        {t(
          "sidepanel:flashcards.selectionHint",
          "Turn the current page selection into editable flashcard drafts."
        )}
      </Text>
      {captureError ? (
        <Text type="danger" className="max-w-xs text-xs" role="status">
          {captureError}
        </Text>
      ) : null}
    </main>
  )
}

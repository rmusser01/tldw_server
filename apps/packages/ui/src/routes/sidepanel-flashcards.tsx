import React from "react"
import { useTranslation } from "react-i18next"
import { ExternalLink, Layers, MessageSquareText, Save } from "lucide-react"
import { Button, Input, Select, Typography } from "antd"
import { browser } from "wxt/browser"
import {
  useCreateFlashcardMutation,
  useDecksQuery
} from "@/components/Flashcards/hooks"

const { Text, Title } = Typography
const { TextArea } = Input

type CaptureDraft = {
  front: string
  back: string
  sourceId: string
  sourceTitle?: string
}

export const readSelectedTextFromPage = () => {
  const activeElement = document.activeElement
  if (
    activeElement instanceof HTMLInputElement ||
    activeElement instanceof HTMLTextAreaElement
  ) {
    const { selectionStart, selectionEnd, value } = activeElement
    if (
      typeof selectionStart === "number" &&
      typeof selectionEnd === "number" &&
      selectionEnd > selectionStart
    ) {
      return value.slice(selectionStart, selectionEnd)
    }
  }

  return window.getSelection()?.toString() ?? ""
}

export default function SidepanelFlashcards() {
  const { t } = useTranslation()
  const [captureError, setCaptureError] = React.useState<string | null>(null)
  const [captureLoading, setCaptureLoading] = React.useState(false)
  const [saveStatus, setSaveStatus] = React.useState<{
    type: "success" | "error"
    message: string
  } | null>(null)
  const [draft, setDraft] = React.useState<CaptureDraft | null>(null)
  const [selectedDeckId, setSelectedDeckId] = React.useState<number | null>(null)
  const decksQuery = useDecksQuery()
  const createFlashcardMutation = useCreateFlashcardMutation()
  const decks = React.useMemo(() => decksQuery.data ?? [], [decksQuery.data])
  const selectedDeck = decks.find((deck) => deck.id === selectedDeckId) ?? null

  React.useEffect(() => {
    if (
      selectedDeckId != null &&
      decks.some((deck) => deck.id === selectedDeckId)
    ) {
      return
    }
    setSelectedDeckId(decks[0]?.id ?? null)
  }, [decks, selectedDeckId])

  const openOptionsHashRoute = React.useCallback(async (route: string) => {
    setCaptureError(null)
    const normalizedRoute = route.startsWith("/") ? route : `/${route}`
    const url = browser.runtime.getURL(`/options.html#${normalizedRoute}`)
    const openFailedMessage = t(
      "sidepanel:flashcards.openFailed",
      "Could not open Flashcards. Check popup permissions and try again."
    )
    const openFallbackWindow = () => {
      const openedWindow = window.open(url, "_blank")
      if (openedWindow) return true
      setCaptureError(openFailedMessage)
      return false
    }

    if (browser.tabs?.create) {
      try {
        await browser.tabs.create({ url })
        return true
      } catch {
        return openFallbackWindow()
      }
    }
    return openFallbackWindow()
  }, [t])

  const openFlashcards = React.useCallback(() => {
    void openOptionsHashRoute("/flashcards")
  }, [openOptionsHashRoute])

  const handleCapturePageSelection = React.useCallback(async () => {
    setCaptureError(null)
    setSaveStatus(null)
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
        func: readSelectedTextFromPage
      })
      const selectedText = String(results?.[0]?.result ?? "").trim()
      if (!selectedText) {
        throw new Error(noSelectionMessage)
      }

      setDraft({
        front:
          activeTab.title ||
          activeTab.url ||
          t("sidepanel:flashcards.defaultFront", "Page selection"),
        back: selectedText,
        sourceId: activeTab.url || String(tabId),
        sourceTitle: activeTab.title || activeTab.url || undefined
      })
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
  }, [t])

  const handleSaveDraft = React.useCallback(async () => {
    if (!draft || selectedDeckId == null) return

    const front = draft.front.trim()
    const back = draft.back.trim()
    if (!front || !back) return

    setSaveStatus(null)
    try {
      await createFlashcardMutation.mutateAsync({
        deck_id: selectedDeckId,
        front,
        back,
        model_type: "basic",
        is_cloze: false,
        reverse: false,
        source_ref_type: "manual",
        source_ref_id: draft.sourceId
      })
      setSaveStatus({
        type: "success",
        message: t("sidepanel:flashcards.saveSuccess", {
          defaultValue: "Saved to {{deckName}}.",
          deckName:
            selectedDeck?.name || t("sidepanel:flashcards.defaultDeck", "deck")
        })
      })
    } catch {
      setSaveStatus({
        type: "error",
        message: t(
          "sidepanel:flashcards.saveFailed",
          "Could not save flashcard. Check your connection and try again."
        )
      })
    }
  }, [createFlashcardMutation, draft, selectedDeck?.name, selectedDeckId, t])

  const hasDraftContent = !!draft?.front.trim() && !!draft?.back.trim()
  const hasDeck = selectedDeckId != null
  const canSaveDraft =
    !!draft && hasDraftContent && hasDeck && !createFlashcardMutation.isPending

  return (
    <main className="flex min-h-full flex-col gap-4 p-4 text-left">
      <div className="flex items-start gap-3">
        <div className="rounded-full border border-border bg-surface2 p-3">
          <Layers className="size-6 text-text-muted" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <Title level={4} className="!mb-1">
            {t("sidepanel:flashcards.title", "Flashcards")}
          </Title>
          <Text type="secondary">
            {t(
              "sidepanel:flashcards.workspaceDescription",
              "Capture selected page text into a deck, or open the full Flashcards workspace."
            )}
          </Text>
        </div>
      </div>
      <div className="flex w-full flex-col gap-2">
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
          onClick={handleCapturePageSelection}
        >
          {t(
            "sidepanel:flashcards.capturePageSelection",
            "Capture page selection"
          )}
        </Button>
      </div>
      <Text type="secondary" className="text-xs">
        {t(
          "sidepanel:flashcards.selectionHint",
          "Create one editable card in the sidepanel. Use full Flashcards for generation, imports, and review."
        )}
      </Text>
      {draft ? (
        <section
          className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3"
          aria-label={t("sidepanel:flashcards.draftSection", "Draft flashcard")}
        >
          <div>
            <Title level={5} className="!mb-1">
              {t("sidepanel:flashcards.draftTitle", "Draft flashcard")}
            </Title>
            <Text type="secondary" className="text-xs">
              {draft.sourceTitle || draft.sourceId}
            </Text>
          </div>
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.deckLabel", "Deck")}
            <Select
              data-testid="sidepanel-flashcards-deck-select"
              aria-label={t("sidepanel:flashcards.deckLabel", "Deck")}
              loading={decksQuery.isLoading}
              disabled={decks.length === 0}
              placeholder={t(
                "sidepanel:flashcards.deckPlaceholder",
                "Choose a deck"
              )}
              value={selectedDeckId ?? undefined}
              options={decks.map((deck) => ({
                label: deck.name,
                value: deck.id
              }))}
              onChange={(value) => setSelectedDeckId(value)}
            />
          </label>
          {decks.length === 0 ? (
            <Text type="warning" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.noDecks",
                "Create a deck in full Flashcards before saving here."
              )}
            </Text>
          ) : null}
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.frontLabel", "Front")}
            <TextArea
              aria-label={t("sidepanel:flashcards.frontLabel", "Front")}
              autoSize={{ minRows: 2, maxRows: 4 }}
              value={draft.front}
              onChange={(event) =>
                setDraft((current) =>
                  current ? { ...current, front: event.target.value } : current
                )
              }
            />
          </label>
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.backLabel", "Back")}
            <TextArea
              aria-label={t("sidepanel:flashcards.backLabel", "Back")}
              autoSize={{ minRows: 4, maxRows: 8 }}
              value={draft.back}
              onChange={(event) =>
                setDraft((current) =>
                  current ? { ...current, back: event.target.value } : current
                )
              }
            />
          </label>
          <Button
            type="primary"
            block
            icon={<Save className="size-4" aria-hidden="true" />}
            loading={createFlashcardMutation.isPending}
            disabled={!canSaveDraft}
            onClick={handleSaveDraft}
          >
            {t("sidepanel:flashcards.saveCard", "Save card")}
          </Button>
          {saveStatus ? (
            <Text
              type={saveStatus.type === "error" ? "danger" : "success"}
              className="text-xs"
              role="status"
            >
              {saveStatus.message}
            </Text>
          ) : null}
        </section>
      ) : null}
      {captureError ? (
        <Text type="danger" className="text-xs" role="status">
          {captureError}
        </Text>
      ) : null}
    </main>
  )
}

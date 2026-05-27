import React from "react"
import { useTranslation } from "react-i18next"
import {
  ExternalLink,
  Layers,
  MessageSquareText,
  Save,
  Trash2
} from "lucide-react"
import { Button, Input, Select, Typography } from "antd"
import { browser } from "wxt/browser"
import {
  useCreateFlashcardMutation,
  useDecksQuery,
  useFlashcardsEnabled
} from "@/components/Flashcards/hooks"
import type { FlashcardCreate } from "@/services/flashcards"

const { Text, Title } = Typography
const { TextArea } = Input

type CaptureDraft = {
  id: string
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
  const draftIdRef = React.useRef(0)
  const [captureError, setCaptureError] = React.useState<string | null>(null)
  const [captureLoading, setCaptureLoading] = React.useState(false)
  const [saveStatus, setSaveStatus] = React.useState<{
    type: "success" | "error"
    message: string
  } | null>(null)
  const [drafts, setDrafts] = React.useState<CaptureDraft[]>([])
  const [savingDraftIds, setSavingDraftIds] = React.useState<Set<string>>(
    () => new Set()
  )
  const [selectedDeckId, setSelectedDeckId] = React.useState<number | null>(null)
  const flashcardsAvailability = useFlashcardsEnabled()
  const decksQuery = useDecksQuery()
  const createFlashcardMutation = useCreateFlashcardMutation()
  const decks = React.useMemo(() => decksQuery.data ?? [], [decksQuery.data])
  const selectedDeck = decks.find((deck) => deck.id === selectedDeckId) ?? null
  const flashcardsUnavailable =
    !flashcardsAvailability.isOnline ||
    flashcardsAvailability.flashcardsUnsupported
  const decksLoading = decksQuery.isLoading || flashcardsAvailability.capsLoading
  const decksLoadFailed = decksQuery.isError
  const hasSelectableDecks =
    !flashcardsUnavailable &&
    !decksLoading &&
    !decksLoadFailed &&
    decks.length > 0
  const showNoDecksMessage =
    !flashcardsUnavailable &&
    !decksLoading &&
    !decksLoadFailed &&
    decks.length === 0

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
      "Open a page tab before capturing page selection."
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

      draftIdRef.current += 1
      const nextDraft: CaptureDraft = {
        id: `capture-${draftIdRef.current}`,
        front:
          activeTab.title ||
          activeTab.url ||
          t("sidepanel:flashcards.defaultFront", "Page selection"),
        back: selectedText,
        sourceId: activeTab.url || String(tabId),
        sourceTitle: activeTab.title || activeTab.url || undefined
      }
      setDrafts((current) => [...current, nextDraft])
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

  const getDeckName = React.useCallback(
    () => selectedDeck?.name || t("sidepanel:flashcards.defaultDeck", "deck"),
    [selectedDeck?.name, t]
  )

  const buildSavePayload = React.useCallback(
    (draft: CaptureDraft): FlashcardCreate | null => {
      if (selectedDeckId == null) return null

      const front = draft.front.trim()
      const back = draft.back.trim()
      if (!front || !back) return null

      return {
        deck_id: selectedDeckId,
        front,
        back,
        model_type: "basic",
        is_cloze: false,
        reverse: false,
        source_ref_type: "manual",
        source_ref_id: draft.sourceId
      }
    },
    [selectedDeckId]
  )

  const hasDeck = selectedDeckId != null && hasSelectableDecks

  const handleUpdateDraft = React.useCallback(
    (draftId: string, field: "front" | "back", value: string) => {
      setDrafts((current) =>
        current.map((draft) =>
          draft.id === draftId ? { ...draft, [field]: value } : draft
        )
      )
    },
    []
  )

  const handleRemoveDraft = React.useCallback((draftId: string) => {
    setSaveStatus(null)
    setDrafts((current) => current.filter((draft) => draft.id !== draftId))
  }, [])

  const handleSaveDraft = React.useCallback(
    async (draftId: string) => {
      if (!hasDeck) return

      const draft = drafts.find((candidate) => candidate.id === draftId)
      if (!draft) return

      const payload = buildSavePayload(draft)
      if (!payload) return

      setSaveStatus(null)
      setSavingDraftIds(new Set([draftId]))
      try {
        await createFlashcardMutation.mutateAsync(payload)
        setSaveStatus({
          type: "success",
          message: t("sidepanel:flashcards.saveSuccess", {
            defaultValue: "Saved to {{deckName}}.",
            deckName: getDeckName()
          })
        })
        setDrafts((current) =>
          current.filter((candidate) => candidate.id !== draftId)
        )
      } catch {
        setSaveStatus({
          type: "error",
          message: t(
            "sidepanel:flashcards.saveFailed",
            "Could not save flashcard. Check your connection and try again."
          )
        })
      } finally {
        setSavingDraftIds(new Set())
      }
    },
    [buildSavePayload, createFlashcardMutation, drafts, getDeckName, hasDeck, t]
  )

  const handleSaveAllDrafts = React.useCallback(async () => {
    if (!hasDeck) return

    const validDrafts = drafts.filter((draft) => buildSavePayload(draft))
    if (!validDrafts.length) return

    setSaveStatus(null)
    setSavingDraftIds(new Set(validDrafts.map((draft) => draft.id)))
    const savedDraftIds = new Set<string>()

    try {
      for (const draft of validDrafts) {
        const payload = buildSavePayload(draft)
        if (!payload) continue

        try {
          await createFlashcardMutation.mutateAsync(payload)
          savedDraftIds.add(draft.id)
        } catch {
          // Keep saving the queue so users get a precise partial-success state.
        }
      }

      if (savedDraftIds.size > 0) {
        setDrafts((current) =>
          current.filter((draft) => !savedDraftIds.has(draft.id))
        )
      }

      const failedCount = validDrafts.length - savedDraftIds.size
      const deckName = getDeckName()
      if (failedCount === 0) {
        setSaveStatus({
          type: "success",
          message: t("sidepanel:flashcards.saveAllSuccess", {
            defaultValue: "Saved {{savedCount}} {{cardLabel}} to {{deckName}}.",
            savedCount: savedDraftIds.size,
            cardLabel: savedDraftIds.size === 1 ? "card" : "cards",
            deckName
          })
        })
      } else if (savedDraftIds.size > 0) {
        setSaveStatus({
          type: "error",
          message: t("sidepanel:flashcards.saveAllPartial", {
            defaultValue:
              "Saved {{savedCount}} {{cardLabel}} to {{deckName}}. {{failedCount}} {{draftLabel}} still needs attention.",
            savedCount: savedDraftIds.size,
            cardLabel: savedDraftIds.size === 1 ? "card" : "cards",
            deckName,
            failedCount,
            draftLabel: failedCount === 1 ? "draft" : "drafts"
          })
        })
      } else {
        setSaveStatus({
          type: "error",
          message: t(
            "sidepanel:flashcards.saveAllFailed",
            "Could not save any flashcards. Check your connection and try again."
          )
        })
      }
    } finally {
      setSavingDraftIds(new Set())
    }
  }, [buildSavePayload, createFlashcardMutation, drafts, getDeckName, hasDeck, t])

  const isSaving = createFlashcardMutation.isPending || savingDraftIds.size > 0
  const canSaveDraft = React.useCallback(
    (draft: CaptureDraft) => !!buildSavePayload(draft) && hasDeck && !isSaving,
    [buildSavePayload, hasDeck, isSaving]
  )
  const validDraftCount = drafts.filter(
    (draft) => buildSavePayload(draft)
  ).length
  const canSaveAllDrafts = validDraftCount > 0 && hasDeck && !isSaving

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
      {drafts.length > 0 ? (
        <section
          className="flex flex-col gap-3"
          aria-label={t("sidepanel:flashcards.draftQueue", "Draft queue")}
        >
          <div className="flex items-center justify-between gap-2">
            <Text strong>
              {t("sidepanel:flashcards.draftQueueCount", {
                defaultValue: "{{count}} draft {{cardLabel}} ready",
                count: drafts.length,
                cardLabel: drafts.length === 1 ? "card" : "cards"
              })}
            </Text>
            <Button
              size="small"
              icon={<Save className="size-4" aria-hidden="true" />}
              loading={isSaving}
              disabled={!canSaveAllDrafts}
              onClick={handleSaveAllDrafts}
            >
              {t("sidepanel:flashcards.saveAllCards", "Save all cards")}
            </Button>
          </div>
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.deckLabel", "Deck")}
            <Select
              data-testid="sidepanel-flashcards-deck-select"
              aria-label={t("sidepanel:flashcards.deckLabel", "Deck")}
              loading={decksLoading}
              disabled={!hasSelectableDecks}
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
          {flashcardsUnavailable ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.unavailable",
                "Flashcards are unavailable. Check the server connection and try again."
              )}
            </Text>
          ) : decksLoadFailed ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.decksLoadFailed",
                "Could not load decks. Check your connection and try again."
              )}
            </Text>
          ) : decksLoading ? (
            <Text type="secondary" className="text-xs" role="status">
              {t("sidepanel:flashcards.decksLoading", "Loading decks...")}
            </Text>
          ) : showNoDecksMessage ? (
            <Text type="warning" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.noDecks",
                "Create a deck in full Flashcards before saving here."
              )}
            </Text>
          ) : null}
          {drafts.map((draft, index) => (
            <section
              key={draft.id}
              className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3"
              aria-label={t("sidepanel:flashcards.draftSection", {
                defaultValue: "Draft flashcard {{index}}",
                index: index + 1
              })}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <Title level={5} className="!mb-1">
                    {t("sidepanel:flashcards.draftTitle", "Draft flashcard")}
                  </Title>
                  <Text type="secondary" className="text-xs">
                    {draft.sourceTitle || draft.sourceId}
                  </Text>
                </div>
                <Button
                  size="small"
                  icon={<Trash2 className="size-4" aria-hidden="true" />}
                  aria-label={t("sidepanel:flashcards.removeDraft", {
                    defaultValue: "Remove draft {{index}}",
                    index: index + 1
                  })}
                  disabled={isSaving}
                  onClick={() => handleRemoveDraft(draft.id)}
                />
              </div>
              <label className="flex flex-col gap-1 text-sm font-medium">
                {t("sidepanel:flashcards.frontLabel", "Front")}
                <TextArea
                  aria-label={t("sidepanel:flashcards.frontLabel", "Front")}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  value={draft.front}
                  onChange={(event) =>
                    handleUpdateDraft(draft.id, "front", event.target.value)
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
                    handleUpdateDraft(draft.id, "back", event.target.value)
                  }
                />
              </label>
              <Button
                type="primary"
                block
                icon={<Save className="size-4" aria-hidden="true" />}
                loading={savingDraftIds.has(draft.id)}
                disabled={!canSaveDraft(draft)}
                onClick={() => void handleSaveDraft(draft.id)}
              >
                {t("sidepanel:flashcards.saveCard", "Save card")}
              </Button>
            </section>
          ))}
        </section>
      ) : null}
      {saveStatus ? (
        <Text
          type={saveStatus.type === "error" ? "danger" : "success"}
          className="text-xs"
          role="status"
        >
          {saveStatus.message}
        </Text>
      ) : null}
      {captureError ? (
        <Text type="danger" className="text-xs" role="status">
          {captureError}
        </Text>
      ) : null}
    </main>
  )
}

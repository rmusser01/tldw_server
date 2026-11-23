import React from "react"
import { Button, Drawer, Space, Typography } from "antd"
import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"

import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { getStudyPackDeckCards } from "@/services/flashcards"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"

const { Text } = Typography
const PAGE_SIZE = 20

const DeckCardList = ({ deckId, deckName }: { deckId: number; deckName?: string }) => {
  const { t } = useTranslation(["option", "common"])
  const filterId = React.useId()
  const sessionId = React.useId()
  const [chapter, setChapter] = React.useState("")
  const [offset, setOffset] = React.useState(0)
  const page = useQuery({
    queryKey: ["flashcards:study-packs:deck-cards", sessionId, deckId, chapter, offset],
    gcTime: 0,
    retry: false,
    placeholderData: keepPreviousData,
    queryFn: async ({ signal }) => {
      const response = await getStudyPackDeckCards(deckId, {
        chapter_id: chapter || undefined,
        limit: PAGE_SIZE,
        offset,
        signal
      })
      signal.throwIfAborted()
      return response
    }
  })
  const { data } = page
  const cards = page.isPlaceholderData || page.isError ? undefined : data
  const total = data?.total ?? data?.count ?? 0
  const missingChapter = chapter && data && !data.chapters.some(({ id }) => id === chapter)
  const multipleSources = new Set(data?.chapters.map(({ source_id }) => source_id)).size > 1
  const allCardsLabel = t("option:flashcards.studyPackAllCards", { defaultValue: "All cards" })
  const withCount = (title: string, count: number) =>
    t("option:flashcards.studyPackCardSetWithCount", { defaultValue: "{{title}} ({{count}})", title, count })
  const chooseChapter = (id: string) => {
    setChapter(id)
    setOffset(0)
  }

  return (
    <div className="space-y-4">
      {deckName ? <Text strong className="block break-words">{deckName}</Text> : null}
      {data ? (
        <div className="space-y-1">
          <label htmlFor={filterId} className="block text-sm font-medium">
            {t("option:flashcards.studyPackCardSet", { defaultValue: "Card set" })}
          </label>
          <select
            id={filterId}
            className="w-full rounded border border-border bg-surface px-2 py-2 text-sm text-text"
            value={chapter}
            onChange={(event) => chooseChapter(event.target.value)}
          >
            <option value="">{withCount(allCardsLabel, data.all_cards_count)}</option>
            {data.chapters.map((set) => (
              <option key={set.id} value={set.id}>
                {withCount(multipleSources && set.source_title
                  ? `${set.source_title} · ${set.title}`
                  : set.title, set.card_count)}
              </option>
            ))}
            {missingChapter ? (
              <option value={chapter}>
                {t("option:flashcards.studyPackChapterUnavailable", { defaultValue: "Chapter unavailable" })}
              </option>
            ) : null}
          </select>
        </div>
      ) : null}
      {page.isError ? (
        <Space wrap>
          <Text role="alert">
            {t("option:flashcards.studyPackBrowseError", { defaultValue: "Could not load cards." })}
          </Text>
          <Button size="small" onClick={() => void page.refetch()} loading={page.isFetching}>
            {t("common:retry", { defaultValue: "Retry" })}
          </Button>
          {chapter || offset > 0 ? (
            <Button size="small" onClick={() => chooseChapter("")}>{allCardsLabel}</Button>
          ) : null}
        </Space>
      ) : !cards ? (
        <Text type="secondary" role="status">
          {t("option:flashcards.studyPackBrowseLoading", { defaultValue: "Loading cards…" })}
        </Text>
      ) : cards.items.length === 0 ? (
        <Text type="secondary" role="status">
          {missingChapter
            ? t("option:flashcards.studyPackChapterMissing", {
                defaultValue: "This chapter is no longer available. Choose another card set."
              })
            : t("option:flashcards.studyPackBrowseEmpty", { defaultValue: "No cards in this set." })}
        </Text>
      ) : (
        <div className="space-y-2">
          {cards.items.map((card) => (
            <article key={card.uuid} className="space-y-2 rounded border border-border bg-surface p-3">
              <div>
                <Text type="secondary" className="mb-1 block text-xs">
                  {t("option:flashcards.front", { defaultValue: "Front" })}
                </Text>
                <MarkdownWithBoundary content={card.front} />
              </div>
              <div className="border-t border-border pt-2">
                <Text type="secondary" className="mb-1 block text-xs">
                  {t("option:flashcards.back", { defaultValue: "Back" })}
                </Text>
                <MarkdownWithBoundary content={card.back} />
                {card.extra ? <MarkdownWithBoundary content={card.extra} size="xs" /> : null}
              </div>
            </article>
          ))}
        </div>
      )}
      {data && (offset > 0 || total > PAGE_SIZE) ? (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Button size="small" disabled={!cards || offset === 0} onClick={() => setOffset(offset - PAGE_SIZE)}>
            {t("common:previous", { defaultValue: "Previous" })}
          </Button>
          {cards && cards.items.length > 0 ? (
            <Text type="secondary" role="status">
              {t("option:flashcards.studyPackBrowseRange", {
                defaultValue: "{{start}}–{{end}} of {{total}}",
                start: offset + 1,
                end: offset + cards.count,
                total
              })}
            </Text>
          ) : null}
          <Button size="small" disabled={!cards || offset + PAGE_SIZE >= total} onClick={() => setOffset(offset + PAGE_SIZE)}>
            {t("common:next", { defaultValue: "Next" })}
          </Button>
        </div>
      ) : null}
    </div>
  )
}

export const StudyPackCardBrowser = ({
  deckId,
  deckName,
  open,
  onClose
}: {
  deckId?: number | null
  deckName?: string
  open: boolean
  onClose: () => void
}) => {
  const { t } = useTranslation("option")
  const titleId = React.useId()
  React.useEffect(() => {
    if (!open) return
    return watchChatAccountChanges((invalidated) => {
      if (invalidated) onClose()
    })
  }, [open, onClose])

  return (
    <Drawer
      open={open}
      aria-labelledby={titleId}
      title={<span id={titleId}>{t("option:flashcards.studyPackBrowseCards", { defaultValue: "Browse cards" })}</span>}
      onClose={onClose}
      size="min(100vw, 32rem)"
    >
      {open && deckId != null ? <DeckCardList key={deckId} deckId={deckId} deckName={deckName} /> : null}
    </Drawer>
  )
}

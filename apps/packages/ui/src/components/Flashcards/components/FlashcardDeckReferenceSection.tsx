import React from "react"
import { Button, Collapse, Input, Spin, Typography } from "antd"
import { useTranslation } from "react-i18next"
import {
  type UseFlashcardQueriesOptions,
  useFlashcardDeckRecentCardsQuery,
  useFlashcardDeckSearchQuery
} from "../hooks"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"

type FlashcardDeckReferenceSectionProps = {
  open: boolean
  deckId: number | null
  deckName?: string | null
} & Pick<UseFlashcardQueriesOptions, "includeWorkspaceItems" | "workspaceId">

const { Text } = Typography
const SECTION_KEY = "deck-reference"
const SEARCH_DEBOUNCE_MS = 300
const RECENT_LIMIT = 6

export const FlashcardDeckReferenceSection: React.FC<
  FlashcardDeckReferenceSectionProps
> = ({ open, deckId, deckName, includeWorkspaceItems, workspaceId }) => {
  const { t } = useTranslation(["option", "common"])
  const [expanded, setExpanded] = React.useState(false)
  const [expandedForDeckId, setExpandedForDeckId] = React.useState<number | null>(null)
  const [searchInput, setSearchInput] = React.useState("")
  const [debouncedSearchInput, setDebouncedSearchInput] = React.useState("")

  React.useEffect(() => {
    setExpanded(false)
    setExpandedForDeckId(null)
    setSearchInput("")
    setDebouncedSearchInput("")
  }, [deckId])

  React.useEffect(() => {
    if (open) return
    setExpanded(false)
    setExpandedForDeckId(null)
    setSearchInput("")
    setDebouncedSearchInput("")
  }, [open])

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearchInput(searchInput)
    }, SEARCH_DEBOUNCE_MS)

    return () => {
      window.clearTimeout(timer)
    }
  }, [searchInput])

  const liveTrimmedSearchInput = searchInput.trim()
  const trimmedDebouncedSearchInput = debouncedSearchInput.trim()
  const sectionExpanded = open && expanded && expandedForDeckId === deckId
  const sectionEnabled = sectionExpanded && deckId != null
  const searchEnabled = sectionEnabled && trimmedDebouncedSearchInput.length > 0
  const searchAreaActive = sectionEnabled && liveTrimmedSearchInput.length > 0
  const searchQuerySettled = liveTrimmedSearchInput === trimmedDebouncedSearchInput

  const recentQuery = useFlashcardDeckRecentCardsQuery(deckId, {
    enabled: sectionEnabled,
    limit: RECENT_LIMIT,
    includeWorkspaceItems,
    workspaceId
  })
  const searchQuery = useFlashcardDeckSearchQuery(
    {
      deckId,
      query: trimmedDebouncedSearchInput
    },
    {
      enabled: searchEnabled,
      includeWorkspaceItems,
      workspaceId
    }
  )

  if (deckId == null) {
    return null
  }

  const recentCards = recentQuery.data ?? []
  const searchCards = searchQuery.data ?? []
  const hasRecentCards = recentCards.length > 0
  const hasSearchResults = searchCards.length > 0
  const isRecentLoading = Boolean(recentQuery.isLoading)
  const isSearchLoading = Boolean(searchQuery.isLoading)
  const isRecentError = Boolean(recentQuery.isError)
  const isSearchError = Boolean(searchQuery.isError)
  const retryLabel = t("common:retry", { defaultValue: "Retry" })
  const frontLabel = t("option:flashcards.front", { defaultValue: "Front" })
  const backLabel = t("option:flashcards.back", { defaultValue: "Back" })

  const renderRecentState = () => {
    if (isRecentLoading) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
          <Spin size="small" />
          <span>
            {t("option:flashcards.deckReferenceRecentLoading", {
              defaultValue: "Loading recent cards..."
            })}
          </span>
        </div>
      )
    }

    if (isRecentError) {
      return (
        <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/20 px-3 py-2 text-xs">
          <Text type="secondary">
            {t("option:flashcards.deckReferenceRecentError", {
              defaultValue: "Unable to load reference cards."
            })}
          </Text>
          <Button
            size="small"
            type="link"
            className="h-auto p-0"
            onClick={() => {
              void recentQuery.refetch()
            }}
          >
            {retryLabel}
          </Button>
        </div>
      )
    }

    if (!hasRecentCards) {
      return (
        <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          {t("option:flashcards.deckReferenceRecentEmpty", {
            defaultValue: "No recent cards in this deck yet."
          })}
        </div>
      )
    }

    return (
      <div className="space-y-2">
        {recentCards.map((card) => (
          <article
            key={card.uuid}
            className="rounded-md border border-border bg-surface px-3 py-2"
          >
            <div className="grid gap-2">
              <div>
                <Text type="secondary" className="mb-1 block text-[11px] uppercase tracking-wide">
                  {frontLabel}
                </Text>
                <MarkdownWithBoundary content={card.front} size="xs" />
              </div>
              <div className="border-t border-border pt-2">
                <Text type="secondary" className="mb-1 block text-[11px] uppercase tracking-wide">
                  {backLabel}
                </Text>
                <MarkdownWithBoundary content={card.back} size="xs" />
              </div>
            </div>
          </article>
        ))}
      </div>
    )
  }

  const renderSearchState = () => {
    if (!searchAreaActive) {
      return null
    }

    if (!searchQuerySettled || isSearchLoading) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
          <Spin size="small" />
          <span>
            {t("option:flashcards.deckReferenceSearchLoading", {
              defaultValue: "Loading search results..."
            })}
          </span>
        </div>
      )
    }

    if (isSearchError) {
      return (
        <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/20 px-3 py-2 text-xs">
          <Text type="secondary">
            {t("option:flashcards.deckReferenceSearchError", {
              defaultValue: "Unable to load search results."
            })}
          </Text>
          <Button
            size="small"
            type="link"
            className="h-auto p-0"
            onClick={() => {
              void searchQuery.refetch()
            }}
          >
            {retryLabel}
          </Button>
        </div>
      )
    }

    if (!hasSearchResults) {
      return (
        <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          {t("option:flashcards.deckReferenceSearchEmpty", {
            defaultValue: "No matching cards."
          })}
        </div>
      )
    }

    return (
      <div className="space-y-2">
        {searchCards.map((card) => (
          <article
            key={card.uuid}
            className="rounded-md border border-border bg-surface px-3 py-2"
          >
            <div className="grid gap-2">
              <div>
                <Text type="secondary" className="mb-1 block text-[11px] uppercase tracking-wide">
                  {frontLabel}
                </Text>
                <MarkdownWithBoundary content={card.front} size="xs" />
              </div>
              <div className="border-t border-border pt-2">
                <Text type="secondary" className="mb-1 block text-[11px] uppercase tracking-wide">
                  {backLabel}
                </Text>
                <MarkdownWithBoundary content={card.back} size="xs" />
              </div>
            </div>
          </article>
        ))}
      </div>
    )
  }

  return (
    <div className="rounded-md border border-border bg-muted/10">
      <Collapse
        activeKey={sectionExpanded ? [SECTION_KEY] : []}
        destroyOnHidden
        bordered={false}
        className="bg-transparent"
        onChange={(nextKeys) => {
          const nextExpanded = Array.isArray(nextKeys)
            ? nextKeys.includes(SECTION_KEY)
            : nextKeys === SECTION_KEY
          setExpanded(nextExpanded)
          setExpandedForDeckId(nextExpanded ? deckId : null)
        }}
        items={[
          {
            key: SECTION_KEY,
            label: (
              <div className="flex min-w-0 flex-col">
                <span className="truncate text-sm font-medium">
                  {t("option:flashcards.deckReferenceTitle", {
                    defaultValue: "Existing cards in this deck"
                  })}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {deckName
                    ? t("option:flashcards.deckReferenceDeckLabel", {
                        defaultValue: "{{deckName}} deck",
                        deckName
                      })
                    : t("option:flashcards.deckReferenceSummary", {
                        defaultValue: "Recent cards and search"
                      })}
                </span>
              </div>
            ),
            children: (
              <div className="space-y-4 px-1 pt-2">
                <section className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <Text className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t("option:flashcards.deckReferenceRecentTitle", {
                        defaultValue: "Recent cards"
                      })}
                    </Text>
                    <Text className="text-[11px] text-muted-foreground">
                      {t("option:flashcards.deckReferenceRecentSubtitle", {
                        defaultValue: "Latest additions in this deck"
                      })}
                    </Text>
                  </div>
                  {renderRecentState()}
                </section>

                <section className="space-y-2 border-t border-border pt-3">
                  <div className="flex items-center justify-between gap-3">
                    <Text className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t("option:flashcards.deckReferenceSearchTitle", {
                        defaultValue: "Search this deck"
                      })}
                    </Text>
                    <Text className="text-[11px] text-muted-foreground">
                      {t("option:flashcards.deckReferenceSearchSubtitle", {
                        defaultValue: "Search across this deck"
                      })}
                    </Text>
                  </div>
                  <Input
                    allowClear
                    value={searchInput}
                    placeholder={t("option:flashcards.deckReferenceSearchPlaceholder", {
                      defaultValue: "Search this deck"
                    })}
                    onChange={(event) => {
                      setSearchInput(event.target.value)
                    }}
                  />
                  {renderSearchState()}
                </section>
              </div>
            )
          }
        ]}
      />
    </div>
  )
}

export default FlashcardDeckReferenceSection

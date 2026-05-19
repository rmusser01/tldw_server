import React from "react"
import { Button, Form, Select, Tabs, Tag } from "antd"
import { ArrowLeft } from "lucide-react"
import { useTranslation } from "react-i18next"
import { WorldBookEntryManager, DEFAULT_ENTRY_FILTER_PRESET } from "./WorldBookEntryManager"
import { WorldBookForm } from "./WorldBookForm"
import { formatWorldBookLastModified } from "./worldBookListUtils"
import type { EntryFilterPreset } from "./WorldBookEntryManager"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WorldBookDetailPanelProps = {
  worldBook: any | null
  attachedCharacters: any[]
  allWorldBooks: any[]
  allCharacters: any[]
  activeTab: WorldBookDetailTabKey
  onActiveTabChange: (tab: WorldBookDetailTabKey) => void
  onUpdateWorldBook: (values: any) => void
  onAttachCharacter: (characterId: number) => Promise<void>
  onDetachCharacter: (characterId: number) => Promise<void>
  onOpenTestMatching: (worldBookId?: number) => void
  maxRecursiveDepth: number
  updating: boolean
  entryFormInstance: any
  settingsFormInstance: any
  entryFilterPreset?: EntryFilterPreset
  settingsBanner?: React.ReactNode
  statsData?: any | null
  statsLoading?: boolean
  statsError?: string | null
  onBack?: () => void
}

export type WorldBookDetailTabKey = "entries" | "attachments" | "stats" | "settings"

// ---------------------------------------------------------------------------
// Attachments tab content
// ---------------------------------------------------------------------------

const AttachmentsTabContent: React.FC<{
  worldBookId: number
  attachedCharacters: any[]
  allCharacters: any[]
  onAttachCharacter: (characterId: number) => Promise<void>
  onDetachCharacter: (characterId: number) => Promise<void>
}> = ({
  worldBookId,
  attachedCharacters,
  allCharacters,
  onAttachCharacter,
  onDetachCharacter
}) => {
  const { t } = useTranslation(["option"])
  const [selectedCharacterId, setSelectedCharacterId] = React.useState<number | null>(null)
  const [attaching, setAttaching] = React.useState(false)

  const attachedIds = React.useMemo(
    () => new Set(attachedCharacters.map((c: any) => c.id)),
    [attachedCharacters]
  )

  const availableCharacters = React.useMemo(
    () => allCharacters.filter((c: any) => !attachedIds.has(c.id)),
    [allCharacters, attachedIds]
  )

  const handleAttach = React.useCallback(async () => {
    if (selectedCharacterId == null) return
    setAttaching(true)
    try {
      await onAttachCharacter(selectedCharacterId)
      setSelectedCharacterId(null)
    } finally {
      setAttaching(false)
    }
  }, [selectedCharacterId, onAttachCharacter])

  return (
    <div className="space-y-4">
      {/* Attached characters list */}
      <div>
        <h3 className="text-sm font-medium mb-2">
          {t("option:worldBooks.detail.attachments.heading", {
            defaultValue: `Attached Characters (${attachedCharacters.length})`
          })}
        </h3>
        {attachedCharacters.length === 0 ? (
          <p className="text-text-muted text-sm">
            {t("option:worldBooks.detail.attachments.empty", {
              defaultValue: "No characters attached."
            })}
          </p>
        ) : (
          <div className="space-y-2">
            {attachedCharacters.map((character: any) => (
              <div
                key={character.id}
                className="flex items-center justify-between rounded border border-border px-3 py-2"
              >
                <a
                  href={`/characters?from=world-books&focusCharacterId=${encodeURIComponent(
                    String(character.id)
                  )}&focusWorldBookId=${encodeURIComponent(String(worldBookId))}`}
                  className="text-sm text-primary hover:underline"
                  aria-label={t("option:worldBooks.detail.attachments.openCharacter", {
                    defaultValue: `Open character ${character.name || `Character ${character.id}`}`
                  })}
                >
                  {character.name}
                </a>
                <Button
                  danger
                  size="small"
                  onClick={() => onDetachCharacter(character.id)}
                >
                  {t("option:worldBooks.detail.attachments.detach", {
                    defaultValue: "Detach"
                  })}
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Attach new character */}
      {availableCharacters.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2">
            {t("option:worldBooks.detail.attachments.attachHeading", {
              defaultValue: "Attach a Character"
            })}
          </h3>
          <div className="flex gap-2">
            <Select
              className="flex-1"
              placeholder={t("option:worldBooks.detail.attachments.selectPlaceholder", {
                defaultValue: "Select a character"
              })}
              value={selectedCharacterId}
              onChange={(value) => setSelectedCharacterId(value)}
              options={availableCharacters.map((c: any) => ({
                label: c.name,
                value: c.id
              }))}
              allowClear
            />
            <Button
              type="primary"
              disabled={selectedCharacterId == null}
              loading={attaching}
              onClick={handleAttach}
            >
              {t("option:worldBooks.detail.attachments.attach", {
                defaultValue: "Attach"
              })}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Settings tab content
// ---------------------------------------------------------------------------

const SettingsTabContent: React.FC<{
  worldBook: any
  allWorldBooks: any[]
  maxRecursiveDepth: number
  updating: boolean
  form: any
  banner?: React.ReactNode
  onUpdateWorldBook: (values: any) => void
}> = ({
  worldBook,
  allWorldBooks,
  maxRecursiveDepth,
  updating,
  form,
  banner,
  onUpdateWorldBook
}) => {
  React.useEffect(() => {
    if (worldBook) {
      form.setFieldsValue({
        name: worldBook.name,
        description: worldBook.description,
        enabled: worldBook.enabled,
        scan_depth: worldBook.scan_depth,
        token_budget: worldBook.token_budget,
        recursive_scanning: worldBook.recursive_scanning
      })
    }
  }, [worldBook, form])

  return (
    <div className="space-y-3">
      {banner}
      <WorldBookForm
        mode="edit"
        form={form}
        worldBooks={allWorldBooks}
        submitting={updating}
        currentWorldBookId={worldBook?.id}
        maxRecursiveDepth={maxRecursiveDepth}
        onSubmit={onUpdateWorldBook}
      />
    </div>
  )
}

const StatsTabContent: React.FC<{
  statsData?: any | null
  loading?: boolean
  error?: string | null
}> = ({ statsData, loading, error }) => {
  const { t } = useTranslation(["option"])

  if (loading) {
    return (
      <div data-testid="stats-tab-content">
        {t("option:worldBooks.detail.stats.loading", {
          defaultValue: "Loading statistics..."
        })}
      </div>
    )
  }

  if (error) {
    return (
      <div data-testid="stats-tab-content" role="alert" className="text-sm text-danger">
        {t("option:worldBooks.detail.stats.error", {
          defaultValue: `Failed to load statistics: ${error}`
        })}
      </div>
    )
  }

  const metricRows = [
    {
      label: t("option:worldBooks.detail.stats.entries", {
        defaultValue: "Entries"
      }),
      value: statsData?.total_entries
    },
    {
      label: t("option:worldBooks.detail.stats.enabledEntries", {
        defaultValue: "Enabled entries"
      }),
      value: statsData?.enabled_entries
    },
    {
      label: t("option:worldBooks.detail.stats.disabledEntries", {
        defaultValue: "Disabled entries"
      }),
      value: statsData?.disabled_entries
    },
    {
      label: t("option:worldBooks.detail.stats.keywords", {
        defaultValue: "Keywords"
      }),
      value: statsData?.total_keywords
    },
    {
      label: t("option:worldBooks.detail.stats.regexEntries", {
        defaultValue: "Regex entries"
      }),
      value: statsData?.regex_entries
    },
    {
      label: t("option:worldBooks.detail.stats.averagePriority", {
        defaultValue: "Average priority"
      }),
      value: statsData?.average_priority
    },
    {
      label: t("option:worldBooks.detail.stats.estimatedTokens", {
        defaultValue: "Estimated tokens"
      }),
      value: statsData?.estimated_tokens
    },
    {
      label: t("option:worldBooks.detail.stats.contentLength", {
        defaultValue: "Content length"
      }),
      value: statsData?.total_content_length
    }
  ].filter((row) => row.value != null)

  const estimatorNote =
    typeof statsData?.token_estimation_method === "string" &&
    statsData.token_estimation_method.trim().length > 0
      ? t("option:worldBooks.detail.stats.estimatorMethod", {
          defaultValue: `Estimated using ${statsData.token_estimation_method}.`
        })
      : t("option:worldBooks.detail.stats.estimatorFallback", {
          defaultValue: "Estimated using ~4 characters per token."
        })

  if (metricRows.length === 0) {
    return (
      <div data-testid="stats-tab-content" className="text-sm text-text-muted">
        {t("option:worldBooks.detail.stats.empty", {
          defaultValue: "No statistics available yet."
        })}
      </div>
    )
  }

  return (
    <div data-testid="stats-tab-content" className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {metricRows.map((row) => (
          <div
            key={row.label}
            className="rounded border border-border px-3 py-2"
          >
            <div className="text-xs text-text-muted">{row.label}</div>
            <div className="text-base font-semibold">{String(row.value)}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-text-muted">{estimatorNote}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const WorldBookDetailPanel: React.FC<WorldBookDetailPanelProps> = ({
  worldBook,
  attachedCharacters,
  allWorldBooks,
  allCharacters,
  activeTab,
  onActiveTabChange,
  onUpdateWorldBook,
  onAttachCharacter,
  onDetachCharacter,
  onOpenTestMatching,
  maxRecursiveDepth,
  updating,
  entryFormInstance,
  settingsFormInstance,
  entryFilterPreset,
  settingsBanner,
  statsData,
  statsLoading,
  statsError,
  onBack
}) => {
  const { t } = useTranslation(["option"])
  const headingRef = React.useRef<HTMLHeadingElement>(null)

  React.useEffect(() => {
    if (worldBook && headingRef.current) {
      headingRef.current.focus()
    }
  }, [worldBook?.id])

  // No-selection state
  if (worldBook == null) {
    return (
      <main aria-label="World book detail" className="flex items-center justify-center h-full">
        <p className="text-text-muted text-center">
          {t("option:worldBooks.detail.empty", {
            defaultValue: "Select a world book to view its entries and settings"
          })}
        </p>
      </main>
    )
  }

  const entryCount = typeof worldBook.entry_count === "number" ? worldBook.entry_count : 0
  const characterCount = attachedCharacters.length
  const lastModifiedDisplay = formatWorldBookLastModified(worldBook.last_modified)

  const tabItems = [
    {
      key: "entries",
      label: t("option:worldBooks.detail.tabs.entries", {
        defaultValue: "Entries"
      }),
      children: (
        <WorldBookEntryManager
          worldBookId={worldBook.id}
          worldBookName={worldBook.name}
          tokenBudget={worldBook.token_budget}
          worldBooks={allWorldBooks}
          entryFilterPreset={entryFilterPreset ?? DEFAULT_ENTRY_FILTER_PRESET}
          form={entryFormInstance}
        />
      )
    },
    {
      key: "attachments",
      label: t("option:worldBooks.detail.tabs.attachments", {
        defaultValue: "Attachments"
      }),
      children: (
        <AttachmentsTabContent
          worldBookId={worldBook.id}
          attachedCharacters={attachedCharacters}
          allCharacters={allCharacters}
          onAttachCharacter={onAttachCharacter}
          onDetachCharacter={onDetachCharacter}
        />
      )
    },
    {
      key: "stats",
      label: t("option:worldBooks.detail.tabs.stats", {
        defaultValue: "Stats"
      }),
      children: (
        <StatsTabContent
          statsData={statsData}
          loading={statsLoading}
          error={statsError}
        />
      )
    },
    {
      key: "settings",
      label: t("option:worldBooks.detail.tabs.settings", {
        defaultValue: "Settings"
      }),
      children: (
        <SettingsTabContent
          worldBook={worldBook}
          allWorldBooks={allWorldBooks}
          maxRecursiveDepth={maxRecursiveDepth}
          updating={updating}
          form={settingsFormInstance}
          banner={settingsBanner}
          onUpdateWorldBook={onUpdateWorldBook}
        />
      )
    }
  ]

  return (
    <main aria-label="World book detail" className="flex flex-col h-full">
      {/* Back button (mobile navigation) */}
      {onBack && (
        <button
          className="mb-2 flex items-center gap-1 text-sm text-primary hover:underline"
          onClick={onBack}
          aria-label={t("option:worldBooks.detail.backAriaLabel", {
            defaultValue: "Back to world books list"
          })}
        >
          <ArrowLeft className="w-4 h-4" />
          {t("option:header.modeWorldBooks", {
            defaultValue: "World Books"
          })}
        </button>
      )}

      {/* Summary bar */}
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-center gap-3">
        <h2 ref={headingRef} tabIndex={-1} className="text-lg font-semibold m-0">
          {worldBook.name}
        </h2>
        <Tag color={worldBook.enabled ? "green" : "default"}>
          {worldBook.enabled
            ? t("option:worldBooks.detail.status.enabled", {
                defaultValue: "Enabled"
              })
            : t("option:worldBooks.detail.status.disabled", {
                defaultValue: "Disabled"
              })}
        </Tag>
        <span className="text-sm text-text-muted">
          {t("option:worldBooks.detail.entrySummary", {
            defaultValue: `${entryCount} ${entryCount === 1 ? "entry" : "entries"}`
          })}
        </span>
        {worldBook.token_budget != null && (
          <span className="text-sm text-text-muted">
            {t("option:worldBooks.detail.tokenBudget", {
              defaultValue: `Budget: ${worldBook.token_budget}`
            })}
          </span>
        )}
        <span className="text-sm text-text-muted">
          {t("option:worldBooks.detail.characterSummary", {
            defaultValue: `${characterCount} ${characterCount === 1 ? "character" : "characters"}`
          })}
        </span>
        {lastModifiedDisplay.timestamp != null && (
          <span
            className="text-xs text-text-muted"
            title={lastModifiedDisplay.absolute || undefined}
          >
            {lastModifiedDisplay.relative}
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="flex-1 overflow-auto px-4">
        <Tabs
          activeKey={activeTab}
          items={tabItems}
          onChange={(key) => onActiveTabChange(key as WorldBookDetailTabKey)}
        />
      </div>
    </main>
  )
}

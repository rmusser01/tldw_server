import React from "react"
import { Button, Checkbox, Drawer, Input, Select, Space, Typography } from "antd"
import { useQuery } from "@tanstack/react-query"
import { ChevronDown, ChevronRight, FileText, Film, MessageSquare, Trash2 } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import {
  getMediaLibraryItemKey,
  normalizeMediaLibraryResponse
} from "@/components/Option/ResearchWorkspace/SourcesPane/media-library-normalization"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useDebounce } from "@/hooks/useDebounce"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { tldwClient } from "@/services/tldw"
import {
  getStudyPackMediaSections,
  type StudyPackMediaSectionsResponse,
  type StudyPackSourceSelection,
  type StudyPackSourceType,
  type StudyPackSummaryResponse
} from "@/services/flashcards"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"
import { useStudyPackCreateMutation, useStudyPackJobQuery } from "../hooks"

const { Text } = Typography

type StudyPackCreateDrawerProps = {
  open: boolean
  onClose: () => void
  initialIntent?: StudyPackIntent | null
  onCreated?: (pack: StudyPackSummaryResponse) => void
}

const DEFAULT_SOURCE_TYPE: StudyPackSourceType = "media"

const SOURCE_TYPE_ICONS: Record<StudyPackSourceType, React.ComponentType<{ className?: string }>> = {
  media: Film,
  note: FileText,
  message: MessageSquare
}

const normalizeIntentSources = (
  intent: StudyPackIntent | null | undefined
): StudyPackSourceSelection[] =>
  Array.isArray(intent?.sourceItems)
    ? intent.sourceItems.map((item) => ({
        source_type: item.sourceType,
        source_id: item.sourceId,
        source_title: item.sourceTitle,
        excerpt_text: item.excerptText,
        locator: item.locator
      }))
    : []

type MediaOption = { value: string; label: string; title?: string }

type SourceRow = {
  rowId: number
  item: StudyPackSourceSelection
  sectionPreview?: StudyPackMediaSectionsResponse
  needsSectionReview?: boolean
}
type SectionSelection = { section_indices: number[]; section_revision: string }

const withSectionSelection = (item: StudyPackSourceSelection, selection: SectionSelection | null) => {
  const locator = { ...item.locator, ...selection }
  if (!selection) {
    delete locator.section_indices
    delete locator.section_revision
  }
  return { ...item, locator: Object.keys(locator).length ? locator : undefined }
}

const isWholeMedia = (source: StudyPackSourceSelection) =>
  source.source_type === "media" &&
  !source.excerpt_text?.trim() &&
  source.locator?.timestamp_seconds == null &&
  source.locator?.chunk_index == null

const StudyPackMediaSections = ({
  rowId,
  item,
  sectionPreview,
  needsSectionReview,
  accountScope,
  onPreview,
  onChange
}: SourceRow & {
  accountScope: string
  onPreview: (rowId: number, result: StudyPackMediaSectionsResponse) => void
  onChange: (rowId: number, selection: SectionSelection | null) => void
}) => {
  const { t } = useTranslation(["option", "common"])
  const disclosureId = React.useId()
  const [expanded, setExpanded] = React.useState(false)
  const preview = useQuery({
    queryKey: ["flashcards:study-packs:media-sections", accountScope, item.source_id],
    enabled: expanded,
    gcTime: 0,
    retry: false,
    queryFn: async ({ signal }) => {
      const response = await getStudyPackMediaSections(item.source_id, { signal })
      signal.throwIfAborted()
      return response
    }
  })
  const reviewLabel = t("option:flashcards.studyPackReviewSections", {
    defaultValue: "Review sections"
  })
  const result = preview.isError ? undefined : sectionPreview
  const fetched = preview.isError ? undefined : preview.data
  React.useEffect(() => {
    if (expanded && fetched) onPreview(rowId, fetched)
  }, [expanded, fetched, onPreview, rowId])

  const storedIndices = item.locator?.section_indices
  const storedSelection = Array.isArray(storedIndices) ? storedIndices : undefined
  const sectionStatus = needsSectionReview
    ? sectionPreview?.sections.length
      ? t("option:flashcards.studyPackSectionsChanged", {
          defaultValue: "This media changed. Choose its sections again."
        })
      : t("option:flashcards.studyPackSectionsNoLongerAvailable", {
          defaultValue: "This media changed and no longer has sections to choose."
        })
    : storedSelection?.length === 0
      ? t("option:flashcards.studyPackSectionsRequired", { defaultValue: "Select at least one section." })
      : null

  return (
    <>
      <button
        type="button"
        className="mt-1 inline-flex items-center gap-1 text-sm text-text-muted hover:text-text"
        aria-expanded={expanded}
        aria-controls={disclosureId}
        onClick={() => setExpanded((current) => !current)}
      >
        {expanded ? (
          <ChevronDown className="size-4" aria-hidden="true" />
        ) : (
          <ChevronRight className="size-4" aria-hidden="true" />
        )}
        {reviewLabel}
      </button>
      {sectionStatus ? (
        <Text type="danger" role="status" className="block">
          {sectionStatus}
        </Text>
      ) : null}
      <div id={disclosureId} hidden={!expanded}>
        {expanded && (
          preview.isError ? (
            <Space className="mt-1" wrap>
              <Text role="alert">
                {t("option:flashcards.studyPackSectionsError", {
                  defaultValue: "Could not load sections."
                })}
              </Text>
              <Button size="small" onClick={() => void preview.refetch()} loading={preview.isFetching}>
                {t("common:retry", { defaultValue: "Retry" })}
              </Button>
            </Space>
          ) : !result ? (
            <Text type="secondary" role="status" className="block">
              {t("option:flashcards.studyPackSectionsLoading", { defaultValue: "Loading sections…" })}
            </Text>
          ) : result.sections.length === 0 ? (
            needsSectionReview ? (
              <Button size="small" className="mt-1" onClick={() => onChange(rowId, null)}>
                {t("option:flashcards.studyPackUseWholeSource", { defaultValue: "Use whole source" })}
              </Button>
            ) : (
              <Text type="secondary" role="status" className="block">
                {t("option:flashcards.studyPackSectionsWholeSource", {
                  defaultValue: "This media will be used as a single source."
                })}
              </Text>
            )
          ) : (
            <Checkbox.Group<number>
              aria-label={reviewLabel}
              className="mt-1 flex-col"
              options={result.sections.map(({ index, title }) => ({ label: title, value: index }))}
              value={storedSelection ?? []}
              onChange={(indices) => onChange(rowId, {
                section_revision: result.revision,
                section_indices: indices
              })}
            />
          )
        )}
      </div>
    </>
  )
}

const StudyPackMediaPicker = ({ onSelect }: { onSelect: (media: MediaOption) => void }) => {
  const { t } = useTranslation(["option", "common"])
  const pickerId = React.useId()
  const [opened, setOpened] = React.useState(false)
  const [search, setSearch] = React.useState("")
  const term = useDebounce(search.trim(), 300)
  const isDebouncing = search.trim() !== term
  const media = useQuery({
    queryKey: ["study-pack-media", pickerId, term],
    enabled: opened && !isDebouncing,
    gcTime: 0,
    queryFn: async ({ signal }) => {
      const page = { page: 1, results_per_page: 50 }
      const response = term
        ? await tldwClient.searchMedia({ query: term }, page, { signal })
        : await tldwClient.listMedia(page, { signal })
      signal.throwIfAborted()
      const { items, totalCount } = normalizeMediaLibraryResponse(response)
      const options = items.flatMap((item): MediaOption[] => {
        const id = getMediaLibraryItemKey(item)
        if (!id) return []
        const rawTitle = (item as { title?: unknown }).title
        const title = typeof rawTitle === "string" ? rawTitle.trim() || undefined : undefined
        return [{
          value: id,
          label: title ?? t("option:flashcards.studyPackUntitledMedia", {
            defaultValue: "Media {{id}}",
            id
          }),
          title
        }]
      })
      return {
        totalCount,
        options: options.map((option) =>
          options.some((other) => other.value !== option.value && other.label === option.label)
            ? { ...option, label: t("option:flashcards.studyPackMediaTitleWithId", {
                defaultValue: "{{title}} (ID {{id}})",
                title: option.label,
                id: option.value
              }) }
            : option
        )
      }
    }
  })

  const loading = isDebouncing || (opened && media.isFetching && !media.data)
  const result = loading || media.isError ? undefined : media.data
  const searchLabel = t("option:flashcards.studyPackMediaSearch", {
    defaultValue: "Search media library"
  })
  const emptyText = term
    ? t("option:flashcards.studyPackNoMediaMatches", { defaultValue: "No matching media." })
    : t("option:flashcards.studyPackMediaEmpty", { defaultValue: "Your media library is empty." })

  return (
    <div className="mb-3">
      <Select<string[], MediaOption>
        aria-label={searchLabel}
        placeholder={searchLabel}
        className="w-full"
        mode="multiple"
        value={[]}
        showSearch={{
          searchValue: search,
          onSearch: setSearch,
          filterOption: false,
          // Keep the failed term visible when Retry takes focus.
          autoClearSearchValue: false
        }}
        onOpenChange={(isOpen) => { if (isOpen) setOpened(true) }}
        onSelect={(_, option) => {
          onSelect(option)
          setSearch("")
        }}
        loading={loading}
        options={result?.options ?? []}
        notFoundContent={loading
          ? t("option:flashcards.studyPackMediaLoading", { defaultValue: "Loading media…" })
          : media.isError ? null : emptyText}
      />
      {result && result.options.length > 0 && result.options.length < result.totalCount ? (
        <Text type="secondary" className="mt-1 block">
          {t("option:flashcards.studyPackMediaResultLimit", {
            defaultValue: "Showing {{shown}} of {{total}} media items. Try a more specific search.",
            shown: result.options.length,
            total: result.totalCount
          })}
        </Text>
      ) : null}
      {!isDebouncing && media.isError ? (
        <Space className="mt-1">
          <Text role="alert">
            {t("option:flashcards.studyPackMediaError", { defaultValue: "Could not load media." })}
          </Text>
          <Button size="small" onClick={() => void media.refetch()} loading={media.isFetching}>
            {t("common:retry", { defaultValue: "Retry" })}
          </Button>
        </Space>
      ) : null}
    </div>
  )
}

export const StudyPackCreateDrawer: React.FC<StudyPackCreateDrawerProps> = ({
  open,
  onClose,
  initialIntent,
  onCreated
}) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const message = useAntdMessage()
  const createMutation = useStudyPackCreateMutation()
  const handledTerminalJobIdRef = React.useRef<number | null>(null)

  const [title, setTitle] = React.useState("")
  const [sources, setSources] = React.useState<SourceRow[]>([])
  const nextSourceRowId = React.useRef(0)
  const makeSourceRow = React.useCallback(
    (item: StudyPackSourceSelection): SourceRow => ({ rowId: nextSourceRowId.current++, item }),
    []
  )
  const [accountEpoch, setAccountEpoch] = React.useState(0)
  // Cache only within this drawer and its current account session.
  const accountScope = `${React.useId()}:${accountEpoch}`
  const [sourceType, setSourceType] = React.useState<StudyPackSourceType>(DEFAULT_SOURCE_TYPE)
  const [sourceId, setSourceId] = React.useState("")
  const [sourceTitle, setSourceTitle] = React.useState("")
  const [jobId, setJobId] = React.useState<number | null>(null)
  // Typing a raw ID is the fallback. Hide it when the drawer opens with
  // sources already chosen elsewhere (NE-08).
  const [manualEntryOpen, setManualEntryOpen] = React.useState(true)
  const manualSourceId = React.useId()

  const jobQuery = useStudyPackJobQuery(jobId, {
    enabled: open && jobId != null
  })
  const sourceTypeOptions = React.useMemo<Array<{ label: string; value: StudyPackSourceType }>>(
    () => [
      {
        label: t("option:flashcards.studyPackSourceType.media", {
          defaultValue: "Media"
        }),
        value: "media"
      },
      {
        label: t("option:flashcards.studyPackSourceType.note", {
          defaultValue: "Note"
        }),
        value: "note"
      },
      {
        label: t("option:flashcards.studyPackSourceType.message", {
          defaultValue: "Message"
        }),
        value: "message"
      }
    ],
    [t]
  )
  const sourceTypeLabel = React.useCallback(
    (type: StudyPackSourceType) =>
      sourceTypeOptions.find((option) => option.value === type)?.label ?? type,
    [sourceTypeOptions]
  )

  React.useEffect(() => {
    if (!open) return
    setTitle(initialIntent?.title ?? "")
    const intentSources = normalizeIntentSources(initialIntent)
    setSources(intentSources.map(makeSourceRow))
    setManualEntryOpen(intentSources.length === 0)
    setSourceType(DEFAULT_SOURCE_TYPE)
    setSourceId("")
    setSourceTitle("")
    setJobId(null)
    handledTerminalJobIdRef.current = null
  }, [initialIntent, makeSourceRow, open])

  React.useEffect(() => {
    if (!open) return
    return watchChatAccountChanges((invalidated) => {
      if (!invalidated) return
      setAccountEpoch((current) => current + 1)
      setSources([])
      onClose()
    })
  }, [open, onClose])

  React.useEffect(() => {
    const response = jobQuery.data
    if (!response) return
    if (handledTerminalJobIdRef.current === response.job.id) return

    if (response.job.status === "completed") {
      handledTerminalJobIdRef.current = response.job.id
      const pack = response.study_pack
      if (!pack?.deck_id) {
        message.error(
          t("option:flashcards.studyPackResultUnavailable", {
            defaultValue: "Study pack completed, but its review deck is unavailable."
          })
        )
        setJobId(null)
        return
      }
      onCreated?.(pack)
      onClose()
      navigate(`/flashcards?tab=review&deck_id=${pack.deck_id}`, {
        replace: true
      })
      return
    }

    if (response.job.status === "failed" || response.job.status === "cancelled") {
      handledTerminalJobIdRef.current = response.job.id
      message.error(
        response.error ||
          t("option:flashcards.studyPackCreateFailed", {
            defaultValue: "Study pack generation failed."
          })
      )
      setJobId(null)
    }
  }, [jobQuery.data, message, navigate, onClose, onCreated, t])

  const trimmedTitle = title.trim()
  const trimmedSourceId = sourceId.trim()
  const trimmedSourceTitle = sourceTitle.trim()
  const hasPendingJob = jobId != null
  let maxPasses = Infinity
  let selectedPasses = 0
  for (const { item, sectionPreview, needsSectionReview } of sources) {
    if (!sectionPreview) {
      selectedPasses += 1
      continue
    }
    if (sectionPreview.sections.length > 0) {
      maxPasses = Math.min(maxPasses, sectionPreview.max_generation_passes ?? Infinity)
    }
    if (needsSectionReview) continue
    if (sectionPreview.sections.length === 0) {
      selectedPasses += 1
    } else {
      const indices = item.locator?.section_indices
      if (!Array.isArray(indices)) continue
      for (const section of sectionPreview.sections) {
        if (indices.includes(section.index)) selectedPasses += section.generation_passes ?? 1
      }
    }
  }
  const exceedsPassBudget = selectedPasses > maxPasses
  const canSubmit =
    trimmedTitle.length > 0 &&
    sources.length > 0 &&
    !exceedsPassBudget &&
    sources.every(({ item, needsSectionReview }) => {
      if (needsSectionReview) return false
      const indices = item.locator?.section_indices
      return !isWholeMedia(item) || !Array.isArray(indices) || indices.length > 0
    }) &&
    !createMutation.isPending &&
    !hasPendingJob
  const pendingJobMessage = jobQuery.isError
    ? t("option:flashcards.studyPackJobStatusUnavailable", {
        defaultValue: "Unable to check study pack progress. Checking again shortly."
      })
    : jobQuery.data?.job.status === "running"
      ? t("option:flashcards.studyPackJobRunning", {
          defaultValue: "Creating your study pack."
        })
      : jobQuery.data?.job.status === "queued"
        ? t("option:flashcards.studyPackJobQueued", {
            defaultValue: "Study pack queued. Waiting for generation to start."
          })
        : t("option:flashcards.studyPackJobAccepted", {
            defaultValue: "Study pack request accepted. Waiting for a status update."
          })

  const handleAddSource = React.useCallback(() => {
    if (!trimmedSourceId) return

    const source = makeSourceRow({
      source_type: sourceType,
      source_id: trimmedSourceId,
      source_title: trimmedSourceTitle || undefined
    })
    setSources((current) => [...current, source])
    setSourceId("")
    setSourceTitle("")
  }, [makeSourceRow, sourceType, trimmedSourceId, trimmedSourceTitle])

  const handleSelectMedia = ({ value, title }: MediaOption) => {
    const source = makeSourceRow({ source_type: "media", source_id: value, source_title: title })
    setSources((current) =>
      current.some(({ item }) => item.source_type === "media" && item.source_id === value)
        ? current
        : [...current, source]
    )
  }

  const handleRemoveSource = React.useCallback((rowId: number) => {
    setSources((current) => current.filter((source) => source.rowId !== rowId))
  }, [])

  const updateSource = React.useCallback((rowId: number, update: (source: SourceRow) => SourceRow) => {
    setSources((current) => current.map((source) => source.rowId === rowId ? update(source) : source))
  }, [])

  const handleSectionPreview = React.useCallback((rowId: number, sectionPreview: StudyPackMediaSectionsResponse) => {
    updateSource(rowId, (source) => {
      if (source.sectionPreview === sectionPreview) return source
      const locator = source.item.locator
      const hasSectionScope = locator?.section_revision != null || Array.isArray(locator?.section_indices)
      const scopeChanged = hasSectionScope && (
        sectionPreview.sections.length === 0 ||
        locator?.section_revision !== sectionPreview.revision ||
        (source.needsSectionReview && source.sectionPreview?.sections.length === 0)
      )
      const updated = {
        ...source,
        sectionPreview,
        needsSectionReview: Boolean(scopeChanged || source.needsSectionReview)
      }
      if (sectionPreview.sections.length === 0 || (hasSectionScope && !scopeChanged)) return updated
      return {
        ...updated,
        item: withSectionSelection(source.item, {
          section_revision: sectionPreview.revision,
          section_indices: scopeChanged ? [] : sectionPreview.sections.map(({ index }) => index)
        })
      }
    })
  }, [updateSource])

  const handleSectionSelection = React.useCallback((rowId: number, selection: SectionSelection | null) => {
    updateSource(rowId, (source) => ({
      ...source,
      item: withSectionSelection(source.item, selection),
      needsSectionReview: source.needsSectionReview && selection?.section_indices.length === 0
    }))
  }, [updateSource])

  const handleSubmit = React.useCallback(async () => {
    if (!canSubmit) return

    try {
      const accepted = await createMutation.mutateAsync({
        title: trimmedTitle,
        source_items: sources.map(({ item }) => item),
        deck_mode: "new"
      })
      setJobId(accepted.job.id)
    } catch (error) {
      const detail =
        typeof error === "object" &&
        error !== null &&
        "response" in error &&
        typeof (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail === "string"
          ? (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null
      message.error(
        detail ||
          t("option:flashcards.studyPackCreateFailed", {
            defaultValue: "Study pack generation failed."
          })
      )
    }
  }, [canSubmit, createMutation, message, sources, t, trimmedTitle])

  const drawerTitle = t("option:flashcards.studyPackDrawerTitle", {
    defaultValue: "Create study pack"
  })

  return (
    <Drawer
      open={open}
      title={drawerTitle}
      onClose={onClose}
      destroyOnClose
      size="large"
      footer={
        <Space>
          <Button onClick={onClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            onClick={() => {
              void handleSubmit()
            }}
            disabled={!canSubmit}
            loading={createMutation.isPending || hasPendingJob}
          >
            {t("option:flashcards.studyPackCreateButton", {
              defaultValue: "Create study pack"
            })}
          </Button>
        </Space>
      }
    >
      <div className="space-y-4">
        {hasPendingJob ? (
          <Text role="status" aria-live="polite" aria-atomic="true" className="block">
            {pendingJobMessage}
          </Text>
        ) : null}
        <div className="space-y-1">
          <Text strong>Title</Text>
          <Input
            aria-label="Title"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder={t("option:flashcards.studyPackTitlePlaceholder", {
              defaultValue: "Study pack title"
            })}
          />
        </div>

        <div className="rounded border border-border bg-surface p-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <Text strong>
              {t("option:flashcards.studyPackSourcesLabel", {
                defaultValue: "Sources"
              })}
            </Text>
            <Text type="secondary">
              {t("option:flashcards.studyPackSourcesCount", {
                defaultValue: "{{count}} selected",
                count: sources.length
              })}
            </Text>
          </div>

          {open ? <StudyPackMediaPicker key={accountScope} onSelect={handleSelectMedia} /> : null}

          <button
            type="button"
            className="mb-2 inline-flex items-center gap-1 text-sm text-text-muted hover:text-text"
            aria-expanded={manualEntryOpen}
            aria-controls={manualSourceId}
            onClick={() => setManualEntryOpen((current) => !current)}
            data-testid="study-pack-manual-source-toggle"
          >
            {manualEntryOpen ? (
              <ChevronDown className="size-4" aria-hidden="true" />
            ) : (
              <ChevronRight className="size-4" aria-hidden="true" />
            )}
            {t("option:flashcards.studyPackManualSourceToggle", {
              defaultValue: "Add a source by ID"
            })}
          </button>
          <div id={manualSourceId} hidden={!manualEntryOpen}>
            {manualEntryOpen ? (
              <>
                <Space.Compact className="w-full">
                  <Select
                    aria-label="Source Type"
                    value={sourceType}
                    onChange={(value) => setSourceType(value)}
                    options={sourceTypeOptions}
                    className="min-w-[120px]"
                  />
                  <Input
                    aria-label={t("option:flashcards.studyPackSourceIdAdvancedLabel", {
                      defaultValue: "Source ID (advanced/manual)"
                    })}
                    value={sourceId}
                    onChange={(event) => setSourceId(event.target.value)}
                    placeholder={t("option:flashcards.studyPackSourceIdAdvancedPlaceholder", {
                      defaultValue: "Source ID (advanced/manual)"
                    })}
                  />
                  <Input
                    aria-label="Source Title"
                    value={sourceTitle}
                    onChange={(event) => setSourceTitle(event.target.value)}
                    placeholder={t("option:flashcards.studyPackSourceTitlePlaceholder", {
                      defaultValue: "Optional title"
                    })}
                  />
                  <Button onClick={handleAddSource}>
                    {t("common:add", { defaultValue: "Add source" })}
                  </Button>
                </Space.Compact>
                <Text
                  type="secondary"
                  className="mt-2 block text-xs"
                  data-testid="study-pack-source-id-guidance"
                >
                  {t("option:flashcards.studyPackSourceIdGuidance", {
                    defaultValue:
                      "Advanced/manual source reference. Choose Media, Note, or Message, then paste the item ID from that source."
                  })}
                </Text>
              </>
            ) : null}
          </div>

          <div className="mt-3 space-y-2">
            {sources.length === 0 ? (
              <Text type="secondary">
                {t("option:flashcards.studyPackSourcesEmpty", {
                  defaultValue:
                    "Add at least one supported source: Media, Note, or Message."
                })}
              </Text>
            ) : (
              sources.map((source, index) => {
                const { rowId, item } = source
                const TypeIcon = SOURCE_TYPE_ICONS[item.source_type] ?? FileText
                const typeLabel = sourceTypeLabel(item.source_type)
                const title = item.source_title?.trim()
                return (
                  <div
                    key={rowId}
                    className="flex items-start justify-between gap-3 rounded border border-border bg-background p-2"
                    data-testid={`study-pack-source-${index}`}
                  >
                    <div className="flex min-w-0 items-start gap-2">
                      <TypeIcon className="mt-0.5 size-4 shrink-0 text-text-muted" aria-hidden="true" />
                      <div className="min-w-0">
                        {/* Name a source by its title; fall back to the ID
                            only when nothing else identifies it (NE-08). */}
                        <Text className="block truncate">
                          {title || `${typeLabel} ${item.source_id}`}
                        </Text>
                        {title ? (
                          <Text type="secondary" className="block text-xs">
                            {typeLabel}
                          </Text>
                        ) : null}
                        {open && isWholeMedia(item) ? (
                          <StudyPackMediaSections
                            {...source}
                            accountScope={accountScope}
                            onPreview={handleSectionPreview}
                            onChange={handleSectionSelection}
                          />
                        ) : null}
                      </div>
                    </div>
                    <Button
                      type="text"
                      aria-label={t("common:remove", { defaultValue: "Remove" })}
                      icon={<Trash2 className="size-4" />}
                      onClick={() => handleRemoveSource(rowId)}
                    />
                  </div>
                )
              })
            )}
          </div>
          {exceedsPassBudget ? (
            <Text type="danger" role="status" className="mt-2 block">
              {t("option:flashcards.studyPackSectionsPassBudget", {
                defaultValue: "Your selection needs {{used}} generation passes. The limit is {{limit}}. Uncheck sections or remove a source.",
                used: selectedPasses,
                limit: maxPasses
              })}
            </Text>
          ) : null}
        </div>
      </div>
    </Drawer>
  )
}

export default StudyPackCreateDrawer

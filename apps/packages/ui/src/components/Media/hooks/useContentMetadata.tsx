import { useMemo } from 'react'
import type { MenuProps } from 'antd'
import type { TFunction } from 'i18next'
import {
  Send,
  Copy,
  Sparkles,
  MessageSquare,
  Clock,
  StickyNote,
  ExternalLink,
  Download,
  UploadCloud
} from 'lucide-react'
import { getTextStats } from '@/utils/text-stats'
import { formatRelativeTime } from '@/utils/dateFormatters'
import { estimateReadingTimeMinutes } from '../mediaMetadataUtils'
import type { MediaResultItem } from '../types'

const firstValidDateString = (...vals: any[]): string | null => {
  for (const v of vals) {
    if (typeof v !== 'string') continue
    const trimmed = v.trim()
    if (!trimmed) continue
    const asDate = new Date(trimmed)
    if (!Number.isNaN(asDate.getTime())) {
      return trimmed
    }
  }
  return null
}

const SAFE_METADATA_PRIORITY_KEYS = [
  'doi',
  'pmid',
  'pmcid',
  'arxiv_id',
  'journal',
  'license'
] as const

const SAFE_METADATA_LABELS: Record<string, string> = {
  doi: 'DOI',
  pmid: 'PMID',
  pmcid: 'PMCID',
  arxiv_id: 'arXiv',
  journal: 'Journal',
  license: 'License'
}

const toDisplayMetadataLabel = (key: string): string => {
  if (SAFE_METADATA_LABELS[key]) return SAFE_METADATA_LABELS[key]
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

export const normalizeVectorProcessingStatus = (value: unknown): string | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value >= 1) return 'completed'
    if (value < 0) return 'failed'
    return 'pending'
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const lowered = value.trim().toLowerCase()
    if (['1', 'true', 'complete', 'completed', 'done', 'success'].includes(lowered)) {
      return 'completed'
    }
    if (['-1', 'false', 'failed', 'error'].includes(lowered)) {
      return 'failed'
    }
    if (['0', 'pending', 'queued', 'in_progress', 'in-progress'].includes(lowered)) {
      return 'pending'
    }
    return lowered
  }
  return null
}

export const normalizeChunkingStatus = (value: unknown): string | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value >= 1) return 'completed'
    if (value < 0) return 'failed'
    return 'pending'
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const lowered = value.trim().toLowerCase()
    if (['1', 'true', 'complete', 'completed', 'done', 'success'].includes(lowered)) {
      return 'completed'
    }
    if (['-1', 'false', 'failed', 'error'].includes(lowered)) {
      return 'failed'
    }
    if (['0', 'pending', 'queued', 'in_progress', 'in-progress'].includes(lowered)) {
      return 'pending'
    }
    return lowered
  }
  return null
}

export const processingStatusClass = (status: string): string => {
  if (status.includes('fail') || status.includes('error')) {
    return 'bg-danger/10 text-danger'
  }
  if (status.includes('complete') || status.includes('success') || status === 'done') {
    return 'bg-success/10 text-success'
  }
  return 'bg-warn/10 text-warn'
}

export const formatProcessingStatus = (status: string): string =>
  status
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase())

const toDisplayMetadataValue = (value: unknown): string => {
  if (value == null) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => toDisplayMetadataValue(entry))
      .filter((entry) => entry.length > 0)
      .join(', ')
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }
  return ''
}

export interface UseContentMetadataDeps {
  selectedMedia: MediaResultItem | null
  content: string
  mediaDetail: any
  isNote: boolean
  editState: {
    selectedAnalysis: { text: string; type: string } | null
  }
  modals: {
    handleExportMedia: () => void
    handleReprocessMedia: () => Promise<void>
    canScheduleSourceRefresh: boolean
    setScheduleRefreshModalOpen: (v: boolean) => void
    setAnalysisModalOpen: (v: boolean) => void
  }
  onChatWithMedia?: () => void
  onChatAboutMedia?: () => void
  onGenerateFlashcardsFromContent?: (payload: {
    text: string
    sourceId?: string
    sourceTitle?: string
  }) => void
  onCreateNoteWithContent?: (content: string, title: string) => void
  onOpenInMultiReview?: () => void
  handleCopyContent: () => void
  handleCopyMetadata: () => void
  t: TFunction
}

export function useContentMetadata(deps: UseContentMetadataDeps) {
  const {
    selectedMedia,
    content,
    mediaDetail,
    isNote,
    editState,
    modals,
    onChatWithMedia,
    onChatAboutMedia,
    onGenerateFlashcardsFromContent,
    onCreateNoteWithContent,
    onOpenInMultiReview,
    handleCopyContent,
    handleCopyMetadata,
    t
  } = deps

  // Text stats
  const { wordCount, charCount, paragraphCount } = useMemo(() => {
    const text = content || ''
    const apiWordCount = mediaDetail?.content?.word_count
    const {
      wordCount: computedWordCount,
      charCount,
      paragraphCount
    } = getTextStats(text)
    const wordCountValue =
      typeof apiWordCount === 'number' ? apiWordCount : computedWordCount
    return {
      wordCount: wordCountValue,
      charCount,
      paragraphCount
    }
  }, [content, mediaDetail])

  const readingTimeMinutes = useMemo(
    () =>
      estimateReadingTimeMinutes({
        wordCount,
        charCount
      }),
    [charCount, wordCount]
  )

  const ingestedAt = useMemo(
    () =>
      firstValidDateString(
        selectedMedia?.meta?.created_at,
        mediaDetail?.created_at,
        mediaDetail?.ingested_at,
        mediaDetail?.metadata?.created_at,
        selectedMedia?.raw?.created_at
      ),
    [
      mediaDetail?.created_at,
      mediaDetail?.ingested_at,
      mediaDetail?.metadata?.created_at,
      selectedMedia?.meta?.created_at,
      selectedMedia?.raw?.created_at
    ]
  )

  const lastModifiedAt = useMemo(
    () =>
      firstValidDateString(
        mediaDetail?.updated_at,
        mediaDetail?.last_modified,
        mediaDetail?.last_modified_at,
        mediaDetail?.modified_at,
        mediaDetail?.metadata?.updated_at,
        mediaDetail?.metadata?.last_modified,
        selectedMedia?.raw?.updated_at,
        selectedMedia?.raw?.last_modified
      ),
    [
      mediaDetail?.last_modified,
      mediaDetail?.last_modified_at,
      mediaDetail?.metadata?.last_modified,
      mediaDetail?.metadata?.updated_at,
      mediaDetail?.modified_at,
      mediaDetail?.updated_at,
      selectedMedia?.raw?.last_modified,
      selectedMedia?.raw?.updated_at
    ]
  )

  const ingestedLabel = ingestedAt
    ? formatRelativeTime(ingestedAt, t, { compact: true })
    : null
  const lastModifiedLabel = lastModifiedAt
    ? formatRelativeTime(lastModifiedAt, t, { compact: true })
    : null
  const readingTimeLabel =
    readingTimeMinutes != null
      ? t('review:mediaPage.readingTime', {
          defaultValue: `${readingTimeMinutes} min read`,
          minutes: readingTimeMinutes
        })
      : null

  const safeMetadata = useMemo(() => {
    const fromDetail = mediaDetail?.safe_metadata
    if (fromDetail && typeof fromDetail === 'object' && !Array.isArray(fromDetail)) {
      return fromDetail as Record<string, unknown>
    }
    const fromRaw = selectedMedia?.raw?.safe_metadata
    if (fromRaw && typeof fromRaw === 'object' && !Array.isArray(fromRaw)) {
      return fromRaw as Record<string, unknown>
    }
    return {} as Record<string, unknown>
  }, [mediaDetail?.safe_metadata, selectedMedia?.raw?.safe_metadata])

  const safeMetadataEntries = useMemo(() => {
    const entries = Object.entries(safeMetadata)
      .map(([key, value]) => ({
        key,
        label: toDisplayMetadataLabel(key),
        value: toDisplayMetadataValue(value)
      }))
      .filter((entry) => entry.value.length > 0)

    const priorityEntries = SAFE_METADATA_PRIORITY_KEYS.flatMap((key) => {
      const match = entries.find((entry) => entry.key === key)
      return match ? [match] : []
    })
    const priorityKeySet = new Set<string>(SAFE_METADATA_PRIORITY_KEYS)
    const remainingEntries = entries
      .filter((entry) => !priorityKeySet.has(entry.key))
      .sort((left, right) => left.label.localeCompare(right.label))

    return [...priorityEntries, ...remainingEntries]
  }, [safeMetadata])

  const chunkingStatus = useMemo(() => {
    const candidates = [
      mediaDetail?.chunking_status,
      mediaDetail?.processing?.chunking_status,
      mediaDetail?.processing?.chunking,
      selectedMedia?.raw?.chunking_status,
      selectedMedia?.raw?.processing?.chunking_status
    ]
    for (const candidate of candidates) {
      const normalized = normalizeChunkingStatus(candidate)
      if (normalized) return normalized
    }
    return null
  }, [
    mediaDetail?.chunking_status,
    mediaDetail?.processing?.chunking,
    mediaDetail?.processing?.chunking_status,
    selectedMedia?.raw?.chunking_status,
    selectedMedia?.raw?.processing?.chunking_status
  ])

  const vectorProcessingStatus = useMemo(() => {
    const candidates = [
      mediaDetail?.vector_processing,
      mediaDetail?.vector_processing_status,
      mediaDetail?.processing?.vector_processing,
      mediaDetail?.processing?.vector_processing_status,
      selectedMedia?.raw?.vector_processing,
      selectedMedia?.raw?.vector_processing_status,
      selectedMedia?.raw?.processing?.vector_processing,
      selectedMedia?.raw?.processing?.vector_processing_status
    ]
    for (const candidate of candidates) {
      const normalized = normalizeVectorProcessingStatus(candidate)
      if (normalized) return normalized
    }
    return null
  }, [
    mediaDetail?.processing?.vector_processing,
    mediaDetail?.processing?.vector_processing_status,
    mediaDetail?.vector_processing,
    mediaDetail?.vector_processing_status,
    selectedMedia?.raw?.processing?.vector_processing,
    selectedMedia?.raw?.processing?.vector_processing_status,
    selectedMedia?.raw?.vector_processing,
    selectedMedia?.raw?.vector_processing_status
  ])

  const processingStatusBadges = useMemo(() => {
    const badges: Array<{ key: 'chunking' | 'vector'; label: string; status: string }> = []
    if (chunkingStatus) {
      badges.push({
        key: 'chunking',
        label: t('review:mediaPage.chunkingStatusLabel', {
          defaultValue: 'Chunking'
        }),
        status: chunkingStatus
      })
    }
    if (vectorProcessingStatus) {
      badges.push({
        key: 'vector',
        label: t('review:mediaPage.vectorStatusLabel', {
          defaultValue: 'Vector'
        }),
        status: vectorProcessingStatus
      })
    }
    return badges
  }, [chunkingStatus, t, vectorProcessingStatus])

  const hasDetailedMetadata =
    processingStatusBadges.length > 0 || safeMetadataEntries.length > 0

  const chatWithLabel = t('review:reviewPage.chatWithMedia', {
    defaultValue: 'Chat with this media'
  })
  const chatWithClarifiedLabel = t('review:reviewPage.chatWithMediaClarified', {
    defaultValue: 'Chat with this media (full content)'
  })
  const chatAboutClarifiedLabel = t('review:reviewPage.chatAboutMediaClarified', {
    defaultValue: 'Chat about this media (RAG context)'
  })

  // Actions dropdown menu items
  const actionMenuItems: MenuProps['items'] = useMemo(() => [
    ...(!isNote && (onChatWithMedia || onChatAboutMedia) ? [{
      key: 'group-use',
      type: 'group' as const,
      label: t('review:mediaPage.menuGroupUse', { defaultValue: 'Use' }),
      children: [
        ...(onChatWithMedia ? [{
          key: 'chat-with',
          label: chatWithClarifiedLabel,
          icon: <Send className="w-4 h-4" />,
          onClick: onChatWithMedia
        }] : []),
        ...(onChatAboutMedia ? [{
          key: 'chat-about',
          label: chatAboutClarifiedLabel,
          icon: <MessageSquare className="w-4 h-4" />,
          onClick: onChatAboutMedia
        }] : [])
      ]
    }] : []),
    ...(!isNote && (onCreateNoteWithContent || onGenerateFlashcardsFromContent)
      ? [
      { type: 'divider' as const },
      {
        key: 'group-create',
        type: 'group' as const,
        label: t('review:mediaPage.menuGroupCreate', { defaultValue: 'Create' }),
        children: [
          ...(onCreateNoteWithContent
            ? [
                {
                  key: 'create-note-content',
                  label: t('review:mediaPage.createNoteWithContent', {
                    defaultValue: 'Create note with content'
                  }),
                  icon: <StickyNote className="w-4 h-4" />,
                  onClick: () => {
                    const title =
                      selectedMedia?.title ||
                      t('review:mediaPage.untitled', { defaultValue: 'Untitled' })
                    onCreateNoteWithContent(content, title)
                  }
                },
                ...(editState.selectedAnalysis
                  ? [
                      {
                        key: 'create-note-content-analysis',
                        label: t('review:mediaPage.createNoteWithContentAnalysis', {
                          defaultValue: 'Create note with content + analysis'
                        }),
                        icon: <StickyNote className="w-4 h-4" />,
                        onClick: () => {
                          const title =
                            selectedMedia?.title ||
                            t('review:mediaPage.untitled', {
                              defaultValue: 'Untitled'
                            })
                          const noteContent = `${content}\n\n---\n\n## Analysis\n\n${editState.selectedAnalysis!.text}`
                          onCreateNoteWithContent(noteContent, title)
                        }
                      }
                    ]
                  : [])
              ]
            : []),
          ...(onGenerateFlashcardsFromContent && content.trim().length > 0
            ? [
                {
                  key: 'generate-flashcards-content',
                  label: t('review:mediaPage.generateFlashcardsFromContent', {
                    defaultValue: 'Generate flashcards from content'
                  }),
                  icon: <Sparkles className="w-4 h-4" />,
                  onClick: () =>
                    onGenerateFlashcardsFromContent({
                      text: content,
                      sourceId:
                        selectedMedia?.id != null ? String(selectedMedia.id) : undefined,
                      sourceTitle: selectedMedia?.title
                    })
                }
              ]
            : [])
        ]
      }
    ]
      : []),
    ...(!isNote &&
    (onChatWithMedia ||
      onChatAboutMedia ||
      onCreateNoteWithContent ||
      onGenerateFlashcardsFromContent)
      ? [{ type: 'divider' as const }]
      : []),
    {
      key: 'group-copy',
      type: 'group' as const,
      label: t('review:mediaPage.menuGroupCopy', { defaultValue: 'Copy' }),
      children: [
        {
          key: 'copy-content',
          label: t('review:mediaPage.copyContent', { defaultValue: 'Copy content' }),
          icon: <Copy className="w-4 h-4" />,
          onClick: handleCopyContent
        },
        {
          key: 'copy-metadata',
          label: t('review:mediaPage.copyMetadata', { defaultValue: 'Copy metadata' }),
          icon: <Copy className="w-4 h-4" />,
          onClick: handleCopyMetadata
        }
      ]
    },
    ...(!isNote ? [
      { type: 'divider' as const },
      {
        key: 'export-media',
        label: t('review:mediaPage.exportMedia', {
          defaultValue: 'Export content'
        }),
        icon: <Download className="w-4 h-4" />,
        onClick: modals.handleExportMedia
      },
      {
        key: 'reprocess-media',
        label: t('review:mediaPage.reprocessMedia', {
          defaultValue: 'Reprocess content'
        }),
        icon: <UploadCloud className="w-4 h-4" />,
        onClick: () => {
          void modals.handleReprocessMedia()
        }
      },
      ...(modals.canScheduleSourceRefresh
        ? [
            {
              key: 'schedule-refresh',
              label: t('review:mediaPage.scheduleSourceRefresh', {
                defaultValue: 'Schedule source refresh'
              }),
              icon: <Clock className="w-4 h-4" />,
              onClick: () => modals.setScheduleRefreshModalOpen(true)
            }
          ]
        : []),
      ...(onOpenInMultiReview
        ? [
            {
              key: 'open-multi-review',
              label: t('review:reviewPage.openInMulti', 'Open in Multi-Item Review'),
              icon: <ExternalLink className="w-4 h-4" />,
              onClick: onOpenInMultiReview
            }
          ]
        : [])
    ] : [])
  ], [
    isNote,
    onChatWithMedia,
    onChatAboutMedia,
    onCreateNoteWithContent,
    onGenerateFlashcardsFromContent,
    onOpenInMultiReview,
    selectedMedia,
    content,
    editState.selectedAnalysis,
    modals,
    handleCopyContent,
    handleCopyMetadata,
    chatWithClarifiedLabel,
    chatAboutClarifiedLabel,
    t
  ])

  return {
    wordCount,
    charCount,
    paragraphCount,
    readingTimeMinutes,
    ingestedAt,
    lastModifiedAt,
    ingestedLabel,
    lastModifiedLabel,
    readingTimeLabel,
    safeMetadataEntries,
    processingStatusBadges,
    hasDetailedMetadata,
    chatWithLabel,
    actionMenuItems
  }
}

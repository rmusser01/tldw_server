import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { message } from 'antd'
import { bgRequest } from '@/services/background-proxy'
import { downloadBlob } from '@/utils/download-blob'
import type { MediaResultItem } from '../types'

type MediaExportFormat = 'json' | 'markdown' | 'text' | 'bibtex'
export type MediaAnnotationColor = 'yellow' | 'green' | 'blue' | 'pink'
type ReingestSchedulePreset = 'hourly' | 'daily' | 'weekly'
type DocumentIntelligenceTab =
  | 'outline'
  | 'insights'
  | 'references'
  | 'figures'
  | 'annotations'

interface MediaAnnotationEntry {
  id: string
  media_id: number
  location: string
  text: string
  color: MediaAnnotationColor
  note?: string
  annotation_type: 'highlight' | 'page_note'
  created_at?: string
  updated_at?: string
}

interface DocumentIntelligencePanelState<T = any> {
  loading: boolean
  error: string | null
  data: T[]
}

type DocumentIntelligencePanelsState = Record<
  DocumentIntelligenceTab,
  DocumentIntelligencePanelState
>

export const DOCUMENT_INTELLIGENCE_TABS: Array<{
  key: DocumentIntelligenceTab
  label: string
}> = [
  { key: 'outline', label: 'Outline' },
  { key: 'insights', label: 'Insights' },
  { key: 'references', label: 'References' },
  { key: 'figures', label: 'Figures' },
  { key: 'annotations', label: 'Annotations' }
]

export const ANNOTATION_COLOR_OPTIONS: Array<{
  value: MediaAnnotationColor
  label: string
}> = [
  { value: 'yellow', label: 'Yellow' },
  { value: 'green', label: 'Green' },
  { value: 'blue', label: 'Blue' },
  { value: 'pink', label: 'Pink' }
]

const REINGEST_CRON_BY_PRESET: Record<ReingestSchedulePreset, string> = {
  hourly: '0 * * * *',
  daily: '0 9 * * *',
  weekly: '0 9 * * MON'
}

const createDefaultDocumentIntelligencePanels =
  (): DocumentIntelligencePanelsState => ({
    outline: { loading: false, error: null, data: [] },
    insights: { loading: false, error: null, data: [] },
    references: { loading: false, error: null, data: [] },
    figures: { loading: false, error: null, data: [] },
    annotations: { loading: false, error: null, data: [] }
  })

const createLoadingDocumentIntelligencePanels =
  (): DocumentIntelligencePanelsState => ({
    outline: { loading: true, error: null, data: [] },
    insights: { loading: true, error: null, data: [] },
    references: { loading: true, error: null, data: [] },
    figures: { loading: true, error: null, data: [] },
    annotations: { loading: true, error: null, data: [] }
  })

const getErrorStatusCode = (error: unknown): number | null => {
  if (!error || typeof error !== 'object') return null
  const candidate = (error as any).status ?? (error as any).statusCode
  return typeof candidate === 'number' && Number.isFinite(candidate)
    ? candidate
    : null
}

const firstNonEmptyString = (...vals: any[]): string => {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim().length > 0) return v
  }
  return ''
}

const toExportFileStem = (selectedMedia: MediaResultItem): string => {
  const fromTitle = String(selectedMedia.title || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
  const safeTitle = fromTitle.length > 0 ? fromTitle : `media-${selectedMedia.id}`
  return `${safeTitle}-${selectedMedia.id}`
}

const sanitizeBibtexValue = (value: string): string =>
  value
    .replace(/\\/g, '\\\\')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim()

const toCitationFieldString = (value: unknown): string => {
  if (value == null) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (Array.isArray(value)) {
    return value
      .flatMap((entry) => {
        if (typeof entry === 'string') return [entry.trim()]
        if (entry && typeof entry === 'object' && 'name' in entry) {
          const name = (entry as { name?: unknown }).name
          return typeof name === 'string' ? [name.trim()] : []
        }
        return []
      })
      .filter(Boolean)
      .join(' and ')
  }
  return ''
}

const toCitationKey = (
  selectedMedia: MediaResultItem,
  safeMetadata: Record<string, unknown>
): string => {
  const doi = toCitationFieldString(safeMetadata.doi)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  if (doi.length > 0) return doi

  const titleSlug = String(selectedMedia.title || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  if (titleSlug.length > 0) {
    const year = toCitationFieldString(
      safeMetadata.year ?? safeMetadata.publication_year ?? safeMetadata.published_year
    )
      .replace(/[^\d]/g, '')
      .slice(0, 4)
    return `${titleSlug}${year ? `_${year}` : ''}`
  }

  return `media_${String(selectedMedia.id).replace(/[^a-z0-9]+/gi, '_')}`
}

const buildBibtexExport = (
  selectedMedia: MediaResultItem,
  exportPayload: {
    title: string
    source: string
    exported_at: string
  },
  safeMetadata: Record<string, unknown>
): string => {
  const entryType = 'article'
  const entryKey = toCitationKey(selectedMedia, safeMetadata)
  const yearField = toCitationFieldString(
    safeMetadata.year ??
      safeMetadata.publication_year ??
      safeMetadata.published_year ??
      safeMetadata.published_at
  )
    .replace(/[^\d]/g, '')
    .slice(0, 4)
  const rawFieldTuples: Array<[string, string]> = [
    ['title', exportPayload.title || `Media ${selectedMedia.id}`],
    ['author', toCitationFieldString(safeMetadata.authors ?? safeMetadata.author)],
    ['journal', toCitationFieldString(safeMetadata.journal)],
    ['year', yearField],
    ['doi', toCitationFieldString(safeMetadata.doi)],
    ['url', toCitationFieldString(safeMetadata.url) || exportPayload.source],
    ['pmid', toCitationFieldString(safeMetadata.pmid)],
    ['eprint', toCitationFieldString(safeMetadata.arxiv_id ?? safeMetadata.arxiv)]
  ]
  const fieldTuples = rawFieldTuples.filter(([, value]) => value.trim().length > 0)

  const body = fieldTuples
    .map(([field, value], index) => {
      const suffix = index === fieldTuples.length - 1 ? '' : ','
      return `  ${field} = {${sanitizeBibtexValue(value)}}${suffix}`
    })
    .join('\n')

  return [`@${entryType}{${entryKey},`, body, '}'].join('\n')
}

export interface UseContentViewerModalsDeps {
  selectedMedia: MediaResultItem | null
  content: string
  mediaDetail: any
  selectedMediaId: string | null
  isNote: boolean
  editingKeywords: string[]
  selectedAnalysis: { type: string; text: string } | null
  collapsedSections: Record<string, boolean>
  setCollapsedSections: (updater: (prev: Record<string, boolean>) => Record<string, boolean>) => void
  contentBodyRef: React.RefObject<HTMLDivElement | null>
  onRefreshMedia?: () => void
  t: (key: string, opts?: Record<string, any>) => string
}

export function useContentViewerModals(deps: UseContentViewerModalsDeps) {
  const {
    selectedMedia,
    content,
    mediaDetail,
    selectedMediaId,
    isNote,
    editingKeywords,
    selectedAnalysis,
    collapsedSections,
    setCollapsedSections,
    contentBodyRef,
    onRefreshMedia,
    t
  } = deps

  const [analysisModalOpen, setAnalysisModalOpen] = useState(false)
  const [diffModalOpen, setDiffModalOpen] = useState(false)
  const [diffLeftText, setDiffLeftText] = useState('')
  const [diffRightText, setDiffRightText] = useState('')
  const [diffLeftLabel, setDiffLeftLabel] = useState('')
  const [diffRightLabel, setDiffRightLabel] = useState('')
  const [diffMetadataSummary, setDiffMetadataSummary] = useState<{
    left: string[]
    right: string[]
    changed: string[]
  } | null>(null)
  const [exportModalOpen, setExportModalOpen] = useState(false)
  const [exportFormat, setExportFormat] = useState<MediaExportFormat>('json')
  const [scheduleRefreshModalOpen, setScheduleRefreshModalOpen] = useState(false)
  const [scheduleRefreshPreset, setScheduleRefreshPreset] =
    useState<ReingestSchedulePreset>('daily')
  const [scheduleRefreshSubmitting, setScheduleRefreshSubmitting] = useState(false)
  const [metadataDetailsExpanded, setMetadataDetailsExpanded] = useState(false)
  const [contentExpanded, setContentExpanded] = useState(true)

  // Document intelligence state
  const [activeIntelligenceTab, setActiveIntelligenceTab] =
    useState<DocumentIntelligenceTab>('outline')
  const [documentIntelligencePanels, setDocumentIntelligencePanels] =
    useState<DocumentIntelligencePanelsState>(() =>
      createDefaultDocumentIntelligencePanels()
    )
  const [loadedDocumentIntelligenceMediaId, setLoadedDocumentIntelligenceMediaId] =
    useState<string | null>(null)

  // Annotation state
  const [annotationSelectionText, setAnnotationSelectionText] = useState('')
  const [annotationSelectionLocation, setAnnotationSelectionLocation] = useState('')
  const [annotationManualText, setAnnotationManualText] = useState('')
  const [annotationDraftNote, setAnnotationDraftNote] = useState('')
  const [annotationDraftColor, setAnnotationDraftColor] =
    useState<MediaAnnotationColor>('yellow')
  const [annotationCreating, setAnnotationCreating] = useState(false)
  const [annotationUpdatingId, setAnnotationUpdatingId] = useState<string | null>(null)
  const [annotationDeletingId, setAnnotationDeletingId] = useState<string | null>(null)
  const [annotationSyncing, setAnnotationSyncing] = useState(false)

  // Embedded media state
  const [embeddedMediaUrl, setEmbeddedMediaUrl] = useState<string | null>(null)
  const [embeddedMediaLoading, setEmbeddedMediaLoading] = useState(false)
  const [embeddedMediaError, setEmbeddedMediaError] = useState<string | null>(null)
  const embeddedMediaObjectUrlRef = useRef<string | null>(null)

  const intelligenceSectionCollapsed = collapsedSections.intelligence ?? true

  // Reset metadata details on media change
  useEffect(() => {
    setMetadataDetailsExpanded(false)
  }, [selectedMedia?.id])

  // Reset document intelligence on media change
  useEffect(() => {
    setActiveIntelligenceTab('outline')
    setDocumentIntelligencePanels(createDefaultDocumentIntelligencePanels())
    setLoadedDocumentIntelligenceMediaId(null)
    setAnnotationSelectionText('')
    setAnnotationSelectionLocation('')
    setAnnotationManualText('')
    setAnnotationDraftNote('')
    setAnnotationDraftColor('yellow')
  }, [selectedMediaId, selectedMedia?.kind])

  const sourceUrlForScheduling = useMemo(() => {
    const candidate = firstNonEmptyString(
      selectedMedia?.raw?.url,
      mediaDetail?.url,
      mediaDetail?.source_url,
      mediaDetail?.sourceUrl
    )
    if (!candidate) return ''
    try {
      const parsed = new URL(candidate)
      if (!['http:', 'https:'].includes(parsed.protocol)) return ''
      return candidate
    } catch {
      return ''
    }
  }, [
    mediaDetail?.sourceUrl,
    mediaDetail?.source_url,
    mediaDetail?.url,
    selectedMedia?.raw?.url
  ])

  const canScheduleSourceRefresh = !isNote && sourceUrlForScheduling.length > 0

  const activeDocumentIntelligencePanel =
    documentIntelligencePanels[activeIntelligenceTab]

  const annotationPanelEntries = useMemo(
    () =>
      Array.isArray(documentIntelligencePanels.annotations.data)
        ? (documentIntelligencePanels.annotations.data as MediaAnnotationEntry[])
        : [],
    [documentIntelligencePanels.annotations.data]
  )

  const setAnnotationPanelEntries = useCallback(
    (
      updater:
        | MediaAnnotationEntry[]
        | ((prev: MediaAnnotationEntry[]) => MediaAnnotationEntry[])
    ) => {
      setDocumentIntelligencePanels((prev) => {
        const previousRows = Array.isArray(prev.annotations.data)
          ? (prev.annotations.data as MediaAnnotationEntry[])
          : []
        const nextRows =
          typeof updater === 'function'
            ? (updater as (prev: MediaAnnotationEntry[]) => MediaAnnotationEntry[])(previousRows)
            : updater
        return {
          ...prev,
          annotations: {
            loading: false,
            error: null,
            data: nextRows
          }
        }
      })
    },
    []
  )

  const clearAnnotationDraft = useCallback(() => {
    setAnnotationSelectionText('')
    setAnnotationSelectionLocation('')
    setAnnotationManualText('')
    setAnnotationDraftNote('')
    setAnnotationDraftColor('yellow')
  }, [])

  // Fetch document intelligence
  const fetchDocumentIntelligence = useCallback(async () => {
    if (!selectedMediaId || isNote) return
    const encodedMediaId = encodeURIComponent(selectedMediaId)
    setDocumentIntelligencePanels(createLoadingDocumentIntelligencePanels())

    const results = await Promise.allSettled([
      bgRequest<any>({
        path: `/api/v1/media/${encodedMediaId}/outline` as any,
        method: 'GET' as any
      }),
      bgRequest<any>({
        path: `/api/v1/media/${encodedMediaId}/insights` as any,
        method: 'POST' as any,
        body: {}
      }),
      bgRequest<any>({
        path: `/api/v1/media/${encodedMediaId}/references?enrich=true&limit=25` as any,
        method: 'GET' as any
      }),
      bgRequest<any>({
        path: `/api/v1/media/${encodedMediaId}/figures?min_size=50` as any,
        method: 'GET' as any
      }),
      bgRequest<any>({
        path: `/api/v1/media/${encodedMediaId}/annotations` as any,
        method: 'GET' as any
      })
    ])

    const nextPanels = createDefaultDocumentIntelligencePanels()
    const assignError = (tab: DocumentIntelligenceTab, error: unknown) => {
      const statusCode = getErrorStatusCode(error)
      if (statusCode === 404 || statusCode === 410 || statusCode === 422) {
        nextPanels[tab] = { loading: false, error: null, data: [] }
        return
      }
      nextPanels[tab] = {
        loading: false,
        error: t('review:mediaPage.intelligenceLoadError', {
          defaultValue: 'Unable to load this panel. Try again.'
        }),
        data: []
      }
    }

    const [
      outlineResult,
      insightsResult,
      referencesResult,
      figuresResult,
      annotationsResult
    ] = results

    if (outlineResult.status === 'fulfilled') {
      nextPanels.outline = {
        loading: false,
        error: null,
        data: Array.isArray(outlineResult.value?.entries)
          ? outlineResult.value.entries
          : []
      }
    } else {
      assignError('outline', outlineResult.reason)
    }

    if (insightsResult.status === 'fulfilled') {
      nextPanels.insights = {
        loading: false,
        error: null,
        data: Array.isArray(insightsResult.value?.insights)
          ? insightsResult.value.insights
          : []
      }
    } else {
      assignError('insights', insightsResult.reason)
    }

    if (referencesResult.status === 'fulfilled') {
      nextPanels.references = {
        loading: false,
        error: null,
        data: Array.isArray(referencesResult.value?.references)
          ? referencesResult.value.references
          : []
      }
    } else {
      assignError('references', referencesResult.reason)
    }

    if (figuresResult.status === 'fulfilled') {
      nextPanels.figures = {
        loading: false,
        error: null,
        data: Array.isArray(figuresResult.value?.figures)
          ? figuresResult.value.figures
          : []
      }
    } else {
      assignError('figures', figuresResult.reason)
    }

    if (annotationsResult.status === 'fulfilled') {
      nextPanels.annotations = {
        loading: false,
        error: null,
        data: Array.isArray(annotationsResult.value?.annotations)
          ? annotationsResult.value.annotations
          : []
      }
    } else {
      assignError('annotations', annotationsResult.reason)
    }

    setDocumentIntelligencePanels(nextPanels)
    setLoadedDocumentIntelligenceMediaId(selectedMediaId)
  }, [isNote, selectedMediaId, t])

  const fetchDocumentIntelligenceRef = useRef(fetchDocumentIntelligence)
  useEffect(() => {
    fetchDocumentIntelligenceRef.current = fetchDocumentIntelligence
  }, [fetchDocumentIntelligence])

  // Auto-fetch intelligence when section is expanded
  useEffect(() => {
    if (intelligenceSectionCollapsed) return
    if (!selectedMediaId || isNote) return
    if (loadedDocumentIntelligenceMediaId === selectedMediaId) return
    void fetchDocumentIntelligenceRef.current()
  }, [
    intelligenceSectionCollapsed,
    isNote,
    loadedDocumentIntelligenceMediaId,
    selectedMediaId
  ])

  // Annotation handlers
  const captureAnnotationSelection = useCallback((selectionText: string, location?: string) => {
    const selectedText = selectionText.trim()
    if (!selectedMediaId || isNote || !selectedText) return

    setAnnotationSelectionText(selectedText.slice(0, 4000))
    setAnnotationSelectionLocation(location || `selection:${Date.now()}`)
    setActiveIntelligenceTab('annotations')
    if (collapsedSections.intelligence ?? true) {
      void setCollapsedSections((prev) => ({
        ...prev,
        intelligence: false
      }))
    }
  }, [collapsedSections.intelligence, isNote, selectedMediaId, setCollapsedSections])

  const handleCaptureAnnotationSelection = useCallback(() => {
    if (!selectedMediaId || isNote) return
    if (typeof window === 'undefined' || typeof window.getSelection !== 'function') {
      return
    }

    const contentNode = contentBodyRef.current
    const selection = window.getSelection()
    if (!contentNode || !selection || selection.rangeCount === 0 || selection.isCollapsed) {
      return
    }

    const range = selection.getRangeAt(0)
    const anchorNode = range.commonAncestorContainer
    if (!contentNode.contains(anchorNode)) {
      return
    }

    const selectedText = selection.toString().trim()
    if (!selectedText) return

    captureAnnotationSelection(selectedText)
  }, [captureAnnotationSelection, contentBodyRef, isNote, selectedMediaId])

  const handleCreateAnnotation = useCallback(async () => {
    if (!selectedMediaId || isNote) return

    const highlightText =
      annotationSelectionText.trim() || annotationManualText.trim()
    const noteText = annotationDraftNote.trim()
    if (!highlightText && !noteText) {
      message.warning(
        t('review:mediaPage.annotationCreateEmpty', {
          defaultValue: 'Select text or enter annotation text first.'
        })
      )
      return
    }

    const annotationType = highlightText ? 'highlight' : 'page_note'
    const annotationLocation =
      annotationSelectionLocation || `manual:${Date.now()}`

    setAnnotationCreating(true)
    try {
      const created = await bgRequest<MediaAnnotationEntry>({
        path: `/api/v1/media/${encodeURIComponent(selectedMediaId)}/annotations` as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          location: annotationLocation,
          text: highlightText || noteText,
          color: annotationDraftColor,
          note: noteText || undefined,
          annotation_type: annotationType
        }
      })
      setAnnotationPanelEntries((prev) => [...prev, created])
      setLoadedDocumentIntelligenceMediaId(selectedMediaId)
      clearAnnotationDraft()
      message.success(
        t('review:mediaPage.annotationSaved', {
          defaultValue: 'Annotation saved.'
        })
      )
    } catch (error) {
      console.error('Failed to create annotation:', error)
      message.error(
        t('review:mediaPage.annotationSaveFailed', {
          defaultValue: 'Unable to save annotation.'
        })
      )
    } finally {
      setAnnotationCreating(false)
    }
  }, [
    annotationDraftColor,
    annotationDraftNote,
    annotationManualText,
    annotationSelectionLocation,
    annotationSelectionText,
    clearAnnotationDraft,
    isNote,
    selectedMediaId,
    setAnnotationPanelEntries,
    t
  ])

  const handleUpdateAnnotationNote = useCallback(
    async (annotation: MediaAnnotationEntry) => {
      if (!selectedMediaId || isNote) return
      const nextNote = window.prompt(
        t('review:mediaPage.annotationEditPrompt', {
          defaultValue: 'Update annotation note'
        }),
        annotation.note || ''
      )
      if (nextNote == null) return

      setAnnotationUpdatingId(annotation.id)
      try {
        const updated = await bgRequest<MediaAnnotationEntry>({
          path: `/api/v1/media/${encodeURIComponent(selectedMediaId)}/annotations/${encodeURIComponent(annotation.id)}` as any,
          method: 'PUT' as any,
          headers: { 'Content-Type': 'application/json' },
          body: { note: nextNote }
        })
        setAnnotationPanelEntries((prev) =>
          prev.map((entry) => (entry.id === annotation.id ? updated : entry))
        )
      } catch (error) {
        console.error('Failed to update annotation:', error)
        message.error(
          t('review:mediaPage.annotationUpdateFailed', {
            defaultValue: 'Unable to update annotation.'
          })
        )
      } finally {
        setAnnotationUpdatingId(null)
      }
    },
    [isNote, selectedMediaId, setAnnotationPanelEntries, t]
  )

  const handleDeleteAnnotation = useCallback(
    async (annotationId: string) => {
      if (!selectedMediaId || isNote) return

      setAnnotationDeletingId(annotationId)
      try {
        await bgRequest({
          path: `/api/v1/media/${encodeURIComponent(selectedMediaId)}/annotations/${encodeURIComponent(annotationId)}` as any,
          method: 'DELETE' as any
        })
        setAnnotationPanelEntries((prev) =>
          prev.filter((entry) => entry.id !== annotationId)
        )
      } catch (error) {
        console.error('Failed to delete annotation:', error)
        message.error(
          t('review:mediaPage.annotationDeleteFailed', {
            defaultValue: 'Unable to delete annotation.'
          })
        )
      } finally {
        setAnnotationDeletingId(null)
      }
    },
    [isNote, selectedMediaId, setAnnotationPanelEntries, t]
  )

  const handleSyncAnnotations = useCallback(async () => {
    if (!selectedMediaId || isNote) return

    setAnnotationSyncing(true)
    try {
      await bgRequest({
        path: `/api/v1/media/${encodeURIComponent(selectedMediaId)}/annotations/sync` as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          annotations: annotationPanelEntries.map((entry) => ({
            location: entry.location,
            text: entry.text,
            color: entry.color,
            note: entry.note,
            annotation_type: entry.annotation_type
          })),
          client_ids: annotationPanelEntries.map((entry) => entry.id)
        }
      })
      message.success(
        t('review:mediaPage.annotationSyncSuccess', {
          defaultValue: 'Annotations synced.'
        })
      )
    } catch (error) {
      console.error('Failed to sync annotations:', error)
      message.error(
        t('review:mediaPage.annotationSyncFailed', {
          defaultValue: 'Unable to sync annotations.'
        })
      )
    } finally {
      setAnnotationSyncing(false)
    }
  }, [annotationPanelEntries, isNote, selectedMediaId, t])

  // Export
  const handleExportMedia = useCallback(() => {
    if (isNote || !selectedMedia) return
    setExportModalOpen(true)
  }, [isNote, selectedMedia])

  const confirmExportMedia = useCallback(() => {
    if (isNote || !selectedMedia) return

    const citationSafeMetadata =
      mediaDetail?.safe_metadata &&
      typeof mediaDetail.safe_metadata === 'object' &&
      !Array.isArray(mediaDetail.safe_metadata)
        ? (mediaDetail.safe_metadata as Record<string, unknown>)
        : selectedMedia?.raw?.safe_metadata &&
            typeof selectedMedia.raw.safe_metadata === 'object' &&
            !Array.isArray(selectedMedia.raw.safe_metadata)
          ? (selectedMedia.raw.safe_metadata as Record<string, unknown>)
          : {}

    const exportPayload = {
      id: selectedMedia.id,
      title: selectedMedia.title || '',
      type: selectedMedia.meta?.type || '',
      source: selectedMedia.meta?.source || '',
      keywords: editingKeywords,
      content,
      analysis: selectedAnalysis?.text || '',
      exported_at: new Date().toISOString()
    }

    let output = ''
    let extension = 'txt'
    let mimeType = 'text/plain;charset=utf-8'

    if (exportFormat === 'json') {
      output = JSON.stringify(exportPayload, null, 2)
      extension = 'json'
      mimeType = 'application/json;charset=utf-8'
    } else if (exportFormat === 'bibtex') {
      output = buildBibtexExport(selectedMedia, exportPayload, citationSafeMetadata)
      extension = 'bib'
      mimeType = 'application/x-bibtex;charset=utf-8'
    } else if (exportFormat === 'markdown') {
      output = [
        `# ${exportPayload.title || `Media ${exportPayload.id}`}`,
        '',
        `- ID: ${exportPayload.id}`,
        `- Type: ${exportPayload.type || 'N/A'}`,
        `- Source: ${exportPayload.source || 'N/A'}`,
        `- Keywords: ${
          Array.isArray(exportPayload.keywords) && exportPayload.keywords.length > 0
            ? exportPayload.keywords.join(', ')
            : 'None'
        }`,
        `- Exported At: ${exportPayload.exported_at}`,
        '',
        '## Content',
        '',
        exportPayload.content || '',
        '',
        '## Analysis',
        '',
        exportPayload.analysis || ''
      ].join('\n')
      extension = 'md'
      mimeType = 'text/markdown;charset=utf-8'
    } else {
      output = [
        `Title: ${exportPayload.title || `Media ${exportPayload.id}`}`,
        `ID: ${exportPayload.id}`,
        `Type: ${exportPayload.type || 'N/A'}`,
        `Source: ${exportPayload.source || 'N/A'}`,
        `Keywords: ${
          Array.isArray(exportPayload.keywords) && exportPayload.keywords.length > 0
            ? exportPayload.keywords.join(', ')
            : 'None'
        }`,
        `Exported At: ${exportPayload.exported_at}`,
        '',
        'Content:',
        exportPayload.content || '',
        '',
        'Analysis:',
        exportPayload.analysis || ''
      ].join('\n')
      extension = 'txt'
      mimeType = 'text/plain;charset=utf-8'
    }

    try {
      const blob = new Blob([output], { type: mimeType })
      downloadBlob(blob, `${toExportFileStem(selectedMedia)}.${extension}`)
      message.success(
        t('review:mediaPage.exportSuccess', {
          defaultValue: 'Export ready.'
        })
      )
      setExportModalOpen(false)
    } catch (error) {
      console.error('Failed to export media:', error)
      message.error(
        t('review:mediaPage.exportFailed', {
          defaultValue: 'Unable to export this item.'
        })
      )
    }
  }, [content, editingKeywords, exportFormat, isNote, mediaDetail, selectedAnalysis, selectedMedia, t])

  // Reprocess
  const handleReprocessMedia = useCallback(async () => {
    if (isNote || !selectedMediaId) return

    try {
      await bgRequest({
        path: `/api/v1/media/${selectedMediaId}/reprocess` as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          perform_chunking: true,
          generate_embeddings: true,
          force_regenerate_embeddings: true
        }
      })
      message.success(
        t('review:mediaPage.reprocessQueued', {
          defaultValue: 'Reprocessing started.'
        })
      )
      if (onRefreshMedia) {
        onRefreshMedia()
      }
    } catch (error) {
      console.error('Failed to start reprocess:', error)
      message.error(
        t('review:mediaPage.reprocessFailed', {
          defaultValue:
            'Unable to start reprocessing. Please try again.'
        })
      )
    }
  }, [isNote, onRefreshMedia, selectedMediaId, t])

  // Schedule source refresh
  const handleScheduleSourceRefresh = useCallback(async () => {
    if (!selectedMedia || selectedMedia.kind === 'note' || !sourceUrlForScheduling) return
    const sourceName =
      selectedMedia.title?.trim() ||
      t('review:mediaPage.untitled', { defaultValue: 'Untitled' })
    const scheduleExpr = REINGEST_CRON_BY_PRESET[scheduleRefreshPreset]
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'

    setScheduleRefreshSubmitting(true)
    try {
      const createdSource = await bgRequest<{ id?: number | string }>({
        path: '/api/v1/watchlists/sources' as any,
        method: 'POST' as any,
        body: {
          name: sourceName,
          url: sourceUrlForScheduling,
          source_type: 'site',
          active: true,
          tags: ['media-refresh']
        }
      })
      const sourceId = Number(createdSource?.id)
      if (!Number.isFinite(sourceId) || sourceId <= 0) {
        throw new Error('Invalid watchlist source id')
      }

      await bgRequest({
        path: '/api/v1/watchlists/jobs' as any,
        method: 'POST' as any,
        body: {
          name: `Refresh: ${sourceName}`,
          description: `Scheduled source refresh for media ${selectedMedia.id}`,
          scope: { sources: [sourceId] },
          schedule_expr: scheduleExpr,
          timezone,
          active: true,
          output_prefs: {
            ingest: {
              persist_to_media_db: true
            }
          }
        }
      })
      message.success(
        t('review:mediaPage.scheduleRefreshSuccess', {
          defaultValue: 'Scheduled source refresh monitor.'
        })
      )
      setScheduleRefreshModalOpen(false)
    } catch (error) {
      console.error('Failed to schedule source refresh:', error)
      message.error(
        t('review:mediaPage.scheduleRefreshFailed', {
          defaultValue: 'Unable to schedule source refresh.'
        })
      )
    } finally {
      setScheduleRefreshSubmitting(false)
    }
  }, [scheduleRefreshPreset, selectedMedia, sourceUrlForScheduling, t])

  // Embedded media
  const mediaType = String(
    selectedMedia?.meta?.type || mediaDetail?.type || mediaDetail?.media_type || ''
  )
    .toLowerCase()
    .trim()
  const isPlayableMediaType = mediaType === 'audio' || mediaType === 'video'
  const hasOriginalFile = Boolean(
    mediaDetail?.has_original_file ??
      mediaDetail?.hasOriginalFile ??
      selectedMedia?.raw?.has_original_file ??
      selectedMedia?.raw?.hasOriginalFile
  )
  const shouldShowEmbeddedPlayer =
    selectedMedia?.kind === 'media' && isPlayableMediaType && hasOriginalFile

  const resolveMediaMimeType = useCallback((mType: string, mDetail: any): string => {
    const candidates = [
      mDetail?.file_mime_type,
      mDetail?.mime_type,
      mDetail?.metadata?.mime_type,
      mDetail?.metadata?.content_type
    ]
    for (const candidate of candidates) {
      if (typeof candidate === 'string' && candidate.trim().length > 0) {
        return candidate.trim()
      }
    }
    if (mType === 'video') return 'video/mp4'
    if (mType === 'audio') return 'audio/mpeg'
    return 'application/octet-stream'
  }, [])

  const embeddedMediaMimeType = useMemo(
    () => resolveMediaMimeType(mediaType, mediaDetail),
    [mediaDetail, mediaType, resolveMediaMimeType]
  )

  const mediaPlayerRef = useRef<HTMLMediaElement | null>(null)

  // Load embedded media file
  useEffect(() => {
    const revokeObjectUrl = () => {
      if (
        embeddedMediaObjectUrlRef.current &&
        typeof URL !== 'undefined' &&
        typeof URL.revokeObjectURL === 'function'
      ) {
        URL.revokeObjectURL(embeddedMediaObjectUrlRef.current)
        embeddedMediaObjectUrlRef.current = null
      }
    }

    mediaPlayerRef.current = null
    setEmbeddedMediaError(null)
    setEmbeddedMediaLoading(false)
    setEmbeddedMediaUrl(null)
    revokeObjectUrl()

    if (!shouldShowEmbeddedPlayer || !selectedMediaId) {
      return () => {
        revokeObjectUrl()
      }
    }

    let cancelled = false
    setEmbeddedMediaLoading(true)

    ;(async () => {
      try {
        const fileBuffer = await bgRequest<ArrayBuffer>({
          path: `/api/v1/media/${selectedMediaId}/file` as any,
          method: 'GET' as any,
          responseType: 'arrayBuffer'
        })
        if (cancelled) return

        const asArrayBuffer =
          fileBuffer instanceof ArrayBuffer
            ? fileBuffer
            : fileBuffer && (fileBuffer as any).buffer instanceof ArrayBuffer
              ? (fileBuffer as any).buffer
              : null
        if (!asArrayBuffer) {
          throw new Error('No media file bytes returned')
        }
        if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
          throw new Error('Object URLs are unavailable in this environment')
        }

        const blob = new Blob([asArrayBuffer], { type: embeddedMediaMimeType })
        const objectUrl = URL.createObjectURL(blob)
        embeddedMediaObjectUrlRef.current = objectUrl
        setEmbeddedMediaUrl(objectUrl)
      } catch (error) {
        if (cancelled) return
        console.debug('Failed to load embedded media file', error)
        setEmbeddedMediaError('Unable to load media preview.')
      } finally {
        if (!cancelled) {
          setEmbeddedMediaLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
      revokeObjectUrl()
    }
  }, [embeddedMediaMimeType, selectedMediaId, shouldShowEmbeddedPlayer])

  // Show diff handler
  const handleShowDiff = useCallback(
    (left: string, right: string, leftLabel: string, rightLabel: string, metadataDiff?: { left: string[]; right: string[]; changed: string[] }) => {
      setDiffLeftText(left)
      setDiffRightText(right)
      setDiffLeftLabel(leftLabel)
      setDiffRightLabel(rightLabel)
      setDiffMetadataSummary(metadataDiff || null)
      setDiffModalOpen(true)
    },
    []
  )

  const closeDiffModal = useCallback(() => {
    setDiffModalOpen(false)
    setDiffMetadataSummary(null)
  }, [])

  return {
    // Analysis modal
    analysisModalOpen,
    setAnalysisModalOpen,
    // Diff modal
    diffModalOpen,
    diffLeftText,
    diffRightText,
    diffLeftLabel,
    diffRightLabel,
    diffMetadataSummary,
    handleShowDiff,
    closeDiffModal,
    // Export modal
    exportModalOpen,
    setExportModalOpen,
    exportFormat,
    setExportFormat,
    handleExportMedia,
    confirmExportMedia,
    // Schedule refresh modal
    scheduleRefreshModalOpen,
    setScheduleRefreshModalOpen,
    scheduleRefreshPreset,
    setScheduleRefreshPreset,
    scheduleRefreshSubmitting,
    handleScheduleSourceRefresh,
    sourceUrlForScheduling,
    canScheduleSourceRefresh,
    // Metadata
    metadataDetailsExpanded,
    setMetadataDetailsExpanded,
    contentExpanded,
    setContentExpanded,
    // Document intelligence
    activeIntelligenceTab,
    setActiveIntelligenceTab,
    documentIntelligencePanels,
    intelligenceSectionCollapsed,
    activeDocumentIntelligencePanel,
    fetchDocumentIntelligence,
    // Annotations
    annotationSelectionText,
    annotationManualText,
    setAnnotationManualText,
    annotationDraftNote,
    setAnnotationDraftNote,
    annotationDraftColor,
    setAnnotationDraftColor,
    annotationCreating,
    annotationUpdatingId,
    annotationDeletingId,
    annotationSyncing,
    annotationPanelEntries,
    captureAnnotationSelection,
    handleCaptureAnnotationSelection,
    handleCreateAnnotation,
    handleUpdateAnnotationNote,
    handleDeleteAnnotation,
    handleSyncAnnotations,
    clearAnnotationDraft,
    // Reprocess
    handleReprocessMedia,
    // Embedded media
    mediaType,
    isPlayableMediaType,
    shouldShowEmbeddedPlayer,
    embeddedMediaUrl,
    embeddedMediaLoading,
    embeddedMediaError,
    embeddedMediaMimeType,
    mediaPlayerRef,
    // Developer tools
    REINGEST_CRON_BY_PRESET: REINGEST_CRON_BY_PRESET as Record<string, string>
  }
}

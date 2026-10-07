import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import type { NoteListItem } from '@/components/Notes/notes-manager-types'
import type {
  ExportFormat,
  ExportProgressState,
} from '../notes-manager-types'
import {
  extractKeywords,
} from '../notes-manager-utils'
import {
  buildSingleNoteCopyText,
  buildSingleNoteJson,
  buildSingleNoteMarkdown,
  buildSingleNotePrintableHtml,
  buildStudioPrintableHtml,
  getDefaultStudioPaperSizeFromLocale,
  type SingleNoteCopyMode,
  type SingleNoteExportFormat,
} from '../export-utils'
import { translateMessage } from '@/i18n/translateMessage'
import { formatFileSize } from '@/utils/format'
import type { ConfirmDangerOptions } from '@/components/Common/confirm-danger'
import type { NoteStudioState, NotesStudioPaperSize } from '../notes-studio-types'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>

/**
 * Notes per export request. GET /api/v1/notes/ accepts limit up to 1000 and
 * GET /api/v1/notes/search/ up to 100, so 100 is a full page for both.
 */
const EXPORT_PAGE_SIZE = 100
/** Hard cap on export requests, whatever the server reports. */
const MAX_EXPORT_REQUESTS = 1000
const EXPORT_PREFLIGHT_NOTE_THRESHOLD = MAX_EXPORT_REQUESTS * EXPORT_PAGE_SIZE

type ExportPage = { items: any[]; total: number | null }

type GatherResult = {
  arr: NoteListItem[]
  limitReached: boolean
  failedBatches: number
  cancelled: boolean
  /** Matching notes the server last reported, or null when it sent no total. */
  expectedTotal: number | null
}

/** list_notes reports the library size as pagination.total (and a top-level total alias). */
const readReportedTotal = (res: any): number | null => {
  const raw = res?.pagination?.total ?? res?.total ?? res?.pagination?.total_items
  if (raw == null) return null
  const value = Number(raw)
  return Number.isFinite(value) && value >= 0 ? value : null
}

const toExportNote = (n: any): NoteListItem => ({
  id: n?.id,
  title: n?.title,
  content: n?.content,
  updated_at: n?.updated_at,
  keywords: extractKeywords(n)
})

export interface UseNotesExportDeps {
  message: MessageInstance
  confirmDanger: ConfirmDanger
  t: (key: string, opts?: Record<string, any>) => string
  /** From list hook */
  listMode: 'active' | 'trash'
  query: string
  effectiveKeywordTokens: string[]
  total: number
  filteredCount: number
  hasActiveFilters: boolean
  selectedBulkNotes: NoteListItem[]
  fetchFilteredNotesRaw: (
    q: string,
    toks: string[],
    page: number,
    pageSize: number,
    signal?: AbortSignal
  ) => Promise<{ items: any[]; total: number }>
  /** From editor hook */
  selectedId: string | number | null
  title: string
  content: string
  editorKeywords: string[]
  selectedStudioState?: NoteStudioState | null
  studioPaperSize?: NotesStudioPaperSize
}

export function useNotesExport(deps: UseNotesExportDeps) {
  const {
    message,
    confirmDanger,
    t,
    listMode,
    query,
    effectiveKeywordTokens,
    total,
    filteredCount,
    hasActiveFilters,
    selectedBulkNotes,
    fetchFilteredNotesRaw,
    selectedId,
    title,
    content,
    editorKeywords,
    selectedStudioState = null,
    studioPaperSize,
  } = deps

  const [exportProgress, setExportProgress] = React.useState<ExportProgressState | null>(null)
  const exportAbortRef = React.useRef<AbortController | null>(null)

  // Stop an in-flight export when the page unmounts.
  React.useEffect(() => () => exportAbortRef.current?.abort(), [])

  const cancelExport = React.useCallback(() => {
    exportAbortRef.current?.abort()
  }, [])

  /**
   * Collects every matching note with limit/offset paging, the only paging the
   * notes list and search endpoints read. The loop ends on the reported total,
   * on a short or empty page (so a missing total cannot loop), when a page adds
   * no new note ids (a server that ignores the offset), on Cancel, or at
   * MAX_EXPORT_REQUESTS.
   */
  const gatherAllMatching = React.useCallback(async (
    format: ExportFormat
  ): Promise<GatherResult> => {
    exportAbortRef.current?.abort()
    const controller = new AbortController()
    exportAbortRef.current = controller
    const { signal } = controller

    const arr: NoteListItem[] = []
    const seenIds = new Set<string>()
    let limitReached = false
    let failedBatches = 0
    let fetchedPages = 0
    let expectedTotal: number | null = null
    const q = query.trim()
    const toks = effectiveKeywordTokens.map((k) => k.toLowerCase())
    const isFiltered = Boolean(q) || toks.length > 0
    const updateProgress = () => {
      setExportProgress({
        format,
        fetchedNotes: arr.length,
        fetchedPages,
        failedBatches,
        totalNotes: expectedTotal
      })
    }

    const fetchPage = async (offset: number): Promise<ExportPage> => {
      if (isFiltered) {
        const page = offset / EXPORT_PAGE_SIZE + 1
        const result = await fetchFilteredNotesRaw(q, toks, page, EXPORT_PAGE_SIZE, signal)
        return { items: result.items, total: readReportedTotal(result) }
      }
      const { bgRequest } = await import('@/services/background-proxy')
      // Oldest first, so notes edited during the export keep their position.
      const params = new URLSearchParams({
        limit: String(EXPORT_PAGE_SIZE),
        offset: String(offset),
        include_keywords: 'true',
        sort_by: 'created_at',
        sort_order: 'asc'
      })
      const res = await bgRequest<any>({
        path: `/api/v1/notes/?${params.toString()}` as `/${string}`,
        method: 'GET' as any,
        abortSignal: signal
      })
      const items = Array.isArray(res?.items) ? res.items : (Array.isArray(res) ? res : [])
      return { items, total: readReportedTotal(res) }
    }

    updateProgress()
    try {
      let offset = 0
      let requests = 0
      let done = false
      while (!done) {
        if (requests >= MAX_EXPORT_REQUESTS) {
          limitReached = true
          break
        }
        requests += 1
        let page: ExportPage
        try {
          page = await fetchPage(offset)
        } catch {
          if (!signal.aborted) {
            failedBatches += 1
            updateProgress()
          }
          break
        }
        if (signal.aborted) break
        if (page.total != null) expectedTotal = page.total
        let added = 0
        for (const note of page.items) {
          if (note?.id != null) {
            const key = String(note.id)
            if (seenIds.has(key)) continue
            seenIds.add(key)
          }
          arr.push(toExportNote(note))
          added += 1
        }
        if (page.items.length > 0) fetchedPages += 1
        updateProgress()
        offset += EXPORT_PAGE_SIZE
        done =
          page.items.length < EXPORT_PAGE_SIZE ||
          added === 0 ||
          (expectedTotal != null && offset >= expectedTotal)
      }
    } finally {
      if (exportAbortRef.current === controller) exportAbortRef.current = null
    }
    return { arr, limitReached, failedBatches, cancelled: signal.aborted, expectedTotal }
  }, [effectiveKeywordTokens, fetchFilteredNotesRaw, query])

  const maybeConfirmExportPreflight = React.useCallback(
    async (format: ExportFormat): Promise<boolean> => {
      if (listMode !== 'active') return true
      const estimatedScope = Math.max(total, filteredCount)
      if (estimatedScope < EXPORT_PREFLIGHT_NOTE_THRESHOLD) return true
      const scopeText = hasActiveFilters
        ? 'current search/filter scope'
        : 'all active notes'
      return confirmDanger({
        title: `Large ${format.toUpperCase()} export`,
        content:
          `This export is estimated to include about ${estimatedScope.toLocaleString()} notes from ${scopeText}. ` +
          'It may take a while and can return partial results if some batches fail. Continue?',
        okText: 'Start export',
        cancelText: 'Cancel'
      })
    },
    [confirmDanger, filteredCount, hasActiveFilters, listMode, total]
  )

  const maybeWarnExportLimits = React.useCallback(
    ({ arr, limitReached, failedBatches, expectedTotal }: GatherResult) => {
      const arrLength = arr.length
      if (limitReached) {
        message.warning(`Export limited to ${arrLength} notes. Some notes may be excluded.`)
      }
      if (failedBatches > 0) {
        message.warning(
          `Export completed with partial data. ${failedBatches} batch${
            failedBatches === 1 ? '' : 'es'
          } failed.`
        )
      }
      if (!limitReached && failedBatches === 0 && expectedTotal != null && arrLength < expectedTotal) {
        message.warning(
          `Exported ${arrLength} of ${expectedTotal} matching notes. Some notes may be missing.`
        )
      }
    },
    [message]
  )

  const exportAll = React.useCallback(async () => {
    try {
      const allowed = await maybeConfirmExportPreflight('md')
      if (!allowed) return
      const gathered = await gatherAllMatching('md')
      if (gathered.cancelled) { message.info('Export cancelled'); return }
      const { arr } = gathered
      if (!arr.length) { message.info('No notes to export'); return }
      maybeWarnExportLimits(gathered)
      const md = arr
        .map((n, idx) => `### ${n.title || `Note ${n.id ?? idx + 1}`}\n\n${String(n.content || '')}`)
        .join('\n\n---\n\n')
      const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `notes-export.md`
      a.click()
      URL.revokeObjectURL(url)
      const sizeDisplay = formatFileSize(blob.size)
      message.success(
        translateMessage(
          t,
          'option:notesSearch.exportSuccess',
          'Exported {{count}} notes ({{size}})',
          { count: arr.length, size: sizeDisplay }
        )
      )
    } catch (e: any) {
      message.error(e?.message || 'Export failed')
    } finally {
      setExportProgress(null)
    }
  }, [gatherAllMatching, maybeConfirmExportPreflight, maybeWarnExportLimits, message, t])

  const exportAllCSV = React.useCallback(async () => {
    try {
      const allowed = await maybeConfirmExportPreflight('csv')
      if (!allowed) return
      const gathered = await gatherAllMatching('csv')
      if (gathered.cancelled) { message.info('Export cancelled'); return }
      const { arr } = gathered
      if (!arr.length) { message.info('No notes to export'); return }
      maybeWarnExportLimits(gathered)
      const escape = (s: any) => '"' + String(s ?? '').replace(/"/g, '""') + '"'
      const header = ['id','title','content','updated_at','keywords']
      const rows = [
        header.join(','),
        ...arr.map((n) =>
          [
            n.id,
            n.title || '',
            (n.content || '').replace(/\r?\n/g, '\\n'),
            n.updated_at || '',
            (n.keywords || []).join('; ')
          ]
            .map(escape)
            .join(',')
        )
      ]
      const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `notes-export.csv`
      a.click()
      URL.revokeObjectURL(url)
      const sizeDisplay = formatFileSize(blob.size)
      message.success(
        translateMessage(
          t,
          'option:notesSearch.exportCsvSuccess',
          'Exported {{count}} notes as CSV ({{size}})',
          { count: arr.length, size: sizeDisplay }
        )
      )
    } catch (e: any) {
      message.error(e?.message || 'Export failed')
    } finally {
      setExportProgress(null)
    }
  }, [gatherAllMatching, maybeConfirmExportPreflight, maybeWarnExportLimits, message, t])

  const exportAllJSON = React.useCallback(async () => {
    try {
      const allowed = await maybeConfirmExportPreflight('json')
      if (!allowed) return
      const gathered = await gatherAllMatching('json')
      if (gathered.cancelled) { message.info('Export cancelled'); return }
      const { arr } = gathered
      if (!arr.length) { message.info('No notes to export'); return }
      maybeWarnExportLimits(gathered)
      const blob = new Blob([JSON.stringify(arr, null, 2)], { type: 'application/json;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `notes-export.json`
      a.click()
      URL.revokeObjectURL(url)
      const sizeDisplay = formatFileSize(blob.size)
      message.success(
        translateMessage(
          t,
          'option:notesSearch.exportJsonSuccess',
          'Exported {{count}} notes as JSON ({{size}})',
          { count: arr.length, size: sizeDisplay }
        )
      )
    } catch (e: any) {
      message.error(e?.message || 'Export failed')
    } finally {
      setExportProgress(null)
    }
  }, [gatherAllMatching, maybeConfirmExportPreflight, maybeWarnExportLimits, message, t])

  const exportSelectedBulk = React.useCallback(() => {
    if (selectedBulkNotes.length === 0) {
      message.info('No selected notes to export')
      return
    }
    const md = selectedBulkNotes
      .map((note, index) => `### ${note.title || `Note ${note.id ?? index + 1}`}\n\n${String(note.content || '')}`)
      .join('\n\n---\n\n')
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `notes-selected-export.md`
    anchor.click()
    URL.revokeObjectURL(url)
    message.success(
      translateMessage(
        t,
        'option:notesSearch.bulkExportSuccess',
        'Exported {{count}} selected notes',
        { count: selectedBulkNotes.length }
      )
    )
  }, [message, selectedBulkNotes, t])

  const printSelected = React.useCallback(() => {
    if (typeof window === 'undefined') {
      message.error(
        t('option:notesSearch.notesPrintUnavailable', {
          defaultValue: 'Print is not available in this environment'
        })
      )
      return
    }
    const printWindow = window.open('', '_blank', 'noopener,noreferrer,width=1024,height=768')
    if (!printWindow) {
      message.error(
        t('option:notesSearch.notesPrintPopupError', {
          defaultValue: 'Unable to open print view. Please allow pop-ups and try again.'
        })
      )
      return
    }

    let printableHtml = ''
    if (selectedStudioState?.studio_document && !selectedStudioState.is_stale) {
      try {
        printableHtml = buildStudioPrintableHtml(
          {
            id: selectedId,
            title,
            content,
            keywords: editorKeywords
          },
          selectedStudioState.studio_document,
          {
            paperSize:
              studioPaperSize ??
              getDefaultStudioPaperSizeFromLocale(
                typeof navigator !== 'undefined' ? navigator.language : ''
              ),
            labels: {
              untitledNote: t('option:notesSearch.notesStudioUntitledPrintNote', {
                defaultValue: 'Untitled note'
              }),
              printTitleSuffix: t('option:notesSearch.notesStudioPrintTitleSuffix', {
                defaultValue: 'Notes Studio Print'
              }),
              exportedLabel: t('option:notesSearch.notesStudioExportedLabel', {
                defaultValue: 'Exported'
              }),
              templateLabel: t('option:notesSearch.notesStudioTemplateMetaLabel', {
                defaultValue: 'Template'
              }),
              paperLabel: t('option:notesSearch.notesStudioPaperMetaLabel', {
                defaultValue: 'Paper'
              }),
              diagramHeading: t('option:notesSearch.notesStudioDiagramHeading', {
                defaultValue: 'Diagram'
              })
            }
          }
        )
      } catch {
        printableHtml = buildSingleNotePrintableHtml({
          id: selectedId,
          title,
          content,
          keywords: editorKeywords
        })
      }
    } else {
      printableHtml = buildSingleNotePrintableHtml({
        id: selectedId,
        title,
        content,
        keywords: editorKeywords
      })
    }

    printWindow.document.open()
    printWindow.document.write(printableHtml)
    printWindow.document.close()
    printWindow.focus()
    printWindow.print()

    message.success(
      t('option:notesSearch.notesPrintOpened', {
        defaultValue: 'Opened print view. Use Save as PDF in your browser to export PDF.'
      })
    )
  }, [content, editorKeywords, message, selectedId, selectedStudioState, studioPaperSize, title])

  const exportSelected = React.useCallback((format: SingleNoteExportFormat = 'md') => {
    if (format === 'print') {
      printSelected()
      return
    }
    const name = (title || `note-${selectedId ?? 'new'}`).replace(/[^a-z0-9-_]+/gi, '-')
    const fileContent =
      format === 'json'
        ? buildSingleNoteJson({
            id: selectedId,
            title,
            content,
            keywords: editorKeywords
          })
        : buildSingleNoteMarkdown({
            id: selectedId,
            title,
            content,
            keywords: editorKeywords
          })
    const blob = new Blob([fileContent], {
      type:
        format === 'json'
          ? 'application/json;charset=utf-8'
          : 'text/markdown;charset=utf-8'
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.${format === 'json' ? 'json' : 'md'}`
    a.click()
    URL.revokeObjectURL(url)
    const sizeDisplay = formatFileSize(blob.size)
    message.success(
      translateMessage(
        t,
        'option:notesSearch.exportNoteSuccess',
        'Exported ({{size}})',
        { size: sizeDisplay }
      )
    )
  }, [content, editorKeywords, message, printSelected, selectedId, t, title])

  const copySelected = React.useCallback(async (mode: SingleNoteCopyMode = 'content') => {
    const payload = buildSingleNoteCopyText(
      {
        id: selectedId,
        title,
        content,
        keywords: editorKeywords
      },
      mode
    )
    try {
      await navigator.clipboard.writeText(payload)
      message.success(mode === 'markdown' ? 'Copied as Markdown' : 'Copied')
    } catch { message.error('Copy failed') }
  }, [content, editorKeywords, message, selectedId, title])

  return {
    exportProgress,
    cancelExport,
    exportAll,
    exportAllCSV,
    exportAllJSON,
    exportSelectedBulk,
    exportSelected,
    copySelected,
    printSelected,
  }
}

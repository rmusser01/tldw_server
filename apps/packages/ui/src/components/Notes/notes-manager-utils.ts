import type { NotesTitleSuggestStrategy, NotesNotebookSetting } from '@/services/settings/ui-settings'
import type { NotesStudioHandwritingMode, NotesStudioTemplateType } from './notes-studio-types'
import type { NoteListItem } from './types'

export type NoteWithKeywords = {
  metadata?: { keywords?: any[] }
  keywords?: any[]
}

export const extractBacklink = (note: any) => {
  const meta = note?.metadata || {}
  const backlinks = meta?.backlinks || meta || {}
  const conversation =
    note?.conversation_id ??
    backlinks?.conversation_id ??
    backlinks?.conversationId ??
    meta?.conversation_id ??
    null
  const message =
    note?.message_id ??
    backlinks?.message_id ??
    backlinks?.messageId ??
    meta?.message_id ??
    null
  return {
    conversation_id: conversation != null ? String(conversation) : null,
    message_id: message != null ? String(message) : null
  }
}

export const extractKeywords = (note: NoteWithKeywords | any): string[] => {
  const rawKeywords = (Array.isArray(note?.metadata?.keywords)
    ? note.metadata.keywords
    : Array.isArray(note?.keywords)
      ? note.keywords
      : []) as any[]
  return (rawKeywords || [])
    .map((item: any) => {
      const raw =
        item?.keyword ??
        item?.keyword_text ??
        item?.text ??
        item
      return typeof raw === 'string' ? raw : null
    })
    .filter((s): s is string => !!s && s.trim().length > 0)
}

// Extract version from note object. Checks multiple candidate fields in order:
// 1. note.version (primary)
// 2. note.expected_version (fallback)
// 3. note.expectedVersion (camelCase variant)
// 4. note.metadata.* (nested variants)
export const toNoteVersion = (note: any): number | null => {
  const candidates = [
    note?.version,
    note?.expected_version,
    note?.expectedVersion,
    note?.metadata?.version,
    note?.metadata?.expected_version,
    note?.metadata?.expectedVersion
  ]
  const validVersions: number[] = []
  for (const candidate of candidates) {
    if (
      typeof candidate === 'number' &&
      Number.isFinite(candidate) &&
      Number.isInteger(candidate) &&
      candidate >= 0
    ) {
      validVersions.push(candidate)
    }
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      const parsed = Number(candidate)
      if (Number.isFinite(parsed) && Number.isInteger(parsed) && parsed >= 0) {
        validVersions.push(parsed)
      }
    }
  }
  if (validVersions.length > 1) {
    const allSame = validVersions.every((version) => version === validVersions[0])
    if (!allSame) {
      console.warn('[toNoteVersion] Multiple conflicting versions found:', validVersions)
    }
  }
  return validVersions[0] ?? null
}

export const toNoteLastModified = (note: any): string | null => {
  const candidates = [
    note?.last_modified,
    note?.updated_at,
    note?.updatedAt,
    note?.lastModified,
    note?.metadata?.last_modified,
    note?.metadata?.updated_at
  ]
  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || candidate.trim().length === 0) continue
    const parsed = new Date(candidate)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString()
    }
  }
  return null
}

export type KeywordSyncWarning = {
  failedCount: number
  failedKeywords: string[]
}

export const toKeywordSyncWarning = (note: any): KeywordSyncWarning | null => {
  const source =
    note?.keyword_sync ??
    note?.keywordSync ??
    note?.metadata?.keyword_sync ??
    null
  if (!source || typeof source !== 'object') return null

  const failedCountCandidate = source?.failed_count ?? source?.failedCount ?? source?.count
  const failedCount = Number(failedCountCandidate)
  if (!Number.isFinite(failedCount) || failedCount <= 0) return null

  const failedKeywords = Array.isArray(source?.failed_keywords)
    ? source.failed_keywords
    : Array.isArray(source?.failedKeywords)
      ? source.failedKeywords
      : []

  const normalizedKeywords = failedKeywords
    .map((keyword: any) => String(keyword || '').trim())
    .filter((keyword: string) => keyword.length > 0)
    .slice(0, 5)

  return {
    failedCount: Math.floor(failedCount),
    failedKeywords: normalizedKeywords
  }
}

export const normalizeConversationId = (value: unknown): string | null => {
  const text = String(value || '').trim()
  return text.length > 0 ? text : null
}

export const toConversationLabel = (chat: any): string | null => {
  const candidates = [
    chat?.title,
    chat?.topic_label,
    chat?.topicLabel,
    chat?.external_ref,
    chat?.source
  ]
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue
    const text = candidate.trim()
    if (!text) continue
    return text
  }
  return null
}

export const toAttachmentMarkdown = (
  fileName: string,
  url: string,
  contentType?: string | null
) => {
  const escapedName = String(fileName || '')
    .replace(/\\/g, '\\\\')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]')
    .replace(/\r?\n/g, ' ')
  const isImage =
    (contentType || '').startsWith('image/') ||
    /\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(fileName)
  if (isImage) {
    return `![${escapedName}](${url})`
  }
  return `[${escapedName}](${url})`
}

export const normalizeGraphNoteId = (rawId: string | number | null | undefined): string => {
  if (rawId == null) return ''
  const text = String(rawId).trim()
  if (text.startsWith('note:')) return text.slice(5)
  return text
}

export const parseSourceNodeId = (
  rawId: string | number | null | undefined
): { source: string; externalRef: string | null } | null => {
  if (rawId == null) return null
  const text = String(rawId).trim()
  if (!text.startsWith('source:')) return null
  const parts = text.split(':')
  if (parts.length < 2) return null
  const source = String(parts[1] || '').trim()
  if (!source) return null
  const externalRefRaw = parts.length > 2 ? parts.slice(2).join(':').trim() : ''
  return {
    source,
    externalRef: externalRefRaw.length > 0 ? externalRefRaw : null
  }
}

// 120px offset accounts for page header and padding
export const MIN_SIDEBAR_HEIGHT = 600
export const NOTE_AUTOSAVE_DELAY_MS = 5000
export const NOTE_SEARCH_DEBOUNCE_MS = 350
export const LARGE_NOTE_PREVIEW_THRESHOLD = 10_000
export const LARGE_NOTE_PREVIEW_DELAY_MS = 120
export const LARGE_NOTES_PAGINATION_THRESHOLD = 100
export const TRASH_LOOKUP_PAGE_SIZE = 100
export const TRASH_LOOKUP_MAX_PAGES = 50
export const NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY = 'tldw:notesOfflineDraftQueue:v1'
export const NOTES_OFFLINE_NEW_DRAFT_KEY = 'draft:new'
export const NOTES_LIST_REGION_ID = 'notes-list-region'
export const NOTES_EDITOR_REGION_ID = 'notes-editor-region'
export const NOTES_SHORTCUTS_SUMMARY_ID = 'notes-shortcuts-summary'
export const NOTES_STUDIO_TEMPLATE_OPTIONS: Array<{
  value: NotesStudioTemplateType
  labelKey: string
  defaultLabel: string
}> = [
  { value: 'lined', labelKey: 'option:notesSearch.notesStudioTemplateLined', defaultLabel: 'Lined' },
  { value: 'grid', labelKey: 'option:notesSearch.notesStudioTemplateGrid', defaultLabel: 'Grid' },
  { value: 'cornell', labelKey: 'option:notesSearch.notesStudioTemplateCornell', defaultLabel: 'Cornell' },
]
export const NOTES_STUDIO_HANDWRITING_OPTIONS: Array<{
  value: NotesStudioHandwritingMode
  labelKey: string
  defaultLabel: string
}> = [
  { value: 'accented', labelKey: 'option:notesSearch.notesStudioHandwritingAccented', defaultLabel: 'Accented' },
  { value: 'off', labelKey: 'option:notesSearch.notesStudioHandwritingOff', defaultLabel: 'Off' },
]

export const shouldIgnoreGlobalShortcut = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) return false
  const element = target as HTMLElement
  if (element.isContentEditable) return true
  const tag = (element.tagName || '').toLowerCase()
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true
  return Boolean(element.closest('input,textarea,select,[contenteditable="true"]'))
}

export const isWithinRegion = (target: EventTarget | null, regionId: string): boolean => {
  if (!(target instanceof Element)) return false
  return Boolean((target as Element).closest(`#${regionId}`))
}

export const isEditorSaveShortcutContext = (target: EventTarget | null): boolean => {
  if (isWithinRegion(target, NOTES_EDITOR_REGION_ID)) return true
  if (typeof document !== 'undefined' && isWithinRegion(document.activeElement, NOTES_EDITOR_REGION_ID)) {
    return true
  }
  return false
}

export const calculateSidebarHeight = () => {
  const vh = typeof window !== 'undefined' ? window.innerHeight : MIN_SIDEBAR_HEIGHT
  return Math.max(MIN_SIDEBAR_HEIGHT, vh - 120)
}

export type SaveNoteOptions = {
  showSuccessMessage?: boolean
}

export type SaveIndicatorState = 'idle' | 'dirty' | 'saving' | 'saved' | 'error'
export type NotesEditorMode = 'edit' | 'split' | 'preview'
export type NotesInputMode = 'markdown' | 'wysiwyg'
export type NotesSortOption = 'modified_desc' | 'created_desc' | 'title_asc' | 'title_desc'
export type KeywordPickerSortMode = 'frequency_desc' | 'alpha_asc' | 'alpha_desc'
export type KeywordFrequencyTone = 'none' | 'low' | 'medium' | 'high'
export type KeywordManagementItem = {
  id: number
  keyword: string
  version: number
  noteCount: number
}
export type KeywordRenameDraft = {
  id: number
  currentKeyword: string
  expectedVersion: number
  nextKeyword: string
}
export type KeywordMergeDraft = {
  source: KeywordManagementItem
  targetKeywordId: number | null
}
export type MarkdownToolbarAction = 'bold' | 'italic' | 'heading' | 'list' | 'link' | 'code'
export type OfflineDraftSyncState = 'queued' | 'syncing' | 'conflict' | 'error'
export type OfflineDraftEntry = {
  key: string
  noteId: string | null
  baseVersion: number | null
  title: string
  content: string
  keywords: string[]
  metadata: Record<string, any> | null
  backlinkConversationId: string | null
  backlinkMessageId: string | null
  updatedAt: string
  syncState: OfflineDraftSyncState
  lastError: string | null
}
export type OfflineDraftSyncResult =
  | {
      status: 'synced'
      key: string
      noteId: string
      version: number | null
      lastSavedAt: string | null
    }
  | {
      status: 'conflict' | 'error'
      key: string
      message: string
    }

export const normalizeOfflineDraftQueue = (rawValue: unknown): Record<string, OfflineDraftEntry> => {
  if (!rawValue || typeof rawValue !== 'object') return {}
  const source = rawValue as Record<string, any>
  const normalized: Record<string, OfflineDraftEntry> = {}
  for (const [key, draft] of Object.entries(source)) {
    if (!draft || typeof draft !== 'object') continue
    const normalizedKey = String(key || '').trim()
    if (!normalizedKey) continue
    const syncStateRaw = String(draft.syncState || '').toLowerCase()
    const syncState: OfflineDraftSyncState =
      syncStateRaw === 'syncing'
        ? 'syncing'
        : syncStateRaw === 'conflict'
          ? 'conflict'
          : syncStateRaw === 'error'
            ? 'error'
            : 'queued'
    normalized[normalizedKey] = {
      key: normalizedKey,
      noteId: draft.noteId != null ? String(draft.noteId) : null,
      baseVersion:
        typeof draft.baseVersion === 'number' && Number.isFinite(draft.baseVersion)
          ? Math.floor(draft.baseVersion)
          : toNoteVersion(draft),
      title: String(draft.title || ''),
      content: String(draft.content || ''),
      keywords: Array.isArray(draft.keywords)
        ? draft.keywords
            .map((keyword: any) => String(keyword || '').trim())
            .filter((keyword: string) => keyword.length > 0)
        : [],
      metadata:
        draft.metadata && typeof draft.metadata === 'object'
          ? { ...(draft.metadata as Record<string, any>) }
          : null,
      backlinkConversationId:
        draft.backlinkConversationId != null ? String(draft.backlinkConversationId) : null,
      backlinkMessageId: draft.backlinkMessageId != null ? String(draft.backlinkMessageId) : null,
      updatedAt: toNoteLastModified(draft) || new Date().toISOString(),
      syncState,
      lastError:
        draft.lastError != null && String(draft.lastError).trim().length > 0
          ? String(draft.lastError)
          : null
    }
  }
  return normalized
}
export type RemoteVersionInfo = { version: number; lastModified: string | null }
export type NotesAssistAction = 'summarize' | 'expand_outline' | 'suggest_keywords'
export type EditProvenanceState =
  | { mode: 'manual' }
  | { mode: 'generated'; action: NotesAssistAction; at: number }
export type MonitoringAlertSeverity = 'info' | 'warning' | 'critical'
export type MonitoringNoticeState = {
  severity: MonitoringAlertSeverity
  title: string
  guidance: string
}
export type ExportFormat = 'md' | 'csv' | 'json'
export type ExportProgressState = {
  format: ExportFormat
  fetchedNotes: number
  fetchedPages: number
  failedBatches: number
}
export type NotesListViewMode = 'list' | 'timeline' | 'moodboard'
export type MoodboardSummary = {
  id: number
  name: string
  description?: string | null
  smart_rule?: Record<string, any> | null
  version?: number
  last_modified?: string
}
export type NotebookFilterOption = NotesNotebookSetting
export type ImportFormat = 'json' | 'markdown'
export type ImportDuplicateStrategy = 'skip' | 'overwrite' | 'create_copy'
export type PendingImportFile = {
  fileName: string
  format: ImportFormat
  content: string
  detectedNotes: number
  parseError: string | null
}
export type NotesImportResponsePayload = {
  files?: Array<{
    file_name?: string | null
    source_format?: ImportFormat
    detected_notes?: number
    created_count?: number
    updated_count?: number
    skipped_count?: number
    failed_count?: number
    errors?: string[]
  }>
  detected_notes?: number
  created_count?: number
  updated_count?: number
  skipped_count?: number
  failed_count?: number
}
export type NotesTitleSettingsResponse = {
  llm_enabled?: boolean
  default_strategy?: string
  effective_strategy?: string
  strategies?: string[]
}
export type NoteTemplateDefinition = {
  id: string
  label: string
  title: string
  content: string
}
export type NotesTocEntry = {
  id: string
  level: number
  text: string
  offset: number
}

export const NOTES_TITLE_STRATEGIES: NotesTitleSuggestStrategy[] = ['heuristic', 'llm', 'llm_fallback']
export const NOTE_TEMPLATES: NoteTemplateDefinition[] = [
  {
    id: 'meeting_notes',
    label: 'Meeting Notes',
    title: 'Meeting Notes',
    content: [
      '## Participants',
      '- ',
      '',
      '## Agenda',
      '- ',
      '',
      '## Key Decisions',
      '- ',
      '',
      '## Action Items',
      '- [ ] '
    ].join('\n')
  },
  {
    id: 'research_brief',
    label: 'Research Brief',
    title: 'Research Brief',
    content: [
      '## Research Question',
      '',
      '## Summary',
      '',
      '## Evidence',
      '- ',
      '',
      '## Open Questions',
      '- ',
      '',
      '## Next Steps',
      '- [ ] '
    ].join('\n')
  },
  {
    id: 'literature_review',
    label: 'Literature Review',
    title: 'Literature Review',
    content: [
      '## Source',
      '- Title:',
      '- Author:',
      '- Year:',
      '- URL:',
      '',
      '## Key Findings',
      '- ',
      '',
      '## Methods',
      '- ',
      '',
      '## Limitations',
      '- ',
      '',
      '## Relevance to Project',
      '- '
    ].join('\n')
  },
  {
    id: 'experiment_log',
    label: 'Experiment Log',
    title: 'Experiment Log',
    content: [
      '## Hypothesis',
      '',
      '## Setup',
      '- ',
      '',
      '## Results',
      '- ',
      '',
      '## Interpretation',
      '- ',
      '',
      '## Follow-up',
      '- [ ] '
    ].join('\n')
  }
]
export const KEYWORD_FREQUENCY_DOT_CLASS: Record<KeywordFrequencyTone, string> = {
  none: 'bg-border',
  low: 'bg-primary/35',
  medium: 'bg-primary/60',
  high: 'bg-primary'
}

export const toKeywordTestIdSegment = (keyword: string) =>
  keyword.toLowerCase().replace(/[^a-z0-9_-]/g, '_')

export const normalizeNotebookKeywords = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const deduped: string[] = []
  const seen = new Set<string>()
  for (const keyword of value) {
    const normalized = String(keyword || '').trim().toLowerCase()
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    deduped.push(normalized)
    if (deduped.length >= 25) break
  }
  return deduped
}

export const normalizeNotebookName = (value: unknown): string => String(value || '').trim()

export const normalizeNotebookOptions = (value: unknown): NotebookFilterOption[] => {
  if (!Array.isArray(value)) return []
  const out: NotebookFilterOption[] = []
  const seenIds = new Set<number>()
  const seenNames = new Set<string>()
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') continue
    const idRaw = Number((entry as any).id)
    const name = normalizeNotebookName((entry as any).name)
    if (!Number.isFinite(idRaw) || idRaw <= 0 || !name) continue
    const id = Math.floor(idRaw)
    const key = name.toLowerCase()
    if (seenIds.has(id) || seenNames.has(key)) continue
    seenIds.add(id)
    seenNames.add(key)
    out.push({
      id,
      name,
      keywords: normalizeNotebookKeywords((entry as any).keywords)
    })
    if (out.length >= 100) break
  }
  return out
}

export const NOTEBOOK_COLLECTION_PAGE_SIZE = 200
export const NOTEBOOK_COLLECTION_MAX_PAGES = 5

export const normalizeNotebookKeywordsFromServer = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const deduped: string[] = []
  const seen = new Set<string>()
  for (const raw of value) {
    const keyword =
      typeof raw === 'string'
        ? raw
        : String(
            (raw as any)?.keyword ??
              (raw as any)?.keyword_text ??
              (raw as any)?.text ??
              ''
          )
    const normalized = keyword.trim().toLowerCase()
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    deduped.push(normalized)
    if (deduped.length >= 25) break
  }
  return deduped
}

export const normalizeNotebookCollectionFromServer = (value: unknown): NotebookFilterOption | null => {
  if (!value || typeof value !== 'object') return null
  const idRaw = Number((value as any).id)
  const name = normalizeNotebookName((value as any).name)
  if (!Number.isFinite(idRaw) || idRaw <= 0 || !name) return null
  return {
    id: Math.floor(idRaw),
    name,
    keywords: normalizeNotebookKeywordsFromServer((value as any).keywords)
  }
}

export const normalizeNotebookCollectionsResponse = (value: unknown): NotebookFilterOption[] => {
  if (Array.isArray(value)) {
    return normalizeNotebookOptions(
      value
        .map((entry) => normalizeNotebookCollectionFromServer(entry))
        .filter((entry): entry is NotebookFilterOption => entry != null)
    )
  }
  if (value && typeof value === 'object') {
    const entries = (value as any).collections
    if (Array.isArray(entries)) {
      return normalizeNotebookOptions(
        entries
          .map((entry) => normalizeNotebookCollectionFromServer(entry))
          .filter((entry): entry is NotebookFilterOption => entry != null)
      )
    }
  }
  return []
}

export const buildNotebookDefaultName = (keywords: string[]): string => {
  const pretty = keywords
    .slice(0, 2)
    .map((keyword) => keyword.replace(/[-_]+/g, ' '))
    .map((keyword) => keyword.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .map((keyword) => keyword.charAt(0).toUpperCase() + keyword.slice(1))
  if (pretty.length === 0) return 'Notebook'
  return `${pretty.join(' + ')} Notebook`
}

export const stripInlineMarkdownForToc = (value: string): string =>
  String(value || '')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim()

export const slugifyHeading = (value: string): string => {
  const normalized = stripInlineMarkdownForToc(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return normalized || 'section'
}

export const extractMarkdownHeadings = (markdown: string): NotesTocEntry[] => {
  const entries: NotesTocEntry[] = []
  const lines = String(markdown || '').split('\n')
  const slugCounts = new Map<string, number>()
  let offset = 0
  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.+?)\s*$/)
    if (match) {
      const level = Math.min(6, Math.max(1, match[1].length))
      const text = stripInlineMarkdownForToc(match[2])
      if (text.length > 0) {
        const baseSlug = slugifyHeading(text)
        const seen = slugCounts.get(baseSlug) || 0
        slugCounts.set(baseSlug, seen + 1)
        entries.push({
          id: seen > 0 ? `${baseSlug}-${seen + 1}` : baseSlug,
          level,
          text,
          offset
        })
      }
    }
    offset += line.length + 1
  }
  return entries
}

export const escapeHtml = (value: string): string =>
  String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const SAFE_URL_PROTOCOLS = /^(https?:|mailto:|tel:|note:|#|\/)/i

const sanitizeUrl = (url: string): string => {
  const trimmed = url.trim()
  if (!trimmed) return ''
  if (SAFE_URL_PROTOCOLS.test(trimmed)) return trimmed
  // Block javascript:, data:, vbscript:, etc.
  if (/^[a-z][a-z0-9+.-]*:/i.test(trimmed)) return ''
  // Relative URLs are safe
  return trimmed
}

export const markdownInlineToHtml = (value: string): string => {
  let html = escapeHtml(value)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, text, url) => {
    const rawUrl = String(url || '').trim()
    const safeUrl = sanitizeUrl(rawUrl)
    if (!safeUrl) {
      // Render as plain text if URL is unsafe
      return escapeHtml(String(text || '').trim() || rawUrl)
    }
    const href = escapeHtml(safeUrl)
    const label = escapeHtml(String(text || '').trim() || href)
    return `<a href="${href}">${label}</a>`
  })
  html = html.replace(/`([^`]+)`/g, (_match, code) => `<code>${escapeHtml(String(code || ''))}</code>`)
  html = html.replace(/\*\*([^*]+)\*\*/g, (_match, strong) => `<strong>${escapeHtml(String(strong || ''))}</strong>`)
  html = html.replace(/\*([^*]+)\*/g, (_match, em) => `<em>${escapeHtml(String(em || ''))}</em>`)
  return html
}

export const markdownToWysiwygHtml = (markdown: string): string => {
  const lines = String(markdown || '').replace(/\r\n/g, '\n').split('\n')
  const slugCounts = new Map<string, number>()
  const out: string[] = []
  let inList = false
  const closeList = () => {
    if (!inList) return
    out.push('</ul>')
    inList = false
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd()
    const headingMatch = line.match(/^(#{1,6})\s+(.+?)\s*$/)
    if (headingMatch) {
      closeList()
      const level = Math.min(6, Math.max(1, headingMatch[1].length))
      const text = stripInlineMarkdownForToc(headingMatch[2])
      const baseSlug = slugifyHeading(text || headingMatch[2])
      const seen = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, seen + 1)
      const slug = seen > 0 ? `${baseSlug}-${seen + 1}` : baseSlug
      out.push(
        `<h${level} data-md-slug="${escapeHtml(slug)}">${markdownInlineToHtml(headingMatch[2])}</h${level}>`
      )
      continue
    }

    const listMatch = line.match(/^\s*-\s+(.+?)\s*$/)
    if (listMatch) {
      if (!inList) {
        out.push('<ul>')
        inList = true
      }
      out.push(`<li>${markdownInlineToHtml(listMatch[1])}</li>`)
      continue
    }

    closeList()
    if (line.length === 0) {
      out.push('<p><br/></p>')
      continue
    }
    out.push(`<p>${markdownInlineToHtml(line)}</p>`)
  }

  closeList()
  if (out.length === 0) return '<p><br/></p>'
  return out.join('')
}

export const inlineNodeToMarkdown = (node: Node): string => {
  if (node.nodeType === Node.TEXT_NODE) {
    return String(node.textContent || '').replace(/\u00A0/g, ' ')
  }
  if (!(node instanceof HTMLElement)) return ''

  const tag = node.tagName.toLowerCase()
  if (tag === 'br') return '\n'
  const inner = Array.from(node.childNodes).map(inlineNodeToMarkdown).join('')
  if (tag === 'strong' || tag === 'b') return `**${inner}**`
  if (tag === 'em' || tag === 'i') return `*${inner}*`
  if (tag === 'code') return `\`${inner}\``
  if (tag === 'a') {
    const href = String(node.getAttribute('href') || '').trim()
    if (!href) return inner
    return `[${inner || href}](${href})`
  }
  return inner
}

export const blockNodeToMarkdown = (node: Node): string => {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = String(node.textContent || '').trim()
    return text
  }
  if (!(node instanceof HTMLElement)) return ''

  const tag = node.tagName.toLowerCase()
  const inline = Array.from(node.childNodes).map(inlineNodeToMarkdown).join('').trim()
  if (/^h[1-6]$/.test(tag)) {
    const level = Number(tag.slice(1))
    return `${'#'.repeat(level)} ${inline}`.trim()
  }
  if (tag === 'ul' || tag === 'ol') {
    const items = Array.from(node.children)
      .filter((child) => child.tagName.toLowerCase() === 'li')
      .map((child) => `- ${Array.from(child.childNodes).map(inlineNodeToMarkdown).join('').trim()}`.trim())
      .filter(Boolean)
    return items.join('\n')
  }
  if (tag === 'pre') {
    const text = String(node.textContent || '').replace(/\u00A0/g, ' ').trim()
    return text ? `\`\`\`\n${text}\n\`\`\`` : ''
  }
  return inline
}

export const wysiwygHtmlToMarkdown = (html: string): string => {
  const source = String(html || '').trim()
  if (!source) return ''

  if (typeof window === 'undefined' || typeof DOMParser === 'undefined') {
    return source.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
  }

  const parser = new DOMParser()
  const doc = parser.parseFromString(`<div>${source}</div>`, 'text/html')
  const root = doc.body.firstElementChild
  if (!root) return ''

  const blocks = Array.from(root.childNodes)
    .map(blockNodeToMarkdown)
    .map((line) => line.trim())
    .filter(Boolean)

  return blocks.join('\n\n').replace(/\n{3,}/g, '\n\n').trim()
}

export const detectImportFormatFromFileName = (fileName: string): ImportFormat => {
  const lower = fileName.toLowerCase()
  if (lower.endsWith('.json')) return 'json'
  return 'markdown'
}

export const estimateDetectedNotesFromImportContent = (format: ImportFormat, content: string): number => {
  if (format === 'markdown') return 1
  try {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed)) return parsed.filter((entry) => entry && typeof entry === 'object').length
    if (parsed && typeof parsed === 'object') {
      for (const key of ['notes', 'data', 'items', 'results']) {
        const candidate = (parsed as Record<string, unknown>)[key]
        if (Array.isArray(candidate)) {
          return candidate.filter((entry) => entry && typeof entry === 'object').length
        }
      }
      return 1
    }
  } catch {
    return 0
  }
  return 0
}

export const NOTE_SORT_API_PARAMS: Record<
  NotesSortOption,
  { sortBy: 'last_modified' | 'created_at' | 'title'; sortOrder: 'asc' | 'desc' }
> = {
  modified_desc: { sortBy: 'last_modified', sortOrder: 'desc' },
  created_desc: { sortBy: 'created_at', sortOrder: 'desc' },
  title_asc: { sortBy: 'title', sortOrder: 'asc' },
  title_desc: { sortBy: 'title', sortOrder: 'desc' }
}
export const NOTE_ASSIST_STOP_WORDS = new Set([
  'the',
  'and',
  'for',
  'with',
  'that',
  'this',
  'from',
  'have',
  'has',
  'you',
  'your',
  'are',
  'was',
  'were',
  'about',
  'into',
  'within',
  'also',
  'they',
  'their',
  'there',
  'will',
  'would',
  'should',
  'could',
  'can',
  'not',
  'but',
  'than',
  'then',
  'when',
  'what',
  'where',
  'which',
  'while',
  'because',
  'using',
  'used',
  'between',
  'through',
  'about',
  'into',
  'over',
  'under',
  'after',
  'before',
  'our',
  'out',
  'its'
])

export const normalizeNotesTitleStrategy = (value: unknown): NotesTitleSuggestStrategy | null => {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'heuristic' || normalized === 'llm' || normalized === 'llm_fallback') {
    return normalized
  }
  return null
}

export const deriveAllowedTitleStrategies = (
  settings: NotesTitleSettingsResponse | null | undefined
): NotesTitleSuggestStrategy[] => {
  const rawStrategies = Array.isArray(settings?.strategies) ? settings.strategies : ['heuristic']
  const base = rawStrategies
    .map((entry) => normalizeNotesTitleStrategy(entry))
    .filter((entry): entry is NotesTitleSuggestStrategy => entry != null)

  const unique = Array.from(new Set(base))
  if (settings?.llm_enabled) {
    return unique.length > 0 ? unique : NOTES_TITLE_STRATEGIES
  }
  return ['heuristic']
}

export const toSortableTimestamp = (candidate: unknown): number => {
  if (typeof candidate !== 'string' || candidate.trim().length === 0) return 0
  const parsed = new Date(candidate)
  const time = parsed.getTime()
  return Number.isNaN(time) ? 0 : time
}

export const toSortableTitle = (candidate: unknown): string =>
  String(candidate || '')
    .trim()
    .toLowerCase()

export const sortNoteRows = (items: any[], sortOption: NotesSortOption): any[] => {
  const next = [...items]
  next.sort((a, b) => {
    if (sortOption === 'title_asc' || sortOption === 'title_desc') {
      const titleA = toSortableTitle(a?.title)
      const titleB = toSortableTitle(b?.title)
      const titleCompare = titleA.localeCompare(titleB)
      if (titleCompare !== 0) {
        return sortOption === 'title_asc' ? titleCompare : -titleCompare
      }
      return String(a?.id || '').localeCompare(String(b?.id || ''))
    }

    const timestampA =
      sortOption === 'created_desc'
        ? toSortableTimestamp(a?.created_at ?? a?.createdAt)
        : toSortableTimestamp(a?.last_modified ?? a?.updated_at ?? a?.updatedAt)
    const timestampB =
      sortOption === 'created_desc'
        ? toSortableTimestamp(b?.created_at ?? b?.createdAt)
        : toSortableTimestamp(b?.last_modified ?? b?.updated_at ?? b?.updatedAt)

    if (timestampA !== timestampB) {
      return timestampB - timestampA
    }

    if (sortOption === 'created_desc') {
      const modifiedA = toSortableTimestamp(a?.last_modified ?? a?.updated_at ?? a?.updatedAt)
      const modifiedB = toSortableTimestamp(b?.last_modified ?? b?.updated_at ?? b?.updatedAt)
      if (modifiedA !== modifiedB) {
        return modifiedB - modifiedA
      }
    }

    return String(a?.id || '').localeCompare(String(b?.id || ''))
  })
  return next
}

export const sortNotesByPinnedIds = (items: NoteListItem[], pinnedIdSet: Set<string>): NoteListItem[] => {
  if (pinnedIdSet.size === 0 || items.length <= 1) return items
  const pinned: NoteListItem[] = []
  const regular: NoteListItem[] = []
  for (const item of items) {
    if (pinnedIdSet.has(String(item.id))) {
      pinned.push(item)
    } else {
      regular.push(item)
    }
  }
  if (pinned.length === 0) return items
  return [...pinned, ...regular]
}

export const buildSummaryDraft = (rawContent: string): string => {
  const normalized = rawContent.replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  const sentences = normalized
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
  const selected = (sentences.length > 0 ? sentences : [normalized]).slice(0, 3)
  if (selected.length === 1) {
    return `Summary: ${selected[0]}`
  }
  return ['Summary:', ...selected.map((sentence) => `- ${sentence}`)].join('\n')
}

export const buildOutlineDraft = (rawContent: string): string => {
  const baseHeadings = rawContent
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^(#+|[-*+]|\d+\.)\s*/, ''))
    .filter((line) => line.length >= 3)
  const uniqueHeadings: string[] = []
  for (const heading of baseHeadings) {
    const normalized = heading.toLowerCase()
    if (uniqueHeadings.some((entry) => entry.toLowerCase() === normalized)) continue
    uniqueHeadings.push(heading)
    if (uniqueHeadings.length >= 3) break
  }
  const headings =
    uniqueHeadings.length > 0 ? uniqueHeadings : ['Main idea', 'Supporting evidence', 'Open questions']
  const sections = headings.map(
    (heading) => `## ${heading}\n- Key point\n- Supporting detail\n- Next action`
  )
  return ['# Expanded Outline', '', ...sections].join('\n\n')
}

export const suggestKeywordsDraft = (rawContent: string, existingKeywords: string[]): string[] => {
  const existing = new Set(
    existingKeywords
      .map((keyword) => String(keyword || '').trim().toLowerCase())
      .filter(Boolean)
  )
  const tokens = (rawContent.toLowerCase().match(/[a-z0-9][a-z0-9-]{2,}/g) || []).filter(
    (token) => !NOTE_ASSIST_STOP_WORDS.has(token) && !/^\d+$/.test(token)
  )
  const counts = new Map<string, number>()
  for (const token of tokens) {
    counts.set(token, (counts.get(token) || 0) + 1)
  }
  const sorted = Array.from(counts.entries()).sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1]
    return a[0].localeCompare(b[0])
  })
  const suggestions: string[] = []
  for (const [token] of sorted) {
    if (existing.has(token)) continue
    suggestions.push(token)
    if (suggestions.length >= 5) break
  }
  return suggestions
}

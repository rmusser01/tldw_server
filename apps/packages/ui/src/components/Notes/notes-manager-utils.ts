import {
  knowledgeNoteHead,
  knowledgeNoteProvenanceMatches,
  retainKnowledgeNoteProvenance,
  resolveKnowledgeNoteProvenance,
  type KnowledgeNoteHead,
} from "@/utils/knowledge-note-provenance";
import React from "react";
import type { NotesGraphEdge } from '@/services/note-graph-suggestions'
import { Modal, Input } from "antd";
import type {
  NotesTitleSuggestStrategy,
  NotesNotebookSetting,
} from "@/services/settings/ui-settings";
import { createSafeStorage } from "@/utils/safe-storage";
import {
  withRecordLock,
  type LocalRegistryRecord,
} from "@/services/settings/local-bucket";
import type { NotesStudioHandwritingMode, NotesStudioTemplateType } from './notes-studio-types'
import type { NoteListItem } from './types'

export const NOTES_TUTORIAL_SHOWN_STORAGE_KEY = 'notes-tutorial-shown'
export const notesUiStorage = createSafeStorage({ area: 'local' })

/**
 * Replacement for window.prompt() using Ant Design modal.
 * Returns the entered string, or null if cancelled.
 */
export function promptModal(options: {
  title: string
  label?: string
  defaultValue?: string
  placeholder?: string
  okText?: string
  cancelText?: string
}): Promise<string | null> {
  return new Promise((resolve) => {
    let currentValue = options.defaultValue ?? ''
    let settled = false
    const finish = (value: string | null) => {
      if (settled) return
      settled = true
      resolve(value)
    }
    const modal = Modal.confirm({
      title: options.title,
      icon: null,
      content: React.createElement('div', { className: 'mt-2' },
        options.label
          ? React.createElement('div', { className: 'mb-1 text-xs text-text-muted' }, options.label)
          : null,
        React.createElement(Input, {
          autoFocus: true,
          defaultValue: currentValue,
          placeholder: options.placeholder,
          onChange: (e: React.ChangeEvent<HTMLInputElement>) => { currentValue = e.target.value },
          onPressEnter: () => {
            modal.destroy()
            finish(currentValue.trim())
          },
        }),
      ),
      okText: options.okText ?? 'OK',
      cancelText: options.cancelText ?? 'Cancel',
      onOk: () => finish(currentValue.trim()),
      onCancel: () => finish(null),
    })
  })
}

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

/**
 * Trim tags, drop blanks and de-duplicate case-insensitively, keeping the
 * first spelling and the original order (matches the server's keyword
 * normalization in notes_schemas.normalized_keywords).
 */
export const normalizeTagList = (tags: readonly unknown[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const tag of tags) {
    if (typeof tag !== 'string') continue
    const text = tag.trim()
    if (!text) continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(text)
  }
  return out
}

/** Union of `existing` and `additions`; existing tags keep their order and spelling. */
export const mergeTagLists = (
  existing: readonly string[],
  additions: readonly string[]
): string[] => normalizeTagList([...existing, ...additions])

/** True when both lists hold the same tags, ignoring case, order and duplicates. */
export const tagSetsMatch = (a: readonly string[], b: readonly string[]): boolean => {
  const left = new Set(normalizeTagList(a).map((tag) => tag.toLowerCase()))
  const right = new Set(normalizeTagList(b).map((tag) => tag.toLowerCase()))
  if (left.size !== right.size) return false
  for (const tag of left) {
    if (!right.has(tag)) return false
  }
  return true
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

export const getNotesGraphEdgeLabel = (rawType: string | null | undefined): string => {
  switch (String(rawType || '').toLowerCase()) {
    case 'manual':
      return 'Manual link'
    case 'wikilink':
      return 'Note link'
    case 'backlink':
      return 'Backlink'
    case 'tag_membership':
      return 'Tag'
    case 'source_membership':
      return 'Source'
    case 'semantic':
      return 'Similar content'
    default:
      return 'Connection'
  }
}

export const toSafeTestId = (value: string | number | null | undefined): string =>
  String(value ?? 'unknown').replace(/[^a-z0-9_-]/gi, '_')

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

export const formatGraphEdgeTypeLabel = (edgeType: string | null | undefined): string => {
  switch (String(edgeType || '').toLowerCase()) {
    case 'manual':
      return 'Manual link'
    case 'wikilink':
      return 'Note link'
    case 'backlink':
      return 'Backlink'
    case 'tag_membership':
      return 'Tag'
    case 'source_membership':
      return 'Source'
    case 'semantic':
      return 'Similar content'
    default:
      return 'Connection'
  }
}

const GROUPED_NOTE_EDGE_TYPES = new Set([
  'manual',
  'wikilink',
  'backlink',
  'semantic'
])
const GRAPH_EDGE_TYPE_ORDER: Record<string, number> = {
  manual: 0,
  wikilink: 1,
  backlink: 2,
  semantic: 3,
  tag_membership: 4,
  source_membership: 5
}

export type NotesGraphEdgeGroup = {
  id: string
  source: string
  target: string
  edges: NotesGraphEdge[]
}

export const groupNotesGraphEdgesByPair = (
  edges: readonly NotesGraphEdge[]
): NotesGraphEdgeGroup[] => {
  const groups = new Map<string, NotesGraphEdgeGroup>()
  edges.forEach((edge) => {
    const canGroup =
      GROUPED_NOTE_EDGE_TYPES.has(edge.type) &&
      edge.source.startsWith('note:') &&
      edge.target.startsWith('note:')
    const pair = [edge.source, edge.target].sort()
    const key = canGroup ? `pair:${pair[0]}:${pair[1]}` : `edge:${edge.id}`
    const current = groups.get(key)
    if (current) {
      current.edges.push(edge)
      return
    }
    groups.set(key, {
      id: key,
      source: canGroup ? pair[0] : edge.source,
      target: canGroup ? pair[1] : edge.target,
      edges: [edge]
    })
  })
  return Array.from(groups.values()).map((group) => ({
    ...group,
    edges: [...group.edges].sort(
      (left, right) =>
        (GRAPH_EDGE_TYPE_ORDER[left.type] ?? 99) -
          (GRAPH_EDGE_TYPE_ORDER[right.type] ?? 99) ||
        left.id.localeCompare(right.id)
    )
  }))
}

const SOURCE_LABEL_BY_TYPE: Record<string, string> = {
  web: 'Web source',
  webpage: 'Web source',
  browser: 'Web source',
  clipper: 'Captured source',
  yt: 'YouTube source',
  youtube: 'YouTube source',
  media: 'Media source'
}

export const formatSourceNodeLabel = (
  sourceId: string | number | null | undefined,
  fallbackLabel?: string | null
): string => {
  const parsed = parseSourceNodeId(sourceId)
  if (!parsed) {
    return String(fallbackLabel || sourceId || 'Linked source')
  }
  const trimmedFallback = String(fallbackLabel || '').trim()
  const rawSourceId = String(sourceId || '').trim()
  const externalRef = parsed.externalRef || ''
  const rawFallbackValues = new Set(
    [
      rawSourceId,
      externalRef,
      `${parsed.source}:${externalRef}`,
      `${parsed.source}: ${externalRef}`
    ]
      .filter(Boolean)
      .map((value) => value.toLowerCase())
  )
  if (trimmedFallback && !rawFallbackValues.has(trimmedFallback.toLowerCase())) {
    return trimmedFallback
  }
  const sourceType = parsed.source.toLowerCase()
  const sourceLabel =
    SOURCE_LABEL_BY_TYPE[sourceType] ||
    `${parsed.source.charAt(0).toUpperCase()}${parsed.source.slice(1)} source`
  return parsed.externalRef ? `${sourceLabel}: ${parsed.externalRef}` : sourceLabel
}

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

export type SaveNoteOptions = {
  restoreProvenance?: boolean
  showSuccessMessage?: boolean
  /** Who asked for the save; see NoteSaveTrigger in notes-save-machine.ts. */
  trigger?: 'auto' | 'manual' | 'retry' | 'leave' | 'switch' | 'keep-mine'
  /** Overwrite this server version (only for an explicit "keep my version"). */
  expectedVersion?: number
}

/**
 * The one save status the editor shows (NS-06). Derived from the save state
 * machine: "saved" only after an acknowledged (2xx) save or load.
 */
export type SaveIndicatorState =
  | 'idle'
  | 'dirty'
  | 'saving'
  | 'saved'
  | 'error'
  | 'retrying'
  | 'conflict'
  | 'offline'
export type SaveRecoveryNotice = {
  kind: 'error' | 'conflict'
  message: string
}
export type NotesEditorMode = 'edit' | 'split' | 'preview'
export type NotesInputMode = 'markdown' | 'wysiwyg'

/**
 * Shows the Markdown/WYSIWYG input-mode toggle. It was hidden (decision D5,
 * #3102) while NE-01 made the WYSIWYG editor type backwards; the editor is now
 * uncontrolled, so the toggle is back. Set to false to hide WYSIWYG again.
 */
export const NOTES_WYSIWYG_INPUT_ENABLED = true
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
export type MarkdownToolbarAction = 'bold' | 'italic' | 'heading' | 'heading1' | 'heading2' | 'heading3' | 'list' | 'numbered' | 'checklist' | 'link' | 'code'

/** Apply block formatting to complete logical lines, retaining unselected text. */
export const formatMarkdownLines = (
  content: string,
  start: number,
  end: number,
  action:
    | "heading"
    | "heading1"
    | "heading2"
    | "heading3"
    | "list"
    | "numbered"
    | "checklist"
): { content: string; start: number; end: number } => {
  const from = start === 0 ? 0 : content.lastIndexOf("\n", start - 1) + 1
  const selectedEnd = end > start && content[end - 1] === "\n" ? end - 1 : end
  const nextBreak = content.indexOf("\n", selectedEnd)
  const to = nextBreak < 0 ? content.length : nextBreak
  let ordinal = 0
  let originalOffset = from
  let changedOffset = from
  let nextStart = start
  let nextEnd = end
  const replacement = content
    .slice(from, to)
    .split("\n")
    .map((line) => {
      const indent = line.match(/^[\t ]*/)?.[0] || ""
      const oldPrefix =
        line
          .slice(indent.length)
          .match(
            /^(?:#{1,6}\s+|(?:[-*+]|\d+\.)\s+(?:\[[ xX]\](?:\s+|$))?)/
          )?.[0] || ""
      const body = line.slice(indent.length + oldPrefix.length)
      const heading = action.startsWith("heading")
      const text = body || (heading ? "Heading" : "List item")
      const prefix = heading
        ? `${"#".repeat(action === "heading" ? 1 : Number(action.slice(-1)))} `
        : action === "numbered"
          ? `${++ordinal}. `
          : action === "checklist"
            ? "- [ ] "
            : "- "
      const keepBlank = !line.trim() && start !== end
      if (keepBlank && action === "numbered") ordinal -= 1
      const changed = keepBlank ? line : `${indent}${prefix}${text}`
      const map = (position: number) =>
        changedOffset +
        (keepBlank
          ? position - originalOffset
          : Math.min(
              changed.length,
              Math.max(
                indent.length + prefix.length,
                position - originalOffset + prefix.length - oldPrefix.length
              )
            ))
      if (start >= originalOffset && start <= originalOffset + line.length)
        nextStart = map(start)
      if (end >= originalOffset && end <= originalOffset + line.length)
        nextEnd = map(end)
      originalOffset += line.length + 1
      changedOffset += changed.length + 1
      return changed
    })
    .join("\n")
  if (start === end && !content.slice(from, to).trim())
    nextEnd = from + replacement.length
  if (end > to) nextEnd = changedOffset
  return {
    content: content.slice(0, from) + replacement + content.slice(to),
    start: nextStart,
    end: nextEnd
  }
}

/** Return a list Enter edit; null leaves ordinary browser editing in charge. */
export const continueMarkdownList = (
  content: string,
  start: number,
  end: number
): { content: string; cursor: number } | null => {
  if (start !== end) return null
  const from = start === 0 ? 0 : content.lastIndexOf("\n", start - 1) + 1
  let fence: { marker: string; length: number } | null = null
  for (const line of content.slice(0, from).split("\n")) {
    const match = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/)
    if (!match) continue
    if (!fence) {
      if (match[1][0] === "`" && match[2].includes("`")) continue
      fence = { marker: match[1][0], length: match[1].length }
    } else if (
      match[1][0] === fence.marker &&
      match[1].length >= fence.length &&
      !match[2].trim()
    )
      fence = null
  }
  if (fence) return null
  const nextBreak = content.indexOf("\n", start)
  const to = nextBreak < 0 ? content.length : nextBreak
  const line = content.slice(from, to)
  const match = line.match(
    /^([\t ]*)([-*+]|\d+\.)[\t ]+(?:\[([ xX])\](?:[\t ]+|$))?/
  )
  if (!match || start < from + match[0].length) return null
  if (!line.slice(match[0].length).trim()) {
    return {
      content: content.slice(0, from) + match[1] + content.slice(to),
      cursor: from + match[1].length
    }
  }
  const number = Number.parseInt(match[2], 10)
  if (
    /^\d/.test(match[2]) &&
    (!Number.isSafeInteger(number) || number >= Number.MAX_SAFE_INTEGER)
  )
    return null
  const marker = /^\d/.test(match[2]) ? `${number + 1}.` : match[2]
  const insertion = `\n${match[1]}${marker} ${match[3] === undefined ? "" : "[ ] "}`
  return {
    content: content.slice(0, start) + insertion + content.slice(start),
    cursor: start + insertion.length
  }
}

export const notesEditorMaxHeight = (
  viewportHeight: number,
  split: boolean
): number => Math.max(280, Math.floor(viewportHeight * (split ? 0.42 : 0.62)))

/** Measure visual wrapping with the textarea's actual computed layout. */
export const revealTextareaOffset = (
  textarea: HTMLTextAreaElement,
  offset: number
): void => {
  const mirror = textarea.ownerDocument.createElement("div")
  const computed =
    textarea.ownerDocument.defaultView?.getComputedStyle(textarea)
  if (!computed) return
  for (const property of Array.from(computed))
    mirror.style.setProperty(property, computed.getPropertyValue(property))
  Object.assign(mirror.style, {
    position: "fixed",
    left: "-10000px",
    top: "0",
    height: "auto",
    maxHeight: "none",
    minHeight: "0",
    overflow: "visible",
    visibility: "hidden",
    whiteSpace: "pre-wrap",
    overflowWrap: "break-word",
    pointerEvents: "none",
    width: `${
      computed.boxSizing === "border-box"
        ? textarea.clientWidth +
          Number.parseFloat(computed.borderLeftWidth) +
          Number.parseFloat(computed.borderRightWidth)
        : textarea.clientWidth -
          Number.parseFloat(computed.paddingLeft) -
          Number.parseFloat(computed.paddingRight)
    }px`
  })
  mirror.setAttribute("aria-hidden", "true")
  mirror.textContent = textarea.value.slice(0, offset)
  const marker = textarea.ownerDocument.createElement("span")
  marker.textContent = "\u200b"
  mirror.append(
    marker,
    textarea.ownerDocument.createTextNode(textarea.value.slice(offset))
  )
  textarea.ownerDocument.body.append(mirror)
  try {
    const top = marker.offsetTop
    textarea.scrollTop = Math.max(0, top - textarea.clientHeight / 2)
  } finally {
    mirror.remove()
  }
}

export type OfflineDraftSyncState = 'queued' | 'syncing' | 'conflict' | 'error'
export type OfflineDraftEntry = {
  pendingWrite?: {
    authorityId?: string;
    operation?: "restore";
    key: string;
    body: Record<string, unknown>;
    expectedVersion: number | null;
    previousTitle?: string | null;
  };
  key: string;
  noteId: string | null;
  baseVersion: number | null;
  title: string;
  content: string;
  keywords: string[];
  metadata: Record<string, any> | null;
  backlinkConversationId: string | null;
  backlinkMessageId: string | null;
  updatedAt: string;
  syncState: OfflineDraftSyncState;
  lastError: string | null;
};
export type OfflineDraftSyncResult =
  | {
      status: 'synced'
      savedTitle: string | null
      head?: KnowledgeNoteHead
      submittedBody?: Record<string, unknown>
      key: string
      noteId: string
      version: number | null
      lastSavedAt: string | null
    }
  | {
      status: 'conflict' | 'error'
      discardPendingWrite?: boolean
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
      ...(typeof draft.pendingWrite?.key === 'string' && draft.pendingWrite.body && typeof draft.pendingWrite.body === 'object' &&
        (typeof draft.pendingWrite.body.content === 'string' || (draft.pendingWrite.operation === 'restore' && draft.noteId != null &&
          Number.isSafeInteger(draft.pendingWrite.body.expected_provenance_version) && draft.pendingWrite.body.expected_provenance_version > 0 &&
          /^sha256:[a-f0-9]{64}$/.test(draft.pendingWrite.body.expected_provenance_hash) && Number.isSafeInteger(draft.pendingWrite.expectedVersion))) &&
        (draft.pendingWrite.expectedVersion === null || Number.isSafeInteger(draft.pendingWrite.expectedVersion))
        ? { pendingWrite: draft.pendingWrite } : {}),
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
// Surface-owned operations share this queue, but only their owning editor acknowledges them.
export const isSurfaceOfflineDraft = (key: string): boolean => key.startsWith('surface:')

export const readOfflineDraftQueue = (storageKey: string): Record<string, OfflineDraftEntry> =>
  normalizeOfflineDraftQueue(JSON.parse(window.localStorage.getItem(storageKey) || '{}'))

export const writeOfflineDraftQueue = (storageKey: string, queue: Record<string, OfflineDraftEntry>): void => {
  const serialized = JSON.stringify(queue)
  window.localStorage.setItem(storageKey, serialized)
  if (window.localStorage.getItem(storageKey) !== serialized)
    throw new Error('Could not retain the pending note operation on this device.')
}

export const writeNotesEditorOfflineDraftQueue = (
  storageKey: string,
  queue: Record<string, OfflineDraftEntry>,
): void => {
  let readableQueue: Record<string, OfflineDraftEntry> = {};
  try {
    readableQueue = readOfflineDraftQueue(storageKey);
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    // Hydration already treats malformed JSON as an empty ordinary queue.
  }
  const surfaceEntries = Object.fromEntries(
    Object.entries(readableQueue).filter(([key]) => isSurfaceOfflineDraft(key)),
  );
  writeOfflineDraftQueue(storageKey, { ...queue, ...surfaceEntries });
};

const surfaceOfflineDraftPrefix = (authorityScope: string): string => `${NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY}:${authorityScope}:`

export const readSurfaceOfflineDraftQueue = async (
  authorityScope: string,
): Promise<Record<string, OfflineDraftEntry>> => {
  const prefix = surfaceOfflineDraftPrefix(authorityScope);
  const entries = await notesUiStorage.getAll();
  const queue: Record<string, OfflineDraftEntry> = {};
  for (const storageKey of Object.keys(entries).filter((key) =>
    key.startsWith(`${prefix}surface:`),
  )) {
    const record =
      await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
        storageKey,
      );
    if (record == null) continue;
    const key = storageKey.slice(prefix.length);
    const draft = normalizeOfflineDraftQueue({ [key]: record.value })[key];
    if (!draft?.pendingWrite && !isQuickNotesRetainedDraft(draft))
      throw new Error(
        "The retained note operation cannot be read safely on this device.",
      );
    queue[key] = draft;
  }
  return queue
}

export const isQuickNotesRetainedDraft = (
  draft?: OfflineDraftEntry,
): boolean => {
  const metadata = draft?.metadata;
  return (
    !!draft &&
    !draft.pendingWrite &&
    typeof metadata?.quickNotesAuthorityId === "string" &&
    !!metadata.quickNotesAuthorityId &&
    typeof (
      metadata.quickNotesAcceptedKey || metadata.quickNotesRejectedKey
    ) === "string" &&
    !!(metadata.quickNotesAcceptedKey || metadata.quickNotesRejectedKey) &&
    typeof metadata.quickNotesDirty === "boolean" &&
    (metadata.quickNotesWorkspaceId === null ||
      typeof metadata.quickNotesWorkspaceId === "string") &&
    typeof metadata.quickNotesWorkspaceTag === "string" &&
    draft.key.startsWith(
      `surface:quick-notes:${JSON.stringify([metadata.quickNotesWorkspaceId, metadata.quickNotesWorkspaceTag])}:`,
    ) &&
    /^(?:[1-9][0-9]*|[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12})$/i.test(
      draft.noteId || "",
    ) &&
    Number.isSafeInteger(draft.baseVersion) &&
    draft.baseVersion! > 0
  );
};

export const retainSurfaceOfflineDraft = async (
  authorityScope: string,
  draft: OfflineDraftEntry,
): Promise<void> => {
  const storageKey = surfaceOfflineDraftPrefix(authorityScope) + draft.key;
  await withRecordLock(storageKey, async () => {
    const previous =
      await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
        storageKey,
      );
    if (
      previous?.value?.metadata?.quickNotesAuthorityId &&
      (typeof navigator === "undefined" || !navigator.locks?.request)
    )
      throw new Error(
        "Safe local note persistence requires a record lock on this device.",
      );
    if (
      previous?.value?.pendingWrite &&
      JSON.stringify(previous.value.pendingWrite) !==
        JSON.stringify(draft.pendingWrite)
    )
      throw new Error(
        "Resolve the retained note operation before replacing it.",
      );
    if (
      isQuickNotesRetainedDraft(previous?.value) &&
      (previous!.value.metadata?.quickNotesAcceptedKey ===
        draft.pendingWrite?.key ||
        previous!.value.metadata?.quickNotesRejectedKey ===
          draft.pendingWrite?.key)
    )
      return;
    // A retry must not replace mutable text with its older submitted snapshot.
    if (
      previous?.value?.metadata?.quickNotesAuthorityId &&
      previous.value.pendingWrite
    )
      return;
    try {
      await notesUiStorage.set(storageKey, {
        value: draft,
        updatedAt: Date.now(),
      });
    } catch {
      throw new Error(
        "Could not retain the pending note operation on this device.",
      );
    }
    const retained =
      await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
        storageKey,
      );
    if (JSON.stringify(retained?.value) !== JSON.stringify(draft))
      throw new Error(
        "Could not retain the pending note operation on this device.",
      );
  });
};

export const retireSurfaceOfflineDraft = async (
  authorityScope: string,
  key: string,
  pendingKey: string,
): Promise<void> => {
  const storageKey = surfaceOfflineDraftPrefix(authorityScope) + key;
  await withRecordLock(storageKey, async () => {
    const previous =
      await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
        storageKey,
      );
    if (
      previous?.value?.metadata?.quickNotesAuthorityId &&
      (typeof navigator === "undefined" || !navigator.locks?.request)
    )
      throw new Error(
        "Safe local note persistence requires a record lock on this device.",
      );
    if (previous?.value?.pendingWrite?.key !== pendingKey) return;
    const draft = previous.value;
    if (
      draft.metadata?.quickNotesAuthorityId &&
      draft.noteId &&
      Number.isSafeInteger(draft.pendingWrite!.expectedVersion) &&
      draft.pendingWrite!.expectedVersion! > 0
    ) {
      const retainedDraft: OfflineDraftEntry = {
        ...draft,
        pendingWrite: undefined,
        baseVersion: draft.pendingWrite!.expectedVersion,
        metadata: {
          ...draft.metadata,
          quickNotesAcceptedKey: undefined,
          quickNotesRejectedKey: pendingKey,
          quickNotesDirty: true,
        },
      };
      if (!isQuickNotesRetainedDraft(retainedDraft))
        throw new Error(
          "The rejected Quick Notes draft binding is incomplete.",
        );
      await notesUiStorage.set(storageKey, {
        value: retainedDraft,
        updatedAt: Date.now(),
      });
      const retained =
        await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
          storageKey,
        );
      if (JSON.stringify(retained?.value) !== JSON.stringify(retainedDraft))
        throw new Error(
          "Could not retire the rejected note operation while preserving its local draft.",
        );
      return;
    }
    try {
      await notesUiStorage.remove(storageKey);
    } catch {
      throw new Error(
        "Could not retire the acknowledged note operation on this device. Retry the same save.",
      );
    }
    const retained =
      await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
        storageKey,
      );
    if (retained?.value?.pendingWrite?.key === pendingKey)
      throw new Error(
        "Could not retire the acknowledged note operation on this device. Retry the same save.",
      );
  });
};

type QuickNotesDraftSnapshot = Pick<
  OfflineDraftEntry,
  "title" | "content" | "keywords" | "metadata"
> & { isDirty: boolean };

export const quickNotesDraftMatchesWrite = (
  draft: Pick<OfflineDraftEntry, "title" | "content" | "keywords" | "metadata">,
  body: Record<string, unknown>,
): boolean => {
  const research = resolveKnowledgeNoteProvenance(draft.metadata, draft.content)
    .provenance?.research;
  const keywords = [
    ...draft.keywords,
    ...(research ? [`workspace:${research.workspace_id}`] : []),
    draft.metadata?.quickNotesWorkspaceTag || "",
  ]
    .map((value) => value.trim())
    .filter(Boolean);
  const normalized = (values: string[]) =>
    values
      .filter(
        (value, index) =>
          values.findIndex(
            (other) => other.toLowerCase() === value.toLowerCase(),
          ) === index,
      )
      .sort();
  if (
    draft.metadata?.pendingKnowledgeProvenance &&
    !knowledgeNoteProvenanceMatches(
      draft.metadata.pendingKnowledgeProvenance,
      body.knowledge_provenance,
    )
  )
    return false;
  return (
    (draft.title || "Untitled Note") === body.title &&
    retainKnowledgeNoteProvenance(draft.content, draft.metadata) ===
      body.content &&
    JSON.stringify(normalized(keywords)) ===
      JSON.stringify(
        normalized(Array.isArray(body.keywords) ? body.keywords : []),
      )
  );
};

export const checkpointQuickNotesOfflineDraft = async (
  authorityScope: string,
  key: string,
  pendingKey: string,
  readCurrentDraft: () => QuickNotesDraftSnapshot | null,
  accepted?: KnowledgeNoteHead & { id: string | number; version?: number },
): Promise<OfflineDraftEntry | null> => {
  const storageKey = surfaceOfflineDraftPrefix(authorityScope) + key;
  return withRecordLock(
    storageKey,
    async () => {
      const record =
        await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
          storageKey,
        );
      const previous = record?.value;
      if (
        !previous ||
        (previous.pendingWrite?.key !== pendingKey &&
          !(
            isQuickNotesRetainedDraft(previous) &&
            (previous.metadata?.quickNotesAcceptedKey === pendingKey ||
              previous.metadata?.quickNotesRejectedKey === pendingKey)
          ))
      )
        return null;
      if (!key.startsWith("surface:quick-notes:"))
        throw new Error("Only Quick Notes owns this retained draft.");
      const current = readCurrentDraft();
      if (
        current &&
        (current.metadata?.quickNotesAuthorityId !==
          (previous.pendingWrite?.authorityId ??
            previous.metadata?.quickNotesAuthorityId) ||
          !key.startsWith(
            `surface:quick-notes:${JSON.stringify([current.metadata?.quickNotesWorkspaceId, current.metadata?.quickNotesWorkspaceTag])}:`,
          ))
      )
        throw new Error(
          "The local draft does not match the retained Workspace and service authority.",
        );
      let next: OfflineDraftEntry = current
        ? {
            ...previous,
            title: current.title,
            content: current.content,
            keywords: [...current.keywords],
            metadata: {
              ...previous.metadata,
              ...current.metadata,
              quickNotesDirty: current.isDirty,
            },
          }
        : previous;
      if (accepted && previous.pendingWrite) {
        if (
          String(accepted.id) !==
            String(previous.pendingWrite.body.id ?? previous.noteId) ||
          !Number.isSafeInteger(accepted.version) ||
          accepted.version! < 1
        )
          throw new Error(
            "The accepted note receipt does not match the retained operation.",
          );
        next = {
          ...next,
          noteId: String(accepted.id),
          baseVersion: accepted.version!,
          pendingWrite: undefined,
          metadata: {
            ...next.metadata,
            ...knowledgeNoteHead(accepted),
            quickNotesAuthorityId: previous.pendingWrite.authorityId,
            quickNotesAcceptedKey: pendingKey,
            quickNotesDirty: !quickNotesDraftMatchesWrite(
              next,
              previous.pendingWrite.body,
            ),
          },
        };
        if (!isQuickNotesRetainedDraft(next))
          throw new Error(
            "The accepted Quick Notes draft binding is incomplete.",
          );
      } else if (!previous.pendingWrite) {
        // A checkpoint queued before ACK keeps its exact accepted base and authority.
        next = {
          ...next,
          metadata: {
            ...next.metadata,
            ...knowledgeNoteHead(previous.metadata),
            quickNotesAuthorityId: previous.metadata!.quickNotesAuthorityId,
            quickNotesAcceptedKey: previous.metadata!.quickNotesAcceptedKey,
          },
        };
      }
      if (JSON.stringify(next) === JSON.stringify(previous)) return previous;
      next = { ...next, updatedAt: new Date().toISOString() };
      await notesUiStorage.set(storageKey, {
        value: next,
        updatedAt: Date.now(),
      });
      const retained =
        await notesUiStorage.get<LocalRegistryRecord<OfflineDraftEntry>>(
          storageKey,
        );
      if (JSON.stringify(retained?.value) !== JSON.stringify(next))
        throw new Error(
          "Could not retain the current local note draft on this device.",
        );
      return next;
    },
    true,
  );
};

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
  /** Matching notes the server reported, once the first page has arrived. */
  totalNotes?: number | null
}
export type NotesListViewMode = 'list' | 'timeline' | 'inbox' | 'moodboard' | 'graph'
export const resolveNotesGraphFocusNoteId = <
  TRecent extends { id: string | number },
  TVisible extends { id: string | number; deleted?: boolean | null }
>(
  selectedId: string | number | null,
  recentNotes: TRecent[],
  visibleNotes: TVisible[]
): string | null => {
  const selected = normalizeGraphNoteId(selectedId)
  if (selected) return selected

  const activeVisibleNotes = visibleNotes.filter((note) => note.deleted !== true)
  const activeIds = new Set(
    activeVisibleNotes.map((note) => normalizeGraphNoteId(note.id))
  )
  const recent = recentNotes.find((note) =>
    activeIds.has(normalizeGraphNoteId(note.id))
  )
  return normalizeGraphNoteId(recent?.id ?? activeVisibleNotes[0]?.id) || null
}
export const hasNotesGraphActiveNotes = <
  TVisible extends { id: string | number; deleted?: boolean | null }
>(
  selectedId: string | number | null,
  activeNoteTotal: number,
  visibleNotes: TVisible[]
): boolean =>
  Boolean(
    normalizeGraphNoteId(selectedId) ||
      activeNoteTotal > 0 ||
      visibleNotes.some((note) => note.deleted !== true)
  )
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
  // Strip C0 control characters (incl. tab/newline/CR) and DEL first: browsers
  // remove them when resolving a URL, so `java\tscript:` would otherwise match
  // neither branch below and be emitted verbatim as an executable scheme.
  let stripped = ''
  for (const ch of url) {
    const code = ch.charCodeAt(0)
    if (code <= 0x1f || code === 0x7f) continue
    stripped += ch
  }
  const trimmed = stripped.trim()
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

/** The WYSIWYG document for an empty note: one empty paragraph to put the caret in. */
export const EMPTY_WYSIWYG_HTML = '<p><br/></p>'

export const markdownToWysiwygHtml = (markdown: string): string => {
  const lines = String(markdown || '').replace(/\r\n/g, '\n').split('\n')
  const slugCounts = new Map<string, number>()
  const out: string[] = []
  const lists: Array<{ indent: number; tag: 'ul' | 'ol'; ordinal: number }> = []
  const closeList = () => {
    while (lists.length) {
      out.push(`</li></${lists[lists.length - 1].tag}>`)
      lists.pop()
    }
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

    const listMatch = line.match(/^([\t ]*)([-*+]|\d+\.)\s+(.*)$/)
    if (listMatch) {
      const indent = listMatch[1].replace(/\t/g, '    ').length
      const tag = /^\d/.test(listMatch[2]) ? 'ol' : 'ul'
      const ordinal = tag === 'ol' ? Number.parseInt(listMatch[2], 10) : 0
      while (lists.length && lists[lists.length - 1].indent > indent) {
        out.push(`</li></${lists[lists.length - 1].tag}>`)
        lists.pop()
      }
      if (lists.length && lists[lists.length - 1].indent === indent && lists[lists.length - 1].tag !== tag) {
        out.push(`</li></${lists[lists.length - 1].tag}>`)
        lists.pop()
      }
      const current = lists[lists.length - 1]
      if (!current || current.indent < indent) {
        out.push(tag === 'ol' ? `<ol start="${ordinal}">` : '<ul>')
        lists.push({ indent, tag, ordinal })
        out.push('<li>')
      } else {
        out.push(`</li><li${tag === 'ol' && ordinal !== current.ordinal + 1 ? ` value="${ordinal}"` : ''}>`)
        current.ordinal = ordinal
      }
      const task = listMatch[3].match(/^\[([ xX])\](?:\s+(.*))?$/)
      if (task && tag === 'ul') {
        out.push(`<input type="checkbox" contenteditable="false" aria-label="${escapeHtml(task[2] || 'Task')}"${task[1] !== ' ' ? ' checked' : ''}>`)
      }
      out.push(markdownInlineToHtml(task && tag === 'ul' ? task[2] || '' : listMatch[3]))
      continue
    }

    closeList()
    if (line.length === 0) {
      out.push(EMPTY_WYSIWYG_HTML)
      continue
    }
    out.push(`<p>${markdownInlineToHtml(line)}</p>`)
  }

  closeList()
  if (out.length === 0) return EMPTY_WYSIWYG_HTML
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
    let ordinal = tag === 'ol' ? Number.parseInt(node.getAttribute('start') || '1', 10) : 0
    if (!Number.isFinite(ordinal)) ordinal = 1
    const items = Array.from(node.children)
      .filter((child) => child.tagName.toLowerCase() === 'li')
      .map((child) => {
        const checkbox = Array.from(child.children).find((element): element is HTMLInputElement =>
          element instanceof HTMLInputElement && element.type === 'checkbox')
        if (tag === 'ol' && child.hasAttribute('value')) {
          const value = Number.parseInt(child.getAttribute('value') || '', 10)
          if (Number.isFinite(value)) ordinal = value
        }
        const prefix = tag === 'ol' ? `${ordinal++}. ` : checkbox ? `- [${checkbox.checked ? 'x' : ' '}] ` : '- '
        const body = Array.from(child.childNodes).filter((part) =>
          !(part instanceof HTMLElement && (part.tagName === 'UL' || part.tagName === 'OL' || part === checkbox)))
          .map(inlineNodeToMarkdown).join('').trim()
        const nested = Array.from(child.children).filter((element) => element.tagName === 'UL' || element.tagName === 'OL')
          .map(blockNodeToMarkdown).map((text) => text.split('\n').map((line) => `    ${line}`).join('\n')).join('\n')
        return `${prefix}${body}${nested ? `\n${nested}` : ''}`.trimEnd()
      })
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

/** Characters of text before the selection's end inside `root`, or null when the selection is elsewhere. */
export const getEditableCaretOffset = (root: HTMLElement): number | null => {
  const selection = root.ownerDocument.defaultView?.getSelection()
  if (!selection || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  if (!root.contains(range.endContainer)) return null
  const prefix = root.ownerDocument.createRange()
  prefix.selectNodeContents(root)
  prefix.setEnd(range.endContainer, range.endOffset)
  return prefix.toString().length
}

/** Collapse the selection `offset` characters into `root`'s text, or at its end if the text is shorter. */
export const setEditableCaretOffset = (root: HTMLElement, offset: number): void => {
  const ownerDocument = root.ownerDocument
  const selection = ownerDocument.defaultView?.getSelection()
  if (!selection) return
  const range = ownerDocument.createRange()
  range.selectNodeContents(root)
  range.collapse(false)
  const walker = ownerDocument.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let remaining = Math.max(0, offset)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const length = node.textContent?.length ?? 0
    if (remaining <= length) {
      range.setStart(node, remaining)
      range.collapse(true)
      break
    }
    remaining -= length
  }
  selection.removeAllRanges()
  selection.addRange(range)
}

/**
 * Write an external revision into the uncontrolled WYSIWYG editor (NE-01).
 * Never call this for the user's own input: replacing the nodes moves the caret.
 * When the editor has focus the caret keeps its text offset, so reloading the
 * same text (for example after a new note's first autosave) does not throw it
 * back to the start of the document.
 */
export const replaceEditableHtml = (root: HTMLElement, html: string): void => {
  const caretOffset =
    root.ownerDocument.activeElement === root ? getEditableCaretOffset(root) : null
  root.innerHTML = html
  if (caretOffset != null) setEditableCaretOffset(root, caretOffset)
}

const WYSIWYG_BLOCK_TAGS = new Set(['P', 'DIV', 'PRE', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'])

/** The block (paragraph, heading, list item…) holding the selection's start inside `root`, or null. */
export const getEditableSelectionBlock = (root: HTMLElement): HTMLElement | null => {
  const selection = root.ownerDocument.defaultView?.getSelection()
  if (!selection || selection.rangeCount === 0) return null
  let node: Node | null = selection.getRangeAt(0).startContainer
  if (!root.contains(node)) return null
  while (node && node !== root) {
    if (node instanceof HTMLElement && WYSIWYG_BLOCK_TAGS.has(node.tagName)) return node
    node = node.parentNode
  }
  return null
}

export const isHeadingElement = (element: Element | null): boolean =>
  Boolean(element && /^H[1-6]$/.test(element.tagName))

/** Checkbox checked state is a property; persist it before recording innerHTML. */
export const syncEditableCheckboxes = (root: HTMLElement): void => {
  for (const checkbox of root.querySelectorAll<HTMLInputElement>(
    'input[type="checkbox"]'
  )) {
    checkbox.toggleAttribute("checked", checkbox.checked)
  }
}

/** Add task controls only to list items intersecting the editor's selection. */
const selectedEditableListItems = (root: HTMLElement): HTMLLIElement[] => {
  const selection = root.ownerDocument.defaultView?.getSelection()
  if (!selection || selection.rangeCount === 0) return []
  const range = selection.getRangeAt(0)
  if (!root.contains(range.commonAncestorContainer)) return []
  if (range.collapsed) {
    const node =
      range.startContainer instanceof Element
        ? range.startContainer
        : range.startContainer.parentElement
    const item = node?.closest("li")
    return item && root.contains(item) ? [item] : []
  }
  return Array.from(root.querySelectorAll("li")).filter((item) =>
    Array.from(item.childNodes).some((body) => {
      if (
        body instanceof Element &&
        (body.tagName === "UL" || body.tagName === "OL")
      )
        return false
      const walker = root.ownerDocument.createTreeWalker(
        body,
        NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT
      )
      let content: Node | null = body
      do {
        if (content.nodeType === Node.TEXT_NODE) {
          const contentRange = root.ownerDocument.createRange()
          contentRange.selectNodeContents(content)
          if (
            !contentRange.collapsed &&
            range.compareBoundaryPoints(Range.START_TO_END, contentRange) > 0 &&
            range.compareBoundaryPoints(Range.END_TO_START, contentRange) < 0
          )
            return true
        } else if (
          content instanceof Element &&
          content.childNodes.length === 0 &&
          range.intersectsNode(content)
        ) {
          return true
        }
        content = walker.nextNode()
      } while (content)
      return false
    })
  )
}

export const makeEditableChecklist = (root: HTMLElement): void => {
  for (const item of selectedEditableListItems(root)) {
    if (
      Array.from(item.children).some(
        (child) =>
          child instanceof HTMLInputElement && child.type === "checkbox"
      )
    )
      continue
    const checkbox = root.ownerDocument.createElement("input")
    checkbox.type = "checkbox"
    checkbox.contentEditable = "false"
    checkbox.setAttribute("aria-label", item.textContent?.trim() || "Task")
    item.prepend(checkbox)
  }
}

export const removeEditableTaskControls = (root: HTMLElement): void => {
  for (const item of selectedEditableListItems(root)) {
    for (const child of Array.from(item.children)) {
      if (child instanceof HTMLInputElement && child.type === "checkbox")
        child.remove()
    }
  }
}

/** Split a rich task in place; native ordinary lists retain browser Enter behavior. */
export const continueEditableChecklist = (root: HTMLElement): boolean => {
  const selection = root.ownerDocument.defaultView?.getSelection()
  if (!selection || selection.rangeCount === 0 || !selection.isCollapsed)
    return false
  const range = selection.getRangeAt(0)
  if (!root.contains(range.startContainer)) return false
  const block =
    range.startContainer instanceof Element
      ? range.startContainer
      : range.startContainer.parentElement
  const item = block?.closest("li")
  if (!item || !root.contains(item) || item.parentElement?.tagName !== "UL")
    return false
  const checkbox = Array.from(item.children).find(
    (child): child is HTMLInputElement =>
      child instanceof HTMLInputElement && child.type === "checkbox"
  )
  if (!checkbox) return false
  const ownerDocument = root.ownerDocument
  let target: HTMLElement
  if (!item.textContent?.trim() && !item.querySelector("ul,ol")) {
    const list = item.parentElement
    const after = ownerDocument.createElement("ul")
    while (item.nextSibling) after.append(item.nextSibling)
    target = ownerDocument.createElement("p")
    target.append(ownerDocument.createElement("br"))
    list.after(target)
    if (after.childNodes.length) target.after(after)
    item.remove()
    if (!list.childNodes.length) list.remove()
  } else {
    const suffix = range.cloneRange()
    suffix.setEnd(item, item.childNodes.length)
    const trailing = suffix.extractContents()
    target = ownerDocument.createElement("li")
    const nextCheckbox = ownerDocument.createElement("input")
    nextCheckbox.type = "checkbox"
    nextCheckbox.contentEditable = "false"
    nextCheckbox.setAttribute(
      "aria-label",
      trailing.textContent?.trim() || "Task"
    )
    target.append(nextCheckbox, trailing)
    if (target.childNodes.length === 1)
      target.append(ownerDocument.createElement("br"))
    item.after(target)
  }
  const caret = ownerDocument.createRange()
  caret.selectNodeContents(target)
  caret.collapse(true)
  if (target.firstChild instanceof HTMLInputElement)
    caret.setStartAfter(target.firstChild)
  selection.removeAllRanges()
  selection.addRange(caret)
  return true
}

/**
 * Chrome's insertUnorderedList builds the list inside the paragraph or heading
 * it came from (`<p><ul><li>…</li></ul></p>`). That is invalid HTML, shows no
 * list and converts to Markdown without the "- ". Move such lists out to the
 * top level, splitting the block around them, and keep the selection in place.
 */
export const unwrapListsFromBlocks = (root: HTMLElement): void => {
  const nested = Array.from(root.querySelectorAll('ul, ol')).filter((list) => {
    const parent = list.parentElement
    if (!parent || parent === root) return false
    return isHeadingElement(parent) || parent.tagName === 'P' ||
      (parent.tagName === 'DIV' && parent.parentElement === root)
  })
  if (nested.length === 0) return
  const selection = root.ownerDocument.defaultView?.getSelection()
  const range = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null
  const saved = range && root.contains(range.startContainer)
    ? {
        startContainer: range.startContainer,
        startOffset: range.startOffset,
        endContainer: range.endContainer,
        endOffset: range.endOffset
      }
    : null
  for (const list of nested) {
    const parent = list.parentElement
    if (!parent) continue
    const tail = parent.cloneNode(false) as HTMLElement
    while (list.nextSibling) tail.appendChild(list.nextSibling)
    parent.after(list)
    if (tail.textContent?.trim()) list.after(tail)
    if (!parent.textContent?.trim() && !parent.querySelector('img')) parent.remove()
  }
  if (saved && selection && root.contains(saved.startContainer) && root.contains(saved.endContainer)) {
    const clamp = (node: Node, offset: number) =>
      Math.min(offset, node.nodeType === Node.TEXT_NODE ? (node.textContent?.length ?? 0) : node.childNodes.length)
    const restored = root.ownerDocument.createRange()
    restored.setStart(saved.startContainer, clamp(saved.startContainer, saved.startOffset))
    restored.setEnd(saved.endContainer, clamp(saved.endContainer, saved.endOffset))
    selection.removeAllRanges()
    selection.addRange(restored)
  }
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

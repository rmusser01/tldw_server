import React from 'react'
import { Button } from 'antd'
import type { MessageInstance } from 'antd/es/message/interface'
import { bgRequest } from '@/services/background-proxy'
import { tldwAuth } from '@/services/tldw/TldwAuth'
import type { ServicePromptTargetConfig, TldwConfig } from '@/services/tldw/TldwApiClient'
import { deriveSingleUserApiKeyCredentialScope } from '@/services/chat-surface-scope'
import { useAntdNotification } from '@/hooks/useAntdNotification'
import { useUndoNotification } from '@/hooks/useUndoNotification'
import { normalizeWikilinkTitle } from '@/components/Notes/wikilinks'
import { createNotesGraphAuthorityScope } from './useNotesGraphAuthorityScope'

/**
 * Offer to update `[[Old title]]` links after a note is renamed (#3110).
 *
 * Links follow titles, so a rename leaves links to the old title unresolved.
 * Nothing is rewritten on its own: once the renamed note has saved, this asks
 * the server how many other notes link to the old title and, if any do, shows
 * a non-blocking prompt. Confirming rewrites the links and shows a result
 * toast with Undo. Dismissing leaves the links as they are, and the same
 * rename is not offered again during this visit.
 */

export type NoteRenamedEvent = {
  noteId: string
  oldTitle: string
  newTitle: string
  /** The verified notes owner the rename was saved for. */
  authorityScope: string
}

export interface UseNotesWikilinkRenameDeps {
  /** Server calls run only while online and for a verified notes owner. */
  isOnline: boolean
  /** Null while the session is being re-checked; that is not a change of owner. */
  authorityScope: string | null | undefined
  connectionConfig: TldwConfig | null | undefined
  message: MessageInstance
  t: (key: string, opts?: Record<string, unknown>) => string
  /** The note open in the editor. Its unsaved edits are never rewritten under it. */
  selectedId: string | number | null
  isDirty: boolean
  /** Whether a note has edits queued offline. Such a note is held back too. */
  hasQueuedDraft?: (noteId: string) => boolean
  /** Reload the open note after its saved text changed on the server. */
  reloadSelectedNote: () => unknown
  /** Refresh the list and link panels after notes were rewritten. */
  onLinksChanged: () => void
}

type LinkingNote = { id: string; title: string; version: number }
type ReferrersPage = { count: number; notes: LinkingNote[]; nextAfterNoteId: string | null }
type TokenReplacement = { token_index: number; original: string }
/** An updated note, with what Undo needs: its new version, the link written, and what it replaced. */
type RewrittenNote = LinkingNote & { replacement: string; replacements: TokenReplacement[] }
type SkipReason =
  | 'skipped_conflict'
  | 'skipped_no_match'
  | 'skipped_not_found'
  | 'skipped_resolved'
  | 'failed'
  | 'unsaved'
type SkippedNote = { id: string; title: string; reason: SkipReason }
type RewriteBatch = {
  updated: RewrittenNote[]
  skipped: SkippedNote[]
  linkForm: string
  newTitle: string
  newTitleShared: boolean
}
/** An open offer. `closedByCode` tells a programmatic close from the user dismissing it. */
type OpenPrompt = { closedByCode: boolean }

const REFERRERS_PATH = '/api/v1/notes/wikilinks/referrers'
const REWRITE_PATH = '/api/v1/notes/wikilinks/rewrite'
const UNDO_PATH = '/api/v1/notes/wikilinks/rewrite/undo'
/** The server accepts at most this many notes per rewrite or undo request. */
const MAX_NOTES_PER_REQUEST = 200
/** Referrer pages one confirmation follows: 10,000 linking notes. */
const MAX_REFERRER_PAGES = 50
const MAX_SKIPPED_NAMES = 5
const RESULT_TOAST_SECONDS = 12
const SKIP_REASONS: readonly string[] = [
  'skipped_conflict',
  'skipped_no_match',
  'skipped_not_found',
  'skipped_resolved',
  'failed'
]
/** Statuses with which the server refused the rewrite itself; asking again would not help. */
const REFUSAL_STATUSES: readonly number[] = [400, 404, 422]

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : null

const parseReferrers = (payload: unknown): ReferrersPage => {
  const body = asRecord(payload)
  const notes: LinkingNote[] = []
  for (const item of Array.isArray(body?.notes) ? body.notes : []) {
    const note = asRecord(item)
    const id = typeof note?.id === 'string' ? note.id : ''
    const version = Number(note?.version)
    if (!id || !Number.isInteger(version) || version < 1) continue
    notes.push({ id, title: typeof note?.title === 'string' ? note.title : '', version })
  }
  const cursor = body?.next_after_note_id
  return {
    count: Number(body?.count) || 0,
    notes,
    nextAfterNoteId: typeof cursor === 'string' && cursor ? cursor : null
  }
}

const toSkipReason = (status: unknown): SkipReason =>
  typeof status === 'string' && SKIP_REASONS.includes(status) ? (status as SkipReason) : 'failed'

const parseReplacements = (value: unknown): TokenReplacement[] => {
  const replacements: TokenReplacement[] = []
  for (const item of Array.isArray(value) ? value : []) {
    const entry = asRecord(item)
    if (typeof entry?.token_index !== 'number' || typeof entry.original !== 'string') continue
    replacements.push({ token_index: entry.token_index, original: entry.original })
  }
  return replacements
}

const parseRewrite = (payload: unknown, sent: LinkingNote[]): RewriteBatch => {
  const body = asRecord(payload)
  if (!body || typeof body.replacement !== 'string' || !Array.isArray(body.results)) {
    throw new Error('The server returned an unexpected response.')
  }
  const replacement = body.replacement
  const sentById = new Map(sent.map((note) => [note.id, note]))
  const updated: RewrittenNote[] = []
  const skipped: SkippedNote[] = []
  for (const item of body.results) {
    const result = asRecord(item)
    const id = typeof result?.id === 'string' ? result.id : ''
    if (!id) continue
    const title = typeof result?.title === 'string' ? result.title : sentById.get(id)?.title || ''
    const version = Number(result?.version)
    const replacements = parseReplacements(result?.replacements)
    // Without a version and its replacements a note could not be undone, so it is not counted as updated.
    if (result?.status === 'updated' && Number.isInteger(version) && replacements.length > 0) {
      updated.push({ id, title, version, replacement, replacements })
    } else {
      skipped.push({ id, title, reason: toSkipReason(result?.status) })
    }
  }
  return {
    updated,
    skipped,
    linkForm: typeof body.link_form === 'string' ? body.link_form : 'title',
    newTitle: typeof body.new_title === 'string' ? body.new_title : '',
    newTitleShared: body.new_title_shared === true
  }
}

const parseUndo = (
  payload: unknown,
  sent: RewrittenNote[]
): { restoredIds: string[]; skipped: SkippedNote[] } => {
  const body = asRecord(payload)
  if (!body || !Array.isArray(body.results)) {
    throw new Error('The server returned an unexpected response.')
  }
  const sentById = new Map(sent.map((note) => [note.id, note]))
  const restoredIds: string[] = []
  const skipped: SkippedNote[] = []
  for (const item of body.results) {
    const result = asRecord(item)
    const id = typeof result?.id === 'string' ? result.id : ''
    if (!id) continue
    if (result?.status === 'restored') {
      restoredIds.push(id)
      continue
    }
    const title = typeof result?.title === 'string' ? result.title : sentById.get(id)?.title || ''
    skipped.push({ id, title, reason: toSkipReason(result?.status) })
  }
  return { restoredIds, skipped }
}

const errorText = (error: unknown): string =>
  error instanceof Error ? error.message : String(error || '')

const isRefusal = (error: unknown): boolean =>
  REFUSAL_STATUSES.includes(Number(asRecord(error)?.status))

/** Each rewrite request decides the link text again, so Undo sends notes back grouped by it. */
const groupByReplacement = (notes: RewrittenNote[]): Array<[string, RewrittenNote[]]> => {
  const groups = new Map<string, RewrittenNote[]>()
  for (const note of notes) {
    const group = groups.get(note.replacement)
    if (group) group.push(note)
    else groups.set(note.replacement, [note])
  }
  return Array.from(groups.entries())
}

export function useNotesWikilinkRename(deps: UseNotesWikilinkRenameDeps) {
  const notification = useAntdNotification()
  const { showUndoNotification } = useUndoNotification()
  // Prompt and toast buttons outlive the render that opened them, so their
  // handlers read the latest values through refs.
  const depsRef = React.useRef(deps)
  depsRef.current = deps
  const notificationRef = React.useRef(notification)
  notificationRef.current = notification
  const showUndoNotificationRef = React.useRef(showUndoNotification)
  showUndoNotificationRef.current = showUndoNotification
  /** Renames the user dismissed: not offered again during this visit. */
  const dismissedRenamesRef = React.useRef(new Set<string>())
  const openPromptsRef = React.useRef(new Map<string, OpenPrompt>())
  // True while the notes page is shown. Counting the linking notes takes
  // requests, and a rename can be saved as the page goes away (the leave
  // flush). An offer whose count answers after the page was left would open
  // on the next page, where nothing closes it, so none opens then.
  const pageShownRef = React.useRef(false)
  // The scope goes null while a session is re-checked (for example when the
  // window regains focus) and then comes back unchanged. Only a different
  // owner, a logout, or leaving the page ends an offer.
  const lastVerifiedScopeRef = React.useRef<string | null>(deps.authorityScope ?? null)

  const scopeIsCurrent = React.useCallback(
    (scope: string): boolean =>
      (depsRef.current.authorityScope ?? lastVerifiedScopeRef.current) === scope,
    []
  )

  const closePrompts = React.useCallback(() => {
    for (const [key, prompt] of openPromptsRef.current) {
      prompt.closedByCode = true
      notificationRef.current.destroy(key)
    }
    openPromptsRef.current.clear()
  }, [])

  /**
   * POST as the verified notes owner, like saving a note does. `stillWanted`
   * is checked once the owner is verified, before anything is sent.
   */
  const ownedRequest = React.useCallback(
    async (
      scope: string,
      path: `/${string}`,
      body: Record<string, unknown>,
      stillWanted?: () => boolean
    ): Promise<unknown> => {
      const ownerChanged = () =>
        new Error(
          depsRef.current.t('option:notesSearch.wikilinkRenameOwnerChanged', {
            defaultValue: 'The signed-in account changed. Reopen the note and try again.'
          })
        )
      const config = depsRef.current.connectionConfig ? { ...depsRef.current.connectionConfig } : null
      if (!config || !scopeIsCurrent(scope)) throw ownerChanged()
      const user = await tldwAuth.getCurrentUser()
      if (stillWanted && !stillWanted()) throw new Error('No longer needed')
      if (
        !user?.is_active ||
        user.id == null ||
        !scopeIsCurrent(scope) ||
        createNotesGraphAuthorityScope(config.serverUrl, user.id) !== scope
      ) {
        throw ownerChanged()
      }
      const servicePromptConfig: ServicePromptTargetConfig = {
        serverUrl: config.serverUrl,
        authMode: config.authMode,
        authSource: config.authSource,
        orgId: config.orgId,
        expectedUserId: user.id,
        expectedSingleUserApiKeyScope:
          config.authMode === 'single-user'
            ? deriveSingleUserApiKeyCredentialScope('single-user', config.apiKey)
            : undefined
      }
      return bgRequest<unknown>({
        path,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-TLDW-Expected-User-ID': String(user.id)
        },
        body,
        servicePromptConfig
      })
    },
    [scopeIsCurrent]
  )

  const describeSkipped = React.useCallback((skipped: SkippedNote[]): string => {
    const { t } = depsRef.current
    const reasonLabel = (reason: SkipReason): string => {
      switch (reason) {
        case 'skipped_conflict':
          return t('option:notesSearch.wikilinkRenameSkipEdited', { defaultValue: 'edited since' })
        case 'unsaved':
          return t('option:notesSearch.wikilinkRenameSkipUnsaved', {
            defaultValue: 'has unsaved changes'
          })
        case 'skipped_not_found':
          return t('option:notesSearch.wikilinkRenameSkipMissing', {
            defaultValue: 'no longer available'
          })
        case 'skipped_no_match':
          return t('option:notesSearch.wikilinkRenameSkipNoLink', {
            defaultValue: 'link already changed'
          })
        case 'skipped_resolved':
          return t('option:notesSearch.wikilinkRenameSkipResolved', {
            defaultValue: 'link now opens another note'
          })
        default:
          return t('option:notesSearch.wikilinkRenameSkipFailed', {
            defaultValue: 'could not be updated'
          })
      }
    }
    const named = skipped
      .slice(0, MAX_SKIPPED_NAMES)
      .map((note) => `${note.title || note.id} (${reasonLabel(note.reason)})`)
    const more = skipped.length - named.length
    if (more > 0) {
      named.push(
        t('option:notesSearch.wikilinkRenameSkipMore', {
          defaultValue: '+{{count}} more',
          count: more
        })
      )
    }
    return named.join(', ')
  }, [])

  /**
   * Hold back notes with local edits the server has not seen: the open note
   * while it is dirty, and notes with a draft queued offline. Rewriting them
   * on the server would turn the user's next save into a conflict.
   */
  const splitNotesWithLocalEdits = React.useCallback(
    <T extends LinkingNote>(notes: T[]): { held: T[]; rest: T[] } => {
      const { selectedId, isDirty, hasQueuedDraft } = depsRef.current
      const openId = isDirty && selectedId != null ? String(selectedId) : null
      const hasLocalEdits = (note: T) => note.id === openId || hasQueuedDraft?.(note.id) === true
      return {
        held: notes.filter(hasLocalEdits),
        rest: notes.filter((note) => !hasLocalEdits(note))
      }
    },
    []
  )

  const afterLinksChanged = React.useCallback((changedIds: string[]) => {
    const { selectedId, isDirty, onLinksChanged, reloadSelectedNote } = depsRef.current
    onLinksChanged()
    if (selectedId != null && !isDirty && changedIds.includes(String(selectedId))) {
      void reloadSelectedNote()
    }
  }, [])

  const undoRewrite = React.useCallback(
    async (event: NoteRenamedEvent, rewritten: RewrittenNote[]) => {
      const { t, message } = depsRef.current
      const restoredIds: string[] = []
      const { held, rest } = splitNotesWithLocalEdits(rewritten)
      const skipped: SkippedNote[] = held.map((note) => ({
        id: note.id,
        title: note.title,
        reason: 'unsaved'
      }))
      try {
        for (const [replacement, notes] of groupByReplacement(rest)) {
          for (let offset = 0; offset < notes.length; offset += MAX_NOTES_PER_REQUEST) {
            const chunk = notes.slice(offset, offset + MAX_NOTES_PER_REQUEST)
            const result = parseUndo(
              await ownedRequest(event.authorityScope, UNDO_PATH, {
                old_title: event.oldTitle,
                replacement,
                notes: chunk.map((note) => ({
                  id: note.id,
                  expected_version: note.version,
                  replacements: note.replacements
                }))
              }),
              chunk
            )
            restoredIds.push(...result.restoredIds)
            skipped.push(...result.skipped)
          }
        }
      } finally {
        if (restoredIds.length > 0) afterLinksChanged(restoredIds)
      }
      if (skipped.length === 0) return
      const notRestored = t('option:notesSearch.wikilinkRenameUndoSkipped', {
        defaultValue: 'Links were not restored in: {{notes}}',
        notes: describeSkipped(skipped)
      })
      // Throwing makes the undo toast report a failure instead of "Restored".
      if (restoredIds.length === 0) throw new Error(notRestored)
      message.warning(notRestored)
    },
    [afterLinksChanged, describeSkipped, ownedRequest, splitNotesWithLocalEdits]
  )

  // `offer` opens a prompt, whose confirm runs the rewrite, which may offer
  // again. The ref breaks that cycle.
  const offerRef = React.useRef<(event: NoteRenamedEvent, renameKey: string) => Promise<void>>(
    async () => {}
  )

  const runRewrite = React.useCallback(
    async (event: NoteRenamedEvent, renameKey: string, firstPage: ReferrersPage) => {
      const { t, message } = depsRef.current
      const scope = event.authorityScope
      const updated: RewrittenNote[] = []
      const skipped: SkippedNote[] = []
      let last: RewriteBatch | null = null
      let failure: string | null = null
      let refused = false
      try {
        let page = firstPage
        for (let pageIndex = 0; pageIndex < MAX_REFERRER_PAGES; pageIndex += 1) {
          const { held, rest } = splitNotesWithLocalEdits(page.notes)
          skipped.push(...held.map((note) => ({ id: note.id, title: note.title, reason: 'unsaved' as const })))
          if (rest.length > 0) {
            last = parseRewrite(
              await ownedRequest(scope, REWRITE_PATH, {
                note_id: event.noteId,
                old_title: event.oldTitle,
                notes: rest.map((note) => ({ id: note.id, expected_version: note.version }))
              }),
              rest
            )
            updated.push(...last.updated)
            skipped.push(...last.skipped)
          }
          if (!page.nextAfterNoteId) break
          page = parseReferrers(
            await ownedRequest(scope, REFERRERS_PATH, {
              title: event.oldTitle,
              exclude_note_id: event.noteId,
              unresolved_only: true,
              after_note_id: page.nextAfterNoteId
            })
          )
          if (page.notes.length === 0) break
        }
      } catch (error) {
        failure = errorText(error)
        refused = isRefusal(error)
      }

      const skippedText =
        skipped.length > 0
          ? t('option:notesSearch.wikilinkRenameSkipped', {
              defaultValue: 'Skipped: {{notes}}.',
              notes: describeSkipped(skipped)
            })
          : ''
      if (updated.length === 0 || !last) {
        if (failure) {
          message.error(
            t('option:notesSearch.wikilinkRenameFailed', {
              defaultValue: 'Could not update links. {{error}}',
              error: failure
            })
          )
        } else {
          message.warning(
            [
              t('option:notesSearch.wikilinkRenameNoneUpdated', {
                defaultValue: 'No links were updated.'
              }),
              skippedText
            ]
              .filter(Boolean)
              .join(' ')
          )
        }
      } else {
        const newTitle = last.newTitle || event.newTitle
        let linkFormText = ''
        if (last.linkForm === 'id') {
          linkFormText = last.newTitleShared
            ? t('option:notesSearch.wikilinkRenameLinkedById', {
                defaultValue: 'Another note is also titled "{{title}}", so the links use this note\'s id.',
                title: newTitle
              })
            : t('option:notesSearch.wikilinkRenameLinkedByIdUnlinkable', {
                defaultValue: '"{{title}}" can\'t be written as a [[link]], so the links use this note\'s id.',
                title: newTitle
              })
        }
        const incompleteText = failure
          ? t('option:notesSearch.wikilinkRenameIncomplete', {
              defaultValue: 'The update stopped early: {{error}}',
              error: failure
            })
          : ''
        afterLinksChanged(updated.map((note) => note.id))
        showUndoNotificationRef.current({
          title:
            updated.length === 1
              ? t('option:notesSearch.wikilinkRenameUpdatedOne', {
                  defaultValue: 'Updated links in 1 note'
                })
              : t('option:notesSearch.wikilinkRenameUpdatedMany', {
                  defaultValue: 'Updated links in {{count}} notes',
                  count: updated.length
                }),
          description: [linkFormText, skippedText, incompleteText].filter(Boolean).join(' ') || undefined,
          duration: RESULT_TOAST_SECONDS,
          onUndo: () => undoRewrite(event, updated)
        })
      }

      // Notes left behind for a reason that can pass (a failed request, an
      // edit since the count, unsaved changes) get a fresh offer, so the user
      // can try them again with current versions.
      const retryable =
        (failure !== null && !refused) ||
        skipped.some((note) => note.reason === 'skipped_conflict' || note.reason === 'unsaved')
      if (retryable) void offerRef.current(event, renameKey)
    },
    [afterLinksChanged, describeSkipped, ownedRequest, splitNotesWithLocalEdits, undoRewrite]
  )

  const openPrompt = React.useCallback(
    (event: NoteRenamedEvent, renameKey: string, firstPage: ReferrersPage) => {
      if (!pageShownRef.current) return
      const { t } = depsRef.current
      const key = `notes-wikilink-rename:${renameKey}`
      const one = firstPage.count === 1
      const prompt: OpenPrompt = { closedByCode: false }
      const confirm = () => {
        if (prompt.closedByCode) return
        prompt.closedByCode = true
        if (openPromptsRef.current.get(key) === prompt) openPromptsRef.current.delete(key)
        notificationRef.current.destroy(key)
        void runRewrite(event, renameKey, firstPage)
      }
      // Opening with the same key replaces the notice: the old one was not dismissed.
      const replaced = openPromptsRef.current.get(key)
      if (replaced) replaced.closedByCode = true
      openPromptsRef.current.set(key, prompt)
      notificationRef.current.open({
        key,
        message: (
          <span className="font-medium text-text" data-testid="notes-wikilink-rename-prompt">
            {one
              ? t('option:notesSearch.wikilinkRenamePromptOne', {
                  defaultValue: '1 note links to "{{title}}"',
                  title: event.oldTitle
                })
              : t('option:notesSearch.wikilinkRenamePromptMany', {
                  defaultValue: '{{count}} notes link to "{{title}}"',
                  count: firstPage.count,
                  title: event.oldTitle
                })}
          </span>
        ),
        description: (
          <span className="text-text-muted">
            {one
              ? t('option:notesSearch.wikilinkRenamePromptDescriptionOne', {
                  defaultValue:
                    'That link no longer opens this note. Update it to the new title, or dismiss to leave it as it is.'
                })
              : t('option:notesSearch.wikilinkRenamePromptDescriptionMany', {
                  defaultValue:
                    'Those links no longer open this note. Update them to the new title, or dismiss to leave them as they are.'
                })}
          </span>
        ),
        icon: null,
        // The offer stays until the user answers it; it never blocks the editor.
        duration: 0,
        placement: 'bottomRight',
        actions: (
          <Button
            type="primary"
            size="small"
            onClick={confirm}
            data-testid="notes-wikilink-rename-confirm"
          >
            {one
              ? t('option:notesSearch.wikilinkRenamePromptActionOne', { defaultValue: 'Update link' })
              : t('option:notesSearch.wikilinkRenamePromptActionMany', { defaultValue: 'Update links' })}
          </Button>
        ),
        onClose: () => {
          if (prompt.closedByCode) return
          // Dismissed by the user: the links stay unresolved, and this rename is not offered again.
          prompt.closedByCode = true
          if (openPromptsRef.current.get(key) === prompt) openPromptsRef.current.delete(key)
          dismissedRenamesRef.current.add(renameKey)
        }
      })
    },
    [runRewrite]
  )

  const offer = React.useCallback(
    async (event: NoteRenamedEvent, renameKey: string) => {
      // Also reached by a retry after a rewrite the user confirmed before leaving.
      if (!pageShownRef.current) return
      let page: ReferrersPage
      try {
        page = parseReferrers(
          await ownedRequest(
            event.authorityScope,
            REFERRERS_PATH,
            {
              title: event.oldTitle,
              exclude_note_id: event.noteId,
              unresolved_only: true
            },
            () => pageShownRef.current
          )
        )
      } catch (error) {
        // The offer is optional: without a count the links simply stay as they are.
        console.debug('[NotesManagerPage] Wikilink referrer count failed:', error)
        return
      }
      if (
        !pageShownRef.current ||
        !scopeIsCurrent(event.authorityScope) ||
        dismissedRenamesRef.current.has(renameKey)
      )
        return
      if (page.count <= 0 || page.notes.length === 0) return
      openPrompt(event, renameKey, page)
    },
    [openPrompt, ownedRequest, scopeIsCurrent]
  )
  offerRef.current = offer

  const handleNoteRenamed = React.useCallback(
    async (event: NoteRenamedEvent) => {
      const oldKey = normalizeWikilinkTitle(event.oldTitle)
      const newKey = normalizeWikilinkTitle(event.newTitle)
      // Links match titles ignoring case and spacing, so such a rename breaks nothing.
      if (!event.noteId || !oldKey || !newKey || oldKey === newKey) return
      // A save that answers after the page was left (its unmount or leave flush).
      if (!pageShownRef.current) return
      if (!depsRef.current.isOnline || !event.authorityScope || !scopeIsCurrent(event.authorityScope)) return
      const renameKey = JSON.stringify([event.authorityScope, event.noteId, oldKey, newKey])
      if (dismissedRenamesRef.current.has(renameKey)) return
      await offer(event, renameKey)
    },
    [offer, scopeIsCurrent]
  )

  // A different verified owner ends the open offers. A scope that goes null
  // and comes back unchanged is only a session re-check.
  React.useEffect(() => {
    const scope = deps.authorityScope
    if (!scope) return
    const previous = lastVerifiedScopeRef.current
    lastVerifiedScopeRef.current = scope
    if (previous && previous !== scope) closePrompts()
  }, [closePrompts, deps.authorityScope])

  // So do a logout and leaving the page. Leaving also stops offers that are
  // still being counted (see pageShownRef).
  React.useEffect(() => {
    pageShownRef.current = true
    const onPrincipalChanged = (event: Event) => {
      if ((event as CustomEvent<{ kind?: string }>).detail?.kind !== 'logout') return
      lastVerifiedScopeRef.current = null
      closePrompts()
    }
    window.addEventListener('tldw:auth-principal-changed', onPrincipalChanged)
    return () => {
      pageShownRef.current = false
      window.removeEventListener('tldw:auth-principal-changed', onPrincipalChanged)
      closePrompts()
    }
  }, [closePrompts])

  return { handleNoteRenamed }
}

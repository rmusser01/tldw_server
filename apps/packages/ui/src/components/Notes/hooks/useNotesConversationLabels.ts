import React from 'react'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from '@/services/service-prompts'
import { toConversationLabel } from '../notes-manager-utils'
import { settleInBatches } from '../notes-conversation-labels'

/**
 * Chat titles for the notes list and the open note's backlink (NL-04, #3101).
 *
 * Titles belong to one account: every lookup runs under the account's
 * request scope, a change of account (or of the chats to look up) aborts the
 * lookups in flight, and the titles, missing chats and retry counts of the
 * old account are dropped, so a title never shows for the wrong account.
 */

const CONVERSATION_LABEL_MAX_RETRIES = 3
const CONVERSATION_LABEL_RETRY_DELAY_MS = 1500

const isMissingConversationLookupError = (error: unknown): boolean => {
  if (!error) return false
  const status =
    typeof error === 'object' && 'status' in error ? Number((error as { status?: unknown }).status) : Number.NaN
  if (status === 404) return true
  const code =
    typeof error === 'object' && 'code' in error ? String((error as { code?: unknown }).code || '') : ''
  if (code.toUpperCase() === 'NOT_FOUND') return true
  const message = error instanceof Error ? error.message : String(error)
  return /\b(not found|404)\b/i.test(message)
}

type LabelState = { scope: string | null; labels: Record<string, string> }

const NO_LABELS: Record<string, string> = Object.freeze({}) as Record<string, string>

type Options = {
  /** The notes account; titles never outlive it. */
  authorityScope: string | null
  isOnline: boolean
  conversationIds: string[]
}

export function useNotesConversationLabels({ authorityScope, isOnline, conversationIds }: Options) {
  const [labelState, setLabelState] = React.useState<LabelState>({ scope: authorityScope, labels: {} })
  // Until the reset below runs, another account's titles are already hidden.
  const conversationLabelById = labelState.scope === authorityScope ? labelState.labels : NO_LABELS
  const labelsRef = React.useRef(conversationLabelById)
  labelsRef.current = conversationLabelById
  const missingIdsRef = React.useRef<Set<string>>(new Set())
  const retryAttemptsRef = React.useRef<Record<string, number>>({})
  const retryTimeoutRef = React.useRef<number | null>(null)
  const [retryTick, setRetryTick] = React.useState(0)
  const controllerRef = React.useRef<AbortController | null>(null)

  const clearRetryTimer = React.useCallback(() => {
    if (retryTimeoutRef.current != null) {
      window.clearTimeout(retryTimeoutRef.current)
      retryTimeoutRef.current = null
    }
  }, [])

  // A new account starts with no titles, no known-missing chats and no retries.
  React.useEffect(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    missingIdsRef.current = new Set()
    retryAttemptsRef.current = {}
    clearRetryTimer()
    setLabelState((current) => (current.scope === authorityScope ? current : { scope: authorityScope, labels: {} }))
  }, [authorityScope, clearRetryTimer])

  React.useEffect(
    () => () => {
      controllerRef.current?.abort()
      clearRetryTimer()
    },
    [clearRetryTimer]
  )

  /** A title the page learned another way (opening the chat). */
  const rememberConversationLabel = React.useCallback(
    (conversationId: string, label: string) => {
      setLabelState((current) => {
        const labels = current.scope === authorityScope ? current.labels : {}
        return { scope: authorityScope, labels: { ...labels, [conversationId]: label } }
      })
    },
    [authorityScope]
  )

  const idsKey = conversationIds.join('\n')

  React.useEffect(() => {
    // Retry counts only matter for chats still on screen.
    const active = new Set(idsKey ? idsKey.split('\n') : [])
    for (const id of Object.keys(retryAttemptsRef.current)) {
      if (!active.has(id)) delete retryAttemptsRef.current[id]
    }
    if (active.size === 0) clearRetryTimer()
  }, [clearRetryTimer, idsKey])

  React.useEffect(() => {
    if (!isOnline || !idsKey || authorityScope == null) return
    const pending = idsKey
      .split('\n')
      .filter((id) => id && !labelsRef.current[id] && !missingIdsRef.current.has(id))
    if (pending.length === 0) return
    const controller = new AbortController()
    controllerRef.current?.abort()
    controllerRef.current = controller
    const { signal } = controller
    const scope = authorityScope
    let snapshot: ServicePromptSnapshot | undefined

    const scheduleRetries = (failedIds: string[]) => {
      const retryable = failedIds.filter((id) => {
        const attempt = (retryAttemptsRef.current[id] ?? 0) + 1
        retryAttemptsRef.current[id] = attempt
        return attempt <= CONVERSATION_LABEL_MAX_RETRIES
      })
      if (retryable.length === 0) return
      clearRetryTimer()
      retryTimeoutRef.current = window.setTimeout(() => {
        retryTimeoutRef.current = null
        setRetryTick((tick) => tick + 1)
      }, CONVERSATION_LABEL_RETRY_DELAY_MS)
    }

    void (async () => {
      try {
        await tldwClient.initialize().catch(() => null)
        snapshot = await loadServicePromptSnapshot([], { signal })
        snapshot.scopeInvalidatedSignal.addEventListener('abort', () => controller.abort(), { once: true })
        if (signal.aborted) return
        const requestScope = snapshot.requestScope
        const settled = await settleInBatches(pending, async (conversationId) => {
          signal.throwIfAborted()
          const chat = await tldwClient.getChat(conversationId, { signal, requestScope })
          return { conversationId, label: toConversationLabel(chat) }
        })
        if (signal.aborted) return
        setLabelState((current) => {
          if (current.scope !== scope) return current
          let next: Record<string, string> | null = null
          for (const result of settled) {
            if (result.status !== 'fulfilled' || !result.value.label) continue
            const { conversationId, label } = result.value
            if (current.labels[conversationId]) continue
            if (next == null) next = { ...current.labels }
            next[conversationId] = label
          }
          return next ? { scope, labels: next } : current
        })
        const failed: string[] = []
        settled.forEach((result, index) => {
          const conversationId = pending[index]
          if (result.status === 'fulfilled') {
            delete retryAttemptsRef.current[conversationId]
          } else if (isMissingConversationLookupError(result.reason)) {
            missingIdsRef.current.add(conversationId)
            delete retryAttemptsRef.current[conversationId]
          } else {
            failed.push(conversationId)
          }
        })
        scheduleRetries(failed)
      } catch {
        // The account could not be confirmed: try again a few times.
        if (!signal.aborted) scheduleRetries(pending)
      } finally {
        snapshot?.release()
        if (controllerRef.current === controller) controllerRef.current = null
      }
    })()

    return () => controller.abort()
  }, [authorityScope, clearRetryTimer, idsKey, isOnline, retryTick])

  return { conversationLabelById, rememberConversationLabel }
}

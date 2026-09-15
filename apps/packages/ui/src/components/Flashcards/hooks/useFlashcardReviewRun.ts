import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { browser } from "wxt/browser"
import { mapFlashcardsUiError } from "../utils/error-taxonomy"
import { createSafeStorage } from "@/utils/safe-storage"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import type { FlashcardReviewContext, FlashcardReviewResponse, FlashcardReviewSessionSummary, FlashcardsRequestOptions } from "@/services/flashcards"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { REFRESH_ROTATION_KEY, REFRESH_SESSION_INVALIDATION_PREFIX } from "@/services/tldw/single-user-credential"
import { servicePromptPrincipalMatches, servicePromptSingleUserApiKeyScopeMatches, servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"

export const isInactiveFlashcardReviewSessionError = (error: unknown) => {
  const { status, rawMessage } = mapFlashcardsUiError(error, { operation: "reviewing a card", fallback: "" })
  return status === 404 && rawMessage === "Flashcard review session is not active"
}

type Rating = { cardUuid: string; rating: number; answerTimeMs?: number }
type Review = (input: Rating & { reviewContext: FlashcardReviewContext; reviewSessionId?: number; options: FlashcardsRequestOptions }) => Promise<FlashcardReviewResponse>
type End = (input: { reviewSessionId: number; options: FlashcardsRequestOptions }) => Promise<FlashcardReviewSessionSummary>
type Run = {
  key: string
  context: FlashcardReviewContext
  controller: AbortController
  lease: Promise<ServicePromptSnapshot>
  leaseFailed?: boolean
  snapshot?: ServicePromptSnapshot
  stopWatching?: () => void
  sessionId: number | null
  rating?: Promise<FlashcardReviewResponse | null>
  ending?: Promise<FlashcardReviewSessionSummary | null>
  closing: boolean
  completed: boolean
  inactive?: boolean
  review: Review
  end: End
}

/** A UI review scope owns its first acknowledged server session and request lease. */
export function useFlashcardReviewRun({ context, enabled, review, end, onCloseError }: {
  context: FlashcardReviewContext
  enabled: boolean
  review: Review
  end: End
  onCloseError: (error: unknown) => void
}) {
  const scopedContext = useMemo(() => ({ review_mode: context.review_mode, deck_id: context.deck_id, tag_filter: context.tag_filter ?? null }),
    [context.review_mode, context.deck_id, context.tag_filter])
  const key = JSON.stringify([enabled, scopedContext])
  const renderedKey = useRef(key)
  renderedKey.current = key
  const current = useRef<Run | null>(null)
  const runs = useRef(new Set<Run>())
  const callbacks = useRef({ review, end, onCloseError })
  callbacks.current = { review, end, onCloseError }
  const mounted = useRef(true)
  const [, redraw] = useState(0)
  const [authorityRevision, setAuthorityRevision] = useState(0)
  const notify = useCallback(() => { if (mounted.current) redraw(n => n + 1) }, [])
  const owns = useCallback((run: Run) => current.current === run && renderedKey.current === run.key && !run.controller.signal.aborted, [])
  const release = useCallback((run: Run) => {
    run.stopWatching?.()
    void run.lease?.then(snapshot => snapshot.release(), () => {})
    runs.current.delete(run)
  }, [])

  const finish = useCallback((run: Run): Promise<FlashcardReviewSessionSummary | null> => {
    if (run.ending) return run.ending
    if (run.completed || run.controller.signal.aborted) return Promise.resolve(null)
    run.closing = true
    const pending = (async () => {
      // The first rating may still be creating the session. Close only after its ID arrives.
      await run.rating?.catch(() => null)
      if (run.sessionId == null || run.controller.signal.aborted) return null
      const lease = await run.lease
      if (lease.scopeSignal.aborted) return null
      const result = await run.end({ reviewSessionId: run.sessionId, options: {
        signal: lease.scopeSignal, requestScope: lease.requestScope
      } })
      if (lease.scopeSignal.aborted) return null
      run.completed = true
      release(run)
      return result
    })()
    run.ending = pending
    notify()
    void pending.finally(() => {
      run.ending = undefined
      run.closing = false
      if (!owns(run)) release(run)
      notify()
    }).catch(() => {})
    return pending
  }, [notify, owns, release])

  const create = useCallback((runKey: string, runContext: FlashcardReviewContext) => {
    const controller = new AbortController()
    const run = {
      key: runKey, context: { ...runContext }, controller, sessionId: null,
      closing: false, completed: false, review: callbacks.current.review, end: callbacks.current.end
    } as Run
    runs.current.add(run)
    current.current = run
    const invalidated = () => {
      controller.abort()
      release(run)
      if (current.current === run) {
        current.current = null
        if (mounted.current) setAuthorityRevision(n => n + 1)
      }
      notify()
    }
    const storage = createSafeStorage({ area: "local" })
    let checkGeneration = 0
    let released = false
    const revalidate = async () => {
      const generation = ++checkGeneration
      const isCurrentCheck = () => !released && !controller.signal.aborted && generation === checkGeneration
      try {
        // Invalidation markers mask effective credentials without rewriting tldwConfig.
        // Treat events as hints: an old marker must not retire a newer login or rotation.
        const config = await tldwClient.ensureConfigForRequest(true)
        if (!isCurrentCheck()) return
        const cookieSession = config.authSource === "cookie-session"
        const expected = run.snapshot?.requestScope
        if ((!cookieSession && config.authMode === "multi-user" && !config.accessToken) ||
          (expected && (!servicePromptTargetsMatch(config, expected.config) ||
            (!cookieSession && (!servicePromptPrincipalMatches(config, expected.userId) ||
              !servicePromptSingleUserApiKeyScopeMatches(config, expected.config.expectedSingleUserApiKeyScope)))))) {
          invalidated()
        }
      } catch {
        if (isCurrentCheck()) invalidated()
      }
    }
    const revalidateHint = () => { void revalidate() }
    const isCredentialKey = (key: string) => key === "tldwConfig" || key === "tldwCookieSessionConfig" ||
      key === REFRESH_ROTATION_KEY || key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)
    const onNativeStorage = (event: StorageEvent) => {
      if (event.key === null || isCredentialKey(event.key)) revalidateHint()
    }
    const onExtensionStorage = (changes: Record<string, unknown>, area: string) => {
      if (area === "local" && Object.keys(changes).some(isCredentialKey)) revalidateHint()
    }
    const watchers = {
      // The verified lease watches target/principal changes once bound. Cover its lookup window too.
      tldwConfig: () => { if (!run.snapshot) invalidated(); else revalidateHint() },
      tldwCookieSessionConfig: invalidated,
      [REFRESH_ROTATION_KEY]: revalidateHint
    }
    storage.watch(watchers)
    window.addEventListener("tldw:config-updated", revalidateHint)
    window.addEventListener("storage", onNativeStorage)
    browser?.storage?.onChanged?.addListener(onExtensionStorage)
    run.stopWatching = () => {
      released = true
      checkGeneration++
      storage.unwatch(watchers)
      window.removeEventListener("tldw:config-updated", revalidateHint)
      window.removeEventListener("storage", onNativeStorage)
      browser?.storage?.onChanged?.removeListener(onExtensionStorage)
    }
    run.lease = loadServicePromptSnapshot([], { signal: controller.signal }).then(snapshot => {
      run.snapshot = snapshot
      snapshot.scopeInvalidatedSignal.addEventListener("abort", invalidated, { once: true })
      if (snapshot.scopeInvalidatedSignal.aborted) invalidated()
      if (controller.signal.aborted) snapshot.release()
      return snapshot
    })
    // The rating reports a genuine scope-resolution failure; mounting must not reject unhandled.
    void run.lease.catch(() => { run.leaseFailed = true })
    notify()
    return run
  }, [notify, release])

  useEffect(() => {
    mounted.current = true
    const invalidate = () => {
      for (const run of runs.current) {
        run.controller.abort()
        release(run)
      }
      current.current = null
      setAuthorityRevision(n => n + 1)
    }
    window.addEventListener("tldw:auth-credentials-changed", invalidate)
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    return () => {
      mounted.current = false
      window.removeEventListener("tldw:auth-credentials-changed", invalidate)
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
    }
  }, [release])

  useEffect(() => {
    const run = enabled ? create(key, scopedContext) : null
    return () => {
      if (!run) return
      // A manual End followed by more ratings creates another run in this same UI scope.
      const active = current.current?.key === key ? current.current : run
      if (current.current === active) current.current = null
      void finish(active).catch(error => {
        if (!active.controller.signal.aborted) callbacks.current.onCloseError(error)
      }).finally(() => release(active))
    }
  }, [key, scopedContext, enabled, authorityRevision, create, finish, release])

  const submit = useCallback(async (rating: Rating) => {
    if (!enabled) return null
    let run = current.current
    if (!run || !owns(run) || run.rating || run.closing || run.inactive) return null
    if (run.completed || run.leaseFailed) {
      run.controller.abort()
      release(run)
      run = create(key, scopedContext)
    }
    const owner = run
    const pending = (async () => {
      // A same-scope server session may be reused until its previous end completes.
      // Do not let that delayed end close a newly started UI run.
      await Promise.all([...runs.current]
        .filter(previous => previous !== owner && previous.key === owner.key && !previous.controller.signal.aborted)
        .map(previous => finish(previous)))
      const lease = await owner.lease
      if (!owns(owner) || lease.scopeSignal.aborted || owner.closing) return null
      const response = await owner.review({ ...rating, reviewContext: owner.context,
        ...(owner.sessionId == null ? {} : { reviewSessionId: owner.sessionId }),
        options: { signal: lease.scopeSignal, requestScope: lease.requestScope }
      })
      if (lease.scopeSignal.aborted) return null
      if (typeof response.review_session_id === "number") owner.sessionId ??= response.review_session_id
      return response
    })()
    owner.rating = pending
    notify()
    try {
      const result = await pending
      return owns(owner) ? result : null
    } catch (error) {
      if (owns(owner)) {
        if (owner.sessionId != null && isInactiveFlashcardReviewSessionError(error)) owner.inactive = true
        throw error
      }
      return null
    } finally {
      owner.rating = undefined
      notify()
    }
  }, [scopedContext, create, enabled, finish, key, notify, owns, release])

  const restart = useCallback(() => {
    const run = current.current
    if (!run || !owns(run) || !run.inactive || run.rating || run.ending) return false
    run.controller.abort()
    release(run)
    create(key, scopedContext)
    return true
  }, [create, key, owns, release, scopedContext])

  const complete = useCallback(async () => {
    const run = current.current
    if (!run || !owns(run)) return null
    try {
      const result = await finish(run)
      return owns(run) ? result : null
    } catch (error) {
      if (owns(run)) throw error
      return null
    }
  }, [finish, owns])
  const run = current.current
  const visible = run && owns(run) && !run.completed ? run : null
  return {
    submit, complete, restart, authorityRevision,
    canRestart: Boolean(visible?.inactive),
    activeSessionId: visible?.sessionId ?? null,
    isPending: Boolean(visible?.rating || visible?.ending)
  }
}

import React from "react"

/**
 * Thin shared submit helper for both composer surfaces.
 *
 * The parent of each composer already owns the `sendMessage` mutation (via
 * React Query's `useMutation`) and the actual payload shape its backend
 * expects — Playground forwards image-generation metadata + research
 * context; Sidepanel forwards `uploadedFiles` + queued `requestOverrides`
 * (see Phase-0 spike 6). This primitive does NOT unify those payload
 * shapes; it wraps the dispatch with `beforeSend` / `afterSend` hooks so
 * both surfaces can standardize on one call shape for the tiny parts they
 * do share (form reset, attachment cleanup, textarea refocus).
 *
 * - `beforeSend` runs synchronously, before the `sendMessage` promise is
 *   awaited. Use it for optimistic UI actions: form.reset(), clear draft
 *   attachments, textAreaFocus(). Playground and Sidepanel both use this
 *   "clear immediately, let the request fly" pattern.
 * - `afterSend` runs only if the dispatch resolves. It receives the resolved
 *   `sendMessage` result so callers can distinguish successful submissions
 *   from handled failed/skipped results. On rejection it's skipped and the
 *   error propagates so the caller can render it.
 */

export interface UseComposerSubmitOptions<TPayload, TResult = unknown> {
  sendMessage: (payload: TPayload) => Promise<TResult>
}

export interface ComposerSubmitHooks<TResult = unknown> {
  beforeSend?: () => void
  afterSend?: (result: TResult) => void
}

export interface UseComposerSubmitResult<TPayload, TResult = unknown> {
  dispatch: (
    payload: TPayload,
    hooks?: ComposerSubmitHooks<TResult>
  ) => Promise<TResult>
}

export function useComposerSubmit<TPayload, TResult = unknown>({
  sendMessage,
}: UseComposerSubmitOptions<TPayload, TResult>): UseComposerSubmitResult<
  TPayload,
  TResult
> {
  const dispatch = React.useCallback(
    async (payload: TPayload, hooks?: ComposerSubmitHooks<TResult>) => {
      hooks?.beforeSend?.()
      const result = await sendMessage(payload)
      hooks?.afterSend?.(result)
      return result
    },
    [sendMessage]
  )

  return { dispatch }
}

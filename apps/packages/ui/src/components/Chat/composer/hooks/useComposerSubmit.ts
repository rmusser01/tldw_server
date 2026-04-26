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
 * - `afterSend` runs only if the dispatch resolves. On rejection it's
 *   skipped and the error propagates so the caller can render it.
 */

export interface UseComposerSubmitOptions<TPayload> {
  sendMessage: (payload: TPayload) => Promise<unknown>
}

export interface ComposerSubmitHooks {
  beforeSend?: () => void
  afterSend?: () => void
}

export interface UseComposerSubmitResult<TPayload> {
  dispatch: (payload: TPayload, hooks?: ComposerSubmitHooks) => Promise<void>
}

export function useComposerSubmit<TPayload>({
  sendMessage,
}: UseComposerSubmitOptions<TPayload>): UseComposerSubmitResult<TPayload> {
  const dispatch = React.useCallback(
    async (payload: TPayload, hooks?: ComposerSubmitHooks) => {
      hooks?.beforeSend?.()
      await sendMessage(payload)
      hooks?.afterSend?.()
    },
    [sendMessage]
  )

  return { dispatch }
}

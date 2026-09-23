import type { TldwConfig } from "./TldwApiClient"
import { createServicePromptScopeChangedError } from "./service-prompt-scope-error"

export type AuthConnectionTarget = Pick<TldwConfig, "serverUrl" | "authMode">
export type AuthConnectionAttempt = Readonly<{
  target: AuthConnectionTarget
  signal?: AbortSignal
  /** Account generation captured by the caller before its own validation awaits. */
  assertCurrent?: () => void
}>

export const normalizeAuthServerUrl = (value: unknown): string =>
  String(value || "")
    .trim()
    .replace(/\/+$/, "")

/** Credentials are authorized for the visible full base URL, not just its origin. */
export const authConnectionTargetsMatch = (
  saved: Partial<AuthConnectionTarget> | null | undefined,
  displayed: Partial<AuthConnectionTarget>,
): boolean =>
  Boolean(
    normalizeAuthServerUrl(saved?.serverUrl) &&
    normalizeAuthServerUrl(saved?.serverUrl) ===
      normalizeAuthServerUrl(displayed.serverUrl) &&
    saved?.authMode === "multi-user" &&
    displayed.authMode === "multi-user",
  )

export const assertAuthConnectionAttempt = (
  attempt: AuthConnectionAttempt,
  saved: Partial<AuthConnectionTarget> | null | undefined,
): void => {
  attempt.assertCurrent?.()
  if (
    attempt.signal?.aborted ||
    !authConnectionTargetsMatch(saved, attempt.target)
  ) {
    throw createServicePromptScopeChangedError()
  }
}

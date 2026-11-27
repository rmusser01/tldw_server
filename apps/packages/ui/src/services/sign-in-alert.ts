import { errorCode, type AuthSession } from "@/services/auth-sessions"
import type { NotificationItem } from "@/services/notifications"

export const NEW_SIGN_IN_KIND = "new_sign_in"
export const NEW_SIGN_IN_PROTECTED_CODE = "new_sign_in_protected"
export const NEW_SIGN_IN_PROTECTED_MESSAGE =
  "This alert can be dismissed from a browser that was signed in before it, or after 24 hours."

export const isProtectedAlertError = (error: unknown): boolean =>
  errorCode(error) === NEW_SIGN_IN_PROTECTED_CODE

/** True when the alert's source session is this browser (the server decides `same_install`). */
export const isOwnSignIn = (
  sourceJobId: string | number | null | undefined,
  sessions: AuthSession[]
): boolean => {
  const current = sessions.find((s) => s.is_current)
  if (!current || sourceJobId == null) return false
  const source = sessions.find((s) => String(s.id) === String(sourceJobId))
  return !!source && (source.id === current.id || source.same_install)
}

export const visibleSignInAlerts = (
  items: NotificationItem[],
  sessions: AuthSession[]
): NotificationItem[] =>
  items
    .filter(
      (i) =>
        i.kind === NEW_SIGN_IN_KIND &&
        !i.read_at &&
        !i.dismissed_at &&
        !isOwnSignIn(i.source_job_id, sessions)
    )
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))

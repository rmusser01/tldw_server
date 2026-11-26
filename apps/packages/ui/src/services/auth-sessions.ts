import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type AuthSession = {
  id: number
  ip_address: string | null
  user_agent: string | null
  device_label: string
  created_at: string
  last_activity: string
  expires_at: string
  is_current: boolean
  same_install: boolean
}

export const INTERACTIVE_REQUIRED_CODE = "interactive_session_required"

export const listAuthSessions = (): Promise<AuthSession[]> =>
  bgRequest<AuthSession[]>({
    path: toAllowedPath("/api/v1/auth/sessions"),
    method: "GET"
  })

export const revokeAuthSession = async (id: number): Promise<void> => {
  await bgRequest<unknown>({
    path: toAllowedPath(`/api/v1/auth/sessions/${id}`),
    method: "DELETE"
  })
}

export const revokeOtherAuthSessions = async (): Promise<{ sessions_revoked: number }> => {
  const res = await bgRequest<{ details?: { sessions_revoked?: number } }>({
    path: appendPathQuery(
      toAllowedPath("/api/v1/auth/sessions/revoke-all"),
      "?keep_current=true"
    ),
    method: "POST"
  })
  return { sessions_revoked: res?.details?.sessions_revoked ?? 0 }
}

/** Code from an ApiError, a bgRequest failure, or a raw `{detail:{code}}` error. */
export const errorCode = (error: unknown): string | undefined => {
  const e = error as {
    errorCode?: unknown
    details?: { detail?: { code?: unknown } }
    detail?: { code?: unknown }
  } | null
  if (typeof e?.errorCode === "string") return e.errorCode
  const code = e?.details?.detail?.code ?? e?.detail?.code
  return typeof code === "string" ? code : undefined
}

export const isInteractiveRequiredError = (error: unknown): boolean =>
  errorCode(error) === INTERACTIVE_REQUIRED_CODE

import type { ApiSendResponse } from "@/services/api-send"

/** Preserve refresh transport failures instead of treating them as revoked sessions. */
export const createTokenRefreshError = (response: ApiSendResponse): Error => {
  const status = response.ok ? 502 : response.status
  return Object.assign(new Error(
    status === 401
      ? "Session expired. Please log in again."
      : response.error || "Token refresh returned no access token. Please try again."
  ), {
    status,
    code: response.code,
    headers: response.headers,
    retryAfterMs: response.retryAfterMs
  })
}

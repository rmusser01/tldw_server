import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { resolveBrowserWebSocketBase } from "@/services/tldw/browser-websocket"

export type PersonaWebSocketConnection = {
  url: string
  /**
   * Values passed as the second `new WebSocket(url, protocols)` argument. The
   * browser sends these as the `Sec-WebSocket-Protocol` request header.
   */
  protocols: string[]
}

/**
 * Build the persona-stream WebSocket URL plus the auth subprotocols.
 *
 * TASK-12106: the auth credential is NO LONGER placed in the URL query string
 * (it would leak into server access logs, proxy logs, and browser history).
 * Instead it is carried in the WebSocket subprotocol as `["bearer", <token>]`,
 * which the backend parses from `Sec-WebSocket-Protocol`
 * (persona.py:3705-3712: splits on ",", requires parts[0]=="bearer" and uses
 * parts[1] as the token; only consulted when no Authorization header is set).
 * `_should_treat_bearer_as_api_key` (persona.py:3663-3679) then maps a
 * single-user / non-JWT bearer onto the API-key path server-side.
 *
 * NEEDS LIVE-SERVER VALIDATION before merge:
 *  - The server does not echo the offered subprotocol; confirm the browser
 *    still completes the handshake against a running backend.
 *
 * Subprotocol values must be RFC 6455 tokens. Default tldw keys (secrets
 * token_urlsafe / token_hex) and JWTs are token-safe, but a user-set custom API
 * key containing separators (space, `,`, `/`, `=` ...) would make
 * `new WebSocket(url, ["bearer", key])` throw. For that case we fall back to the
 * legacy query-string token (so persona voice keeps working) rather than crash.
 */

// RFC 6455 token charset (valid WebSocket subprotocol characters).
const WS_SUBPROTOCOL_TOKEN_RE = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/

const isSubprotocolSafe = (value: string): boolean =>
  value.length > 0 && WS_SUBPROTOCOL_TOKEN_RE.test(value)

export const buildPersonaWebSocketUrl = (
  config: Pick<TldwConfig, "serverUrl" | "authMode" | "apiKey" | "accessToken">
): PersonaWebSocketConnection => {
  const serverUrl = String(config.serverUrl || "").trim()
  if (!serverUrl) {
    throw new Error("tldw server is not configured")
  }

  const base = resolveBrowserWebSocketBase(serverUrl)

  let credential: string
  if (config.authMode === "multi-user") {
    credential = String(config.accessToken || "").trim()
    if (!credential) {
      throw new Error("Not authenticated. Please log in under Settings.")
    }
  } else {
    credential = String(config.apiKey || "").trim()
    if (!credential) {
      throw new Error("API key missing. Update Settings -> tldw server.")
    }
  }

  if (isSubprotocolSafe(credential)) {
    return {
      url: `${base}/api/v1/persona/stream`,
      protocols: ["bearer", credential]
    }
  }

  // Credential can't be sent as a WebSocket subprotocol without throwing; keep
  // persona voice working via the legacy query-string token for this case.
  if (typeof console !== "undefined") {
    console.warn(
      "[persona-stream] API key is not WebSocket-subprotocol-safe; falling back to query-string auth (the credential will appear in the connection URL). Use a token-safe API key to keep it out of the URL."
    )
  }
  const params = new URLSearchParams()
  params.set(config.authMode === "multi-user" ? "token" : "api_key", credential)
  return {
    url: `${base}/api/v1/persona/stream?${params.toString()}`,
    protocols: []
  }
}

import { resolveBrowserWebSocketBase } from "@/services/tldw/browser-websocket"

type AudioWebSocketSender = {
  send: (payload: string) => void
}

const normalizeAudioWebSocketPath = (path: string): string => {
  const normalizedPath = String(path || "").trim()
  if (!normalizedPath) {
    throw new Error("Audio WebSocket path is required")
  }

  const pathWithSlash = normalizedPath.startsWith("/")
    ? normalizedPath
    : `/${normalizedPath}`
  const url = new URL(pathWithSlash, "ws://tldw.local")
  if (url.searchParams.has("token")) {
    throw new Error("Audio WebSocket tokens must be sent in the auth frame")
  }

  if (!url.pathname.startsWith("/api/v1/audio/")) {
    throw new Error("Audio WebSocket path must target an audio endpoint")
  }

  return `${url.pathname}${url.search}${url.hash}`
}

export const buildAudioWebSocketUrl = (
  serverUrl: string,
  path: string
): string => {
  const base = resolveBrowserWebSocketBase(serverUrl).replace(/\/$/, "")
  if (!base) {
    throw new Error("tldw server not configured")
  }

  return `${base}${normalizeAudioWebSocketPath(path)}`
}

export const sendAudioWebSocketAuthFrame = (
  ws: AudioWebSocketSender,
  token: string
): void => {
  const normalizedToken = String(token || "").trim()
  if (!normalizedToken) {
    throw new Error("Audio WebSocket auth token is required")
  }

  ws.send(JSON.stringify({ type: "auth", token: normalizedToken }))
}

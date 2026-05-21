import React from "react"

import { buildPersonaWebSocketUrl } from "@/services/persona-stream"
import {
  createPersonaLiveSession,
  focusPersonaLiveSession,
  listPersonaLiveSessions,
  stopPersonaLiveSession
} from "@/services/persona-live-control"
import type {
  PersonaLiveSessionList,
  PersonaLiveSessionSummary
} from "@/services/persona-live-control"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type PersonaLiveStreamState = "closed" | "connecting" | "open" | "error"

export type PersonaLiveControlOptions = {
  autoLoad?: boolean
  defaultPersonaId?: string | null
  surface?: string | null
}

export type PersonaLiveSendTextResult =
  | {
      ok: true
      clientMessageId: string
    }
  | {
      ok: false
      clientMessageId: string
      error: string
    }

export type PersonaLiveSendTextOptions = {
  clientMessageId?: string | null
}

const SEND_TEXT_ACTION = "send_text_ws"
const STREAM_CONNECT_ERROR = "Persona live stream failed to connect"

const terminalLifecycles = new Set(["stopping", "stopped", "error"])

const normalizeOptionalString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed || null
}

const generateStableId = (prefix: string): string => {
  const cryptoApi = globalThis.crypto
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return `${prefix}:${cryptoApi.randomUUID()}`
  }
  return `${prefix}:${Date.now().toString(36)}:${Math.random()
    .toString(36)
    .slice(2)}`
}

const chooseFocusedSessionId = (payload: PersonaLiveSessionList): string | null => {
  const focusedFromPayload = normalizeOptionalString(payload.focusedSessionId)
  if (focusedFromPayload) return focusedFromPayload
  return payload.sessions.find((session) => session.isFocused)?.sessionId ?? null
}

const isTextSendableSession = (
  session: PersonaLiveSessionSummary | null | undefined
): session is PersonaLiveSessionSummary => {
  if (!session) return false
  if (!session.capabilities.text) return false
  if (terminalLifecycles.has(session.lifecycle)) return false
  return session.allowedActions.includes(SEND_TEXT_ACTION)
}

const upsertSession = (
  sessions: PersonaLiveSessionSummary[],
  nextSession: PersonaLiveSessionSummary,
  options: { focused?: boolean } = {}
): PersonaLiveSessionSummary[] => {
  let found = false
  const updated = sessions.map((session) => {
    if (session.sessionId !== nextSession.sessionId) {
      return options.focused ? { ...session, isFocused: false } : session
    }
    found = true
    return options.focused ? { ...nextSession, isFocused: true } : nextSession
  })
  if (found) return updated
  return [
    ...updated,
    options.focused ? { ...nextSession, isFocused: true } : nextSession
  ]
}

const getSessionErrorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? error.message || fallback : fallback

export function usePersonaLiveControl(options: PersonaLiveControlOptions = {}) {
  const {
    autoLoad = true,
    defaultPersonaId = null,
    surface = null
  } = options
  const normalizedSurface = normalizeOptionalString(surface)
  const normalizedDefaultPersonaId = normalizeOptionalString(defaultPersonaId)

  const [sessions, setSessions] = React.useState<PersonaLiveSessionSummary[]>([])
  const [focusedSessionId, setFocusedSessionId] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(autoLoad)
  const [error, setError] = React.useState<string | null>(null)
  const [lastSendError, setLastSendError] = React.useState<string | null>(null)
  const [streamState, setStreamState] =
    React.useState<PersonaLiveStreamState>("closed")
  const [pendingFocusSessionId, setPendingFocusSessionId] =
    React.useState<string | null>(null)

  const sessionsRef = React.useRef(sessions)
  const focusedSessionIdRef = React.useRef(focusedSessionId)
  const wsRef = React.useRef<WebSocket | null>(null)
  const streamConnectPromiseRef = React.useRef<Promise<WebSocket> | null>(null)

  React.useEffect(() => {
    sessionsRef.current = sessions
  }, [sessions])

  React.useEffect(() => {
    focusedSessionIdRef.current = focusedSessionId
  }, [focusedSessionId])

  const focusedSession = React.useMemo(
    () => sessions.find((session) => session.sessionId === focusedSessionId) ?? null,
    [sessions, focusedSessionId]
  )

  const voiceAvailable = Boolean(focusedSession?.capabilities.voice)
  const canSendText = isTextSendableSession(focusedSession)

  const applyFocusedSession = React.useCallback(
    (session: PersonaLiveSessionSummary) => {
      setSessions((current) => upsertSession(current, session, { focused: true }))
      setFocusedSessionId(session.sessionId)
    },
    []
  )

  const reload = React.useCallback(async (): Promise<PersonaLiveSessionList> => {
    setLoading(true)
    setError(null)
    try {
      const payload = await listPersonaLiveSessions({
        personaId: normalizedDefaultPersonaId,
        surface: normalizedSurface
      })
      setSessions(payload.sessions)
      setFocusedSessionId(chooseFocusedSessionId(payload))
      return payload
    } catch (err) {
      const message = getSessionErrorMessage(
        err,
        "Failed to load Persona live sessions"
      )
      setError(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [normalizedDefaultPersonaId, normalizedSurface])

  React.useEffect(() => {
    if (!autoLoad) {
      setLoading(false)
      return
    }
    void reload().catch(() => undefined)
  }, [autoLoad, reload])

  React.useEffect(
    () => () => {
      streamConnectPromiseRef.current = null
      const ws = wsRef.current
      wsRef.current = null
      if (ws && ws.readyState < WebSocket.CLOSING) {
        ws.close()
      }
    },
    []
  )

  const focusSession = React.useCallback(
    async (sessionId: string): Promise<PersonaLiveSessionSummary> => {
      const normalizedSessionId = normalizeOptionalString(sessionId)
      if (!normalizedSessionId) {
        throw new Error("sessionId is required")
      }
      setPendingFocusSessionId(normalizedSessionId)
      setError(null)
      try {
        const session = await focusPersonaLiveSession(normalizedSessionId)
        applyFocusedSession(session)
        return session
      } catch (err) {
        const message = getSessionErrorMessage(
          err,
          "Failed to focus Persona live session"
        )
        setError(message)
        throw err
      } finally {
        setPendingFocusSessionId(null)
      }
    },
    [applyFocusedSession]
  )

  const startTextSession = React.useCallback(
    async (personaId?: string | null): Promise<PersonaLiveSessionSummary> => {
      const normalizedPersonaId =
        normalizeOptionalString(personaId) ?? normalizedDefaultPersonaId
      if (!normalizedPersonaId) {
        throw new Error("personaId is required")
      }
      setError(null)
      const session = await createPersonaLiveSession({
        personaId: normalizedPersonaId,
        reusePolicy: "resume_compatible",
        idempotencyKey: generateStableId("persona-live"),
        surface: normalizedSurface
      })
      applyFocusedSession(session)
      return session
    },
    [applyFocusedSession, normalizedDefaultPersonaId, normalizedSurface]
  )

  const stopSession = React.useCallback(
    async (sessionId?: string | null): Promise<PersonaLiveSessionSummary> => {
      const normalizedSessionId =
        normalizeOptionalString(sessionId) ?? focusedSessionIdRef.current
      if (!normalizedSessionId) {
        throw new Error("sessionId is required")
      }
      setError(null)
      const stoppedSession = await stopPersonaLiveSession(normalizedSessionId)
      setSessions((current) => upsertSession(current, stoppedSession))
      if (focusedSessionIdRef.current === normalizedSessionId) {
        setFocusedSessionId(null)
      }
      await reload().catch(() => undefined)
      return stoppedSession
    },
    [reload]
  )

  const ensureStreamSocket = React.useCallback(async (): Promise<WebSocket> => {
    const current = wsRef.current
    if (current?.readyState === WebSocket.OPEN) {
      return current
    }
    if (streamConnectPromiseRef.current) {
      return streamConnectPromiseRef.current
    }

    setStreamState("connecting")
    const connectPromise = tldwClient
      .ensureConfigForRequest(true)
      .then((config) => {
        const url = buildPersonaWebSocketUrl(config)
        const ws = new WebSocket(url)
        wsRef.current = ws

        return new Promise<WebSocket>((resolve, reject) => {
          let settled = false
          const failConnect = () => {
            if (settled) return
            settled = true
            streamConnectPromiseRef.current = null
            if (wsRef.current === ws) {
              wsRef.current = null
            }
            setStreamState("error")
            reject(new Error(STREAM_CONNECT_ERROR))
          }

          ws.onopen = () => {
            settled = true
            streamConnectPromiseRef.current = null
            setStreamState("open")
            resolve(ws)
          }
          ws.onerror = failConnect
          ws.onclose = () => {
            if (!settled) {
              failConnect()
              return
            }
            if (wsRef.current === ws) {
              setStreamState("closed")
            }
          }
        })
      })
      .catch((err) => {
        streamConnectPromiseRef.current = null
        setStreamState("error")
        throw err
      })

    streamConnectPromiseRef.current = connectPromise
    return connectPromise
  }, [])

  const ensureSendableSession = React.useCallback(async () => {
    const focusedId = focusedSessionIdRef.current
    const currentFocused =
      sessionsRef.current.find((session) => session.sessionId === focusedId) ?? null
    if (isTextSendableSession(currentFocused)) {
      return currentFocused
    }
    const personaId =
      normalizedDefaultPersonaId ?? currentFocused?.personaId ?? null
    return startTextSession(personaId)
  }, [normalizedDefaultPersonaId, startTextSession])

  const sendText = React.useCallback(
    async (
      text: string,
      sendOptions: PersonaLiveSendTextOptions = {}
    ): Promise<PersonaLiveSendTextResult> => {
      const clientMessageId =
        normalizeOptionalString(sendOptions.clientMessageId) ??
        generateStableId("persona-live-message")
      const trimmedText = normalizeOptionalString(text)
      if (!trimmedText) {
        return {
          ok: false,
          clientMessageId,
          error: "Message text is required"
        }
      }
      setLastSendError(null)
      try {
        const session = await ensureSendableSession()
        const ws = await ensureStreamSocket()
        ws.send(
          JSON.stringify({
            type: "user_message",
            session_id: session.sessionId,
            client_message_id: clientMessageId,
            text: trimmedText
          })
        )
        return { ok: true, clientMessageId }
      } catch (err) {
        const message = getSessionErrorMessage(
          err,
          "Failed to send Persona live message"
        )
        setLastSendError(message)
        return {
          ok: false,
          clientMessageId,
          error: message
        }
      }
    },
    [ensureSendableSession, ensureStreamSocket]
  )

  return {
    sessions,
    focusedSessionId,
    focusedSession,
    loading,
    error,
    lastSendError,
    streamState,
    pendingFocusSessionId,
    canSendText,
    voiceAvailable,
    reload,
    focusSession,
    startTextSession,
    stopSession,
    sendText
  }
}

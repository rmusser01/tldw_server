/**
 * useACPSession - React hook for ACP session management
 *
 * Provides WebSocket-based real-time communication with ACP sessions,
 * handling connection state, updates, and permission flows.
 */

import React from "react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { buildACPAuthParams, resolveACPServerUrl } from "@/services/acp/connection"
import type {
  ACPSessionState,
  ACPWSServerMessage,
  ACPWSUpdateMessage,
  ACPWSPermissionRequestMessage,
  ACPWSPromptCompleteMessage,
  ACPWSErrorMessage,
  ACPWSConnectedMessage,
  ACPPendingPermission,
  ACPUpdate,
  ACPPermissionTier,
} from "@/services/acp/types"
import { WS_CONFIG, SESSION_CONFIG, shouldRetryACPWebSocketClose } from "@/services/acp/constants"

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export interface UseACPSessionOptions {
  /** Session ID to connect to (if undefined, no connection will be made) */
  sessionId?: string
  /** Auto-connect when sessionId is provided */
  autoConnect?: boolean
  /** Callbacks */
  onStateChange?: (state: ACPSessionState) => void
  onUpdate?: (update: ACPWSUpdateMessage) => void
  onPermissionRequest?: (request: ACPWSPermissionRequestMessage) => void
  onPromptComplete?: (result: ACPWSPromptCompleteMessage) => void
  onError?: (error: ACPWSErrorMessage) => void
  onConnected?: (message: ACPWSConnectedMessage) => void
  onDisconnected?: () => void
}

export interface ReconnectInfo {
  isReconnecting: boolean
  attempt: number
  maxAttempts: number
}

export interface UseACPSessionReturn {
  /** Current connection state */
  state: ACPSessionState
  /** Whether connected to the WebSocket */
  isConnected: boolean
  /** Agent capabilities (available after connection) */
  agentCapabilities: Record<string, unknown> | null
  /** List of updates received */
  updates: ACPUpdate[]
  /** Pending permission requests */
  pendingPermissions: ACPPendingPermission[]
  /** Last error message */
  error: string | null
  /** Auto-reconnection progress (null when not reconnecting) */
  reconnectInfo: ReconnectInfo | null
  /** Connect to the session */
  connect: () => Promise<void>
  /** Disconnect from the session */
  disconnect: () => void
  /** Send a prompt to the session */
  sendPrompt: (messages: Array<{ role: "system" | "user" | "assistant"; content: string }>) => void
  /** Approve a permission request */
  approvePermission: (requestId: string, batchApproveTier?: ACPPermissionTier) => void
  /** Deny a permission request */
  denyPermission: (requestId: string) => void
  /** Cancel the current operation */
  cancel: () => void
  /** Clear updates list */
  clearUpdates: () => void
}

// -----------------------------------------------------------------------------
// Hook Implementation
// -----------------------------------------------------------------------------

export function useACPSession(options: UseACPSessionOptions = {}): UseACPSessionReturn {
  const {
    sessionId,
    autoConnect = true,
    onStateChange,
    onUpdate,
    onPermissionRequest,
    onPromptComplete,
    onError,
    onConnected,
    onDisconnected,
  } = options

  const { config: connectionConfig } = useCanonicalConnectionConfig()

  // State
  const [state, setState] = React.useState<ACPSessionState>("disconnected")
  const [agentCapabilities, setAgentCapabilities] = React.useState<Record<string, unknown> | null>(null)
  const [updates, setUpdates] = React.useState<ACPUpdate[]>([])
  const [pendingPermissions, setPendingPermissions] = React.useState<ACPPendingPermission[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [reconnectInfo, setReconnectInfo] = React.useState<ReconnectInfo | null>(null)

  // Refs
  const wsRef = React.useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptsRef = React.useRef(0)
  const isClosingRef = React.useRef(false)
  const sessionIdRef = React.useRef(sessionId)

  // Callback refs (to avoid stale closures)
  const callbacksRef = React.useRef({
    onStateChange,
    onUpdate,
    onPermissionRequest,
    onPromptComplete,
    onError,
    onConnected,
    onDisconnected,
  })

  // Update refs when dependencies change
  React.useEffect(() => {
    callbacksRef.current = {
      onStateChange,
      onUpdate,
      onPermissionRequest,
      onPromptComplete,
      onError,
      onConnected,
      onDisconnected,
    }
  }, [onStateChange, onUpdate, onPermissionRequest, onPromptComplete, onError, onConnected, onDisconnected])

  React.useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  // State change handler
  const updateState = React.useCallback((newState: ACPSessionState) => {
    setState(newState)
    callbacksRef.current.onStateChange?.(newState)
  }, [])

  // Build WebSocket URL with auth
  const buildWsUrl = React.useCallback(() => {
    if (!sessionId || !connectionConfig) return null

    const wsUrl = resolveACPServerUrl(connectionConfig).replace(/^http/i, "ws")
    const params = new URLSearchParams()
    const authParams = buildACPAuthParams(connectionConfig)

    if (authParams.api_key) {
      params.set("api_key", authParams.api_key)
    }
    if (authParams.token) {
      params.set("token", authParams.token)
    }

    const query = params.toString()
    return `${wsUrl}/api/v1/acp/sessions/${sessionId}/stream${query ? `?${query}` : ""}`
  }, [sessionId, connectionConfig])

  // Handle incoming messages
  const handleMessage = React.useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data) as ACPWSServerMessage

      switch (message.type) {
        case "connected":
          setAgentCapabilities(message.agent_capabilities ?? null)
          updateState("connected")
          callbacksRef.current.onConnected?.(message)
          break

        case "update": {
          const update: ACPUpdate = {
            timestamp: new Date(),
            type: message.update_type,
            data: message.data,
          }
          setUpdates((prev) => {
            const next = [...prev, update]
            // Limit updates in memory
            if (next.length > SESSION_CONFIG.MAX_UPDATES_PER_SESSION) {
              return next.slice(-SESSION_CONFIG.MAX_UPDATES_PER_SESSION)
            }
            return next
          })
          callbacksRef.current.onUpdate?.(message)

          // Update state if running
          if (state === "connected") {
            updateState("running")
          }
          break
        }

        case "permission_request": {
          const pending: ACPPendingPermission = {
            request_id: message.request_id,
            tool_name: message.tool_name,
            tool_arguments: message.tool_arguments,
            tier: message.tier,
            approval_requirement: message.approval_requirement,
            governance_reason: message.governance_reason,
            deny_reason: message.deny_reason,
            provenance_summary: message.provenance_summary,
            runtime_narrowing_reason: message.runtime_narrowing_reason,
            policy_snapshot_fingerprint: message.policy_snapshot_fingerprint,
            timeout_seconds: message.timeout_seconds,
            requestedAt: new Date(),
          }
          setPendingPermissions((prev) => [...prev, pending])
          updateState("waiting_permission")
          callbacksRef.current.onPermissionRequest?.(message)
          break
        }

        case "prompt_complete":
          // Clear pending permissions when prompt completes
          setPendingPermissions([])
          updateState("connected")
          callbacksRef.current.onPromptComplete?.(message)
          break

        case "error":
          setError(message.message)
          updateState("error")
          callbacksRef.current.onError?.(message)
          break

        case "done":
          // Server closing connection
          break
      }
    } catch (e) {
      console.error("Failed to parse ACP WebSocket message:", e)
    }
  }, [state, updateState])

  // Connect to WebSocket
  const connect = React.useCallback(async () => {
    const url = buildWsUrl()
    if (!url) {
      setError("Missing ACP connection configuration")
      updateState("error")
      return
    }

    // Close existing connection
    if (wsRef.current) {
      isClosingRef.current = true
      wsRef.current.close()
      wsRef.current = null
    }

    isClosingRef.current = false
    updateState("connecting")
    setError(null)

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        reconnectAttemptsRef.current = 0
        setReconnectInfo(null)
        // State will be updated to "connected" when we receive the "connected" message
      }

      ws.onclose = (event) => {
        if (!isClosingRef.current) {
          // Unexpected close - attempt reconnect
          if (
            shouldRetryACPWebSocketClose(event.code) &&
            reconnectAttemptsRef.current < WS_CONFIG.MAX_RECONNECT_ATTEMPTS
          ) {
            const nextAttempt = reconnectAttemptsRef.current + 1
            setReconnectInfo({
              isReconnecting: true,
              attempt: nextAttempt,
              maxAttempts: WS_CONFIG.MAX_RECONNECT_ATTEMPTS,
            })
            const delay = Math.min(
              WS_CONFIG.RECONNECT_DELAY_MS * Math.pow(WS_CONFIG.RECONNECT_BACKOFF_MULTIPLIER, reconnectAttemptsRef.current),
              WS_CONFIG.MAX_RECONNECT_DELAY_MS
            )
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttemptsRef.current++
              connect()
            }, delay)
          } else {
            setReconnectInfo(null)
            updateState("disconnected")
            callbacksRef.current.onDisconnected?.()
          }
        } else {
          setReconnectInfo(null)
          updateState("disconnected")
          callbacksRef.current.onDisconnected?.()
        }
      }

      ws.onerror = () => {
        // Error will be followed by close event
        setError("WebSocket connection error")
      }

      ws.onmessage = handleMessage
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed")
      updateState("error")
    }
  }, [buildWsUrl, handleMessage, updateState])

  // Disconnect from WebSocket
  const disconnect = React.useCallback(() => {
    isClosingRef.current = true

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setReconnectInfo(null)
    updateState("disconnected")
    setPendingPermissions([])
  }, [updateState])

  // Send a message via WebSocket
  const sendMessage = React.useCallback((message: Record<string, unknown>) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected")
    }
    wsRef.current.send(JSON.stringify(message))
  }, [])

  // Send a prompt
  const sendPrompt = React.useCallback((
    messages: Array<{ role: "system" | "user" | "assistant"; content: string }>
  ) => {
    if (!sessionIdRef.current) {
      throw new Error("No session ID")
    }
    sendMessage({
      type: "prompt",
      session_id: sessionIdRef.current,
      prompt: messages,
    })
    updateState("running")
  }, [sendMessage, updateState])

  // Approve a permission request
  const approvePermission = React.useCallback((
    requestId: string,
    batchApproveTier?: ACPPermissionTier
  ) => {
    sendMessage({
      type: "permission_response",
      request_id: requestId,
      approved: true,
      batch_approve_tier: batchApproveTier,
    })
    // Remove the permission and check if we should transition state
    setPendingPermissions((prev) => {
      const filtered = prev.filter((p) => p.request_id !== requestId)
      if (filtered.length === 0) {
        updateState("running")
      }
      return filtered
    })
  }, [sendMessage, updateState])

  // Deny a permission request
  const denyPermission = React.useCallback((requestId: string) => {
    sendMessage({
      type: "permission_response",
      request_id: requestId,
      approved: false,
    })
    // Remove the permission and check if we should transition state
    setPendingPermissions((prev) => {
      const filtered = prev.filter((p) => p.request_id !== requestId)
      if (filtered.length === 0) {
        updateState("running")
      }
      return filtered
    })
  }, [sendMessage, updateState])

  // Cancel the current operation
  const cancel = React.useCallback(() => {
    if (!sessionIdRef.current) return
    sendMessage({
      type: "cancel",
      session_id: sessionIdRef.current,
    })
  }, [sendMessage])

  // Clear updates
  const clearUpdates = React.useCallback(() => {
    setUpdates([])
  }, [])

  // Auto-connect when sessionId changes
  React.useEffect(() => {
    if (autoConnect && sessionId && connectionConfig) {
      void connect()
    }
    return () => {
      disconnect()
    }
    // `connect` depends on message handlers and session state; re-running on every
    // callback identity change would thrash the ACP socket.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoConnect, connectionConfig, disconnect, sessionId])

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      disconnect()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    state,
    isConnected: state === "connected" || state === "running" || state === "waiting_permission",
    agentCapabilities,
    updates,
    pendingPermissions,
    error,
    reconnectInfo,
    connect,
    disconnect,
    sendPrompt,
    approvePermission,
    denyPermission,
    cancel,
    clearUpdates,
  }
}

export default useACPSession

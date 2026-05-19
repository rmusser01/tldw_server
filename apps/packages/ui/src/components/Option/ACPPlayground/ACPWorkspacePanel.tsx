import React from "react"
import { Empty } from "antd"
import { Terminal as TerminalIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { buildACPAuthHeaders, resolveACPServerUrl } from "@/services/acp/connection"
import { useACPSessionsStore } from "@/store/acp-sessions"

type WebSocketWithHeaders = new (
  url: string,
  protocols?: string | string[],
  options?: { headers?: Record<string, string> }
) => WebSocket

const buildWsUrl = (baseUrl: string, path: string): string => {
  const wsBase = baseUrl.replace(/^http/, "ws").replace(/\/$/, "")
  const url = new URL(path.startsWith("/") ? path : `/${path}`, wsBase)
  return url.toString()
}

const buildAuthProtocols = (headers: Record<string, string>): string[] | undefined => {
  const authHeader = headers.Authorization || headers.authorization
  if (authHeader?.toLowerCase().startsWith("bearer ")) {
    return ["bearer", authHeader.slice(7).trim()]
  }
  const apiKey = headers["X-API-KEY"] || headers["x-api-key"]
  if (apiKey) {
    return ["x-api-key", apiKey]
  }
  return undefined
}

const loadTerminalRuntime = () =>
  Promise.all([
    import("xterm"),
    import("@xterm/addon-fit"),
    import("xterm/css/xterm.css"),
  ])

const createWebSocket = (
  url: string,
  headers: Record<string, string>,
  protocols?: string[]
): WebSocket => {
  const WsCtor = WebSocket as unknown as WebSocketWithHeaders
  try {
    return new WsCtor(url, protocols, { headers })
  } catch {
    return new WebSocket(url, protocols)
  }
}

const resolveTokenColor = (tokenName: string, fallbackRgb: string): string => {
  if (typeof window === "undefined") return fallbackRgb
  const tokenValue = getComputedStyle(document.documentElement).getPropertyValue(tokenName).trim()
  return tokenValue ? `rgb(${tokenValue})` : fallbackRgb
}

export const ACPWorkspacePanel: React.FC = () => {
  const { t } = useTranslation("playground")
  const { config: connectionConfig } = useCanonicalConnectionConfig()
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const resizeTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const fitTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const activeSessionId = useACPSessionsStore((s) => s.activeSessionId)
  const activeSession = useACPSessionsStore((s) =>
    s.activeSessionId ? s.getSession(s.activeSessionId) : undefined
  )

  const sshPath = activeSession?.sshWsUrl || ""

  const [wsStatus, setWsStatus] = React.useState<"connecting" | "connected" | "disconnected">("disconnected")
  const [reconnectKey, setReconnectKey] = React.useState(0)

  const handleReconnect = React.useCallback(() => {
    setReconnectKey((k) => k + 1)
  }, [])

  React.useEffect(() => {
    if (!activeSessionId || !sshPath || !containerRef.current || !connectionConfig) return

    let disposed = false
    let disposeTerminal: (() => void) | null = null
    const container = containerRef.current
    setWsStatus("connecting")

    void (async () => {
      try {
        const [{ Terminal }, { FitAddon }] = await loadTerminalRuntime()
        if (disposed) return

        const term = new Terminal({
          fontFamily: "JetBrains Mono, Menlo, Monaco, monospace",
          fontSize: 13,
          theme: {
            background: resolveTokenColor("--color-bg", "rgb(11 15 26)"),
            foreground: resolveTokenColor("--color-text", "rgb(220 226 240)"),
            cursor: resolveTokenColor("--color-focus", "rgb(110 231 255)")
          },
          cursorBlink: true,
        })
        const fitAddon = new FitAddon()
        term.loadAddon(fitAddon)
        term.open(container)
        fitAddon.fit()

        const headers = buildACPAuthHeaders(connectionConfig)
        const protocols = buildAuthProtocols(headers)
        const wsUrl = buildWsUrl(resolveACPServerUrl(connectionConfig), sshPath)
        const ws = createWebSocket(wsUrl, headers, protocols)
        ws.binaryType = "arraybuffer"

        ws.onopen = () => {
          if (!disposed) {
            setWsStatus("connected")
            term.focus()
          }
        }
        ws.onmessage = (event) => {
          if (typeof event.data === "string") {
            term.write(event.data)
            return
          }
          const data = new Uint8Array(event.data)
          term.write(data)
        }
        ws.onerror = () => {
          if (!disposed) {
            setWsStatus("disconnected")
          }
          term.write("\\r\\n[SSH connection error]\\r\\n")
        }
        ws.onclose = () => {
          if (!disposed) {
            setWsStatus("disconnected")
          }
          term.write("\\r\\n[SSH connection closed - click Reconnect to retry]\\r\\n")
        }

        const disposeInput = term.onData((data) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(data)
          }
        })

        const scheduleResize = (cols: number, rows: number) => {
          if (resizeTimeoutRef.current) {
            clearTimeout(resizeTimeoutRef.current)
          }
          resizeTimeoutRef.current = setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: "resize", cols, rows }))
            }
          }, 100)
        }

        const disposeResize = term.onResize(({ cols, rows }) => {
          scheduleResize(cols, rows)
        })

        const scheduleFit = () => {
          if (fitTimeoutRef.current) {
            clearTimeout(fitTimeoutRef.current)
          }
          fitTimeoutRef.current = setTimeout(() => {
            fitAddon.fit()
          }, 100)
        }

        const handleResize = () => scheduleFit()
        window.addEventListener("resize", handleResize)

        disposeTerminal = () => {
          disposeInput.dispose()
          disposeResize.dispose()
          window.removeEventListener("resize", handleResize)
          if (resizeTimeoutRef.current) {
            clearTimeout(resizeTimeoutRef.current)
            resizeTimeoutRef.current = null
          }
          if (fitTimeoutRef.current) {
            clearTimeout(fitTimeoutRef.current)
            fitTimeoutRef.current = null
          }
          ws.close()
          term.dispose()
        }

        if (disposed && disposeTerminal) {
          disposeTerminal()
          disposeTerminal = null
        }
      } catch (err) {
        if (!disposed) {
          setWsStatus("disconnected")
          console.error("Failed to initialize workspace terminal:", err)
        }
      }
    })()

    return () => {
      disposed = true
      if (disposeTerminal) {
        disposeTerminal()
        disposeTerminal = null
      }
    }
  }, [activeSessionId, connectionConfig, sshPath, reconnectKey])

  if (!activeSessionId) {
    return (
      <div className="flex h-full items-center justify-center">
        <Empty
          description={t(
            "playground:acp.workspace.selectSession",
            "Select a session to open the workspace terminal."
          )}
        />
      </div>
    )
  }

  if (!activeSession?.sshWsUrl) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div className="space-y-2 text-center">
              <p>
                {t(
                  "playground:acp.workspace.unavailable",
                  "Workspace terminal is not available for this session."
                )}
              </p>
              <p className="text-xs text-text-muted">
                {t(
                  "playground:acp.workspace.sandboxRequired",
                  "The workspace terminal requires sandbox mode, which runs the agent inside a Docker container with SSH access. To enable it, set [ACP-SANDBOX] enabled = true in config.txt."
                )}
              </p>
            </div>
          }
        />
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-surface px-3 py-2 text-sm">
        <div className="flex items-center gap-2 text-text-muted">
          <TerminalIcon className="h-4 w-4" />
          {t("playground:acp.workspace.title", "Workspace Terminal")}
          <span
            className={`h-2 w-2 rounded-full ${
              wsStatus === "connected" ? "bg-success" :
              wsStatus === "connecting" ? "bg-info animate-pulse" :
              "bg-error"
            }`}
            aria-label={
              wsStatus === "connected" ? "Connected" :
              wsStatus === "connecting" ? "Connecting" :
              "Disconnected"
            }
            role="status"
          />
        </div>
        {wsStatus === "disconnected" && (
          <button
            type="button"
            onClick={handleReconnect}
            className="rounded px-2 py-1 text-xs text-primary hover:bg-surface2"
          >
            {t("playground:acp.workspace.reconnect", "Reconnect")}
          </button>
        )}
      </div>
      <div ref={containerRef} className="flex-1 bg-black" />
    </div>
  )
}

export default ACPWorkspacePanel

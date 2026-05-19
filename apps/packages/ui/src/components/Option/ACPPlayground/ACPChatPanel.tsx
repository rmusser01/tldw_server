import React, { useState, useRef, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Button, Input, Empty, Spin } from "antd"
import { Send, Square, Bot, User, Wrench, AlertCircle, Copy, Check } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeHighlight from "rehype-highlight"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPUpdate, ACPSessionState, ParsedUpdateMessage } from "@/services/acp/types"
import type { ReconnectInfo } from "@/hooks/useACPSession"

const { TextArea } = Input

// ---------------------------------------------------------------------------
// CopyButton - hover-visible clipboard copy for messages
// ---------------------------------------------------------------------------
// TODO: CopyButton default label should be localized via useTranslation once
// this component is promoted to a shared location.
const CopyButton: React.FC<{ text: string; className?: string; label?: string }> = ({ text, className, label = "Copy" }) => {
  const [copied, setCopied] = React.useState(false)

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* ignore */
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`rounded p-1 text-text-muted transition-colors hover:bg-surface2 hover:text-text ${className || ""}`}
      title={label}
    >
      {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  )
}

// ---------------------------------------------------------------------------
// extractTextFromChildren - extract plain text from React children tree
// ---------------------------------------------------------------------------
const extractTextFromChildren = (children: React.ReactNode): string => {
  if (typeof children === "string") return children
  if (Array.isArray(children)) return children.map(extractTextFromChildren).join("")
  if (React.isValidElement(children) && children.props?.children) {
    return extractTextFromChildren(children.props.children)
  }
  return String(children ?? "")
}

// ---------------------------------------------------------------------------
// ToolArgumentsDisplay - smart rendering of tool call arguments
// ---------------------------------------------------------------------------
const ToolArgumentsDisplay: React.FC<{ args: Record<string, unknown> }> = ({ args }) => {
  const entries = Object.entries(args)
  const isSimple =
    entries.length > 0 &&
    entries.length <= 10 &&
    entries.every(([, v]) => v === null || typeof v !== "object")

  if (isSimple) {
    return (
      <div className="rounded bg-surface2 p-2">
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          {entries.map(([key, value]) => (
            <React.Fragment key={key}>
              <dt className="font-medium text-text-muted">{key}</dt>
              <dd className="truncate font-mono text-text">{String(value ?? "")}</dd>
            </React.Fragment>
          ))}
        </dl>
      </div>
    )
  }

  return (
    <div className="relative group/tool rounded-lg bg-surface2 p-2">
      <pre className="overflow-x-auto text-xs text-text-muted">
        {JSON.stringify(args, null, 2)}
      </pre>
      <CopyButton
        text={JSON.stringify(args, null, 2)}
        className="absolute right-2 top-2 opacity-0 group-hover/tool:opacity-100"
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// parseUpdateMessage - type-safe message parsing from raw ACP updates
// ---------------------------------------------------------------------------
export function parseUpdateMessage(
  update: ACPUpdate,
  cancelledLabel: string,
): ParsedUpdateMessage {
  const data = update.data as Record<string, unknown>

  switch (update.type) {
    case "text":
    case "assistant_text":
      return {
        role: "assistant",
        content: String(data.text ?? data.content ?? ""),
      }

    case "user_text":
      return {
        role: "user",
        content: String(data.text ?? data.content ?? ""),
      }

    case "tool_call":
      return {
        role: "tool",
        toolName: String(data.name ?? data.tool ?? "unknown"),
        toolArgs: (data.arguments ?? data.input ?? {}) as Record<string, unknown>,
      }

    case "tool_result":
      return {
        role: "tool_result",
        toolName: String(data.name ?? data.tool ?? ""),
        result: data.result ?? data.output ?? "",
      }

    case "error":
      return {
        role: "error",
        content: String(data.message ?? data.error ?? "Unknown error"),
      }

    case "cancelled":
      return {
        role: "system",
        content: cancelledLabel,
      }

    default:
      return {
        role: "system",
        content: JSON.stringify(data, null, 2),
      }
  }
}

interface ACPChatPanelProps {
  state: ACPSessionState
  isConnected: boolean
  updates: ACPUpdate[]
  sendPrompt: (messages: Array<{ role: "system" | "user" | "assistant"; content: string }>) => void
  cancel: () => void
  connect: () => Promise<void>
  error?: string | null
  reconnectInfo?: ReconnectInfo | null
}

export const ACPChatPanel: React.FC<ACPChatPanelProps> = ({
  state,
  isConnected,
  updates,
  sendPrompt,
  cancel,
  connect,
  error,
  reconnectInfo,
}) => {
  const { t } = useTranslation(["playground", "option", "common"])
  const [inputValue, setInputValue] = useState("")
  const [isReconnecting, setIsReconnecting] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Store
  const activeSessionId = useACPSessionsStore((s) => s.activeSessionId)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [updates])

  const handleSend = () => {
    if (!inputValue.trim() || !isConnected) return

    sendPrompt([{ role: "user", content: inputValue.trim() }])
    setInputValue("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleCancel = () => {
    cancel()
  }

  const handleReconnect = async () => {
    setIsReconnecting(true)
    try {
      await connect()
    } finally {
      setIsReconnecting(false)
    }
  }

  const isRunning = state === "running" || state === "waiting_permission"

  // No active session
  if (!activeSessionId) {
    return (
      <div className="flex h-full items-center justify-center">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div className="space-y-2 text-sm text-text-muted">
              <p>{t("playground:acp.selectOrCreate", "Select or create a session to start")}</p>
              <p>
                {t(
                  "playground:acp.selectOrCreateHelp",
                  "ACP connects you to AI coding agents like Claude Code. Create a session, point it at a project directory, and start prompting."
                )}
              </p>
            </div>
          }
        />
      </div>
    )
  }

  // Connecting
  if (state === "connecting") {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <Spin size="large" />
          <div className="mt-4 text-text-muted">
            {t("playground:acp.connecting", "Connecting to session...")}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <div className="custom-scrollbar flex-1 overflow-y-auto p-4">
        {updates.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-text-muted">
              <Bot className="mx-auto h-12 w-12 opacity-50" />
              <p className="mt-4">
                {t("playground:acp.startPrompt", "Send a message to start the conversation")}
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-4">
            {updates.map((update, index) => (
              <UpdateMessage key={index} update={update} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-border bg-surface p-4">
        <div className="mx-auto max-w-3xl">
          <div className="flex gap-2">
            <TextArea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t("playground:acp.inputPlaceholder", "Type your message...")}
              autoSize={{ minRows: 1, maxRows: 6 }}
              disabled={!isConnected || isRunning}
              className="flex-1"
            />
            {isRunning ? (
              <Button
                type="primary"
                danger
                icon={<Square className="h-4 w-4" />}
                onClick={handleCancel}
              >
                {t("playground:acp.stop", "Stop")}
              </Button>
            ) : (
              <Button
                type="primary"
                icon={<Send className="h-4 w-4" />}
                onClick={handleSend}
                disabled={!inputValue.trim() || !isConnected}
              >
                {t("playground:acp.send", "Send")}
              </Button>
            )}
          </div>

          {!isConnected && (
            <div className="mt-2 flex items-center justify-between gap-2 text-xs text-warning">
              <span className="flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                {reconnectInfo?.isReconnecting
                  ? t(
                      "playground:acp.reconnecting",
                      "Reconnecting (attempt {{attempt}}/{{max}})...",
                      { attempt: reconnectInfo.attempt, max: reconnectInfo.maxAttempts }
                    )
                  : t("playground:acp.notConnected", "Not connected to session")}
              </span>
              {!reconnectInfo?.isReconnecting && (
                <Button size="small" onClick={handleReconnect} loading={isReconnecting}>
                  {t("playground:acp.reconnect", "Reconnect")}
                </Button>
              )}
            </div>
          )}

          {error && (
            <div className="mt-2 text-xs text-error">{error}</div>
          )}
        </div>
      </div>
    </div>
  )
}

interface UpdateMessageProps {
  update: ACPUpdate
}

const UpdateMessage: React.FC<UpdateMessageProps> = ({ update }) => {
  const { t } = useTranslation(["playground"])

  const message = parseUpdateMessage(
    update,
    t("playground:acp.operationCancelled", "Operation cancelled"),
  )

  // User message
  if (message.role === "user") {
    return (
      <div className="group flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
          <User className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <div className="text-xs font-medium text-text-muted">{t("playground:acp.role.you", "You")}</div>
            <CopyButton text={message.content} className="opacity-0 group-hover:opacity-100" label={t("playground:acp.copy", "Copy")} />
          </div>
          <div className="mt-1 whitespace-pre-wrap text-text">
            {message.content}
          </div>
        </div>
      </div>
    )
  }

  // Assistant message
  if (message.role === "assistant") {
    return (
      <div className="group flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-success/10">
          <Bot className="h-4 w-4 text-success" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <div className="text-xs font-medium text-text-muted">{t("playground:acp.role.agent", "Agent")}</div>
            <CopyButton text={message.content} className="opacity-0 group-hover:opacity-100" label={t("playground:acp.copy", "Copy")} />
          </div>
          <div className="mt-1 text-text prose prose-sm dark:prose-invert max-w-none [&_code]:bg-surface2 [&_code]:px-1 [&_code]:rounded">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre: ({ children, ...props }) => (
                  <div className="relative group/code">
                    <pre className="overflow-x-auto rounded-lg bg-surface2 p-3 text-sm" {...props}>
                      {children}
                    </pre>
                    <CopyButton
                      text={extractTextFromChildren(children)}
                      className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100"
                    />
                  </div>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    )
  }

  // Tool call
  if (message.role === "tool") {
    const argsText = JSON.stringify(message.toolArgs, null, 2)
    return (
      <div className="group flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-info/10">
          <Wrench className="h-4 w-4 text-info" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <div className="text-xs font-medium text-text-muted">
              {t("playground:acp.role.tool", "Tool")}: {message.toolName}
            </div>
            <CopyButton text={argsText} className="opacity-0 group-hover:opacity-100" label={t("playground:acp.copy", "Copy")} />
          </div>
          <div className="mt-1">
            <ToolArgumentsDisplay args={message.toolArgs} />
          </div>
        </div>
      </div>
    )
  }

  // Tool result
  if (message.role === "tool_result") {
    const resultText =
      typeof message.result === "string"
        ? message.result
        : JSON.stringify(message.result, null, 2)
    return (
      <div className="group flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-info/10">
          <Wrench className="h-4 w-4 text-info" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <div className="text-xs font-medium text-text-muted">
              {t("playground:acp.role.result", "Result")}: {message.toolName}
            </div>
            <CopyButton text={resultText} className="opacity-0 group-hover:opacity-100" label={t("playground:acp.copy", "Copy")} />
          </div>
          <div className="mt-1 rounded-lg bg-surface2 p-2">
            <pre className="overflow-x-auto text-xs text-text-muted">
              {resultText}
            </pre>
          </div>
        </div>
      </div>
    )
  }

  // Error message
  if (message.role === "error") {
    return (
      <div className="group flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-error/10">
          <AlertCircle className="h-4 w-4 text-error" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <div className="text-xs font-medium text-error">{t("playground:acp.role.error", "Error")}</div>
            <CopyButton text={message.content} className="opacity-0 group-hover:opacity-100" label={t("playground:acp.copy", "Copy")} />
          </div>
          <div className="mt-1 whitespace-pre-wrap text-error">
            {message.content}
          </div>
        </div>
      </div>
    )
  }

  // System message
  return (
    <div className="flex justify-center">
      <div className="rounded-lg bg-surface2 px-3 py-1 text-xs text-text-muted">
        {message.content}
      </div>
    </div>
  )
}

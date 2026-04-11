import { useState, useRef, useEffect, useCallback } from "react"
import { Button, Empty, Input, Segmented, Spin, Typography } from "antd"
import { Send } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { TldwChatService } from "@/services/tldw/TldwChat"
import { useStoreChatModelSettings } from "@/store/model"
import { useStorage } from "@plasmohq/storage/hook"
import {
  getManuscriptScene,
  listManuscriptCharacters,
  listManuscriptWorldInfo,
  type ManuscriptCharacter,
  type ManuscriptCharacterListResponse,
  type ManuscriptSceneResponse,
  type ManuscriptWorldInfoItem,
  type ManuscriptWorldInfoListResponse,
} from "@/services/writing-playground"

type AIAgentTabProps = { isOnline: boolean }
type AgentMode = "quick" | "planning" | "brainstorm"
type AgentMessage = { role: "user" | "assistant"; content: string }

const SYSTEM_PROMPTS: Record<AgentMode, string> = {
  quick: "You are a writing assistant. Give brief, direct answers (3 sentences max). The WRITER writes. You ASSIST and ADVISE.",
  planning: "You are a story planning assistant. Help with plot structure, character arcs, and world-building. Provide structured suggestions. The WRITER writes. You ASSIST and ADVISE.",
  brainstorm: "You are a creative brainstorming partner. Generate ideas freely, suggest alternatives, explore possibilities. The WRITER writes. You ASSIST and ADVISE.",
}

const chatService = new TldwChatService()

export function AIAgentTab({ isOnline }: AIAgentTabProps) {
  const { activeProjectId, activeNodeId } = useWritingPlaygroundStore()
  const [selectedModel] = useStorage<string>("selectedModel")
  const apiProvider = useStoreChatModelSettings((state) => state.apiProvider)

  const [mode, setMode] = useState<AgentMode>("quick")
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (typeof messagesEndRef.current?.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  const buildContextSnippet = useCallback(async (): Promise<string> => {
    const parts: string[] = []
    try {
      if (activeNodeId) {
        const scene: ManuscriptSceneResponse = await getManuscriptScene(activeNodeId)
        const sceneContent = scene.content_plain || ""
        if (sceneContent) {
          const snippet = sceneContent.length > 2000
            ? `${sceneContent.slice(0, 2000)}...`
            : sceneContent
          parts.push(`[Current Scene: ${scene.title || "Untitled"}]\n${snippet}`)
        }
      }
      if (activeProjectId) {
        const [charsResp, worldResp] = await Promise.all([
          listManuscriptCharacters(activeProjectId),
          listManuscriptWorldInfo(activeProjectId),
        ])
        const chars = charsResp.characters || []
        if (chars.length > 0) {
          const charList = chars.slice(0, 10).map((c: ManuscriptCharacter) =>
            `- ${c.name} (${c.role || "unknown role"})`
          ).join("\n")
          parts.push(`[Characters]\n${charList}`)
        }
        const items = worldResp.items || []
        if (items.length > 0) {
          const worldList = items.slice(0, 10).map((w: ManuscriptWorldInfoItem) =>
            `- ${w.name} (${w.kind || "info"})`
          ).join("\n")
          parts.push(`[World Info]\n${worldList}`)
        }
      }
    } catch {
      // Context is best-effort; ignore errors
    }
    return parts.length > 0 ? "\n\n--- Manuscript Context ---\n" + parts.join("\n\n") : ""
  }, [activeProjectId, activeNodeId])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: AgentMessage = { role: "user", content: text }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setLoading(true)

    try {
      const context = await buildContextSnippet()
      const systemPrompt = SYSTEM_PROMPTS[mode] + context

      const chatMessages = [
        ...messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content })),
        { role: "user" as const, content: text },
      ]

      const response = await chatService.sendMessage(
        chatMessages,
        {
          model: selectedModel || "default",
          systemPrompt,
          apiProvider: apiProvider || undefined,
          temperature: mode === "brainstorm" ? 0.9 : mode === "quick" ? 0.3 : 0.6,
          maxTokens: mode === "quick" ? 256 : 1024,
        }
      )

      setMessages((prev) => [...prev, { role: "assistant", content: response }])
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err?.message || "Failed to get response"}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!isOnline) {
    return <Empty description="Server offline" />
  }

  return (
    <div className="flex flex-col gap-2 h-full">
      <Segmented
        block
        size="small"
        value={mode}
        onChange={(v) => {
          setMode(v as AgentMode)
          setMessages([])
        }}
        options={[
          { value: "quick", label: "Quick" },
          { value: "planning", label: "Planning" },
          { value: "brainstorm", label: "Brainstorm" },
        ]}
      />

      <div className="flex-1 min-h-0 overflow-y-auto rounded border border-border bg-surface p-2 flex flex-col gap-2">
        {messages.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <Typography.Text type="secondary" className="text-xs">
              {mode === "quick"
                ? "Ask a quick question about your story."
                : mode === "planning"
                ? "Discuss plot structure, arcs, and world-building."
                : "Brainstorm ideas freely."}
            </Typography.Text>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`rounded-md px-2 py-1.5 text-xs max-w-[90%] whitespace-pre-wrap ${
              msg.role === "user"
                ? "self-end bg-primary/10 text-text"
                : "self-start bg-surface-hover text-text"
            }`}
          >
            {msg.content}
          </div>
        ))}
        {loading && (
          <div className="self-start">
            <Spin size="small" />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-1">
        <Input.TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          autoSize={{ minRows: 1, maxRows: 3 }}
          disabled={loading}
          className="flex-1"
          size="small"
        />
        <Button
          type="primary"
          size="small"
          icon={<Send className="h-3.5 w-3.5" />}
          onClick={handleSend}
          disabled={!input.trim() || loading}
          className="self-end"
        />
      </div>
    </div>
  )
}

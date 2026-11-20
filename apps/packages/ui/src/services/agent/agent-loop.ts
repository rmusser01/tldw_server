/**
 * Agent Loop Orchestration
 *
 * Manages the agent execution loop:
 * 1. Send prompt + tools to LLM
 * 2. Execute tool calls from LLM response
 * 3. Collect approvals for write/exec tools
 * 4. Feed results back to LLM
 * 5. Repeat until complete or max steps
 */

import { bgStream } from "@/services/background-proxy"
import * as nativeClient from "@/services/native/native-client"
import { preserveGoogleToolCallSignature } from "@/utils/chat-tools"
import type {
  AgentEvent,
  AgentMessage,
  AgentResult,
  AgentSession,
  AgentSettings,
  PendingApproval,
  ToolCall,
  ToolDefinition
} from "./types"
import { DEFAULT_AGENT_SETTINGS, TOOL_TIERS } from "./types"

const MOONSHOT_MODEL = /^(?:moonshot[:/].+|(?:kimi-k|moonshot-v)[0-9][a-z0-9._-]*)$/i
const MAX_REASONING_LENGTH = 100_000

const AGENT_SYSTEM_PROMPT = `You are an AI coding assistant with access to workspace tools. You can:
- Read and search files in the user's codebase
- Write files and apply patches (with user approval)
- Run git commands
- Execute allowlisted commands like tests and linters

When asked to make changes:
1. First explore the codebase to understand the context
2. Propose changes using clear unified diffs
3. Run tests after making changes

Always explain your reasoning before taking actions. Be precise and avoid unnecessary changes.`

/**
 * Get tool definitions for the LLM
 */
export function getToolDefinitions(): ToolDefinition[] {
  return [
    // Filesystem read tools
    {
      type: "function",
      function: {
        name: "fs_list",
        description: "List directory contents",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Directory path to list (default: current directory)" },
            depth: { type: "integer", description: "Maximum depth to recurse (default: 1)" },
            include_hidden: { type: "boolean", description: "Include hidden files" }
          }
        }
      }
    },
    {
      type: "function",
      function: {
        name: "fs_read",
        description: "Read file contents",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "File path to read" },
            start_line: { type: "integer", description: "Starting line number (1-indexed)" },
            end_line: { type: "integer", description: "Ending line number (inclusive)" }
          },
          required: ["path"]
        }
      }
    },
    // Search tools
    {
      type: "function",
      function: {
        name: "search_grep",
        description: "Search file contents using regex pattern",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Search pattern (regex)" },
            paths: { type: "array", items: { type: "string" }, description: "Paths to search in" },
            glob: { type: "string", description: "File glob pattern (e.g., *.ts)" },
            case_sensitive: { type: "boolean", description: "Case sensitive search" },
            max_results: { type: "integer", description: "Maximum results to return" }
          },
          required: ["pattern"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "search_glob",
        description: "Find files matching a glob pattern",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Glob pattern to match" },
            path: { type: "string", description: "Base path to search from" }
          },
          required: ["pattern"]
        }
      }
    },
    // Git read tools
    {
      type: "function",
      function: {
        name: "git_status",
        description: "Get git repository status",
        parameters: { type: "object", properties: {} }
      }
    },
    {
      type: "function",
      function: {
        name: "git_diff",
        description: "Show git diff",
        parameters: {
          type: "object",
          properties: {
            paths: { type: "array", items: { type: "string" }, description: "Paths to diff" },
            staged: { type: "boolean", description: "Show staged changes" }
          }
        }
      }
    },
    {
      type: "function",
      function: {
        name: "git_log",
        description: "Show recent commits",
        parameters: {
          type: "object",
          properties: {
            count: { type: "integer", description: "Number of commits to show" },
            path: { type: "string", description: "Filter by path" }
          }
        }
      }
    },
    // Write tools
    {
      type: "function",
      function: {
        name: "fs_write",
        description: "Write content to a file (requires approval)",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "File path to write" },
            content: { type: "string", description: "Content to write" }
          },
          required: ["path", "content"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "fs_apply_patch",
        description: "Apply a unified diff patch (requires approval)",
        parameters: {
          type: "object",
          properties: {
            patch: { type: "string", description: "Unified diff to apply" }
          },
          required: ["patch"]
        }
      }
    },
    // Git write tools
    {
      type: "function",
      function: {
        name: "git_add",
        description: "Stage files for commit (requires approval)",
        parameters: {
          type: "object",
          properties: {
            paths: { type: "array", items: { type: "string" }, description: "Paths to stage" }
          },
          required: ["paths"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "git_commit",
        description: "Create a git commit (requires approval)",
        parameters: {
          type: "object",
          properties: {
            message: { type: "string", description: "Commit message" }
          },
          required: ["message"]
        }
      }
    },
    // Exec tools
    {
      type: "function",
      function: {
        name: "exec_run",
        description: "Run an allowlisted command (requires explicit approval). Available commands: pytest, npm_test, go_test, cargo_test, ruff, eslint, prettier, black",
        parameters: {
          type: "object",
          properties: {
            command_id: { type: "string", description: "ID of the allowlisted command (e.g., pytest, npm_test)" },
            args: { type: "array", items: { type: "string" }, description: "Additional arguments" },
            cwd: { type: "string", description: "Working directory" }
          },
          required: ["command_id"]
        }
      }
    }
  ]
}

/**
 * Map function name from LLM to MCP tool name
 */
function mapToolName(llmName: string): string {
  const mapping: Record<string, string> = {
    "fs_list": "fs.list",
    "fs_read": "fs.read",
    "fs_write": "fs.write",
    "fs_apply_patch": "fs.apply_patch",
    "fs_mkdir": "fs.mkdir",
    "fs_delete": "fs.delete",
    "search_grep": "search.grep",
    "search_glob": "search.glob",
    "git_status": "git.status",
    "git_diff": "git.diff",
    "git_log": "git.log",
    "git_branch": "git.branch",
    "git_add": "git.add",
    "git_commit": "git.commit",
    "exec_run": "exec.run"
  }
  return mapping[llmName] || llmName
}

/**
 * Agent loop class
 */
export class AgentLoop {
  private session: AgentSession
  private settings: AgentSettings
  private eventHandler: (event: AgentEvent) => void
  private cancelled = false

  constructor(
    workspaceId: string,
    task: string,
    settings: Partial<AgentSettings> = {},
    onEvent: (event: AgentEvent) => void
  ) {
    this.settings = { ...DEFAULT_AGENT_SETTINGS, ...settings }
    this.eventHandler = onEvent
    this.session = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      workspaceId,
      task,
      messages: [
        { role: "system", content: AGENT_SYSTEM_PROMPT },
        { role: "user", content: task }
      ],
      status: "idle",
      currentStep: 0,
      pendingApprovals: [],
      createdAt: new Date(),
      updatedAt: new Date()
    }
  }

  /**
   * Cancel the agent loop
   */
  cancel(): void {
    this.cancelled = true
    this.session.status = "cancelled"
  }

  /**
   * Get current session state
   */
  getSession(): AgentSession {
    return { ...this.session }
  }

  /**
   * Approve pending tool calls
   */
  approvePending(toolCallIds: string[]): void {
    for (const approval of this.session.pendingApprovals) {
      if (toolCallIds.includes(approval.toolCallId)) {
        approval.status = "approved"
      }
    }
  }

  /**
   * Reject pending tool calls
   */
  rejectPending(toolCallIds: string[]): void {
    for (const approval of this.session.pendingApprovals) {
      if (toolCallIds.includes(approval.toolCallId)) {
        approval.status = "rejected"
      }
    }
  }

  /**
   * Run the agent loop
   */
  async run(): Promise<AgentResult> {
    this.session.status = "running"

    try {
      while (this.session.currentStep < this.settings.maxSteps && !this.cancelled) {
        this.session.currentStep++
        this.emit({ type: "step_start", step: this.session.currentStep })

        // Call LLM
        this.emit({ type: "llm_start" })
        const llmResponse = await this.callLLM()
        this.emit({
          type: "llm_complete",
          content: llmResponse.content,
          tool_calls: llmResponse.tool_calls
        })

        // Check if we're done (no tool calls)
        if (!llmResponse.tool_calls || llmResponse.tool_calls.length === 0) {
          this.session.status = "complete"
          const result: AgentResult = {
            status: "complete",
            response: llmResponse.content,
            stepsCompleted: this.session.currentStep
          }
          this.emit({ type: "complete", result })
          return result
        }

        // Record calls before their tool responses so providers can match them.
        this.session.messages.push({ role: "assistant", ...llmResponse })
        await this.processToolCalls(llmResponse.tool_calls)

        this.session.updatedAt = new Date()
      }

      // Max steps reached
      this.session.status = "complete"
      const result: AgentResult = {
        status: this.cancelled ? "cancelled" : "max_steps_reached",
        stepsCompleted: this.session.currentStep
      }
      this.emit({ type: "complete", result })
      return result
    } catch (e: any) {
      this.session.status = "error"
      const result: AgentResult = {
        status: "error",
        error: e.message,
        stepsCompleted: this.session.currentStep
      }
      this.emit({ type: "error", error: e.message })
      return result
    }
  }

  /**
   * Call the LLM
   */
  private async callLLM(): Promise<Pick<AgentMessage, "content" | "tool_calls" | "reasoning_content">> {
    let content = ""
    let reasoning: string | null = ""
    let tool_calls: ToolCall[] = []

    // Use streaming to collect response
    const stream = bgStream({
      path: "/api/v1/chat/completions",
      method: "POST",
      body: {
        model: this.settings.model,
        messages: this.session.messages,
        tools: getToolDefinitions(),
        tool_choice: "auto",
        stream: true
      }
    })

    for await (const chunk of stream) {
      // Parse SSE data
      if (chunk.startsWith("data: ")) {
        const data = chunk.slice(6)
        if (data === "[DONE]") break

        try {
          const parsed = JSON.parse(data)
          const delta = parsed.choices?.[0]?.delta

          if (delta?.content) {
            content += delta.content
            this.emit({ type: "llm_chunk", content: delta.content })
          }

          const reasoningDelta = delta?.reasoning_content ?? parsed.additional_kwargs?.reasoning_content
          const model = this.settings.model || parsed.model
          if (
            reasoning !== null &&
            typeof reasoningDelta === "string" &&
            typeof model === "string" &&
            MOONSHOT_MODEL.test(model.trim())
          ) {
            reasoning = reasoning.length + reasoningDelta.length <= MAX_REASONING_LENGTH
              ? reasoning + reasoningDelta
              : null
          }

          if (delta?.tool_calls) {
            // Aggregate tool calls across chunks
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0
              if (!tool_calls[idx]) {
                tool_calls[idx] = {
                  id: tc.id || `tool_${idx}`,
                  type: "function",
                  function: { name: "", arguments: "" }
                }
              }
              if (tc.id) tool_calls[idx].id = tc.id
              preserveGoogleToolCallSignature(
                tool_calls[idx],
                tc.extra_content?.google?.thought_signature
              )
              if (tc.function?.name) tool_calls[idx].function.name += tc.function.name
              if (tc.function?.arguments) tool_calls[idx].function.arguments += tc.function.arguments
            }
          }
        } catch {
          // Ignore parse errors for non-JSON lines
        }
      }
    }

    // Filter out empty tool calls
    tool_calls = tool_calls.filter(tc => tc.function.name)

    if (tool_calls.length === 0) return { content }
    return {
      content,
      tool_calls,
      ...(reasoning ? { reasoning_content: reasoning } : {})
    }
  }

  /**
   * Process tool calls from LLM response
   */
  private async processToolCalls(toolCalls: ToolCall[]): Promise<void> {
    // Categorize tool calls by approval tier
    const autoApprove: ToolCall[] = []
    const needsApproval: PendingApproval[] = []

    for (const tc of toolCalls) {
      const mcpName = mapToolName(tc.function.name)
      const tier = TOOL_TIERS[mcpName] || "individual"

      const shouldAutoApprove =
        tier === "auto" ||
        (tier === "batch" && this.settings.autoApproveWrites) ||
        (tier === "individual" && this.settings.autoApproveExec)

      if (shouldAutoApprove) {
        autoApprove.push(tc)
      } else {
        needsApproval.push({
          toolCallId: tc.id,
          toolName: mcpName,
          args: JSON.parse(tc.function.arguments || "{}"),
          tier,
          status: "pending"
        })
      }
    }

    // Execute auto-approved tools
    for (const tc of autoApprove) {
      await this.executeToolCall(tc)
    }

    // Request approval for remaining tools
    if (needsApproval.length > 0) {
      this.session.pendingApprovals = needsApproval
      this.session.status = "waiting_approval"
      this.emit({ type: "approval_needed", approvals: needsApproval })

      // Wait for approvals
      await this.waitForApprovals()

      // Execute approved tools
      for (const approval of this.session.pendingApprovals) {
        const tc = toolCalls.find(t => t.id === approval.toolCallId)
        if (tc && approval.status === "approved") {
          await this.executeToolCall(tc)
        } else if (tc && approval.status === "rejected") {
          // Add rejection message
          this.session.messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: JSON.stringify({ ok: false, error: "User rejected this action" })
          })
        }
      }

      this.session.pendingApprovals = []
      this.session.status = "running"
    }
  }

  /**
   * Execute a single tool call
   */
  private async executeToolCall(tc: ToolCall): Promise<void> {
    this.emit({ type: "tool_start", tool_call: tc })

    const mcpName = mapToolName(tc.function.name)
    const args = JSON.parse(tc.function.arguments || "{}")

    try {
      const result = await nativeClient.executeTool(mcpName, args)
      this.session.messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: JSON.stringify(result)
      })
      this.emit({ type: "tool_complete", tool_call_id: tc.id, result })
    } catch (e: any) {
      const errorResult = { ok: false, error: e.message }
      this.session.messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: JSON.stringify(errorResult)
      })
      this.emit({ type: "tool_complete", tool_call_id: tc.id, result: errorResult })
    }
  }

  /**
   * Wait for all pending approvals to be resolved
   */
  private async waitForApprovals(): Promise<void> {
    while (
      this.session.pendingApprovals.some(a => a.status === "pending") &&
      !this.cancelled
    ) {
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }

  /**
   * Emit an event
   */
  private emit(event: AgentEvent): void {
    try {
      this.eventHandler(event)
    } catch (e) {
      console.error("[agent-loop] Error in event handler:", e)
    }
  }
}

/**
 * Create and run an agent
 */
export async function runAgent(
  workspaceId: string,
  task: string,
  settings: Partial<AgentSettings> = {},
  onEvent: (event: AgentEvent) => void
): Promise<AgentResult> {
  const agent = new AgentLoop(workspaceId, task, settings, onEvent)
  return agent.run()
}

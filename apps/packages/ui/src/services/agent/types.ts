/**
 * Types for the agent loop orchestration
 */

import type { ToolCall as ChatToolCall } from "@/types/tool-calls"

/**
 * Message in the conversation history
 */
export interface AgentMessage {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  tool_calls?: ToolCall[]
  reasoning_content?: string
  tool_call_id?: string
  name?: string
}

/**
 * Tool call from the LLM
 */
export type ToolCall = ChatToolCall & { function: { arguments: string } }

/**
 * Tool definition for the LLM
 */
export interface ToolDefinition {
  type: "function"
  function: {
    name: string
    description: string
    parameters: Record<string, any>
  }
}

/**
 * Agent session state
 */
export interface AgentSession {
  id: string
  workspaceId: string
  task: string
  messages: AgentMessage[]
  status: AgentStatus
  currentStep: number
  pendingApprovals: PendingApproval[]
  createdAt: Date
  updatedAt: Date
}

export type AgentStatus =
  | "idle"
  | "running"
  | "waiting_approval"
  | "complete"
  | "error"
  | "cancelled"

/**
 * Pending approval for a tool call
 */
export interface PendingApproval {
  toolCallId: string
  toolName: string
  args: Record<string, any>
  tier: ApprovalTier
  status: "pending" | "approved" | "rejected"
}

export type ApprovalTier = "auto" | "batch" | "individual"

/**
 * Tool tier configuration
 */
export const TOOL_TIERS: Record<string, ApprovalTier> = {
  // Tier 0: Auto-approve (read-only)
  "workspace.list": "auto",
  "workspace.pwd": "auto",
  "workspace.chdir": "auto",
  "fs.list": "auto",
  "fs.read": "auto",
  "search.grep": "auto",
  "search.glob": "auto",
  "git.status": "auto",
  "git.diff": "auto",
  "git.log": "auto",
  "git.branch": "auto",

  // Tier 1: Batch approval (show diff, approve all at once)
  "fs.write": "batch",
  "fs.apply_patch": "batch",
  "fs.mkdir": "batch",
  "git.add": "batch",
  "git.commit": "batch",

  // Tier 2: Individual approval
  "fs.delete": "individual",
  "exec.run": "individual",
  "git.push": "individual"
}

/**
 * Agent loop result
 */
export interface AgentResult {
  status: "complete" | "max_steps_reached" | "cancelled" | "error"
  response?: string
  error?: string
  stepsCompleted: number
}

/**
 * Event emitted during agent execution
 */
export type AgentEvent =
  | { type: "step_start"; step: number }
  | { type: "llm_start" }
  | { type: "llm_chunk"; content: string }
  | { type: "llm_complete"; content: string; tool_calls?: ToolCall[] }
  | { type: "tool_start"; tool_call: ToolCall }
  | { type: "tool_complete"; tool_call_id: string; result: any }
  | { type: "approval_needed"; approvals: PendingApproval[] }
  | { type: "complete"; result: AgentResult }
  | { type: "error"; error: string }

/**
 * Settings for the agent
 */
export interface AgentSettings {
  maxSteps: number
  autoApproveReads: boolean
  autoApproveWrites: boolean
  autoApproveExec: boolean
  showToolCallsInChat: boolean
  model?: string
}

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
  maxSteps: 20,
  autoApproveReads: true,
  autoApproveWrites: false,
  autoApproveExec: false,
  showToolCallsInChat: true
}

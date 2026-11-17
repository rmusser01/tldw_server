/**
 * ACP Constants and Configuration
 */

import type { ACPPermissionTier, ACPMCPServerConfig } from "./types"

// -----------------------------------------------------------------------------
// Tool Tier Mapping
// -----------------------------------------------------------------------------

/**
 * Default tool tier mappings based on tool name patterns.
 * These determine whether a tool requires approval and at what level.
 *
 * - auto: No approval needed (read-only operations)
 * - batch: Can approve multiple at once (write operations)
 * - individual: Must approve each one (destructive/execute operations)
 */
export const TOOL_TIER_PATTERNS: Array<{
  pattern: RegExp
  tier: ACPPermissionTier
}> = [
  // Auto-approve: read-only operations
  { pattern: /^(read|get|list|search|find|view|show|glob|grep)/i, tier: "auto" },
  { pattern: /(read|view|list|search)$/i, tier: "auto" },

  // Individual approval: destructive/execute operations
  { pattern: /^(delete|remove|exec|run|shell|bash|terminal|push|force)/i, tier: "individual" },
  { pattern: /(delete|remove|execute|run|push)$/i, tier: "individual" },

  // Everything else defaults to batch
]

/**
 * Explicit tool name to tier mappings
 */
export const TOOL_TIERS: Record<string, ACPPermissionTier> = {
  // File system - read
  "fs.read": "auto",
  "fs.list": "auto",
  "fs.stat": "auto",
  "fs.exists": "auto",

  // File system - write
  "fs.write": "batch",
  "fs.mkdir": "batch",
  "fs.copy": "batch",
  "fs.move": "batch",
  "fs.patch": "batch",
  "fs.apply_patch": "batch",

  // File system - destructive
  "fs.delete": "individual",
  "fs.remove": "individual",
  "fs.rmdir": "individual",

  // Search
  "search.grep": "auto",
  "search.glob": "auto",
  "search.find": "auto",

  // Git - read
  "git.status": "auto",
  "git.diff": "auto",
  "git.log": "auto",
  "git.branch": "auto",
  "git.show": "auto",

  // Git - write
  "git.add": "batch",
  "git.commit": "batch",
  "git.checkout": "batch",
  "git.merge": "batch",
  "git.rebase": "batch",
  "git.stash": "batch",

  // Git - destructive
  "git.push": "individual",
  "git.reset": "individual",
  "git.force_push": "individual",

  // Execution
  "exec.run": "individual",
  "exec.shell": "individual",
  "terminal.run": "individual",
  "bash.run": "individual",

  // Network
  "http.get": "auto",
  "http.post": "batch",
  "http.put": "batch",
  "http.delete": "individual",

  // Workspace
  "workspace.list": "auto",
  "workspace.pwd": "auto",
  "workspace.chdir": "auto",
}

/**
 * Get the permission tier for a tool
 */
export function getToolTier(toolName: string): ACPPermissionTier {
  // Check explicit mapping first
  if (toolName in TOOL_TIERS) {
    return TOOL_TIERS[toolName]
  }

  // Check patterns
  for (const { pattern, tier } of TOOL_TIER_PATTERNS) {
    if (pattern.test(toolName)) {
      return tier
    }
  }

  // Default to batch (safer than auto, less disruptive than individual)
  return "batch"
}

// -----------------------------------------------------------------------------
// WebSocket Configuration
// -----------------------------------------------------------------------------

export const WS_CONFIG = {
  /** Reconnection delay in ms */
  RECONNECT_DELAY_MS: 1000,
  /** Maximum reconnection delay in ms */
  MAX_RECONNECT_DELAY_MS: 30000,
  /** Reconnection delay multiplier for exponential backoff */
  RECONNECT_BACKOFF_MULTIPLIER: 1.5,
  /** Maximum reconnection attempts before giving up */
  MAX_RECONNECT_ATTEMPTS: 10,
  /** Heartbeat interval from server (should match backend) */
  HEARTBEAT_INTERVAL_MS: 30000,
  /** Consider connection dead if no heartbeat for this duration */
  HEARTBEAT_TIMEOUT_MS: 45000,
} as const

// 4403: valid credentials the server will not admit (an API key without the route's scope,
// or a cookie from an untrusted origin); retrying cannot change the answer.
export const ACP_NON_RETRYABLE_CLOSE_CODES = new Set([4401, 4403, 4404, 4429])

export function shouldRetryACPWebSocketClose(code: number): boolean {
  return !ACP_NON_RETRYABLE_CLOSE_CODES.has(code)
}

// -----------------------------------------------------------------------------
// Session Configuration
// -----------------------------------------------------------------------------

export const SESSION_CONFIG = {
  /** Session expiry time in ms (24 hours) */
  SESSION_EXPIRY_MS: 24 * 60 * 60 * 1000,
  /** Maximum number of updates to keep in memory per session */
  MAX_UPDATES_PER_SESSION: 1000,
  /** Permission request timeout in seconds (should match backend) */
  PERMISSION_TIMEOUT_SECONDS: 300,
} as const

// -----------------------------------------------------------------------------
// UI Configuration
// -----------------------------------------------------------------------------

export const UI_CONFIG = {
  /** Permission tier badge colors */
  TIER_COLORS: {
    auto: "green",
    batch: "yellow",
    individual: "red",
  } as const,
  /** Permission tier labels */
  TIER_LABELS: {
    auto: "Auto-approve",
    batch: "Batch",
    individual: "Review Required",
  } as const,
  /** Permission tier descriptions */
  TIER_DESCRIPTIONS: {
    auto: "This operation is read-only and will be automatically approved.",
    batch: "This operation modifies files. You can approve multiple at once.",
    individual: "This operation is potentially destructive. Review carefully.",
  } as const,
}

// -----------------------------------------------------------------------------
// MCP Server Presets
// -----------------------------------------------------------------------------

export const MCP_SERVER_PRESETS: ACPMCPServerConfig[] = [
  {
    name: "TLDW Server",
    type: "websocket",
    url: "ws://localhost:8000/mcp",
  },
  {
    name: "Filesystem",
    type: "stdio",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  },
  {
    name: "Custom",
    type: "websocket",
    url: "",
  },
]

// -----------------------------------------------------------------------------
// Agent Type Display Info
// -----------------------------------------------------------------------------

export const AGENT_TYPE_INFO = {
  claude_code: {
    name: "Claude Code",
    icon: "anthropic",
    color: "amber",
  },
  codex: {
    name: "OpenAI Codex",
    icon: "openai",
    color: "green",
  },
  opencode: {
    name: "OpenCode",
    icon: "code",
    color: "purple",
  },
  custom: {
    name: "Custom",
    icon: "settings",
    color: "blue",
  },
} as const

// -----------------------------------------------------------------------------
// Error Codes
// -----------------------------------------------------------------------------

export const ACP_ERROR_CODES = {
  // Connection errors
  CONNECTION_FAILED: "connection_failed",
  CONNECTION_TIMEOUT: "connection_timeout",
  AUTH_FAILED: "auth_failed",

  // Session errors
  SESSION_NOT_FOUND: "session_not_found",
  SESSION_CREATION_FAILED: "session_creation_failed",
  INVALID_CWD: "invalid_cwd",
  CWD_NOT_FOUND: "cwd_not_found",
  CWD_PERMISSION_DENIED: "cwd_permission_denied",

  // Agent errors
  AGENT_NOT_CONFIGURED: "agent_not_configured",
  AGENT_START_FAILED: "agent_start_failed",
  AGENT_NOT_RESPONDING: "agent_not_responding",

  // MCP errors
  MCP_CONNECTION_FAILED: "mcp_connection_failed",
  MCP_SERVER_ERROR: "mcp_server_error",

  // Generic
  INTERNAL_ERROR: "internal_error",
  UNKNOWN_ERROR: "unknown_error",
} as const

export type ACPErrorCode = typeof ACP_ERROR_CODES[keyof typeof ACP_ERROR_CODES]

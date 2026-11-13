import { useQuery } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { isRecoverableAuthConfigError } from "@/services/auth-errors"
import { useConnectionStore } from "@/store/connection"
import type { SlashCommandItem } from "@/types/slash-command"

/**
 * Server chat commands (`GET /api/v1/chat/commands`, e.g. /time, /weather).
 * They are RBAC-gated and off by default, so an empty list or a failed read
 * just leaves the composer with its local commands. A server command with no
 * local action is sent as typed; `/chat/completions` runs it.
 */

type ServerSlashCommand = {
  id?: string | number
  command?: string
  name?: string
  label?: string
  description?: string
  keywords?: string[]
  permissions?: string[] | string
  required_permissions?: string[] | string
  requiredPermissions?: string[] | string
  permission?: string
  allowed?: boolean
  enabled?: boolean
}

export type ServerCommandsResponse = {
  commands?: ServerSlashCommand[]
  data?: ServerSlashCommand[]
  results?: ServerSlashCommand[]
  permissions?: string[]
  user_permissions?: string[]
  userPermissions?: string[]
  allowed_commands?: string[]
  allowedCommands?: string[]
}

export type SlashCommandMeta = {
  label: string
  description: string
  keywords: string[]
  insertText?: string
}

export const SERVER_SLASH_COMMANDS_QUERY_KEY = ["tldw:chat:slashCommands"] as const

export const normalizeSlashCommandName = (value: string) =>
  value.replace(/^\/+/, "").trim().toLowerCase()

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const normalizeStringArray = (value?: unknown): string[] => {
  if (value == null) return []
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item).trim())
      .filter((item) => item.length > 0)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed ? [trimmed] : []
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return [String(value)]
  }
  return []
}

const isServerSlashCommand = (value: unknown): value is ServerSlashCommand => {
  if (!isRecord(value)) return false
  const id = value.id
  if (id != null && typeof id !== "string" && typeof id !== "number") {
    return false
  }
  const stringFields = ["command", "name", "label", "description"] as const
  for (const field of stringFields) {
    const fieldValue = value[field]
    if (fieldValue != null && typeof fieldValue !== "string") {
      return false
    }
  }
  const booleanFields = ["allowed", "enabled"] as const
  for (const field of booleanFields) {
    const fieldValue = value[field]
    if (fieldValue != null && typeof fieldValue !== "boolean") {
      return false
    }
  }
  return true
}

const extractServerCommands = (payload: unknown): ServerSlashCommand[] => {
  if (Array.isArray(payload)) {
    return payload.filter(isServerSlashCommand)
  }
  if (!isRecord(payload)) return []
  if (Array.isArray(payload.commands)) {
    return payload.commands.filter(isServerSlashCommand)
  }
  if (Array.isArray(payload.data)) {
    return payload.data.filter(isServerSlashCommand)
  }
  if (Array.isArray(payload.results)) {
    return payload.results.filter(isServerSlashCommand)
  }
  return []
}

const extractPermissions = (payload: unknown): string[] => {
  if (!isRecord(payload)) return []
  const perms =
    payload.permissions || payload.user_permissions || payload.userPermissions
  return normalizeStringArray(perms)
}

const extractAllowedCommands = (payload: unknown): string[] | null => {
  if (!isRecord(payload)) return null
  const allowed = payload.allowed_commands || payload.allowedCommands
  const list = normalizeStringArray(allowed)
  return list.length > 0 ? list.map(normalizeSlashCommandName) : null
}

export const normalizeServerCommandsPayload = (
  payload: unknown
): ServerCommandsResponse | null => {
  if (payload == null) return null
  if (Array.isArray(payload)) {
    return { commands: payload.filter(isServerSlashCommand) }
  }
  if (!isRecord(payload)) return null
  const hasCommandArray =
    Array.isArray(payload.commands) ||
    Array.isArray(payload.data) ||
    Array.isArray(payload.results)
  const commands = extractServerCommands(payload)
  const permissions = extractPermissions(payload)
  const allowedCommands = extractAllowedCommands(payload)
  if (!hasCommandArray && permissions.length === 0 && !allowedCommands) {
    return null
  }
  return {
    ...(hasCommandArray ? { commands } : {}),
    ...(permissions.length ? { permissions } : {}),
    ...(allowedCommands ? { allowed_commands: allowedCommands } : {})
  }
}

/**
 * Read the server's chat commands once per minute while `enabled`. Auth and
 * configuration errors ask the connection store to re-check and resolve to
 * no commands; other failures stay in the query's error state.
 */
export const useServerSlashCommandPayload = (enabled: boolean) => {
  const checkConnection = useConnectionStore((state) => state.checkOnce)
  const { data } = useQuery({
    queryKey: SERVER_SLASH_COMMANDS_QUERY_KEY,
    queryFn: async (): Promise<ServerCommandsResponse | null> => {
      try {
        const payload = (await tldwClient.listChatCommands()) as unknown
        const normalized = normalizeServerCommandsPayload(payload)
        if (!normalized && payload != null) {
          console.warn("Unable to parse slash commands response:", payload)
        }
        return normalized
      } catch (error) {
        if (isRecoverableAuthConfigError(error)) {
          void checkConnection().catch(() => null)
          return null
        }
        throw error
      }
    },
    enabled,
    staleTime: 60_000,
    retry: (failureCount, error) =>
      !isRecoverableAuthConfigError(error) && failureCount < 2
  })
  return data ?? null
}

/**
 * The commands the server offers this user. A command the composer also
 * defines keeps its local action, and its local description and keywords
 * unless the server sends its own.
 */
export const buildServerSlashCommands = (
  payload: ServerCommandsResponse | null,
  fallbackMeta: Record<string, SlashCommandMeta | undefined>,
  localActions: Record<string, (() => void) | undefined>
): SlashCommandItem[] => {
  if (!payload) return []
  const commands = extractServerCommands(payload)
  if (commands.length === 0) return []
  const allowedCommands = extractAllowedCommands(payload)
  const userPermissions = extractPermissions(payload)

  return commands
    .map((command, index) => {
      const commandName = normalizeSlashCommandName(
        command.command || command.name || ""
      )
      if (!commandName) return null
      if (command.allowed === false || command.enabled === false) return null
      if (allowedCommands && !allowedCommands.includes(commandName)) {
        return null
      }
      const requiredPermissions = normalizeStringArray(
        command.required_permissions ||
          command.requiredPermissions ||
          command.permissions ||
          command.permission
      )
      if (requiredPermissions.length > 0) {
        if (!userPermissions || userPermissions.length === 0) {
          return null
        }
        if (
          !requiredPermissions.every((perm) => userPermissions.includes(perm))
        ) {
          return null
        }
      }
      const fallback = fallbackMeta[commandName]
      return {
        id: String(command.id ?? `server-${commandName}-${index}`),
        command: commandName,
        label: command.label || command.name || fallback?.label || commandName,
        description: command.description || fallback?.description || undefined,
        keywords: normalizeStringArray(command.keywords || fallback?.keywords),
        insertText: fallback?.insertText,
        action: localActions[commandName],
        source: "server"
      } satisfies SlashCommandItem
    })
    .filter(Boolean) as SlashCommandItem[]
}

/** Server commands first; local commands fill in the names the server lacks. */
export const mergeSlashCommands = (
  serverCommands: SlashCommandItem[],
  localCommands: SlashCommandItem[]
): SlashCommandItem[] => {
  if (serverCommands.length === 0) return localCommands
  const merged = new Map<string, SlashCommandItem>()
  serverCommands.forEach((command) => merged.set(command.command, command))
  localCommands.forEach((command) => {
    if (!merged.has(command.command)) merged.set(command.command, command)
  })
  return Array.from(merged.values())
}

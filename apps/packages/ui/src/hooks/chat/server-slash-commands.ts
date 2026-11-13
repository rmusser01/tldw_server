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
  /** How to call it, e.g. "/weather <place>". */
  usage?: string
  args?: string[]
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
  const stringFields = ["command", "name", "label", "description", "usage"] as const
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

/** The server's description, with how to call the command when it says. */
const describeServerCommand = (
  command: ServerSlashCommand,
  name: string,
  localDescription?: string
) => {
  const args = normalizeStringArray(command.args)
  const usage =
    command.usage?.trim() ||
    (args.length > 0 ? `/${name} ${args.map((arg) => `<${arg}>`).join(" ")}` : "")
  // The server falls back to the bare name when a command has no description.
  const rawName = (command.command || command.name || "").replace(/^\/+/, "").trim()
  const description =
    command.description && command.description.trim() !== rawName
      ? command.description.trim()
      : localDescription
  return [description, usage].filter(Boolean).join(" · ") || undefined
}

/**
 * The commands the server offers this user. A command the composer also
 * defines keeps the local command's action and behaviour (standalone,
 * insert text), and its description and keywords unless the server sends
 * its own.
 */
export const buildServerSlashCommands = (
  payload: ServerCommandsResponse | null,
  localCommands: SlashCommandItem[]
): SlashCommandItem[] => {
  if (!payload) return []
  const commands = extractServerCommands(payload)
  if (commands.length === 0) return []
  const allowedCommands = extractAllowedCommands(payload)
  const userPermissions = extractPermissions(payload)
  const localByName = new Map(localCommands.map((command) => [command.command, command]))

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
      const local = localByName.get(commandName)
      const keywords = normalizeStringArray(command.keywords)
      return {
        ...local,
        id: String(command.id ?? `server-${commandName}-${index}`),
        command: commandName,
        label: command.label || command.name || local?.label || commandName,
        description: describeServerCommand(command, commandName, local?.description),
        keywords: keywords.length > 0 ? keywords : local?.keywords ?? [],
        source: "server"
      } satisfies SlashCommandItem
    })
    .filter(Boolean) as SlashCommandItem[]
}

/**
 * One list of server and local commands; a server command replaces the local
 * one of the same name. By default server commands come first (the side
 * panel). With `localFirst`, the local commands keep their places and the
 * server's own commands follow, so a list that arrives later does not move
 * the highlighted row (the web composer).
 */
export const mergeSlashCommands = (
  serverCommands: SlashCommandItem[],
  localCommands: SlashCommandItem[],
  { localFirst = false }: { localFirst?: boolean } = {}
): SlashCommandItem[] => {
  if (serverCommands.length === 0) return localCommands
  if (localFirst) {
    const serverByName = new Map(serverCommands.map((command) => [command.command, command]))
    const localNames = new Set(localCommands.map((command) => command.command))
    return [
      ...localCommands.map((command) => serverByName.get(command.command) ?? command),
      ...serverCommands.filter((command) => !localNames.has(command.command))
    ]
  }
  const merged = new Map<string, SlashCommandItem>()
  serverCommands.forEach((command) => merged.set(command.command, command))
  localCommands.forEach((command) => {
    if (!merged.has(command.command)) merged.set(command.command, command)
  })
  return Array.from(merged.values())
}

/**
 * Pure utility functions, constants, and types for the Characters Manager.
 *
 * These are module-level helpers with no React dependencies.
 */

import { parseCharacterTags } from "./tag-manager-utils"
import {
  CHARACTER_PROMPT_PRESETS,
  DEFAULT_CHARACTER_PROMPT_PRESET,
  isCharacterPromptPresetId,
  type CharacterPromptPresetId
} from "@/data/character-prompt-presets"
export { resolveCharacterSelectionId } from "@/utils/default-character-preference"
import { CHARACTER_TEMPLATES } from "@/data/character-templates"
import { extractAvatarValues } from "./AvatarField"
import { validateAndCreateImageDataUrl } from "@/utils/image-utils"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const MAX_NAME_LENGTH = 500
export const MAX_NAME_DISPLAY_LENGTH = 75
export const MAX_DESCRIPTION_LENGTH = 65
export const MAX_TAG_LENGTH = 20
export const MAX_TAGS_DISPLAYED = 6
export const MAX_TABLE_TAGS_DISPLAYED = 2
export const DEFAULT_PAGE_SIZE = 10
export const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const
export const SERVER_QUERY_ROLLOUT_FLAG_KEY = "ff_characters_server_query"
export const TEMPLATE_CHOOSER_SEEN_KEY = "characters-template-chooser-seen"
export const SYSTEM_PROMPT_EXAMPLE =
  CHARACTER_TEMPLATES.find((template) => template.id === "writing-assistant")
    ?.system_prompt ??
  "You are a skilled writing assistant who helps users improve drafts with clear, specific, and encouraging feedback."

export const CHARACTER_FOLDER_TOKEN_PREFIX = "__tldw_folder_id:"
export const CHARACTER_FOLDER_TOKEN_PREFIXES = [
  CHARACTER_FOLDER_TOKEN_PREFIX,
  "__tldw_folder:"
] as const

export const IMPORT_ALLOWED_EXTENSIONS = [
  ".png",
  ".webp",
  ".jpeg",
  ".jpg",
  ".json",
  ".yaml",
  ".yml",
  ".md",
  ".txt"
] as const
export const IMPORT_ALLOWED_EXTENSION_SET = new Set<string>(IMPORT_ALLOWED_EXTENSIONS)
export const IMPORT_UPLOAD_ACCEPT = IMPORT_ALLOWED_EXTENSIONS.join(",")
export const IMPORT_IMAGE_EXTENSIONS = new Set([".png", ".webp", ".jpeg", ".jpg"])

export const EMPTY_CHARACTER_WORLD_BOOK_DATA: {
  options: CharacterWorldBookOption[]
  attachedIds: number[]
} = {
  options: [],
  attachedIds: []
}

export const DATE_INPUT_PATTERN = /^\d{4}-\d{2}-\d{2}$/

export const CHARACTER_VERSION_DIFF_FIELD_KEYS = [
  "name",
  "description",
  "system_prompt",
  "first_message",
  "personality",
  "scenario",
  "post_history_instructions",
  "message_example",
  "creator_notes",
  "alternate_greetings",
  "tags",
  "creator",
  "character_version",
  "extensions"
] as const

export const CHARACTER_VERSION_FIELD_LABELS: Record<
  (typeof CHARACTER_VERSION_DIFF_FIELD_KEYS)[number],
  string
> = {
  name: "Name",
  description: "Description",
  system_prompt: "System prompt",
  first_message: "Greeting",
  personality: "Personality",
  scenario: "Scenario",
  post_history_instructions: "Post-history instructions",
  message_example: "Message example",
  creator_notes: "Creator notes",
  alternate_greetings: "Alternate greetings",
  tags: "Tags",
  creator: "Creator",
  character_version: "Character version",
  extensions: "Extensions"
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AdvancedSectionKey = "promptControl" | "generationSettings" | "metadata"
export type CharacterListScope = "active" | "deleted"
export type TableDensity = "comfortable" | "compact" | "dense"

export type DefaultCharacterPreferenceQueryResult = {
  defaultCharacterId: string | null
}

export type PersonaProfileSummary = {
  id?: string | number | null
  name?: string | null
  character_card_id?: number | null
  origin_character_id?: number | null
}

export type PersonaGardenAction = "open" | "create"

export type PersonaGardenActionContext = {
  characterId: number
  characterName: string
  profiles: PersonaProfileSummary[]
  existingPersona: PersonaProfileSummary | null
}

export type CharacterComparisonRow = {
  field: string
  label: string
  leftValue: string
  rightValue: string
  different: boolean
}

export type CharacterComparisonFieldDefinition = {
  field: string
  label: string
  getValue: (record: any) => unknown
}

export type CharacterImportPreview = {
  id: string
  file: File
  fileName: string
  format: string
  name: string
  description: string
  tagCount: number
  fieldCount: number
  avatarUrl: string | null
  parseError: {
    key: string
    fallback: string
    values?: Record<string, string | number>
  } | null
}

export type CharacterWorldBookOption = {
  id: number
  name: string
  enabled?: boolean
}

export type CharacterFolderOption = {
  id: number
  name: string
}

export type CharacterRecoveryTelemetryAction =
  | "delete"
  | "undo"
  | "restore"
  | "restore_failed"
  | "bulk_delete"
  | "bulk_undo"
  | "bulk_restore"
  | "bulk_restore_failed"

export type CharacterGenerationSettings = {
  temperature?: number
  top_p?: number
  repetition_penalty?: number
  stop?: string[]
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

export const normalizePageSize = (value: unknown): number => {
  const parsed =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number.parseInt(value, 10)
        : Number.NaN
  return PAGE_SIZE_OPTIONS.includes(parsed as (typeof PAGE_SIZE_OPTIONS)[number])
    ? parsed
    : DEFAULT_PAGE_SIZE
}

export const emitCharacterRecoveryTelemetry = (
  action: CharacterRecoveryTelemetryAction,
  payload?: Record<string, unknown>
) => {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent("tldw:characters-recovery", {
      detail: {
        action,
        timestamp: new Date().toISOString(),
        ...(payload || {})
      }
    })
  )
}

export const getImportFileExtension = (fileName: string): string => {
  const idx = fileName.lastIndexOf(".")
  return idx >= 0 ? fileName.slice(idx).toLowerCase() : ""
}

export const buildDefaultImportName = (fileName: string): string => {
  const idx = fileName.lastIndexOf(".")
  const base = idx >= 0 ? fileName.slice(0, idx) : fileName
  return base.trim() || fileName
}

export const toIsoBoundaryFromDateInput = (
  value: string,
  boundary: "start" | "end"
): string | undefined => {
  const normalized = value.trim()
  if (!normalized || !DATE_INPUT_PATTERN.test(normalized)) return undefined
  return `${normalized}${
    boundary === "start" ? "T00:00:00.000Z" : "T23:59:59.999Z"
  }`
}

export const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0

export const normalizeWorldBookIds = (value: unknown): number[] => {
  if (!Array.isArray(value)) return []
  return Array.from(
    new Set(
      value
        .map((entry) => Number(entry))
        .filter((id) => Number.isFinite(id) && id > 0)
        .map((id) => Math.trunc(id))
    )
  ).sort((a, b) => a - b)
}

export const toCharacterWorldBookOption = (value: any): CharacterWorldBookOption | null => {
  const id = Number(value?.world_book_id ?? value?.id)
  if (!Number.isFinite(id) || id <= 0) return null
  const rawName = value?.world_book_name ?? value?.name
  const name =
    typeof rawName === "string" && rawName.trim().length > 0
      ? rawName.trim()
      : `World Book ${Math.trunc(id)}`
  return {
    id: Math.trunc(id),
    name,
    enabled: typeof value?.enabled === "boolean" ? value.enabled : undefined
  }
}

export const normalizeImportTags = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((tag) => String(tag).trim())
      .filter((tag) => tag.length > 0)
  }
  if (isNonEmptyString(value)) {
    try {
      const parsed = JSON.parse(value)
      if (Array.isArray(parsed)) {
        return parsed
          .map((tag) => String(tag).trim())
          .filter((tag) => tag.length > 0)
      }
    } catch {
      // fall through
    }
    return value
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0)
  }
  return []
}

export const normalizeCharacterFolderId = (value: unknown): string | undefined => {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return String(Math.trunc(value))
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return undefined
    const numeric = Number(trimmed)
    if (Number.isFinite(numeric) && numeric > 0) {
      return String(Math.trunc(numeric))
    }
    return trimmed
  }
  return undefined
}

export const parseCharacterFolderIdFromToken = (token: unknown): string | undefined => {
  if (typeof token !== "string") return undefined
  const normalized = token.trim()
  if (!normalized) return undefined
  for (const prefix of CHARACTER_FOLDER_TOKEN_PREFIXES) {
    if (!normalized.startsWith(prefix)) continue
    const rawFolderId = normalized.slice(prefix.length).trim()
    if (!rawFolderId) return undefined
    return normalizeCharacterFolderId(rawFolderId)
  }
  return undefined
}

export const isCharacterFolderToken = (tag: unknown): boolean =>
  typeof parseCharacterFolderIdFromToken(tag) === "string"

export const getCharacterVisibleTags = (tags: unknown): string[] =>
  parseCharacterTags(tags).filter((tag) => !isCharacterFolderToken(tag))

export const getCharacterFolderIdFromTags = (tags: unknown): string | undefined => {
  const parsedTags = parseCharacterTags(tags)
  for (const tag of parsedTags) {
    const folderId = parseCharacterFolderIdFromToken(tag)
    if (folderId) return folderId
  }
  return undefined
}

export const buildCharacterFolderToken = (folderId: unknown): string | undefined => {
  const normalizedFolderId = normalizeCharacterFolderId(folderId)
  if (!normalizedFolderId) return undefined
  return `${CHARACTER_FOLDER_TOKEN_PREFIX}${normalizedFolderId}`
}

export const applyCharacterFolderToTags = (
  tags: unknown,
  folderId: unknown
): string[] => {
  const visibleTags = getCharacterVisibleTags(tags)
  const nextFolderToken = buildCharacterFolderToken(folderId)
  return nextFolderToken ? [...visibleTags, nextFolderToken] : visibleTags
}

export const countPopulatedImportFields = (record: Record<string, unknown>): number =>
  Object.values(record).reduce<number>((count, value) => {
    if (value == null) return count
    if (typeof value === "string") {
      return value.trim().length > 0 ? count + 1 : count
    }
    if (Array.isArray(value)) {
      return value.length > 0 ? count + 1 : count
    }
    if (typeof value === "object") {
      return Object.keys(value as Record<string, unknown>).length > 0
        ? count + 1
        : count
    }
    return count + 1
  }, 0)

export const toPreviewAvatarUrl = (value: unknown): string | null => {
  if (!isNonEmptyString(value)) return null
  const trimmed = value.trim()
  if (trimmed.startsWith("data:image/")) return trimmed
  return `data:image/png;base64,${trimmed}`
}

export const normalizeImportPayload = (rawPayload: unknown): Record<string, unknown> => {
  if (!rawPayload || typeof rawPayload !== "object" || Array.isArray(rawPayload)) {
    return {}
  }
  const record = rawPayload as Record<string, unknown>
  if (record.data && typeof record.data === "object" && !Array.isArray(record.data)) {
    return record.data as Record<string, unknown>
  }
  return record
}

export const extractLooseTextCharacterFields = (text: string): Record<string, unknown> => {
  const fields: Record<string, unknown> = {}
  const nameMatch = text.match(/^\s*(?:name|char_name|character_name)\s*:\s*(.+)$/im)
  if (nameMatch?.[1]) {
    fields.name = nameMatch[1].trim()
  }

  const descriptionMatch = text.match(/^\s*description\s*:\s*(.+)$/im)
  if (descriptionMatch?.[1]) {
    fields.description = descriptionMatch[1].trim()
  }

  const inlineTagsMatch = text.match(/^\s*tags\s*:\s*\[(.+)\]\s*$/im)
  if (inlineTagsMatch?.[1]) {
    fields.tags = inlineTagsMatch[1]
      .split(",")
      .map((tag) => tag.trim().replace(/^["']|["']$/g, ""))
      .filter((tag) => tag.length > 0)
    return fields
  }

  const blockTagLines = text.match(
    /^\s*tags\s*:\s*$(?:\r?\n)([\s\S]*?)(?:\r?\n\s*\S.*:|$)/im
  )?.[1]
  if (blockTagLines) {
    const tags = blockTagLines
      .split(/\r?\n/)
      .map((line) => line.match(/^\s*-\s*(.+)$/)?.[1]?.trim())
      .filter((value): value is string => Boolean(value))
      .map((tag) => tag.replace(/^["']|["']$/g, ""))
    if (tags.length > 0) {
      fields.tags = tags
    }
  }

  return fields
}

export const detectMalformedYamlPreview = (text: string): string | null => {
  const lines = text.split(/\r?\n/)

  for (const [lineIndex, rawLine] of lines.entries()) {
    const lineNumber = lineIndex + 1
    const trimmed = rawLine.trim()
    if (!trimmed || trimmed.startsWith("#")) continue

    if (rawLine.includes("\t")) {
      return `Line ${lineNumber}: tabs are not supported for indentation.`
    }

    const withoutComment = rawLine.replace(/\s+#.*$/, "")
    const openSquare = (withoutComment.match(/\[/g) || []).length
    const closeSquare = (withoutComment.match(/\]/g) || []).length
    if (openSquare !== closeSquare) {
      return `Line ${lineNumber}: inline list syntax is malformed.`
    }

    const openCurly = (withoutComment.match(/\{/g) || []).length
    const closeCurly = (withoutComment.match(/\}/g) || []).length
    if (openCurly !== closeCurly) {
      return `Line ${lineNumber}: inline object syntax is malformed.`
    }
  }

  return null
}

export const parseCharacterImportPreview = async (
  file: File,
  index: number
): Promise<CharacterImportPreview> => {
  const extension = getImportFileExtension(file.name)
  const format = extension ? extension.slice(1).toUpperCase() : "FILE"
  const defaultName = buildDefaultImportName(file.name)
  if (!extension || !IMPORT_ALLOWED_EXTENSION_SET.has(extension)) {
    const allowed = IMPORT_ALLOWED_EXTENSIONS.join(", ")
    return {
      id: `${file.name}-${file.lastModified}-${index}`,
      file,
      fileName: file.name,
      format,
      name: defaultName,
      description: "",
      tagCount: 0,
      fieldCount: 0,
      avatarUrl: null,
      parseError: {
        key: "settings:manageCharacters.import.previewUnsupportedType",
        fallback:
          "Unsupported file type: {{extension}}. Supported formats: {{allowed}}",
        values: {
          extension: extension || "no extension",
          allowed
        }
      }
    }
  }

  if (IMPORT_IMAGE_EXTENSIONS.has(extension)) {
    const avatarUrl =
      typeof URL !== "undefined" && typeof URL.createObjectURL === "function"
        ? URL.createObjectURL(file)
        : null
    return {
      id: `${file.name}-${file.lastModified}-${index}`,
      file,
      fileName: file.name,
      format,
      name: defaultName,
      description: "",
      tagCount: 0,
      fieldCount: 0,
      avatarUrl,
      parseError: null
    }
  }

  let text = ""
  try {
    text = await file.text()
  } catch {
    return {
      id: `${file.name}-${file.lastModified}-${index}`,
      file,
      fileName: file.name,
      format,
      name: defaultName,
      description: "",
      tagCount: 0,
      fieldCount: 0,
      avatarUrl: null,
      parseError: {
        key: "settings:manageCharacters.import.previewReadError",
        fallback: "Unable to read file contents for preview."
      }
    }
  }

  let rawPayload: unknown = null
  if (extension === ".json") {
    try {
      rawPayload = JSON.parse(text)
    } catch (error) {
      const message =
        error instanceof Error && error.message
          ? error.message
          : "Invalid JSON"
      return {
        id: `${file.name}-${file.lastModified}-${index}`,
        file,
        fileName: file.name,
        format,
        name: defaultName,
        description: "",
        tagCount: 0,
        fieldCount: 0,
        avatarUrl: null,
        parseError: {
          key: "settings:manageCharacters.import.previewInvalidJson",
          fallback: "Invalid JSON syntax: {{message}}",
          values: { message }
        }
      }
    }
  } else if (extension === ".yaml" || extension === ".yml") {
    const yamlValidationError = detectMalformedYamlPreview(text)
    if (yamlValidationError) {
      return {
        id: `${file.name}-${file.lastModified}-${index}`,
        file,
        fileName: file.name,
        format,
        name: defaultName,
        description: "",
        tagCount: 0,
        fieldCount: 0,
        avatarUrl: null,
        parseError: {
          key: "settings:manageCharacters.import.previewInvalidYaml",
          fallback: "Malformed YAML content: {{message}}",
          values: { message: yamlValidationError }
        }
      }
    }
    const trimmed = text.trim()
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        rawPayload = JSON.parse(trimmed)
      } catch {
        rawPayload = extractLooseTextCharacterFields(text)
      }
    } else {
      rawPayload = extractLooseTextCharacterFields(text)
    }
  } else if (extension === ".md" || extension === ".txt") {
    const trimmed = text.trim()
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        rawPayload = JSON.parse(trimmed)
      } catch {
        rawPayload = extractLooseTextCharacterFields(text)
      }
    } else {
      rawPayload = extractLooseTextCharacterFields(text)
    }
  } else {
    const trimmed = text.trim()
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        rawPayload = JSON.parse(trimmed)
      } catch {
        rawPayload = extractLooseTextCharacterFields(text)
      }
    } else {
      rawPayload = extractLooseTextCharacterFields(text)
    }
  }

  const payload = normalizeImportPayload(rawPayload)
  const inferredName = isNonEmptyString(payload.name)
    ? (payload.name as string).trim()
    : defaultName
  const description = isNonEmptyString(payload.description)
    ? (payload.description as string).trim()
    : ""
  const tags = normalizeImportTags(payload.tags)
  const avatarUrl =
    toPreviewAvatarUrl(payload.image_base64) || toPreviewAvatarUrl(payload.avatar)
  const fieldCount = countPopulatedImportFields(payload)

  return {
    id: `${file.name}-${file.lastModified}-${index}`,
    file,
    fileName: file.name,
    format,
    name: inferredName,
    description,
    tagCount: tags.length,
    fieldCount,
    avatarUrl,
    parseError: null
  }
}

export const toCharactersSortBy = (column: string | null): import("@/services/tldw/TldwApiClient").CharacterListSortBy => {
  switch (column) {
    case "creator":
      return "creator"
    case "activity":
      return "last_used_at"
    case "createdAt":
      return "created_at"
    case "updatedAt":
      return "updated_at"
    case "lastUsedAt":
      return "last_used_at"
    case "conversations":
      return "conversation_count"
    case "name":
    default:
      return "name"
  }
}

export const toCharactersSortOrder = (
  order: "ascend" | "descend" | null
): import("@/services/tldw/TldwApiClient").CharacterListSortOrder => (order === "descend" ? "desc" : "asc")

export const withCharacterNameInLabel = (
  localizedLabel: string,
  fallbackTemplate: string,
  characterName: string
): string => {
  const name = String(characterName || "").trim()
  const localized = String(localizedLabel || "").trim()
  const fallback = String(fallbackTemplate || "").trim()

  if (!name) return localized || fallback
  if (localized.includes(name)) return localized

  if (localized.includes("{{name}}")) {
    return localized.replace(/\{\{\s*name\s*\}\}/g, name)
  }

  if (fallback.includes("{{name}}")) {
    return fallback.replace(/\{\{\s*name\s*\}\}/g, name)
  }

  return `${fallback || localized} ${name}`.trim()
}

export const getCharacterQueryErrorStatusCode = (error: unknown): number | null => {
  const candidate = error as
    | {
        status?: unknown
        response?: { status?: unknown }
        message?: unknown
        details?: unknown
      }
    | null
    | undefined
  const rawStatus = candidate?.status ?? candidate?.response?.status
  const statusCodeFromNumberLike =
    typeof rawStatus === "number"
      ? rawStatus
      : typeof rawStatus === "string"
        ? Number(rawStatus)
        : Number.NaN
  if (Number.isFinite(statusCodeFromNumberLike)) {
    return statusCodeFromNumberLike
  }
  const message = String(candidate?.message || "")
  const detailsText = (() => {
    const details = candidate?.details
    if (typeof details === "string") return details
    if (details == null) return ""
    try {
      return JSON.stringify(details)
    } catch {
      return String(details)
    }
  })()
  const statusMatch = `${message} ${detailsText}`.match(/\b(404|405|422)\b/)
  const statusCode = statusMatch ? Number(statusMatch[1]) : Number.NaN
  return Number.isFinite(statusCode) ? statusCode : null
}

export const isCharacterQueryRouteConflictError = (error: unknown): boolean => {
  const candidate = error as
    | {
        message?: unknown
        details?: unknown
      }
    | null
    | undefined
  const statusCode = getCharacterQueryErrorStatusCode(error)
  const normalizedMessage = String(candidate?.message || "").toLowerCase()
  const normalizedDetails = (() => {
    const details = candidate?.details
    if (typeof details === "string") return details.toLowerCase()
    if (details == null) return ""
    try {
      return JSON.stringify(details).toLowerCase()
    } catch {
      return String(details).toLowerCase()
    }
  })()
  return (
    statusCode === 404 ||
    statusCode === 405 ||
    statusCode === 422 ||
    normalizedMessage.includes("path.character_id") ||
    normalizedMessage.includes("unable to parse string as an integer") ||
    normalizedMessage.includes('input":"query"') ||
    normalizedMessage.includes("/api/v1/characters/query") ||
    normalizedDetails.includes("path.character_id") ||
    normalizedDetails.includes("unable to parse string as an integer") ||
    normalizedDetails.includes('input":"query"') ||
    normalizedDetails.includes("/api/v1/characters/query")
  )
}

export const truncateText = (value?: string, max?: number) => {
  if (!value) return ""
  if (!max || value.length <= max) return value
  return `${value.slice(0, max)}...`
}

export const buildCharacterSelectionPayload = (record: any) => {
  const normalizedExtensions = parseExtensionsObject(record.extensions)

  return {
    id: record.id || record.slug || record.name,
    name: record.name || record.title || record.slug,
    system_prompt:
      record.system_prompt ||
      record.systemPrompt ||
      record.instructions ||
      "",
    greeting:
      record.greeting ||
      record.first_message ||
      record.greet ||
      "",
    alternate_greetings: normalizeAlternateGreetings(record.alternate_greetings),
    extensions:
      normalizedExtensions && Object.keys(normalizedExtensions).length > 0
        ? normalizedExtensions
        : null,
    image_base64:
      typeof record.image_base64 === "string" ? record.image_base64 : null,
    image_mime:
      typeof record.image_mime === "string" ? record.image_mime : null,
    avatar_url:
      record.avatar_url ||
      validateAndCreateImageDataUrl(record.image_base64) ||
      "",
    description: record.description || "",
    version: typeof record.version === "number" ? record.version : undefined
  }
}

export const resolveChatWorkspaceUrl = (): string => {
  if (typeof window === "undefined") return "/"
  try {
    const currentUrl = new URL(window.location.href)
    if (currentUrl.hash.startsWith("#/")) {
      currentUrl.hash = "#/"
      return currentUrl.toString()
    }
    if (/options\.html$/i.test(currentUrl.pathname)) {
      return `${currentUrl.origin}${currentUrl.pathname}#/`
    }
    return `${currentUrl.origin}/`
  } catch {
    return "/"
  }
}

export const normalizeVersionSnapshotValue = (value: unknown): string => {
  if (value === null || typeof value === "undefined") return ""
  if (typeof value === "string") return value.trim()
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export const formatVersionSnapshotValue = (value: unknown): string => {
  if (value === null || typeof value === "undefined") return "\u2014"
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? value : "\u2014"
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

export const resolveCharacterNumericId = (record: any): number | null => {
  const raw = record?.id ?? record?.character_id ?? record?.characterId
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null
  }
  return parsed
}

export const normalizeAlternateGreetings = (value: any): string[] => {
  if (!value) return []
  if (Array.isArray(value)) {
    return value.map((v) => String(v)).filter((v) => v.trim().length > 0)
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value)
      if (Array.isArray(parsed)) {
        return parsed.map((v) => String(v)).filter((v) => v.trim().length > 0)
      }
    } catch {
      // fall through to newline splitting
    }
    return value
      .split(/\r?\n|;/)
      .map((v) => v.trim())
      .filter((v) => v.length > 0)
  }
  return []
}

export const isPlainObject = (value: unknown): value is Record<string, any> =>
  !!value && typeof value === "object" && !Array.isArray(value)

export const parseExtensionsObject = (
  value: unknown
): Record<string, any> | null => {
  if (!value) return {}
  if (isPlainObject(value)) return { ...value }
  if (typeof value === "string") {
    if (!value.trim()) return {}
    try {
      const parsed = JSON.parse(value)
      return isPlainObject(parsed) ? { ...parsed } : {}
    } catch {
      return null
    }
  }
  return {}
}

export const normalizeCharacterComparisonValue = (value: unknown): string => {
  if (Array.isArray(value)) {
    return value
      .map((entry) => normalizeVersionSnapshotValue(entry))
      .join("\n")
      .trim()
  }
  return normalizeVersionSnapshotValue(value)
}

export const formatCharacterComparisonValue = (value: unknown): string => {
  if (value === null || typeof value === "undefined") return "\u2014"

  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? value : "\u2014"
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return "\u2014"
    const hasComplexEntries = value.some(
      (entry) => typeof entry === "object" && entry !== null
    )
    if (!hasComplexEntries) {
      return value.map((entry) => String(entry)).join(", ")
    }
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

export const toComparisonFilenameSegment = (value: unknown): string => {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return normalized || "character"
}

export const CHARACTER_COMPARISON_FIELDS: CharacterComparisonFieldDefinition[] = [
  { field: "name", label: "Name", getValue: (record) => record?.name },
  {
    field: "description",
    label: "Description",
    getValue: (record) => record?.description
  },
  {
    field: "system_prompt",
    label: "System prompt",
    getValue: (record) => record?.system_prompt
  },
  {
    field: "first_message",
    label: "Greeting",
    getValue: (record) => record?.first_message
  },
  {
    field: "personality",
    label: "Personality",
    getValue: (record) => record?.personality
  },
  {
    field: "scenario",
    label: "Scenario",
    getValue: (record) => record?.scenario
  },
  {
    field: "post_history_instructions",
    label: "Post-history instructions",
    getValue: (record) => record?.post_history_instructions
  },
  {
    field: "message_example",
    label: "Message example",
    getValue: (record) => record?.message_example
  },
  {
    field: "creator_notes",
    label: "Creator notes",
    getValue: (record) => record?.creator_notes
  },
  {
    field: "tags",
    label: "Tags",
    getValue: (record) => getCharacterVisibleTags(record?.tags).sort()
  },
  {
    field: "prompt_preset",
    label: "Prompt preset",
    getValue: (record) => record?.prompt_preset
  },
  { field: "model", label: "Model", getValue: (record) => record?.model },
  {
    field: "temperature",
    label: "Temperature",
    getValue: (record) => record?.temperature
  },
  { field: "top_p", label: "Top P", getValue: (record) => record?.top_p },
  {
    field: "max_tokens",
    label: "Max tokens",
    getValue: (record) => record?.max_tokens
  },
  { field: "creator", label: "Creator", getValue: (record) => record?.creator },
  {
    field: "character_version",
    label: "Character version",
    getValue: (record) => record?.character_version
  },
  {
    field: "extensions",
    label: "Extensions",
    getValue: (record) => parseExtensionsObject(record?.extensions) ?? {}
  }
]

export const normalizePromptPresetId = (
  value: unknown
): CharacterPromptPresetId =>
  isCharacterPromptPresetId(value)
    ? value
    : DEFAULT_CHARACTER_PROMPT_PRESET

export const readPromptPresetFromExtensions = (
  extensions: unknown
): CharacterPromptPresetId => {
  const parsed = parseExtensionsObject(extensions)
  if (!parsed) return DEFAULT_CHARACTER_PROMPT_PRESET
  const tldw = isPlainObject(parsed.tldw)
    ? parsed.tldw
    : undefined
  const nestedPreset = tldw
    ? tldw.prompt_preset || tldw.promptPreset
    : undefined
  const topPreset = parsed.prompt_preset || parsed.promptPreset
  return normalizePromptPresetId(nestedPreset || topPreset)
}

export const readDefaultAuthorNoteFromExtensions = (extensions: unknown): string => {
  const parsed = parseExtensionsObject(extensions)
  if (!parsed) return ""

  const tldw = isPlainObject(parsed.tldw) ? parsed.tldw : undefined
  const candidates: unknown[] = [
    parsed.default_author_note,
    parsed.defaultAuthorNote,
    parsed.author_note,
    parsed.authorNote,
    parsed.memory_note,
    parsed.memoryNote,
    tldw?.default_author_note,
    tldw?.defaultAuthorNote,
    tldw?.author_note,
    tldw?.authorNote,
    tldw?.memory_note,
    tldw?.memoryNote
  ]

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim()
    }
  }
  return ""
}

export const readDefaultAuthorNoteFromRecord = (record: any): string => {
  const directCandidates: unknown[] = [
    record?.default_author_note,
    record?.defaultAuthorNote,
    record?.author_note,
    record?.authorNote,
    record?.memory_note,
    record?.memoryNote
  ]
  for (const candidate of directCandidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim()
    }
  }
  return readDefaultAuthorNoteFromExtensions(record?.extensions)
}

export const parseFavoriteValue = (value: unknown): boolean => {
  if (typeof value === "boolean") return value
  if (typeof value === "number") return value === 1
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    return normalized === "1" || normalized === "true"
  }
  return false
}

export const readFavoriteFromExtensions = (extensions: unknown): boolean => {
  const parsed = parseExtensionsObject(extensions)
  if (!parsed) return false
  const tldw = isPlainObject(parsed.tldw) ? parsed.tldw : undefined
  return parseFavoriteValue(tldw?.favorite ?? parsed.favorite)
}

export const readFavoriteFromRecord = (record: any): boolean =>
  readFavoriteFromExtensions(record?.extensions)

export const applyFavoriteToExtensions = (
  rawExtensions: unknown,
  favorite: boolean
): Record<string, any> | undefined | null => {
  const parsed = parseExtensionsObject(rawExtensions)
  const hadRawString =
    typeof rawExtensions === "string" &&
    rawExtensions.trim().length > 0 &&
    parsed === null

  if (hadRawString) {
    return null
  }

  const next: Record<string, any> = parsed && parsed !== null ? { ...parsed } : {}
  const tldw = isPlainObject(next.tldw) ? { ...next.tldw } : {}
  delete next.favorite

  if (favorite) {
    tldw.favorite = true
  } else {
    delete tldw.favorite
  }

  if (Object.keys(tldw).length > 0) {
    next.tldw = tldw
  } else {
    delete next.tldw
  }

  return Object.keys(next).length > 0 ? next : undefined
}

export const parseGenerationNumber = (
  value: unknown,
  minimum: number,
  maximum: number
): number | undefined => {
  if (typeof value === "boolean" || value === null || value === undefined) {
    return undefined
  }
  const parsed =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number.parseFloat(value)
        : Number.NaN
  if (!Number.isFinite(parsed)) return undefined
  if (parsed < minimum || parsed > maximum) return undefined
  return parsed
}

export const normalizeGenerationStopStrings = (value: unknown): string[] | undefined => {
  if (value === null || value === undefined) return undefined
  if (Array.isArray(value)) {
    const normalized = value
      .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
      .filter((entry) => entry.length > 0)
    return normalized.length > 0 ? normalized : undefined
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return undefined
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) {
        const normalized = parsed
          .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
          .filter((entry) => entry.length > 0)
        if (normalized.length > 0) return normalized
      }
    } catch {
      // fall back to simple splitting
    }
    const normalized = trimmed
      .split(/\r?\n|;/)
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)
    return normalized.length > 0 ? normalized : undefined
  }
  return undefined
}

export const resolveGenerationSetting = <T,>(
  containers: Record<string, any>[],
  keys: string[],
  parser: (value: unknown) => T | undefined
): T | undefined => {
  for (const container of containers) {
    for (const key of keys) {
      if (!(key in container)) continue
      const parsed = parser(container[key])
      if (typeof parsed !== "undefined") {
        return parsed
      }
    }
  }
  return undefined
}

export const readGenerationSettingsFromRecord = (
  record: any
): CharacterGenerationSettings => {
  const parsed = parseExtensionsObject(record?.extensions)
  const parsedObject = parsed && isPlainObject(parsed) ? parsed : undefined
  const tldw = parsedObject && isPlainObject(parsedObject.tldw)
    ? parsedObject.tldw
    : undefined
  const generation = tldw && isPlainObject(tldw.generation)
    ? tldw.generation
    : undefined

  const containers: Record<string, any>[] = []
  if (generation) containers.push(generation)
  if (parsedObject) containers.push(parsedObject)
  if (isPlainObject(record)) containers.push(record)

  const temperature = resolveGenerationSetting(
    containers,
    ["temperature", "generation_temperature"],
    (value) => parseGenerationNumber(value, 0, 2)
  )
  const topP = resolveGenerationSetting(
    containers,
    ["top_p", "topP", "generation_top_p"],
    (value) => parseGenerationNumber(value, 0, 1)
  )
  const repetitionPenalty = resolveGenerationSetting(
    containers,
    ["repetition_penalty", "repetitionPenalty", "generation_repetition_penalty"],
    (value) => parseGenerationNumber(value, 0, 3)
  )
  const stop = resolveGenerationSetting(
    containers,
    [
      "stop",
      "stop_strings",
      "stopStrings",
      "stop_sequences",
      "stopSequences",
      "generation_stop_strings"
    ],
    normalizeGenerationStopStrings
  )

  const settings: CharacterGenerationSettings = {}
  if (typeof temperature !== "undefined") settings.temperature = temperature
  if (typeof topP !== "undefined") settings.top_p = topP
  if (typeof repetitionPenalty !== "undefined") {
    settings.repetition_penalty = repetitionPenalty
  }
  if (stop && stop.length > 0) settings.stop = stop
  return settings
}

export const readGenerationSettingsFromFormValues = (
  values: Record<string, any>
): CharacterGenerationSettings => {
  const settings: CharacterGenerationSettings = {}
  const temperature = parseGenerationNumber(values.generation_temperature, 0, 2)
  const topP = parseGenerationNumber(values.generation_top_p, 0, 1)
  const repetitionPenalty = parseGenerationNumber(
    values.generation_repetition_penalty,
    0,
    3
  )
  const stop = normalizeGenerationStopStrings(values.generation_stop_strings)

  if (typeof temperature !== "undefined") settings.temperature = temperature
  if (typeof topP !== "undefined") settings.top_p = topP
  if (typeof repetitionPenalty !== "undefined") {
    settings.repetition_penalty = repetitionPenalty
  }
  if (stop && stop.length > 0) settings.stop = stop
  return settings
}

export const removeLegacyGenerationKeys = (target: Record<string, any>) => {
  delete target.generation
  delete target.top_p
  delete target.topP
  delete target.repetition_penalty
  delete target.repetitionPenalty
  delete target.stop_strings
  delete target.stopStrings
  delete target.stop_sequences
  delete target.stopSequences
}

export const applyCharacterMetadataToExtensions = (
  rawExtensions: unknown,
  params: {
    preset: CharacterPromptPresetId
    defaultAuthorNote?: unknown
    generation?: CharacterGenerationSettings
  }
): Record<string, any> | string | undefined => {
  const parsed = parseExtensionsObject(rawExtensions)
  const hadRawString =
    typeof rawExtensions === "string" &&
    rawExtensions.trim().length > 0 &&
    parsed === null

  if (hadRawString) {
    return rawExtensions as string
  }

  let next: Record<string, any> = parsed && parsed !== null ? { ...parsed } : {}

  const tldw = isPlainObject(next.tldw) ? { ...next.tldw } : {}

  if (params.preset === DEFAULT_CHARACTER_PROMPT_PRESET) {
    delete tldw.prompt_preset
    delete tldw.promptPreset
    delete next.prompt_preset
    delete next.promptPreset
  } else {
    tldw.prompt_preset = params.preset
    delete next.prompt_preset
    delete next.promptPreset
  }

  const defaultAuthorNote =
    typeof params.defaultAuthorNote === "string"
      ? params.defaultAuthorNote.trim()
      : ""
  if (defaultAuthorNote) {
    next.default_author_note = defaultAuthorNote
    delete next.defaultAuthorNote
  } else {
    delete next.default_author_note
    delete next.defaultAuthorNote
  }

  const generation = params.generation || {}
  const normalizedGeneration: Record<string, any> = {}
  const normalizedTemp = parseGenerationNumber(generation.temperature, 0, 2)
  const normalizedTopP = parseGenerationNumber(generation.top_p, 0, 1)
  const normalizedRepetition = parseGenerationNumber(
    generation.repetition_penalty,
    0,
    3
  )
  const normalizedStop = normalizeGenerationStopStrings(generation.stop)

  if (typeof normalizedTemp !== "undefined") {
    normalizedGeneration.temperature = normalizedTemp
  }
  if (typeof normalizedTopP !== "undefined") {
    normalizedGeneration.top_p = normalizedTopP
  }
  if (typeof normalizedRepetition !== "undefined") {
    normalizedGeneration.repetition_penalty = normalizedRepetition
  }
  if (normalizedStop && normalizedStop.length > 0) {
    normalizedGeneration.stop = normalizedStop
  }

  removeLegacyGenerationKeys(next)
  if (Object.keys(normalizedGeneration).length > 0) {
    tldw.generation = normalizedGeneration
  } else {
    delete tldw.generation
  }

  if (Object.keys(tldw).length > 0) {
    next.tldw = tldw
  } else {
    delete next.tldw
  }

  if (Object.keys(next).length > 0) {
    return next
  }

  return undefined
}

export const hasAdvancedData = (record: any, extensionsValue: string): boolean => {
  const normalizedRecord = {
    ...record,
    extensions: record?.extensions ?? extensionsValue
  }
  const defaultAuthorNote = readDefaultAuthorNoteFromRecord(normalizedRecord)
  const generationSettings = readGenerationSettingsFromRecord(normalizedRecord)
  const hasGenerationSettings =
    typeof generationSettings.temperature !== "undefined" ||
    typeof generationSettings.top_p !== "undefined" ||
    typeof generationSettings.repetition_penalty !== "undefined" ||
    (Array.isArray(generationSettings.stop) &&
      generationSettings.stop.length > 0)
  return !!(
    record.personality ||
    record.scenario ||
    record.post_history_instructions ||
    record.message_example ||
    record.creator_notes ||
    (record.alternate_greetings && record.alternate_greetings.length > 0) ||
    record.creator ||
    record.character_version ||
    hasGenerationSettings ||
    defaultAuthorNote ||
    extensionsValue
  )
}

export const buildCharacterPayload = (values: any): Record<string, any> => {
  const tagsWithFolderAssignment = applyCharacterFolderToTags(
    values.tags,
    values.folder_id
  )
  const payload: Record<string, any> = {
    name: values.name,
    description: values.description,
    personality: values.personality,
    scenario: values.scenario,
    system_prompt: values.system_prompt,
    post_history_instructions: values.post_history_instructions,
    first_message: values.greeting || values.first_message,
    message_example: values.message_example,
    creator_notes: values.creator_notes,
    tags: tagsWithFolderAssignment,
    alternate_greetings: Array.isArray(values.alternate_greetings)
      ? values.alternate_greetings.filter((g: string) => g && g.trim().length > 0)
      : values.alternate_greetings,
    creator: values.creator,
    character_version: values.character_version
  }

  // Extract avatar values from unified avatar field
  if (values.avatar) {
    const avatarValues = extractAvatarValues(values.avatar)
    if (avatarValues.avatar_url) {
      payload.avatar_url = avatarValues.avatar_url
    }
    if (avatarValues.image_base64) {
      payload.image_base64 = avatarValues.image_base64
    }
  } else {
    // Fallback for legacy form structure
    if (values.avatar_url) {
      payload.avatar_url = values.avatar_url
    }
    if (values.image_base64) {
      payload.image_base64 = values.image_base64
    }
  }

  // Keep compatibility with mock server / older deployments
  if (values.greeting) {
    payload.greeting = values.greeting
  }

  Object.keys(payload).forEach((key) => {
    if (key === "extensions") return
    const v = payload[key]
    if (
      typeof v === "undefined" ||
      v === null ||
      (typeof v === "string" && v.trim().length === 0) ||
      (Array.isArray(v) && v.length === 0)
    ) {
      delete payload[key]
    }
  })

  const resolvedPreset = normalizePromptPresetId(values.prompt_preset)
  const generationSettings = readGenerationSettingsFromFormValues(values)
  const mergedExtensions = applyCharacterMetadataToExtensions(values.extensions, {
    preset: resolvedPreset,
    defaultAuthorNote: values.default_author_note,
    generation: generationSettings
  })
  if (typeof mergedExtensions !== "undefined") {
    payload.extensions = mergedExtensions
  }

  return payload
}

export const characterIdentifier = (record: any): string =>
  String(record?.id ?? record?.slug ?? record?.name ?? "")

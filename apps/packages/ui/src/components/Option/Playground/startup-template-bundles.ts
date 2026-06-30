import type { Prompt } from "@/db/dexie/types"
import type { ChatModelSettings } from "@/store/model"
import type { ActorSettings } from "@/types/actor"
import type { Character } from "@/types/character"
import type { RagPinnedResult } from "@/utils/rag-format"
import type { PresetKey } from "./ParameterPresets"

export type StartupTemplatePromptSource =
  | "none"
  | "system-template"
  | "prompt-library"
  | "prompt-studio"

export type StartupTemplateBundleSource =
  | "startup-template"
  | "role-play-setup"

export type StartupTemplateRolePlayIdentity = {
  kind: "character" | "persona"
  id: string | number
  name: string
}

export type StartupTemplateRolePlayBehavior = {
  source: "template" | "custom" | "modified-template"
  templateId: string | null
  templateTitle: string | null
  templateCategory: string | null
  systemPrompt: string
  modified: boolean
}

export type StartupTemplateRolePlayGeneration = {
  presetKey: PresetKey
  settings: Partial<ChatModelSettings>
}

export type StartupTemplateRolePlayContext = {
  ragPinnedCount: number
  ragPinnedResultIds: string[]
}

export type StartupTemplateRolePlayMetadata = {
  source: "role-play-setup"
  identity: StartupTemplateRolePlayIdentity | null
  behavior: StartupTemplateRolePlayBehavior | null
  scene: ActorSettings | null
  generation: StartupTemplateRolePlayGeneration | null
  context: StartupTemplateRolePlayContext | null
}

export type RolePlaySetupPreviewDescription = {
  identity: string
  behavior: string
  scene: string
  generation: string
  context: string
}

export type StartupTemplateBundle = {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  source: StartupTemplateBundleSource
  selectedModel: string | null
  systemPrompt: string
  selectedSystemPromptId: string | null
  promptStudioPromptId: number | null
  promptTitle: string | null
  promptSource: StartupTemplatePromptSource
  presetKey: PresetKey
  character: Character | null
  ragPinnedResults: RagPinnedResult[]
  rolePlay: StartupTemplateRolePlayMetadata | null
}

export type StartupTemplateCreateInput = {
  name: string
  source?: StartupTemplateBundleSource
  selectedModel: string | null
  systemPrompt: string
  selectedSystemPromptId?: string | null
  promptStudioPromptId?: number | null
  promptTitle?: string | null
  promptSource?: StartupTemplatePromptSource
  presetKey?: PresetKey
  character?: Character | null
  ragPinnedResults?: RagPinnedResult[]
  rolePlay?: StartupTemplateRolePlayMetadata | null
}

export type StartupTemplatePromptResolution = {
  prompt: Prompt | null
  source: StartupTemplatePromptSource
  promptTitle: string | null
  promptStudioPromptId: number | null
}

const MAX_TEMPLATE_NAME_LENGTH = 80
const FALLBACK_TEMPLATE_NAME = "New startup template"

const PRESET_LABELS: Record<PresetKey, string> = {
  creative: "Creative",
  balanced: "Balanced",
  precise: "Precise",
  custom: "Custom"
}

const nowTimestamp = () => Date.now()

const createTemplateId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `startup-template-${nowTimestamp()}-${Math.random().toString(16).slice(2)}`
}

const normalizePromptSource = (
  value: unknown
): StartupTemplatePromptSource => {
  if (
    value === "none" ||
    value === "system-template" ||
    value === "prompt-library" ||
    value === "prompt-studio"
  ) {
    return value
  }
  return "none"
}

const normalizeBundleSource = (value: unknown): StartupTemplateBundleSource =>
  value === "role-play-setup" ? "role-play-setup" : "startup-template"

const isPresetKey = (value: unknown): value is PresetKey =>
  value === "creative" ||
  value === "balanced" ||
  value === "precise" ||
  value === "custom"

const normalizePresetKey = (value: unknown): PresetKey => {
  if (isPresetKey(value)) {
    return value
  }
  return "custom"
}

const normalizeString = (value: unknown): string =>
  typeof value === "string" ? value : ""

const normalizeNullableString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const normalizeRagPinnedResult = (value: unknown): RagPinnedResult | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const id = normalizeString(record.id).trim()
  const snippet = normalizeString(record.snippet).trim()
  if (!id || !snippet) return null

  return {
    id,
    snippet,
    title: normalizeNullableString(record.title) ?? undefined,
    source: normalizeNullableString(record.source) ?? undefined,
    url: normalizeNullableString(record.url) ?? undefined,
    type: normalizeNullableString(record.type) ?? undefined,
    mediaId:
      typeof record.mediaId === "number" && Number.isFinite(record.mediaId)
        ? record.mediaId
        : undefined
  }
}

const normalizeRagPinnedResults = (value: unknown): RagPinnedResult[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => normalizeRagPinnedResult(entry))
    .filter((entry): entry is RagPinnedResult => Boolean(entry))
    .slice(0, 12)
}

const normalizeCharacter = (value: unknown): Character | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const id = record.id
  if (!(typeof id === "string" || typeof id === "number")) {
    return null
  }
  return value as Character
}

const normalizeRolePlayIdentity = (
  value: unknown
): StartupTemplateRolePlayIdentity | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  if (record.kind !== "character" && record.kind !== "persona") return null
  const id = record.id
  const name = normalizeString(record.name).trim()
  if (!(typeof id === "string" || typeof id === "number")) return null
  if (!String(id).trim() || !name) return null
  return {
    kind: record.kind,
    id,
    name
  }
}

const normalizeRolePlayBehavior = (
  value: unknown
): StartupTemplateRolePlayBehavior | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  if (
    record.source !== "template" &&
    record.source !== "custom" &&
    record.source !== "modified-template"
  ) {
    return null
  }
  const systemPrompt = normalizeString(record.systemPrompt).trim()
  const templateId = normalizeNullableString(record.templateId)
  const templateTitle = normalizeNullableString(record.templateTitle)
  if (record.source === "template" && !templateId && !templateTitle) return null
  if (record.source !== "template" && !systemPrompt && !templateTitle) return null

  return {
    source: record.source,
    templateId,
    templateTitle,
    templateCategory: normalizeNullableString(record.templateCategory),
    systemPrompt,
    modified: Boolean(record.modified)
  }
}

const normalizeActorSettings = (value: unknown): ActorSettings | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  if (typeof record.isEnabled !== "boolean") return null
  if (!Array.isArray(record.aspects)) return null
  if (typeof record.notes !== "string") return null
  return value as ActorSettings
}

const normalizeRolePlaySettings = (
  value: unknown
): Partial<ChatModelSettings> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  const settings: Partial<ChatModelSettings> = {}
  const writableSettings = settings as Record<string, unknown>
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (
      typeof entry === "string" ||
      typeof entry === "boolean" ||
      (typeof entry === "number" && Number.isFinite(entry))
    ) {
      writableSettings[key] = entry
    }
  }
  return settings
}

const normalizeRolePlayGeneration = (
  value: unknown
): StartupTemplateRolePlayGeneration | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  if (!isPresetKey(record.presetKey)) return null
  return {
    presetKey: record.presetKey,
    settings: normalizeRolePlaySettings(record.settings)
  }
}

const normalizeRolePlayContext = (
  value: unknown
): StartupTemplateRolePlayContext | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const rawCount = record.ragPinnedCount
  const ids = Array.isArray(record.ragPinnedResultIds)
    ? record.ragPinnedResultIds
        .map((id) => (typeof id === "string" ? id.trim() : ""))
        .filter(Boolean)
    : []
  const count =
    typeof rawCount === "number" && Number.isFinite(rawCount) && rawCount > 0
      ? Math.floor(rawCount)
      : ids.length
  if (count <= 0 && ids.length === 0) return null
  return {
    ragPinnedCount: Math.max(count, ids.length),
    ragPinnedResultIds: ids
  }
}

const normalizeRolePlayMetadata = (
  value: unknown
): StartupTemplateRolePlayMetadata | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  if (record.source !== "role-play-setup") return null

  const identity = normalizeRolePlayIdentity(record.identity)
  const behavior = normalizeRolePlayBehavior(record.behavior)
  const scene = normalizeActorSettings(record.scene)
  const generation = normalizeRolePlayGeneration(record.generation)
  const context = normalizeRolePlayContext(record.context)

  if (!identity && !behavior && !scene && !generation && !context) return null

  return {
    source: "role-play-setup",
    identity,
    behavior,
    scene,
    generation,
    context
  }
}

const resolveStoredBundleSource = (
  requestedSource: unknown,
  rolePlay: StartupTemplateRolePlayMetadata | null,
  character: Character | null
): StartupTemplateBundleSource => {
  if (normalizeBundleSource(requestedSource) !== "role-play-setup") {
    return "startup-template"
  }
  return rolePlay || character ? "role-play-setup" : "startup-template"
}

export const sanitizeStartupTemplateName = (
  value: string,
  fallback = FALLBACK_TEMPLATE_NAME
): string => {
  const normalized = value.replace(/\s+/g, " ").trim()
  if (!normalized) return fallback
  if (normalized.length <= MAX_TEMPLATE_NAME_LENGTH) return normalized
  return normalized.slice(0, MAX_TEMPLATE_NAME_LENGTH).trim()
}

export const inferStartupTemplatePromptSource = (
  selectedPrompt: Prompt | null,
  hasSystemPromptContent: boolean
): StartupTemplatePromptSource => {
  if (selectedPrompt) {
    if (
      selectedPrompt.sourceSystem === "studio" ||
      selectedPrompt.studioPromptId != null ||
      selectedPrompt.serverId != null
    ) {
      return "prompt-studio"
    }
    return "prompt-library"
  }
  if (hasSystemPromptContent) {
    return "system-template"
  }
  return "none"
}

export const createStartupTemplateBundle = (
  input: StartupTemplateCreateInput,
  options?: {
    id?: string
    now?: number
  }
): StartupTemplateBundle => {
  const now = options?.now ?? nowTimestamp()
  const character = normalizeCharacter(input.character)
  const rolePlay = normalizeRolePlayMetadata(input.rolePlay)
  return {
    id: options?.id ?? createTemplateId(),
    name: sanitizeStartupTemplateName(input.name),
    createdAt: now,
    updatedAt: now,
    source: resolveStoredBundleSource(input.source, rolePlay, character),
    selectedModel:
      typeof input.selectedModel === "string" && input.selectedModel.trim().length > 0
        ? input.selectedModel
        : null,
    systemPrompt: normalizeString(input.systemPrompt),
    selectedSystemPromptId: normalizeNullableString(input.selectedSystemPromptId),
    promptStudioPromptId:
      typeof input.promptStudioPromptId === "number" &&
      Number.isFinite(input.promptStudioPromptId)
        ? input.promptStudioPromptId
        : null,
    promptTitle: normalizeNullableString(input.promptTitle),
    promptSource: normalizePromptSource(input.promptSource),
    presetKey: normalizePresetKey(input.presetKey),
    character,
    ragPinnedResults: normalizeRagPinnedResults(input.ragPinnedResults),
    rolePlay
  }
}

export const normalizeStartupTemplateBundle = (
  value: unknown
): StartupTemplateBundle | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const id = normalizeNullableString(record.id)
  if (!id) return null

  const createdAt =
    typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
      ? record.createdAt
      : nowTimestamp()
  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? record.updatedAt
      : createdAt
  const character = normalizeCharacter(record.character)
  const rolePlay = normalizeRolePlayMetadata(record.rolePlay)

  return {
    id,
    name: sanitizeStartupTemplateName(normalizeString(record.name)),
    createdAt,
    updatedAt,
    source: resolveStoredBundleSource(record.source, rolePlay, character),
    selectedModel: normalizeNullableString(record.selectedModel),
    systemPrompt: normalizeString(record.systemPrompt),
    selectedSystemPromptId: normalizeNullableString(record.selectedSystemPromptId),
    promptStudioPromptId:
      typeof record.promptStudioPromptId === "number" &&
      Number.isFinite(record.promptStudioPromptId)
        ? record.promptStudioPromptId
        : null,
    promptTitle: normalizeNullableString(record.promptTitle),
    promptSource: normalizePromptSource(record.promptSource),
    presetKey: normalizePresetKey(record.presetKey),
    character,
    ragPinnedResults: normalizeRagPinnedResults(record.ragPinnedResults),
    rolePlay
  }
}

export const parseStartupTemplateBundles = (
  value: unknown
): StartupTemplateBundle[] => {
  let parsedValue = value
  if (typeof value === "string") {
    try {
      parsedValue = JSON.parse(value)
    } catch {
      return []
    }
  }
  if (!Array.isArray(parsedValue)) return []

  return parsedValue
    .map((entry) => normalizeStartupTemplateBundle(entry))
    .filter((entry): entry is StartupTemplateBundle => Boolean(entry))
    .sort((a, b) => b.updatedAt - a.updatedAt)
}

export const serializeStartupTemplateBundles = (
  bundles: StartupTemplateBundle[]
): string => JSON.stringify(bundles)

export const upsertStartupTemplateBundle = (
  existing: StartupTemplateBundle[],
  incoming: StartupTemplateBundle
): StartupTemplateBundle[] => {
  const next = existing.filter((entry) => entry.id !== incoming.id)
  return [incoming, ...next].sort((a, b) => b.updatedAt - a.updatedAt)
}

export const removeStartupTemplateBundle = (
  existing: StartupTemplateBundle[],
  id: string
): StartupTemplateBundle[] => existing.filter((entry) => entry.id !== id)

export const isRolePlayRelevantBundle = (
  template: StartupTemplateBundle
): boolean => {
  if (template.source === "role-play-setup") return true
  if (template.character) return true
  if (!template.rolePlay) return false

  return Boolean(
    template.rolePlay.identity ||
      template.rolePlay.behavior ||
      template.rolePlay.scene ||
      template.rolePlay.context
  )
}

const formatRolePlayIdentity = (template: StartupTemplateBundle): string => {
  const identity = template.rolePlay?.identity
  if (identity) {
    const label = identity.kind === "persona" ? "Persona" : "Character"
    return `${label}: ${identity.name || identity.id}`
  }
  if (template.character?.name) return `Character: ${template.character.name}`
  return "None"
}

const formatRolePlayBehavior = (template: StartupTemplateBundle): string => {
  const behavior = template.rolePlay?.behavior
  if (!behavior) {
    return template.systemPrompt.trim().length > 0 ? "Custom behavior" : "No behavior"
  }
  if (behavior.templateTitle) {
    return behavior.modified
      ? `${behavior.templateTitle} modified`
      : behavior.templateTitle
  }
  if (behavior.source === "modified-template") return "Modified template"
  if (behavior.source === "custom") return "Custom behavior"
  return "Behavior template"
}

const formatRolePlayScene = (template: StartupTemplateBundle): string =>
  template.rolePlay?.scene?.isEnabled ? "Scene: active" : "No scene"

const formatRolePlayGeneration = (template: StartupTemplateBundle): string => {
  const presetKey = template.rolePlay?.generation?.presetKey ?? template.presetKey
  return PRESET_LABELS[presetKey] ?? "Custom"
}

const formatPinnedCount = (count: number): string => {
  if (count <= 0) return "No pinned sources"
  if (count === 1) return "1 pinned source"
  return `${count} pinned sources`
}

const formatRolePlayContext = (template: StartupTemplateBundle): string => {
  const metadataCount = template.rolePlay?.context?.ragPinnedCount
  const count =
    typeof metadataCount === "number" && Number.isFinite(metadataCount)
      ? metadataCount
      : template.ragPinnedResults.length
  return formatPinnedCount(count)
}

export const describeRolePlaySetupPreview = (
  template: StartupTemplateBundle
): RolePlaySetupPreviewDescription => ({
  identity: formatRolePlayIdentity(template),
  behavior: formatRolePlayBehavior(template),
  scene: formatRolePlayScene(template),
  generation: formatRolePlayGeneration(template),
  context: formatRolePlayContext(template)
})

export const resolveStartupTemplatePrompt = (
  template: StartupTemplateBundle,
  prompts: Prompt[]
): StartupTemplatePromptResolution => {
  const byId = template.selectedSystemPromptId
    ? prompts.find((prompt) => prompt.id === template.selectedSystemPromptId) || null
    : null
  if (byId) {
    return {
      prompt: byId,
      source: inferStartupTemplatePromptSource(byId, byId.content.trim().length > 0),
      promptTitle: byId.title || template.promptTitle,
      promptStudioPromptId:
        byId.studioPromptId ?? byId.serverId ?? template.promptStudioPromptId
    }
  }

  const studioPromptId = template.promptStudioPromptId
  const byStudioId =
    studioPromptId == null
      ? null
      :
          prompts.find(
            (prompt) =>
              prompt.studioPromptId === studioPromptId ||
              prompt.serverId === studioPromptId
          ) || null

  if (byStudioId) {
    return {
      prompt: byStudioId,
      source: "prompt-studio",
      promptTitle: byStudioId.title || template.promptTitle,
      promptStudioPromptId: byStudioId.studioPromptId ?? byStudioId.serverId ?? studioPromptId
    }
  }

  return {
    prompt: null,
    source: normalizePromptSource(template.promptSource),
    promptTitle: template.promptTitle,
    promptStudioPromptId: studioPromptId
  }
}

export const describeStartupTemplatePrompt = (
  template: StartupTemplateBundle,
  prompts: Prompt[]
): string => {
  const resolution = resolveStartupTemplatePrompt(template, prompts)

  if (resolution.source === "prompt-studio") {
    return resolution.promptTitle
      ? `Prompt Studio: ${resolution.promptTitle}`
      : "Prompt Studio prompt"
  }
  if (resolution.source === "prompt-library") {
    return resolution.promptTitle
      ? `Prompt library: ${resolution.promptTitle}`
      : "Prompt library"
  }
  if (template.systemPrompt.trim().length > 0) {
    return "Custom system prompt"
  }
  return "No prompt"
}

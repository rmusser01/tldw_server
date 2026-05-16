import {
  coerceBoolean,
  coerceOptionalString,
  coerceNumber,
  coerceString,
  defineSetting
} from "@/services/settings/registry"
import { DEFAULT_SPLASH_CARD_NAMES } from "@/data/splash-cards"
import {
  normalizeMediaChatHandoffPayload,
  type MediaChatHandoffPayload
} from "@/services/tldw/media-chat-handoff"
import {
  normalizeWatchlistChatHandoffPayload,
  type WatchlistChatHandoffPayload
} from "@/services/tldw/watchlist-chat-handoff"
import { migrateThemeUserPreferences } from "@/themes/user-preference-migration"
import type { ThemeDefinition } from "@/themes/types"
import { validateThemeDefinition } from "@/themes/validation"

const THEME_VALUES = ["system", "dark", "light"] as const
export type ThemeValue = (typeof THEME_VALUES)[number]

const normalizeThemeValue = (value: unknown, fallback: ThemeValue) => {
  const normalized = String(value || "").toLowerCase()
  return THEME_VALUES.includes(normalized as ThemeValue)
    ? (normalized as ThemeValue)
    : fallback
}

export const THEME_SETTING = defineSetting(
  "theme",
  "dark" as ThemeValue,
  (value) => normalizeThemeValue(value, "dark"),
  {
    area: "local",
    validate: (value) => THEME_VALUES.includes(value),
    localStorageKey: "theme",
    mirrorToLocalStorage: true,
    localStorageSerialize: (value) => value
  }
)

export const THEME_PRESET_SETTING = defineSetting(
  "tldw:themePreset",
  "default",
  (value) => coerceString(value, "default"),
  {
    area: "local",
    beforeGet: migrateThemeUserPreferences,
    localStorageKey: "tldw:themePreset",
    mirrorToLocalStorage: true
  }
)

const coerceThemeArray = (value: unknown): ThemeDefinition[] => {
  if (!Array.isArray(value)) return []
  return value.filter(validateThemeDefinition)
}

export const CUSTOM_THEMES_SETTING = defineSetting<ThemeDefinition[]>(
  "tldw:customThemes",
  [],
  coerceThemeArray,
  {
    area: "local",
    localStorageKey: "tldw:customThemes",
    mirrorToLocalStorage: true
  }
)

export const I18N_LANGUAGE_SETTING = defineSetting(
  "i18nextLng",
  "en",
  (value) => coerceString(value, "en"),
  {
    area: "local",
    localStorageKey: "i18nextLng",
    mirrorToLocalStorage: true
  }
)

export const CHAT_BACKGROUND_IMAGE_SETTING = defineSetting(
  "chatBackgroundImage",
  undefined as string | undefined,
  coerceOptionalString
)

export const CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB = 15
export const CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH =
  CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB * 1_000_000

const SPLASH_CARD_NAME_SET = new Set(DEFAULT_SPLASH_CARD_NAMES)

const coerceSplashCardNameArray = (value: unknown): string[] => {
  const normalized = coerceStringArray(value, DEFAULT_SPLASH_CARD_NAMES)
  const seen = new Set<string>()
  const next: string[] = []
  for (const name of normalized) {
    if (!SPLASH_CARD_NAME_SET.has(name) || seen.has(name)) continue
    seen.add(name)
    next.push(name)
  }
  return next
}

export const SPLASH_DISABLED_SETTING = defineSetting(
  "tldw:splash:disabled",
  false,
  (value) => coerceBoolean(value, false),
  {
    area: "local",
    localStorageKey: "tldw_splash_disabled",
    mirrorToLocalStorage: true
  }
)

export const SPLASH_ENABLED_CARD_NAMES_SETTING = defineSetting(
  "tldw:splash:enabledCards",
  DEFAULT_SPLASH_CARD_NAMES,
  (value) => coerceSplashCardNameArray(value),
  {
    area: "local",
    validate: (value) =>
      Array.isArray(value) &&
      value.every((name) => SPLASH_CARD_NAME_SET.has(name)),
    localStorageKey: "tldw:splash:enabledCards",
    mirrorToLocalStorage: true
  }
)

export const SPLASH_DURATION_SECONDS_MIN = 1
export const SPLASH_DURATION_SECONDS_MAX = 10
export const SPLASH_DURATION_SECONDS_DEFAULT = 3

const coerceSplashDurationSeconds = (value: unknown): number => {
  const parsed = Math.round(coerceNumber(value, SPLASH_DURATION_SECONDS_DEFAULT))
  if (!Number.isFinite(parsed)) return SPLASH_DURATION_SECONDS_DEFAULT
  return Math.min(
    SPLASH_DURATION_SECONDS_MAX,
    Math.max(SPLASH_DURATION_SECONDS_MIN, parsed)
  )
}

export const SPLASH_DURATION_SECONDS_SETTING = defineSetting(
  "tldw:splash:durationSeconds",
  SPLASH_DURATION_SECONDS_DEFAULT,
  (value) => coerceSplashDurationSeconds(value),
  {
    area: "local",
    validate: (value) =>
      Number.isInteger(value) &&
      value >= SPLASH_DURATION_SECONDS_MIN &&
      value <= SPLASH_DURATION_SECONDS_MAX,
    localStorageKey: "tldw:splash:durationSeconds",
    mirrorToLocalStorage: true
  }
)

export const CONTEXT_FILE_SIZE_MB_SETTING = defineSetting(
  "tldw:contextFileMaxSizeMb",
  10,
  (value) => coerceNumber(value, 10),
  {
    area: "local",
    validate: (value) => Number.isFinite(value) && value > 0
  }
)

const coerceOptionalNumber = (
  value: unknown,
  fallback: number | null
): number | null => {
  if (value === null || value === undefined || value === "") return fallback
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return fallback
    const parsed = Number(trimmed)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

const coerceStringArray = (
  value: unknown,
  fallback: string[]
): string[] => {
  const normalize = (items: unknown[]): string[] => {
    const seen = new Set<string>()
    const result: string[] = []
    for (const item of items) {
      if (typeof item !== "string") continue
      const trimmed = item.trim()
      if (!trimmed || seen.has(trimmed)) continue
      seen.add(trimmed)
      result.push(trimmed)
    }
    return result
  }

  if (Array.isArray(value)) {
    return normalize(value)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return fallback
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      try {
        const parsed = JSON.parse(trimmed)
        if (Array.isArray(parsed)) {
          return normalize(parsed)
        }
      } catch {
        // ignore JSON parse errors
      }
    }
    return normalize(trimmed.split(","))
  }
  return fallback
}

export const MCP_TOOL_CATALOG_SETTING = defineSetting(
  "tldw:mcp:catalog",
  "",
  (value) => coerceString(value, ""),
  {
    area: "local"
  }
)

export type McpDisabledToolScopePreference = {
  disabledToolNames: string[]
  updatedAt?: string
}

export type McpDisabledToolPreferences = {
  version: 1
  scopes: Record<string, McpDisabledToolScopePreference>
}

const DEFAULT_MCP_DISABLED_TOOL_PREFERENCES: McpDisabledToolPreferences = {
  version: 1,
  scopes: {}
}

const coerceMcpDisabledToolPreferences = (
  value: unknown
): McpDisabledToolPreferences => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return DEFAULT_MCP_DISABLED_TOOL_PREFERENCES
  }
  const data = value as Record<string, unknown>
  if (data.version !== 1 || !data.scopes || typeof data.scopes !== "object") {
    return DEFAULT_MCP_DISABLED_TOOL_PREFERENCES
  }
  const scopes: Record<string, McpDisabledToolScopePreference> = {}
  for (const [scope, rawPreference] of Object.entries(
    data.scopes as Record<string, unknown>
  )) {
    const normalizedScope = scope.trim()
    if (!normalizedScope) continue
    if (!rawPreference || typeof rawPreference !== "object") continue
    const preference = rawPreference as Record<string, unknown>
    const disabledToolNames = coerceStringArray(
      preference.disabledToolNames,
      []
    )
    const updatedAt =
      typeof preference.updatedAt === "string" && preference.updatedAt.trim()
        ? preference.updatedAt.trim()
        : undefined
    scopes[normalizedScope] = {
      disabledToolNames,
      ...(updatedAt ? { updatedAt } : {})
    }
  }
  return {
    version: 1,
    scopes
  }
}

export const MCP_DISABLED_TOOLS_SETTING = defineSetting(
  "tldw:mcp:disabledTools:v1",
  DEFAULT_MCP_DISABLED_TOOL_PREFERENCES,
  coerceMcpDisabledToolPreferences,
  {
    area: "local",
    localStorageKey: "tldw:mcp:disabledTools:v1",
    mirrorToLocalStorage: true
  }
)

export const MCP_TOOL_CATALOG_ID_SETTING = defineSetting(
  "tldw:mcp:catalogId",
  null as number | null,
  (value) => coerceOptionalNumber(value, null),
  {
    area: "local"
  }
)

export const MCP_TOOL_MODULE_SETTING = defineSetting(
  "tldw:mcp:module",
  [] as string[],
  (value) => coerceStringArray(value, []),
  {
    area: "local"
  }
)

export const MCP_TOOL_CATALOG_STRICT_SETTING = defineSetting(
  "tldw:mcp:catalogStrict",
  false,
  (value) => coerceBoolean(value, false),
  {
    area: "local"
  }
)

const UI_MODE_VALUES = ["sidePanel", "webui"] as const
export type UiModeValue = (typeof UI_MODE_VALUES)[number]

const normalizeUiModeValue = (value: unknown, fallback: UiModeValue) => {
  const normalized = String(value || "")
  return UI_MODE_VALUES.includes(normalized as UiModeValue)
    ? (normalized as UiModeValue)
    : fallback
}

export const UI_MODE_SETTING = defineSetting(
  "uiMode",
  "sidePanel" as UiModeValue,
  (value) => normalizeUiModeValue(value, "sidePanel"),
  {
    validate: (value) => UI_MODE_VALUES.includes(value)
  }
)

const SIDEBAR_TAB_VALUES = ["server", "folders"] as const
type SidebarTabValue = (typeof SIDEBAR_TAB_VALUES)[number]

export const SIDEBAR_ACTIVE_TAB_SETTING = defineSetting(
  "tldw:sidebar:activeTab",
  "server" as SidebarTabValue,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return SIDEBAR_TAB_VALUES.includes(normalized as SidebarTabValue)
      ? (normalized as SidebarTabValue)
      : "server"
  },
  {
    area: "local",
    validate: (value) => SIDEBAR_TAB_VALUES.includes(value)
  }
)

const SIDEBAR_SERVER_CHAT_FILTER_VALUES = [
  "all",
  "character",
  "non_character",
  "trash"
] as const
export type SidebarServerChatFilterValue =
  (typeof SIDEBAR_SERVER_CHAT_FILTER_VALUES)[number]

export const SIDEBAR_SERVER_CHAT_FILTER_SETTING = defineSetting(
  "tldw:sidebar:serverChatFilter",
  "all" as SidebarServerChatFilterValue,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return SIDEBAR_SERVER_CHAT_FILTER_VALUES.includes(
      normalized as SidebarServerChatFilterValue
    )
      ? (normalized as SidebarServerChatFilterValue)
      : "all"
  },
  {
    area: "local",
    validate: (value) =>
      SIDEBAR_SERVER_CHAT_FILTER_VALUES.includes(value)
  }
)

export const SIDEBAR_SHORTCUTS_COLLAPSED_SETTING = defineSetting(
  "tldw:sidebar:shortcutsCollapsed",
  false,
  (value) => coerceBoolean(value, false),
  {
    area: "local"
  }
)

export const PERSONA_BUDDY_SHELL_ENABLED_SETTING = defineSetting(
  "tldw:personaBuddyShellEnabled",
  true,
  (value) => coerceBoolean(value, true),
  {
    area: "local",
    localStorageKey: "tldw:personaBuddyShellEnabled",
    mirrorToLocalStorage: true
  }
)

export const SIDEBAR_SHORTCUT_MAX_COUNT = 15

export const HEADER_SHORTCUT_IDS = [
  "chat",
  "chat-workspace",
  "prompts",
  "prompt-studio",
  "characters",
  "chat-dictionaries",
  "world-books",
  "deep-research",
  "knowledge-qa",
  "media",
  "document-workspace",
  "repo2txt",
  "multi-item-review",
  "flashcards",
  "notes",
  "watchlists",
  "integrations",
  "mcp-hub",
  "scheduled-tasks",
  "collections",
  "skills",
  "model-playground",
  "workspace-playground",
  "writing-playground",
  "quizzes",
  "evaluations",
  "stt-playground",
  "tts-playground",
  "chunking-playground",
  "kanban-playground",
  "data-tables",
  "audiobook-studio",
  "presentation-studio",
  "acp-playground",
  "workflows",
  "admin-server",
  "admin-integrations",
  "documentation",
  "chatbooks-playground",
  "moderation-review",
  "moderation-rules",
  "family-guardrails",
  "guardian",
  "admin-llamacpp",
  "admin-mlx",
  "admin-monitoring",
  "admin-vllm",
  "settings",
  "account",
  "billing"
] as const
export type HeaderShortcutId = (typeof HEADER_SHORTCUT_IDS)[number]

export const DEFAULT_HEADER_SHORTCUT_SELECTION = [
  ...HEADER_SHORTCUT_IDS
] as HeaderShortcutId[]

const REQUIRED_HEADER_SHORTCUT_IDS: HeaderShortcutId[] = [
  "workflows",
  "acp-playground",
  "integrations",
  "scheduled-tasks",
  "admin-integrations"
]

const coerceHeaderShortcutSelection = (
  value: unknown,
  fallback: HeaderShortcutId[]
): HeaderShortcutId[] => {
  if (!Array.isArray(value)) return fallback
  const allowed = new Set<HeaderShortcutId>(HEADER_SHORTCUT_IDS)
  const required = new Set<HeaderShortcutId>(REQUIRED_HEADER_SHORTCUT_IDS)
  const legacyMap: Record<string, HeaderShortcutId[]> = {
    "moderation-playground": ["moderation-review", "moderation-rules"]
  }
  const unique = new Set<HeaderShortcutId>()
  for (const entry of value) {
    if (typeof entry !== "string") continue
    const mappedEntries = legacyMap[entry] ?? [entry as HeaderShortcutId]
    for (const mapped of mappedEntries) {
      if (allowed.has(mapped)) {
        unique.add(mapped)
      }
    }
  }
  for (const requiredId of required) {
    unique.add(requiredId)
  }
  return HEADER_SHORTCUT_IDS.filter((id) => unique.has(id))
}

export const HEADER_SHORTCUT_SELECTION_SETTING = defineSetting(
  "tldw:headerShortcuts:selection",
  DEFAULT_HEADER_SHORTCUT_SELECTION,
  (value) =>
    coerceHeaderShortcutSelection(value, DEFAULT_HEADER_SHORTCUT_SELECTION),
  {
    area: "local"
  }
)

export const SIDEBAR_SHORTCUT_IDS = [
  "quick-ingest",
  ...HEADER_SHORTCUT_IDS
] as const
export type SidebarShortcutId = (typeof SIDEBAR_SHORTCUT_IDS)[number]

const LEGACY_DEFAULT_SIDEBAR_SHORTCUT_SELECTION: SidebarShortcutId[] = [
  "quick-ingest",
  "chat",
  "prompts",
  "prompt-studio",
  "characters",
  "chat-dictionaries",
  "world-books",
  "knowledge-qa",
  "media",
  "document-workspace"
]

export const DEFAULT_SIDEBAR_SHORTCUT_SELECTION: SidebarShortcutId[] = [
  "quick-ingest",
  "chat",
  "chat-workspace",
  "prompts",
  "characters",
  "deep-research",
  "world-books",
  "knowledge-qa",
  "media",
  "watchlists",
  "document-workspace",
  "flashcards",
  "moderation-review",
  "tts-playground",
  "stt-playground"
]

const areShortcutSelectionsEqual = (
  first: SidebarShortcutId[],
  second: SidebarShortcutId[]
): boolean =>
  first.length === second.length &&
  first.every((entry, index) => entry === second[index])

const coerceSidebarShortcutSelection = (
  value: unknown,
  fallback: SidebarShortcutId[]
): SidebarShortcutId[] => {
  if (!Array.isArray(value)) return fallback
  const allowed = new Set<SidebarShortcutId>(SIDEBAR_SHORTCUT_IDS)
  const legacyMap: Record<string, SidebarShortcutId> = {
    knowledge: "knowledge-qa",
    "multi-item": "multi-item-review",
    "moderation-playground": "moderation-rules"
  }
  const unique = new Set<SidebarShortcutId>()
  for (const entry of value) {
    if (typeof entry !== "string") continue
    const mapped = legacyMap[entry] ?? entry
    if (allowed.has(mapped as SidebarShortcutId)) {
      unique.add(mapped as SidebarShortcutId)
    }
  }
  const normalized = SIDEBAR_SHORTCUT_IDS.filter((id) => unique.has(id)).slice(
    0,
    SIDEBAR_SHORTCUT_MAX_COUNT
  )
  if (
    areShortcutSelectionsEqual(normalized, LEGACY_DEFAULT_SIDEBAR_SHORTCUT_SELECTION)
  ) {
    return DEFAULT_SIDEBAR_SHORTCUT_SELECTION
  }
  if (normalized.length === 0 && value.length > 0) {
    return fallback
  }
  return normalized
}

export const SIDEBAR_SHORTCUT_SELECTION_SETTING = defineSetting(
  "tldw:sidebar:shortcutSelection",
  DEFAULT_SIDEBAR_SHORTCUT_SELECTION,
  (value) =>
    coerceSidebarShortcutSelection(value, DEFAULT_SIDEBAR_SHORTCUT_SELECTION),
  {
    area: "local"
  }
)

const VOICE_CHAT_TTS_MODE_VALUES = ["stream", "full"] as const
export type VoiceChatTtsMode = (typeof VOICE_CHAT_TTS_MODE_VALUES)[number]

const coerceVoiceChatTtsMode = (
  value: unknown,
  fallback: VoiceChatTtsMode
): VoiceChatTtsMode => {
  const normalized = String(value || "").toLowerCase()
  return VOICE_CHAT_TTS_MODE_VALUES.includes(normalized as VoiceChatTtsMode)
    ? (normalized as VoiceChatTtsMode)
    : fallback
}

export const VOICE_CHAT_ENABLED_SETTING = defineSetting(
  "voiceChatEnabled",
  false,
  (value) => coerceBoolean(value, false),
  { area: "local" }
)

export const VOICE_CHAT_MODEL_SETTING = defineSetting(
  "voiceChatModel",
  "",
  (value) => coerceString(value, ""),
  { area: "local" }
)

export const VOICE_CHAT_PAUSE_MS_SETTING = defineSetting(
  "voiceChatPauseMs",
  900,
  (value) => coerceNumber(value, 900),
  {
    area: "local",
    validate: (value) => Number.isFinite(value) && value > 0
  }
)

export const VOICE_CHAT_TRIGGER_PHRASES_SETTING = defineSetting(
  "voiceChatTriggerPhrases",
  [] as string[],
  (value) => coerceStringArray(value, []),
  { area: "local" }
)

export const VOICE_CHAT_AUTO_RESUME_SETTING = defineSetting(
  "voiceChatAutoResume",
  true,
  (value) => coerceBoolean(value, true),
  { area: "local" }
)

export const VOICE_CHAT_BARGE_IN_SETTING = defineSetting(
  "voiceChatBargeIn",
  false,
  (value) => coerceBoolean(value, false),
  { area: "local" }
)

export const VOICE_CHAT_TTS_MODE_SETTING = defineSetting(
  "voiceChatTtsMode",
  "stream" as VoiceChatTtsMode,
  (value) => coerceVoiceChatTtsMode(value, "stream"),
  { area: "local" }
)

export const HEADER_SHORTCUTS_EXPANDED_SETTING = defineSetting(
  "headerShortcutsExpanded",
  false,
  (value) => coerceBoolean(value, false),
  {
    area: "local"
  }
)

const HEADER_SHORTCUTS_LAUNCHER_VIEW_VALUES = ["current", "legacy"] as const
export type HeaderShortcutsLauncherViewValue =
  (typeof HEADER_SHORTCUTS_LAUNCHER_VIEW_VALUES)[number]

export const HEADER_SHORTCUTS_LAUNCHER_VIEW_SETTING = defineSetting(
  "headerShortcutsLauncherView",
  "current" as HeaderShortcutsLauncherViewValue,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return HEADER_SHORTCUTS_LAUNCHER_VIEW_VALUES.includes(
      normalized as HeaderShortcutsLauncherViewValue
    )
      ? (normalized as HeaderShortcutsLauncherViewValue)
      : "current"
  },
  {
    area: "local"
  }
)

const coerceBooleanRecord = (value: unknown): Record<string, boolean> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  const entries = Object.entries(value as Record<string, unknown>)
  if (entries.length === 0) return value as Record<string, boolean>
  const hasInvalid = entries.some(([, entry]) => typeof entry !== "boolean")
  if (!hasInvalid) return value as Record<string, boolean>
  const normalized: Record<string, boolean> = {}
  for (const [key, entry] of entries) {
    if (typeof entry === "boolean") normalized[key] = entry
  }
  return normalized
}

export const SEEN_HINTS_SETTING = defineSetting(
  "tldw:seenHints",
  {} as Record<string, boolean>,
  (value) => coerceBooleanRecord(value),
  {
    area: "local"
  }
)

export const MEDIA_REVIEW_ORIENTATION_SETTING = defineSetting(
  "media-review-orientation",
  "vertical" as "vertical" | "horizontal",
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return normalized === "horizontal" ? "horizontal" : "vertical"
  },
  {
    area: "local"
  }
)

const VIEW_MODE_VALUES = ["spread", "list", "all"] as const
export type ViewModeValue = (typeof VIEW_MODE_VALUES)[number]

export const MEDIA_REVIEW_VIEW_MODE_SETTING = defineSetting(
  "media-review-view-mode",
  "spread" as ViewModeValue,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return VIEW_MODE_VALUES.includes(normalized as ViewModeValue)
      ? (normalized as ViewModeValue)
      : "spread"
  },
  {
    area: "local"
  }
)

export const MEDIA_REVIEW_FILTERS_COLLAPSED_SETTING = defineSetting(
  "media-review-filters-collapsed",
  true,
  (value) => coerceBoolean(value, false),
  {
    area: "local"
  }
)

export const MEDIA_REVIEW_AUTO_VIEW_MODE_SETTING = defineSetting(
  "media-review-auto-view-mode",
  true,
  (value) => coerceBoolean(value, true),
  {
    area: "local"
  }
)

const coerceIdArray = (value: unknown): Array<string | number> => {
  if (!Array.isArray(value)) return []
  return value.filter(
    (entry) => typeof entry === "string" || typeof entry === "number"
  )
}

export const MEDIA_REVIEW_SELECTION_SETTING = defineSetting(
  "media-review-selection",
  [] as Array<string | number>,
  coerceIdArray,
  {
    area: "local"
  }
)

export const MEDIA_REVIEW_FOCUSED_ID_SETTING = defineSetting(
  "media-review-focused-id",
  null as string | number | null,
  (value) => {
    if (value === null || value === undefined) return null
    if (typeof value === "string" || typeof value === "number") return value
    return null
  },
  {
    area: "local"
  }
)

export type DiscussMediaPrompt = MediaChatHandoffPayload

export const DISCUSS_MEDIA_PROMPT_SETTING = defineSetting(
  "tldw:discussMediaPrompt",
  undefined as DiscussMediaPrompt | undefined,
  (value) => normalizeMediaChatHandoffPayload(value),
  {
    area: "local",
    localStorageKey: "tldw:discussMediaPrompt",
    mirrorToLocalStorage: true
  }
)

export type DiscussWatchlistPrompt = WatchlistChatHandoffPayload

export const DISCUSS_WATCHLIST_PROMPT_SETTING = defineSetting(
  "tldw:discussWatchlistPrompt",
  undefined as DiscussWatchlistPrompt | undefined,
  (value) => normalizeWatchlistChatHandoffPayload(value),
  {
    area: "local",
    localStorageKey: "tldw:discussWatchlistPrompt",
    mirrorToLocalStorage: true
  }
)

export const LAST_MEDIA_ID_SETTING = defineSetting(
  "tldw:lastMediaId",
  undefined as string | undefined,
  coerceOptionalString,
  {
    area: "local",
    localStorageKey: "tldw:lastMediaId",
    mirrorToLocalStorage: true
  }
)

export const LAST_NOTE_ID_SETTING = defineSetting(
  "tldw:lastNoteId",
  undefined as string | undefined,
  coerceOptionalString,
  {
    area: "local",
    localStorageKey: "tldw:lastNoteId",
    mirrorToLocalStorage: true
  }
)

const NOTES_PAGE_SIZE_VALUES = [20, 50, 100] as const
export type NotesPageSize = (typeof NOTES_PAGE_SIZE_VALUES)[number]

const coerceNotesPageSize = (value: unknown): NotesPageSize => {
  const parsed = Math.round(coerceNumber(value, 20))
  if (NOTES_PAGE_SIZE_VALUES.includes(parsed as NotesPageSize)) {
    return parsed as NotesPageSize
  }
  return 20
}

export const NOTES_PAGE_SIZE_SETTING = defineSetting(
  "tldw:notesPageSize",
  20 as NotesPageSize,
  (value) => coerceNotesPageSize(value),
  {
    area: "local",
    localStorageKey: "tldw:notesPageSize",
    mirrorToLocalStorage: true
  }
)

const NOTES_TITLE_STRATEGY_VALUES = ["heuristic", "llm", "llm_fallback"] as const
export type NotesTitleSuggestStrategy = (typeof NOTES_TITLE_STRATEGY_VALUES)[number]

export const NOTES_TITLE_SUGGEST_STRATEGY_SETTING = defineSetting(
  "tldw:notesTitleSuggestStrategy",
  "heuristic" as NotesTitleSuggestStrategy,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    if (NOTES_TITLE_STRATEGY_VALUES.includes(normalized as NotesTitleSuggestStrategy)) {
      return normalized as NotesTitleSuggestStrategy
    }
    return "heuristic"
  },
  {
    area: "local",
    localStorageKey: "tldw:notesTitleSuggestStrategy",
    mirrorToLocalStorage: true
  }
)

export type NotesRecentOpenedEntry = {
  id: string
  title: string
}

const coerceNotesPinnedIds = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const deduped: string[] = []
  const seen = new Set<string>()
  for (const entry of value) {
    const rawId = String(entry || "").trim()
    if (!rawId || seen.has(rawId)) continue
    seen.add(rawId)
    deduped.push(rawId)
    if (deduped.length >= 500) break
  }
  return deduped
}

export const NOTES_PINNED_IDS_SETTING = defineSetting(
  "tldw:notesPinnedIds",
  [] as string[],
  coerceNotesPinnedIds,
  {
    area: "local",
    localStorageKey: "tldw:notesPinnedIds",
    mirrorToLocalStorage: true
  }
)

export type NotesNotebookSetting = {
  id: number
  name: string
  keywords: string[]
}

const coerceNotesNotebookSettings = (value: unknown): NotesNotebookSetting[] => {
  if (!Array.isArray(value)) return []
  const out: NotesNotebookSetting[] = []
  const seenIds = new Set<number>()
  const seenNames = new Set<string>()
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue
    const idCandidate = Number((entry as any).id)
    const name = String((entry as any).name || "").trim()
    if (!Number.isFinite(idCandidate) || idCandidate <= 0) continue
    if (!name) continue
    const id = Math.floor(idCandidate)
    if (seenIds.has(id)) continue
    const nameKey = name.toLowerCase()
    if (seenNames.has(nameKey)) continue
    const keywordsRaw = Array.isArray((entry as any).keywords)
      ? (entry as any).keywords
      : []
    const keywordSeen = new Set<string>()
    const keywords: string[] = []
    for (const keyword of keywordsRaw) {
      const normalizedKeyword = String(keyword || "").trim().toLowerCase()
      if (!normalizedKeyword || keywordSeen.has(normalizedKeyword)) continue
      keywordSeen.add(normalizedKeyword)
      keywords.push(normalizedKeyword)
      if (keywords.length >= 25) break
    }
    seenIds.add(id)
    seenNames.add(nameKey)
    out.push({
      id,
      name,
      keywords
    })
    if (out.length >= 100) break
  }
  return out
}

export const NOTES_NOTEBOOKS_SETTING = defineSetting(
  "tldw:notesNotebooks",
  [] as NotesNotebookSetting[],
  coerceNotesNotebookSettings,
  {
    area: "local",
    localStorageKey: "tldw:notesNotebooks",
    mirrorToLocalStorage: true
  }
)

const coerceNotesRecentOpened = (value: unknown): NotesRecentOpenedEntry[] => {
  if (!Array.isArray(value)) return []
  const deduped: NotesRecentOpenedEntry[] = []
  const seen = new Set<string>()
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue
    const rawId = String((entry as any).id || "").trim()
    const rawTitle = String((entry as any).title || "").trim()
    if (!rawId || !rawTitle) continue
    if (seen.has(rawId)) continue
    seen.add(rawId)
    deduped.push({
      id: rawId,
      title: rawTitle
    })
    if (deduped.length >= 5) break
  }
  return deduped
}

export const NOTES_RECENT_OPENED_SETTING = defineSetting(
  "tldw:notesRecentOpened",
  [] as NotesRecentOpenedEntry[],
  coerceNotesRecentOpened,
  {
    area: "local",
    localStorageKey: "tldw:notesRecentOpened",
    mirrorToLocalStorage: true
  }
)

export const LAST_DECK_ID_SETTING = defineSetting(
  "tldw:lastDeckId",
  undefined as string | undefined,
  coerceOptionalString,
  {
    area: "local",
    localStorageKey: "tldw:lastDeckId",
    mirrorToLocalStorage: true
  }
)

export const FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING = defineSetting(
  "tldw:flashcards:reviewOnboardingDismissed",
  false,
  (value) => coerceBoolean(value, false),
  {
    area: "local",
    localStorageKey: "tldw:flashcards:reviewOnboardingDismissed",
    mirrorToLocalStorage: true
  }
)

const FLASHCARDS_SHORTCUT_HINT_DENSITY_VALUES = [
  "expanded",
  "compact",
  "hidden"
] as const
export type FlashcardsShortcutHintDensity =
  (typeof FLASHCARDS_SHORTCUT_HINT_DENSITY_VALUES)[number]

export const FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING = defineSetting(
  "tldw:flashcards:shortcutHintDensity",
  "expanded" as FlashcardsShortcutHintDensity,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    if (
      FLASHCARDS_SHORTCUT_HINT_DENSITY_VALUES.includes(
        normalized as FlashcardsShortcutHintDensity
      )
    ) {
      return normalized as FlashcardsShortcutHintDensity
    }
    return "expanded"
  },
  {
    area: "local",
    localStorageKey: "tldw:flashcards:shortcutHintDensity",
    mirrorToLocalStorage: true
  }
)

export const DEFAULT_MEDIA_COLLAPSED_SECTIONS: Record<string, boolean> = {
  statistics: false,
  content: false,
  metadata: true,
  analysis: false,
  intelligence: true
}

export const MEDIA_COLLAPSED_SECTIONS_SETTING = defineSetting(
  "tldw:media:collapsedSections",
  DEFAULT_MEDIA_COLLAPSED_SECTIONS,
  (value) => coerceBooleanRecord(value),
  {
    area: "local"
  }
)

const MEDIA_TEXT_SIZE_PRESET_VALUES = ["s", "m", "l"] as const
export type MediaTextSizePreset = (typeof MEDIA_TEXT_SIZE_PRESET_VALUES)[number]

const coerceMediaTextSizePreset = (
  value: unknown
): MediaTextSizePreset => {
  const normalized = String(value || "").toLowerCase()
  if (MEDIA_TEXT_SIZE_PRESET_VALUES.includes(normalized as MediaTextSizePreset)) {
    return normalized as MediaTextSizePreset
  }
  return "m"
}

export const MEDIA_TEXT_SIZE_PRESET_SETTING = defineSetting(
  "tldw:media:textSizePreset",
  "m" as MediaTextSizePreset,
  (value) => coerceMediaTextSizePreset(value),
  {
    area: "local"
  }
)

export const MEDIA_HIDE_TRANSCRIPT_TIMINGS_SETTING = defineSetting(
  "tldw:media:hideTranscriptTimings",
  true,
  (value) => coerceBoolean(value, true),
  {
    area: "local"
  }
)

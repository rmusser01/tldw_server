/**
 * Two-tier label system for world book settings.
 *
 * Provides friendly (user-facing) and technical (power-user) labels,
 * descriptions, and notes for each configurable setting key.
 */

type SettingKey = "scan_depth" | "token_budget" | "recursive_scanning"

const FRIENDLY_LABELS: Record<SettingKey, string> = {
  scan_depth: "Messages to search",
  token_budget: "Context size limit",
  recursive_scanning: "Chain matching",
}

const TECHNICAL_LABELS: Record<SettingKey, string> = {
  scan_depth: "Scan Depth",
  token_budget: "Token Budget",
  recursive_scanning: "Recursive Scanning",
}

const FRIENDLY_DESCRIPTIONS: Record<SettingKey, string> = {
  scan_depth:
    "How far back in the conversation to look for keyword matches. Higher = more matches found, slower processing.",
  token_budget:
    "Maximum amount of world info added to each response. Higher = more lore available to the AI, but uses more of the conversation window.",
  recursive_scanning:
    "When a matched entry contains keywords from other entries, also include those. Useful for interconnected lore.",
}

const TECHNICAL_DESCRIPTIONS: Record<SettingKey, string> = {
  scan_depth:
    "scan_depth: 1-20. Number of recent messages to search for keyword matches.",
  token_budget:
    "token_budget: 50-5000 (tokens). Maximum tokens of world info injected into context.",
  recursive_scanning:
    "recursive_scanning: Also search matched content for additional keyword matches. Max depth configurable server-side.",
}

const TECHNICAL_NOTES: Record<SettingKey, string> = {
  scan_depth: "scan_depth: 1-20",
  token_budget: "token_budget: 50-5000 (tokens)",
  recursive_scanning: "recursive_scanning: max depth configurable",
}

function isSettingKey(key: string): key is SettingKey {
  return key === "scan_depth" || key === "token_budget" || key === "recursive_scanning"
}

/** Returns a friendly or technical label for the given setting key. Falls back to `key` for unknown settings. */
export function getSettingLabel(key: string, showTechnical: boolean): string {
  if (!isSettingKey(key)) return key
  return showTechnical ? TECHNICAL_LABELS[key] : FRIENDLY_LABELS[key]
}

/** Returns a friendly or technical description for the given setting key. Returns `""` for unknown settings. */
export function getSettingDescription(key: string, showTechnical: boolean): string {
  if (!isSettingKey(key)) return ""
  return showTechnical ? TECHNICAL_DESCRIPTIONS[key] : FRIENDLY_DESCRIPTIONS[key]
}

/** Returns the technical note (API field name and range) for the given setting key. Returns `""` for unknown settings. */
export function getSettingTechnicalNote(key: string): string {
  if (!isSettingKey(key)) return ""
  return TECHNICAL_NOTES[key]
}

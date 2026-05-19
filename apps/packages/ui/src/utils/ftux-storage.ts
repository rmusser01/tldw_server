export const FIRST_INGEST_DISMISS_KEY = "tldw:media:first-ingest-dismissed"
const LEGACY_FIRST_INGEST_DISMISS_KEYS = ["tldw_first_ingest_tutorial_dismissed"]

export const MCP_HUB_EXPLAINER_DISMISSED_KEY = "tldw:mcp-hub:explainer-dismissed"
const LEGACY_MCP_HUB_EXPLAINER_DISMISSED_KEYS = ["tldw_mcp_hub_explainer_dismissed"]

const readBooleanFlagWithMigration = (
  currentKey: string,
  legacyKeys: string[]
): boolean => {
  try {
    if (localStorage.getItem(currentKey) === "true") {
      return true
    }

    for (const legacyKey of legacyKeys) {
      if (localStorage.getItem(legacyKey) !== "true") {
        continue
      }

      localStorage.setItem(currentKey, "true")
      localStorage.removeItem(legacyKey)
      return true
    }
  } catch {
    return false
  }

  return false
}

const persistBooleanFlag = (key: string): void => {
  try {
    localStorage.setItem(key, "true")
  } catch {
    // ignore storage failures
  }
}

export const readFirstIngestDismissed = (): boolean =>
  readBooleanFlagWithMigration(FIRST_INGEST_DISMISS_KEY, LEGACY_FIRST_INGEST_DISMISS_KEYS)

export const persistFirstIngestDismissed = (): void => {
  persistBooleanFlag(FIRST_INGEST_DISMISS_KEY)
}

export const clearFirstIngestDismissed = (): void => {
  try {
    localStorage.removeItem(FIRST_INGEST_DISMISS_KEY)
  } catch {
    // ignore storage failures
  }
}

export const readMcpHubExplainerDismissed = (): boolean =>
  readBooleanFlagWithMigration(
    MCP_HUB_EXPLAINER_DISMISSED_KEY,
    LEGACY_MCP_HUB_EXPLAINER_DISMISSED_KEYS
  )

export const persistMcpHubExplainerDismissed = (): void => {
  persistBooleanFlag(MCP_HUB_EXPLAINER_DISMISSED_KEY)
}

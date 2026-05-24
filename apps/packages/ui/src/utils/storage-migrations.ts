import { createSafeStorage } from "@/utils/safe-storage"

export const MODEL_PLAYGROUND_SIDEBAR_KEY = "modelPlaygroundSidebarOpen"
export const MODEL_PLAYGROUND_DEBUG_KEY = "modelPlaygroundDebugOpen"

// Research Workspace pane state keys
export const WORKSPACE_LEFT_PANE_KEY = "workspaceLeftPaneOpen"
export const WORKSPACE_RIGHT_PANE_KEY = "workspaceRightPaneOpen"

// Landing Hub preference key
export const SKIP_LANDING_HUB_KEY = "tldw_skip_landing_hub"

const LEGACY_SIDEBAR_KEY = "workspaceSidebarOpen"
const LEGACY_DEBUG_KEY = "workspaceDebugOpen"
const LEGACY_SKIP_LANDING_HUB_KEY = "skipLandingHub"

const storage = createSafeStorage({ area: "local" })

const isUnset = (value: unknown): value is null | undefined =>
  value === undefined || value === null

export const migrateModelPlaygroundPrefs = async (
  storageInstance = storage
): Promise<void> => {
  try {
    const [newSidebar, newDebug] = await Promise.all([
      storageInstance.get<boolean | null | undefined>(MODEL_PLAYGROUND_SIDEBAR_KEY),
      storageInstance.get<boolean | null | undefined>(MODEL_PLAYGROUND_DEBUG_KEY)
    ])

    const legacySidebar = await storageInstance.get<boolean | null | undefined>(
      LEGACY_SIDEBAR_KEY
    )
    const legacyDebug = await storageInstance.get<boolean | null | undefined>(
      LEGACY_DEBUG_KEY
    )

    if (isUnset(newSidebar) && !isUnset(legacySidebar)) {
      await storageInstance.set(MODEL_PLAYGROUND_SIDEBAR_KEY, legacySidebar)
    }

    if (isUnset(newDebug) && !isUnset(legacyDebug)) {
      await storageInstance.set(MODEL_PLAYGROUND_DEBUG_KEY, legacyDebug)
    }

    if (!isUnset(legacySidebar)) {
      await storageInstance.remove(LEGACY_SIDEBAR_KEY)
    }

    if (!isUnset(legacyDebug)) {
      await storageInstance.remove(LEGACY_DEBUG_KEY)
    }
  } catch {
    // ignore storage failures; defaults still apply
  }
}

export const migrateLandingHubPrefs = async (
  storageInstance = storage
): Promise<void> => {
  try {
    const [newValue, legacyValue] = await Promise.all([
      storageInstance.get<boolean | null | undefined>(SKIP_LANDING_HUB_KEY),
      storageInstance.get<boolean | null | undefined>(LEGACY_SKIP_LANDING_HUB_KEY)
    ])

    if (isUnset(newValue) && !isUnset(legacyValue)) {
      await storageInstance.set(SKIP_LANDING_HUB_KEY, legacyValue)
    }

    if (!isUnset(legacyValue)) {
      await storageInstance.remove(LEGACY_SKIP_LANDING_HUB_KEY)
    }
  } catch {
    // ignore storage failures; defaults still apply
  }
}

export const runStorageMigrations = async (): Promise<void> => {
  await migrateModelPlaygroundPrefs()
  await migrateLandingHubPrefs()
}

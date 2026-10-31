/**
 * Stored server URL lookup, kept separate from `tldw-server.ts`.
 *
 * `tldw-server.ts` subscribes to model-cache invalidation at module scope, so
 * importing it eagerly pulls in `tldwClient`/`tldwModels` and, through the
 * `services/tldw` barrel, every API domain module. Connection bootstrap runs on
 * the app-shell path and needs only this one function, so it lives here to keep
 * the API client out of the bundle that every page downloads.
 */
import { readTldwSetting } from "@/services/tldw-settings-storage"

/**
 * Read any previously stored tldw server URL from extension storage,
 * without falling back to the hard-coded default.
 *
 * This is used by connection bootstrap code to distinguish a true
 * first-run (no URL configured anywhere) from a misconfigured server.
 */
export const getStoredTldwServerURL = async (): Promise<string | null> => {
  try {
    const url = await readTldwSetting<string>("tldwServerUrl")
    if (typeof url === "string") {
      const trimmed = url.trim()
      if (trimmed.length > 0) {
        return trimmed
      }
    }
  } catch {
    // Ignore storage read failures; caller will treat as "no URL".
  }
  return null
}

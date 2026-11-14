import { Storage } from "@plasmohq/storage"
import type { SerdeOptions, StorageOptions } from "@plasmohq/storage"

/**
 * Deserializer that tolerates both:
 * - Values written via Plasmo Storage (JSON-stringified)
 * - Plain objects written via chrome.storage.local.set({...})
 *
 * This prevents runtime JSON.parse errors such as:
 *   SyntaxError: "[object Object]" is not valid JSON
 */
export const safeStorageSerde: SerdeOptions & { deserializer: <T = unknown>(value: unknown) => T } = {
  serializer: JSON.stringify,
  deserializer: (value: any) => {
    if (typeof value === "string") {
      try {
        return JSON.parse(value)
      } catch {
        // If the stored string is not valid JSON, return it as-is
        return value
      }
    }
    // Non-string values (objects, numbers, booleans) are already structured
    return value
  }
}

/**
 * Helper for creating a Storage instance that can safely read values
 * written either through Plasmo Storage or directly via chrome.storage.*.
 */
export const createSafeStorage = (options: StorageOptions = {}): Storage => {
  const { serde, ...rest } = options || {}

  return new Storage({
    ...rest,
    serde: {
      ...safeStorageSerde,
      ...(serde || {})
    }
  })
}

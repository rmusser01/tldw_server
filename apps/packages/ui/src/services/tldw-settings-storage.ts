import { createSafeStorage } from "@/utils/safe-storage"

const localStorage = createSafeStorage({ area: "local" })
const legacySyncStorage = createSafeStorage({ area: "sync" })

export const LEGACY_TLDW_SETTING_KEYS = [
  "tldwServerUrl",
  "pageShareUrl",
  "systemPromptForNonRag",
  "systemPromptForNonRagOption",
  "systemPromptForRag",
  "questionPromptForRag"
] as const

export type LegacyTldwSettingKey = (typeof LEGACY_TLDW_SETTING_KEYS)[number]

export const readTldwSetting = async <T>(
  key: LegacyTldwSettingKey
): Promise<T | undefined> => {
  const current = await localStorage.get<T>(key)
  if (current !== undefined) return current

  const legacy = await legacySyncStorage.get<T>(key)
  if (legacy === undefined) return undefined

  await localStorage.set(key, legacy)
  await legacySyncStorage.remove(key).catch(() => undefined)
  return legacy
}

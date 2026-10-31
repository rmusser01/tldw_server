import type { SettingDef } from '@/services/settings/registry'

/** Never import browser-wide private metadata into an unidentified account. */
export function notesAuthoritySetting<T>(setting: SettingDef<T>, scope: string | null | undefined): SettingDef<T> | null {
  if (!scope) return null
  return {
    ...setting,
    key: `${setting.key}:${scope}`,
    localStorageKey: setting.localStorageKey ? `${setting.localStorageKey}:${scope}` : undefined
  }
}

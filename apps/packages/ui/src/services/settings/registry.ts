import type { StorageOptions } from "@plasmohq/storage"
import { createSafeStorage } from "@/utils/safe-storage"

type CoerceFn<T> = (value: unknown) => T
type ValidateFn<T> = (value: T) => boolean
type LocalStorageSerializeFn<T> = (value: T) => string
type LocalStorageDeserializeFn = (value: string) => unknown
type BeforeGetFn = () => void | Promise<void>

type SettingOptions<T> = {
  coerce?: CoerceFn<T>
  validate?: ValidateFn<T>
  area?: StorageOptions["area"]
  localStorageKey?: string
  mirrorToLocalStorage?: boolean
  localStorageSerialize?: LocalStorageSerializeFn<T>
  localStorageDeserialize?: LocalStorageDeserializeFn
  beforeGet?: BeforeGetFn
}

export type SettingDef<T> = {
  key: string
  defaultValue: T
  coerce?: CoerceFn<T>
  validate?: ValidateFn<T>
  area?: StorageOptions["area"]
  localStorageKey?: string
  mirrorToLocalStorage?: boolean
  localStorageSerialize?: LocalStorageSerializeFn<T>
  localStorageDeserialize?: LocalStorageDeserializeFn
  beforeGet?: BeforeGetFn
}

const storageCache = new Map<StorageOptions["area"] | undefined, ReturnType<typeof createSafeStorage>>()

// WP-20: per-context cache of normalized setting values. Reads hit the cache
// instead of re-issuing chrome.storage reads + localStorage mirror writes on
// every get. The cache is invalidated by `chrome.storage.onChanged`, so writes
// from OTHER contexts (options page vs sidepanel) are picked up on the next
// read-through miss.
const valueCache = new Map<string, unknown>()

// Last localStorage mirror value this context wrote or observed (serialized
// form). The mirror is only written when the normalized value differs from
// this cached form.
const mirrorCache = new Map<string, string | null>()

const settingCacheKey = (setting: SettingDef<unknown>): string =>
  `${setting.area ?? "local"}::${setting.key}`

let storageChangeListenerBound = false
const ensureStorageChangeListener = (): void => {
  if (storageChangeListenerBound) return
  storageChangeListenerBound = true
  try {
    const globalApi = globalThis as {
      chrome?: { storage?: { onChanged?: { addListener?: (listener: (changes: Record<string, unknown>, areaName: string) => void) => void } } }
      browser?: { storage?: { onChanged?: { addListener?: (listener: (changes: Record<string, unknown>, areaName: string) => void) => void } } }
    }
    const onChanged =
      globalApi.chrome?.storage?.onChanged ??
      globalApi.browser?.storage?.onChanged
    if (typeof onChanged?.addListener !== "function") return
    onChanged.addListener((changes, areaName) => {
      for (const key of Object.keys(changes || {})) {
        valueCache.delete(`${areaName}::${key}`)
      }
    })
  } catch {
    // Non-extension contexts have no storage.onChanged; the cache simply
    // lives for this page's lifetime.
  }
}

export const getStorageForSetting = (setting: SettingDef<unknown>) => {
  const area = setting.area
  if (storageCache.has(area)) {
    return storageCache.get(area)!
  }
  const storage = createSafeStorage(area ? { area } : {})
  storageCache.set(area, storage)
  return storage
}

export const defineSetting = <T>(
  key: string,
  defaultValue: T,
  coerceOrOptions?: CoerceFn<T> | SettingOptions<T>,
  maybeOptions?: SettingOptions<T>
): SettingDef<T> => {
  if (typeof coerceOrOptions === "function") {
    return {
      key,
      defaultValue,
      coerce: coerceOrOptions,
      ...(maybeOptions || {})
    }
  }
  return {
    key,
    defaultValue,
    ...(coerceOrOptions || {})
  }
}

const isUnset = (value: unknown): boolean =>
  value === undefined || value === null || value === ""

const safeParse = (value: string): unknown => {
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

const readLocalStorageValue = <T>(setting: SettingDef<T>): unknown => {
  if (!setting.localStorageKey) return undefined
  if (typeof window === "undefined") return undefined
  try {
    const raw = window.localStorage.getItem(setting.localStorageKey)
    if (raw === null) return undefined
    return setting.localStorageDeserialize
      ? setting.localStorageDeserialize(raw)
      : safeParse(raw)
  } catch {
    return undefined
  }
}

const clearLocalStorageValue = (setting: SettingDef<unknown>) => {
  if (!setting.localStorageKey) return
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(setting.localStorageKey)
  } catch {
    // ignore localStorage errors
  }
}

const writeLocalStorageValue = <T>(setting: SettingDef<T>, value: T) => {
  if (!setting.localStorageKey || !setting.mirrorToLocalStorage) return
  if (typeof window === "undefined") return
  try {
    const mirrorKey = setting.localStorageKey
    if (isUnset(value)) {
      // Compare against the cached mirror so a repeated unset read does not
      // issue removeItem on every get.
      let last = mirrorCache.get(mirrorKey)
      if (last === undefined) {
        last = window.localStorage.getItem(mirrorKey)
      }
      if (last !== null) {
        window.localStorage.removeItem(mirrorKey)
      }
      mirrorCache.set(mirrorKey, null)
      return
    }
    const serialized = setting.localStorageSerialize
      ? setting.localStorageSerialize(value)
      : typeof value === "string"
        ? value
        : JSON.stringify(value)
    // Write-only-on-change: the mirror is compared against the cached value
    // (what this context last wrote or observed), not re-written per read.
    let last = mirrorCache.get(mirrorKey)
    if (last === undefined) {
      last = window.localStorage.getItem(mirrorKey)
    }
    if (last !== serialized) {
      window.localStorage.setItem(mirrorKey, serialized)
    }
    mirrorCache.set(mirrorKey, serialized)
  } catch {
    // ignore localStorage errors
  }
}

export const normalizeSettingValue = <T>(
  setting: SettingDef<T>,
  value: unknown
): T => {
  if (isUnset(value)) return setting.defaultValue
  const coerced = setting.coerce ? setting.coerce(value) : (value as T)
  if (setting.validate && !setting.validate(coerced)) {
    return setting.defaultValue
  }
  return coerced
}

/** Opt in to read failures when absence must not authorize overwriting stored data. */
export const getSetting = async <T,>(
  setting: SettingDef<T>,
  options?: { throwOnError?: boolean; isCurrent?: () => boolean }
): Promise<T> => {
  try {
    ensureStorageChangeListener()
    const cacheKey = settingCacheKey(setting as unknown as SettingDef<unknown>)
    if (valueCache.has(cacheKey)) {
      // Invariant: cached values are SHARED references, not copies (copying
      // per read would defeat the cache). Consumers must treat getSetting
      // results as read-only and never mutate object/array values in place —
      // no mutating consumer exists today; keep it that way.
      return valueCache.get(cacheKey) as T
    }
    await setting.beforeGet?.()
    if (options?.isCurrent?.() === false) return setting.defaultValue
    const storage = getStorageForSetting(setting)
    const raw = await storage.get(setting.key)
    // An old read must not migrate or mirror over a newer explicit choice.
    if (options?.isCurrent?.() === false) return setting.defaultValue
    let normalized: T
    let cacheable = true
    if (isUnset(raw)) {
      const legacy = readLocalStorageValue(setting)
      if (!isUnset(legacy)) {
        normalized = normalizeSettingValue(setting, legacy)
        await storage.set(setting.key, normalized)
        writeLocalStorageValue(setting, normalized)
        if (!setting.mirrorToLocalStorage) clearLocalStorageValue(setting)
      } else {
        normalized = setting.defaultValue
        // An absent setting is not cached: absence stays observable for
        // future writes/first-set migrations without extra invalidation.
        cacheable = false
      }
    } else {
      normalized = normalizeSettingValue(setting, raw)
      if (normalized !== raw) {
        await storage.set(setting.key, normalized)
      }
      writeLocalStorageValue(setting, normalized)
    }
    if (cacheable) {
      valueCache.set(cacheKey, normalized)
    }
    return normalized
  } catch (error) {
    if (options?.throwOnError) throw error
    return setting.defaultValue
  }
}

export const setSetting = async <T>(setting: SettingDef<T>, value: T) => {
  ensureStorageChangeListener()
  const storage = getStorageForSetting(setting)
  const normalized = normalizeSettingValue(setting, value)
  await storage.set(setting.key, normalized)
  valueCache.set(settingCacheKey(setting as unknown as SettingDef<unknown>), normalized)
  writeLocalStorageValue(setting, normalized)
}

export const clearSetting = async (setting: SettingDef<unknown>) => {
  ensureStorageChangeListener()
  const storage = getStorageForSetting(setting)
  await storage.remove(setting.key)
  valueCache.delete(settingCacheKey(setting))
  if (setting.localStorageKey) mirrorCache.delete(setting.localStorageKey)
  clearLocalStorageValue(setting)
}

/**
 * Test-only seam: drop the module-level value/mirror caches. Suites that seed
 * storage directly (bypassing `setSetting`) or that swap the storage mock have
 * no `storage.onChanged` invalidation path in jsdom, so a cached value from an
 * earlier test would be served across the reset. Production never needs this:
 * the cache is invalidated by `storage.onChanged`, `setSetting` and
 * `clearSetting`.
 */
export const __clearSettingCachesForTests = (): void => {
  valueCache.clear()
  mirrorCache.clear()
}

export const coerceBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === "boolean") return value
  if (typeof value === "string") return value === "true"
  return fallback
}

export const coerceBooleanOrNull = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    if (value === "true") return true
    if (value === "false") return false
  }
  if (typeof value === "number") {
    if (value === 1) return true
    if (value === 0) return false
  }
  return null
}

export const coerceNumber = (value: unknown, fallback: number): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

export const coerceString = (value: unknown, fallback: string): string => {
  if (typeof value === "string" && value.length > 0) return value
  return fallback
}

export const coerceOptionalString = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.length > 0) return value
  return undefined
}

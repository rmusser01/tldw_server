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
    if (isUnset(value)) {
      window.localStorage.removeItem(setting.localStorageKey)
      return
    }
    const serialized = setting.localStorageSerialize
      ? setting.localStorageSerialize(value)
      : typeof value === "string"
        ? value
        : JSON.stringify(value)
    window.localStorage.setItem(setting.localStorageKey, serialized)
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

export const getSetting = async <T>(setting: SettingDef<T>): Promise<T> => {
  try {
    await setting.beforeGet?.()
    const storage = getStorageForSetting(setting)
    const raw = await storage.get(setting.key)
    if (isUnset(raw)) {
      const legacy = readLocalStorageValue(setting)
      if (!isUnset(legacy)) {
        const normalized = normalizeSettingValue(setting, legacy)
        await storage.set(setting.key, normalized)
        writeLocalStorageValue(setting, normalized)
        if (!setting.mirrorToLocalStorage) clearLocalStorageValue(setting)
        return normalized
      }
      return setting.defaultValue
    }
    const normalized = normalizeSettingValue(setting, raw)
    if (normalized !== raw) {
      await storage.set(setting.key, normalized)
    }
    writeLocalStorageValue(setting, normalized)
    return normalized
  } catch {
    return setting.defaultValue
  }
}

export const setSetting = async <T>(setting: SettingDef<T>, value: T) => {
  const storage = getStorageForSetting(setting)
  const normalized = normalizeSettingValue(setting, value)
  await storage.set(setting.key, normalized)
  writeLocalStorageValue(setting, normalized)
}

export const clearSetting = async (setting: SettingDef<unknown>) => {
  const storage = getStorageForSetting(setting)
  await storage.remove(setting.key)
  clearLocalStorageValue(setting)
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

import { useCallback, useLayoutEffect, useMemo, useState } from "react"
import { useStorage } from "@plasmohq/storage/hook"
import type { SettingDef } from "@/services/settings/registry"
import {
  getSetting,
  getStorageForSetting,
  normalizeSettingValue,
  setSetting
} from "@/services/settings/registry"

export const useSetting = <T,>(setting: SettingDef<T>) => {
  const storage = getStorageForSetting(setting)
  const [publication] = useState(() => {
    let binding = setting
    let revision = 0
    let latest = setting.defaultValue
    type Watchers = Parameters<typeof storage.watch>[0]
    const subscriptions = new WeakMap<
      Watchers,
      { wrapped: Watchers; current: boolean; storage: typeof storage }
    >()
    const instance = new Proxy(storage, {
      get(target, property) {
        if (property === "get")
          return async (key: string) => {
            if (key !== binding.key) return latest
            const original = revision
            const value = await getSetting(binding, {
              isCurrent: () => revision === original
            })
            if (revision !== original) return latest
            latest = normalizeSettingValue(binding, value)
            return latest
          }
        if (property === "set")
          return (key: string, value: unknown) =>
            getStorageForSetting(binding).set(key, value)
        if (property === "remove")
          return (key: string) => getStorageForSetting(binding).remove(key)
        if (property === "watch")
          return (callbacks: Watchers) => {
            const watched = binding
            const subscription = {
              wrapped: {} as Watchers,
              current: true,
              storage: getStorageForSetting(watched)
            }
            for (const key of Object.keys(callbacks)) {
              subscription.wrapped[key] = (change, ...args: unknown[]) => {
                if (
                  !subscription.current ||
                  binding.key !== watched.key ||
                  getStorageForSetting(binding).area !==
                    subscription.storage.area
                )
                  return
                revision++
                latest = normalizeSettingValue(binding, change.newValue)
                Reflect.apply(callbacks[key], undefined, [change, ...args])
              }
            }
            subscriptions.set(callbacks, subscription)
            const unwatch: unknown = subscription.storage.watch(
              subscription.wrapped
            )
            if (typeof unwatch === "function")
              return () => {
                revision++
                subscription.current = false
                subscriptions.delete(callbacks)
                unwatch()
              }
            return unwatch
          }
        if (property === "unwatch")
          return (callbacks: Watchers) => {
            revision++
            const subscription = subscriptions.get(callbacks)
            if (subscription) subscription.current = false
            subscriptions.delete(callbacks)
            return (subscription?.storage ?? target).unwatch(
              subscription?.wrapped ?? callbacks
            )
          }
        return Reflect.get(target, property, target)
      }
    })
    return {
      instance,
      current: () => latest,
      publish: (value: T) => {
        revision++
        latest = value
      },
      bind: (next: SettingDef<T>) => {
        if (
          binding.key !== next.key ||
          getStorageForSetting(binding).area !== getStorageForSetting(next).area
        ) {
          revision++
          latest = next.defaultValue
        } else latest = normalizeSettingValue(next, latest)
        binding = next
      }
    }
  })
  // shortcut: the native SDK requires remounting for same-key area swaps; add explicit rebinding if a caller needs them.
  useLayoutEffect(() => {
    publication.bind(setting)
  }, [publication, setting])
  const [rawValue, , meta] = useStorage<T | undefined>(
    { key: setting.key, instance: publication.instance },
    setting.defaultValue
  )

  const value = useMemo(
    () => normalizeSettingValue(setting, rawValue),
    [setting, rawValue]
  )

  const setValue = useCallback(
    async (next: T | ((prev: T) => T)) => {
      const resolved =
        typeof next === "function"
          ? (next as (prev: T) => T)(publication.current())
          : next
      const normalized = normalizeSettingValue(setting, resolved)
      publication.publish(normalized)
      // Keep the explicit choice ahead of older hydration in either storage backend.
      meta?.setRenderValue?.(normalized)
      await setSetting(setting, normalized)
    },
    [meta, publication, setting]
  )

  return [value, setValue, meta] as const
}

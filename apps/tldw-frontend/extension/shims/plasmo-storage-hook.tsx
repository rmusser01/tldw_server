import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Storage } from "./plasmo-storage"

type UseStorageOptions<T> = {
  key: string
  instance?: Storage
  defaultValue?: T
}

type UseStorageMeta<T> = {
  isLoading: boolean
  setRenderValue: (value: T | undefined) => void
}

type SetValue<T> = (
  value: T | ((prev: T | undefined) => T)
) => Promise<void>

export function useStorage<T = unknown>(
  keyOrOptions: string | UseStorageOptions<T>,
  defaultValue?: T
): [T | undefined, SetValue<T>, UseStorageMeta<T>] {
  const options: UseStorageOptions<T> =
    typeof keyOrOptions === "string"
      ? { key: keyOrOptions, defaultValue }
      : keyOrOptions

  const storage = useMemo(
    () => options.instance ?? new Storage(),
    [options.instance]
  )

  const defaultValueRef = useRef<T | undefined>(options.defaultValue)
  const [value, setValue] = useState<T | undefined>(defaultValueRef.current)
  const [isLoading, setIsLoading] = useState(true)

  // Track the freshest value so functional updates (`setValue(v => ...)`) don't
  // read a stale render closure and drop updates.
  const valueRef = useRef<T | undefined>(value)
  const applyValue = useCallback((next: T | undefined) => {
    valueRef.current = next
    setValue(next)
  }, [])

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    storage
      .get<T>(options.key)
      .then((stored) => {
        if (cancelled) return
        applyValue(stored === undefined ? defaultValueRef.current : stored)
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false)
        }
      })

    // Subscribe so cross-instance / cross-tab writes apply without a reload.
    const unwatch = storage.watch({
      [options.key]: (change) => {
        if (cancelled) return
        applyValue(
          change.newValue === undefined
            ? defaultValueRef.current
            : (change.newValue as T)
        )
      }
    })

    return () => {
      cancelled = true
      unwatch()
    }
  }, [options.key, storage, applyValue])

  const setStoredValue = useCallback<SetValue<T>>(
    async (next) => {
      const resolved =
        typeof next === "function"
          ? (next as (prev: T | undefined) => T)(valueRef.current)
          : next
      applyValue(resolved)
      await storage.set(options.key, resolved)
    },
    [options.key, storage, applyValue]
  )

  return [value, setStoredValue, { isLoading, setRenderValue: applyValue }]
}

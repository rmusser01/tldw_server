const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

type ExtensionStorageArea = {
  get?: (
    key: string,
    callback?: (items: Record<string, unknown>) => void
  ) => Promise<Record<string, unknown>> | void
  set?: (
    items: Record<string, unknown>,
    callback?: () => void
  ) => Promise<void> | void
  remove?: (key: string, callback?: () => void) => Promise<void> | void
}

type ExtensionChrome = {
  storage?: {
    session?: ExtensionStorageArea | null
    local?: ExtensionStorageArea | null
  }
  runtime?: { lastError?: unknown }
}

const getExtensionChrome = (): ExtensionChrome | undefined =>
  (globalThis as { chrome?: ExtensionChrome }).chrome

const getExtensionStorageArea = (): ExtensionStorageArea | null => {
  const storage = getExtensionChrome()?.storage
  return storage?.session ?? storage?.local ?? null
}

const reportStorageFailure = (operation: string, error: unknown): void => {
  console.warn(`Web clipper agent-task handoff ${operation} failed`, error)
}

const getChromeRuntimeError = (): unknown =>
  getExtensionChrome()?.runtime?.lastError ?? null

export const readExtensionStorageValue = async (key: string): Promise<unknown> => {
  const storage = getExtensionStorageArea()
  const get = storage?.get
  if (!get) return undefined

  return new Promise((resolve) => {
    let settled = false
    const settle = (items: unknown) => {
      if (settled) return
      settled = true
      resolve(isRecord(items) ? items[key] : undefined)
    }

    try {
      const maybePromise = get.call(storage, key, (items) => {
        const runtimeError = getChromeRuntimeError()
        if (runtimeError) {
          reportStorageFailure("extension read", runtimeError)
          settle(undefined)
          return
        }
        settle(items)
      })
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(settle).catch((error) => {
          reportStorageFailure("extension read", error)
          settle(undefined)
        })
      }
    } catch (error) {
      reportStorageFailure("extension read", error)
      settle(undefined)
    }
  })
}

export const writeExtensionStorageValue = async (
  key: string,
  value: unknown
): Promise<boolean> => {
  const storage = getExtensionStorageArea()
  const set = storage?.set
  if (!set) return false

  return new Promise((resolve) => {
    let settled = false
    const settle = (success: boolean) => {
      if (settled) return
      settled = true
      resolve(success)
    }

    try {
      const maybePromise = set.call(storage, { [key]: value }, () => {
        const runtimeError = getChromeRuntimeError()
        if (runtimeError) {
          reportStorageFailure("extension write", runtimeError)
          settle(false)
          return
        }
        settle(true)
      })
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(() => settle(true)).catch((error) => {
          reportStorageFailure("extension write", error)
          settle(false)
        })
      }
    } catch (error) {
      reportStorageFailure("extension write", error)
      settle(false)
    }
  })
}

export const removeExtensionStorageValue = async (key: string): Promise<boolean> => {
  const storage = getExtensionStorageArea()
  const remove = storage?.remove
  if (!remove) return false

  return new Promise((resolve) => {
    let settled = false
    const settle = (success: boolean) => {
      if (settled) return
      settled = true
      resolve(success)
    }

    try {
      const maybePromise = remove.call(storage, key, () => {
        const runtimeError = getChromeRuntimeError()
        if (runtimeError) {
          reportStorageFailure("extension remove", runtimeError)
          settle(false)
          return
        }
        settle(true)
      })
      if (maybePromise && typeof maybePromise.then === "function") {
        void maybePromise.then(() => settle(true)).catch((error) => {
          reportStorageFailure("extension remove", error)
          settle(false)
        })
      }
    } catch (error) {
      reportStorageFailure("extension remove", error)
      settle(false)
    }
  })
}

import { vi } from "vitest"

export const ensureLocalStorageApi = (): Storage => {
  const existing = window.localStorage as Partial<Storage>
  if (
    typeof existing?.getItem === "function" &&
    typeof existing?.setItem === "function" &&
    typeof existing?.removeItem === "function" &&
    typeof existing?.clear === "function" &&
    typeof existing?.key === "function"
  ) {
    return existing as Storage
  }

  const storage = new Map<string, string>()
  const shim = {
    getItem: vi.fn((key: string) => storage.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => {
      storage.set(key, String(value))
    }),
    removeItem: vi.fn((key: string) => {
      storage.delete(key)
    }),
    clear: vi.fn(() => {
      storage.clear()
    }),
    key: vi.fn((index: number) => Array.from(storage.keys())[index] ?? null),
    get length() {
      return storage.size
    }
  } as Storage

  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: shim
  })
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: shim
  })

  return shim
}

/** Seed document-memory storage for manual-key UAT, never native persistence.
 * These five admin probes do not test durable storage or cross-tab sync. Each
 * navigation starts fresh and is reseeded by Playwright before app boot.
 * Keep this callback self-contained because Playwright serializes it.
 */
export function seedManualUatBrowser({ webUrl, serverUrl, apiKey, legacyBootstrap = false }) {
  // Init scripts also execute in new frames and after cross-origin navigation.
  // Only the explicitly selected WebUI may receive the operator's credential.
  const target = new URL(webUrl)
  if (!["http:", "https:"].includes(target.protocol)) return
  if (location.origin !== target.origin) return
  const createMemoryStorage = () => {
    const values = new Map()
    const methods = {
      getItem: key => values.get(String(key)) ?? null,
      setItem: (key, value) => { values.set(String(key), String(value)) },
      removeItem: key => { values.delete(String(key)) },
      clear: () => { values.clear() },
      key: index => Array.from(values.keys())[Number(index) >>> 0] ?? null,
      get length() { return values.size }
    }
    // Preserve named properties and key enumeration as well as Storage methods.
    return new Proxy(Object.create(methods), {
      get: (object, key) => key in object ? Reflect.get(object, key) : values.get(key),
      set: (_object, key, value) => { values.set(String(key), String(value)); return true },
      deleteProperty: (_object, key) => { values.delete(String(key)); return true },
      ownKeys: () => Array.from(values.keys()),
      getOwnPropertyDescriptor: (_object, key) => values.has(key)
        ? { value: values.get(key), writable: true, enumerable: true, configurable: true }
        : undefined
    })
  }
  const memoryLocal = createMemoryStorage()
  const memorySession = createMemoryStorage()
  for (const name of ["localStorage", "sessionStorage"]) {
    if (Object.getOwnPropertyDescriptor(globalThis, name)?.configurable === false) {
      throw new Error(`Cannot install document-memory UAT ${name}`)
    }
  }
  // Install both before seeding. Never fall back to a native storage backend.
  Object.defineProperties(globalThis, {
    localStorage: { value: memoryLocal, configurable: true, enumerable: true },
    sessionStorage: { value: memorySession, configurable: true, enumerable: true }
  })
  const config = { serverUrl, authMode: "single-user", apiKey }
  memoryLocal.setItem("tldwConfig", JSON.stringify(config))
  memoryLocal.setItem("isMigrated", "true")
  if (!legacyBootstrap) return
  memoryLocal.setItem("serverUrl", serverUrl)
  memoryLocal.setItem("tldwServerUrl", serverUrl)
  memoryLocal.setItem("tldw-api-host", serverUrl)
  memoryLocal.setItem("authMode", "single-user")
  memoryLocal.setItem("apiKey", apiKey)
  memoryLocal.setItem("__tldw_first_run_complete", "true")
  memoryLocal.setItem("assistant_setup_dismissed", "true")
}

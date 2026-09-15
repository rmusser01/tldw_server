import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/services/__tests__/background-proxy.test.ts"
const probe = `
describe("UAT034 actual completion transport", () => {
  it("rejects a real non-2xx direct fetch and retains raw failure text in error/diagnostics", async () => {
    mocks.runtimeId = null
    const cfg = { serverUrl: "http://127.0.0.1:19999", authMode: "single-user", credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: "http://127.0.0.1:19999", apiKey: "synthetic-test-key" }
    mocks.storageGet.mockImplementation(async (key) => key === "tldwConfig" ? cfg : null)
    const actual = await vi.importActual<typeof import("@/services/tldw/request-core")>("@/services/tldw/request-core")
    mocks.tldwRequest.mockImplementation(actual.tldwRequest)
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ detail: "Provider failure at /Users/private/stack.txt", traceback: "Traceback secret" }), { status: 500, headers: { "content-type": "application/json" } }))
    vi.stubGlobal("fetch", fetchMock)
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    try {
      const { TldwApiClient } = await import("@/services/tldw/TldwApiClient")
      const client = new TldwApiClient()
      const outcome = await client.createChatCompletion({ model: "auto", messages: [{ role: "user", content: "hello" }] }).then(value => ({ value }), error => ({ error }))
      expect(outcome).toHaveProperty("error")
      expect((outcome as any).error.status).toBe(500)
      expect((outcome as any).error.message).toContain("/Users/private/stack.txt")
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect(warn.mock.calls.flat().join(" ")).toContain("/Users/private/stack.txt")
      expect(mocks.storageSet.mock.calls.some(([key, value]) => key === "__tldwLastRequestError" && JSON.stringify(value).includes("/Users/private/stack.txt"))).toBe(true)
    } finally { vi.unstubAllGlobals() }
  })
})
`
export default { ...base, plugins: [{ name: "uat034-non2xx-boundary", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  code = code.replace('createSafeStorage: (options?: { area?: string }) => ({', 'safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }, createSafeStorage: (options?: { area?: string }) => ({')
  return { code: code + probe, map: null }
} }], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target] } }

import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
import fs from "node:fs"
const root = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = root + "/src/services/__tests__/background-proxy.test.ts"
const production = root + "/src/services/background-proxy.ts"
const probe = `
    it.each(["direct", "extension"] as const)("independent cancellation: preserves %s classified text-only aborts after sanitization", async transport => {
      const { bgRequest, warn, fetchMock } = await setupTransport(transport, { detail: "Provider\\nrequest aborted." }, 499)
      try {
        await expect(bgRequest({ path: "/api/v1/chat/completions", method: "POST", body: request }))
          .rejects.toMatchObject({ name: "AbortError", code: "REQUEST_ABORTED", status: 499 })
        expect(warn).not.toHaveBeenCalled()
        expect(mocks.storageSet.mock.calls.some(([key]) => key === "__tldwLastRequestError")).toBe(false)
        expect(fetchMock).toHaveBeenCalledTimes(1)
      } finally { vi.unstubAllGlobals() }
    })
`
export default { ...base, plugins: [{ name: "independent-chat-cancellation", enforce: "pre" as const,
  load(id: string) { if (process.env.UAT036_BASELINE === "1" && id === production) return fs.readFileSync("/private/tmp/uat036-baseline-background-proxy.ts", "utf8") },
  transform(code: string, id: string) {
    if (id !== target) return
    const needle = '    it.each(["direct", "extension"] as const)('
    if (!code.includes(needle)) throw new Error("Cancellation probe insertion point missing")
    return { code: code.replace(needle, probe + "\n" + needle), map: null }
  }
}], test: { ...base.test, setupFiles: [root + "/vitest.setup.ts"], include: [target] } }

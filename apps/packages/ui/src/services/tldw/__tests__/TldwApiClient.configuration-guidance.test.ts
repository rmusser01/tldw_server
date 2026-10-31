import { afterEach, describe, expect, it, vi } from "vitest"
import { TldwApiClient } from "../TldwApiClient"

vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false,
  invalidateCookieSessionConfig: vi.fn()
}))

describe("unconfigured client guidance on the current surface", () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it.each(["http:", "https:", "chrome-extension:", "moz-extension:"])(
    "rejects and warns with accurate Settings guidance on %s", async protocol => {
      vi.stubGlobal("window", { location: { protocol } })
      const client = new TldwApiClient()
      vi.spyOn(client, "getConfig").mockResolvedValue(null)
      const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined)
      const expected = protocol.endsWith("extension:")
        ? "tldw server is not configured. Open Settings → tldw server in the extension and set the server URL and API key."
        : "tldw server is not configured. Open Settings → tldw server and set the server URL and API key."
      await expect(client.ensureConfigForRequest(true)).rejects.toThrow(expected)
      expect(warning).toHaveBeenCalledExactlyOnceWith(expected)
    }
  )

  it("keeps configured unauthenticated discovery available without a warning", async () => {
    const client = new TldwApiClient()
    const config = { serverUrl: "https://configured.test", authMode: "single-user" as const }
    vi.spyOn(client, "getConfig").mockResolvedValue(config)
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    await expect(client.ensureConfigForRequest(false)).resolves.toEqual(config)
    expect(warning).not.toHaveBeenCalled()
  })
})

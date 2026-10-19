import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request, bgStream: vi.fn(), bgUpload: vi.fn() }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: vi.fn(async () => null), set: vi.fn(), remove: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))

import { TldwApiClient, TldwApiClientBase } from "../TldwApiClient"

const options = {
  configSnapshot: { serverUrl: "https://captured-admin.test", authMode: "multi-user" as const, authSource: "cookie-session" as const },
  headers: { "X-TLDW-Expected-User-ID": "42" },
  abortSignal: new AbortController().signal
}

describe("captured Admin query transport", () => {
  beforeEach(() => mocks.request.mockReset())

  it.each([
    ["stats", (client: TldwApiClient) => client.getSystemStats({ ...options, timeoutMs: 10_000 })],
    ["users", (client: TldwApiClient) => client.listAdminUsers({ page: 2 }, options)],
    ["roles", (client: TldwApiClient) => client.listAdminRoles(options)],
    ["permissions", (client: TldwApiClient) => client.listPermissions(options)],
    ["security", (client: TldwApiClient) => client.getSecurityAlertStatus(options)],
    ["backups", (client: TldwApiClient) => client.listBackups(undefined, options)],
    ["llamacpp", (client: TldwApiClient) => client.getLlamacppStatus(options)],
    ["mlx", (client: TldwApiClient) => client.getMlxStatus(options)],
    ["coverage", (client: TldwApiClient) => client.getGovernorCoverage(undefined, options)],
    ["schedules", (client: TldwApiClient) => client.listBackupSchedules(options)],
    ["alerts", (client: TldwApiClient) => client.listAlertRules(options)]
  ] as const)("pins %s to the captured principal and connection", async (name, invoke) => {
    await invoke(new TldwApiClient())
    expect(mocks.request).toHaveBeenCalledOnce()
    const request = mocks.request.mock.calls[0][0]
    expect(request).toMatchObject(options)
    expect(request).not.toHaveProperty("servicePromptConfig")
    if (name === "stats") expect(request.timeoutMs).toBe(10_000)
    if (name === "llamacpp") expect(request.expectedStatuses).toEqual([503])
  })

  it("keeps the base diagnostic implementations captured too", async () => {
    const client = new TldwApiClientBase()
    await client.getSystemStats({ ...options, timeoutMs: 10_000 })
    await client.getLlamacppStatus(options)
    await client.getMlxStatus(options)
    expect(mocks.request).toHaveBeenCalledTimes(3)
    for (const [request] of mocks.request.mock.calls) expect(request).toMatchObject(options)
    expect(mocks.request.mock.calls[0][0].timeoutMs).toBe(10_000)
  })
})

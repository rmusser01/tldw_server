import { beforeEach, describe, expect, it, vi } from "vitest"
import { TldwAuthService } from "../tldw/TldwAuth"
import { buildChatSurfaceScopeKeyFromConfig } from "../chat-surface-scope"
import type { TldwConfig } from "../tldw/TldwApiClient"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(), updateConfig: vi.fn(), bgRequest: vi.fn(),
  hosted: false, token: "", snapshots: [] as TldwConfig[]
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getConfig: mocks.getConfig, updateConfig: mocks.updateConfig } }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => mocks.hosted }))
vi.mock("@/services/splash-events", () => ({ emitSplashAfterLoginSuccess: vi.fn() }))

const tokenFor = (claims: Record<string, unknown>) => `header.${btoa(JSON.stringify({ sub: "2", ...claims }))}.signature`
const login = (auth: TldwAuthService, method: string) => method === "password"
  ? auth.login({ username: "alice", password: "owned-test" })
  : auth.verifyMagicLink("owned-magic-link")

beforeEach(() => {
  vi.clearAllMocks()
  mocks.hosted = false
  mocks.snapshots = []
  mocks.token = tokenFor({ active_org_id: 2, org_ids: [2] })
  let current: TldwConfig = { serverUrl: "http://localhost:8000", authMode: "multi-user", orgId: 99 }
  mocks.getConfig.mockImplementation(async () => current)
  mocks.updateConfig.mockImplementation(async patch => {
    current = { ...current, ...patch }
    mocks.snapshots.push(current)
  })
  mocks.bgRequest.mockImplementation(async ({ path }) => path === "/api/v1/orgs"
    ? { items: [{ id: 2 }] }
    : { access_token: mocks.token, refresh_token: "owned-refresh", token_type: "bearer" })
})

describe.each(["password", "magic-link"])("%s login workspace publication", method => {
  it("publishes the complete workspace before any authenticated draft scope is visible", async () => {
    await login(new TldwAuthService(), method)
    expect(mocks.snapshots.map(config => config.orgId)).toEqual([2])
    const finalScope = buildChatSurfaceScopeKeyFromConfig(mocks.snapshots[0])
    expect(mocks.snapshots.map(buildChatSurfaceScopeKeyFromConfig)).toEqual([finalScope])
    expect(mocks.bgRequest.mock.calls.some(([request]) => request.path === "/api/v1/orgs")).toBe(false)
  })

  it.each([
    [{ active_org_id: 3, org_ids: [2, 3, 9] }, 3],
    [{ org_ids: [2, 9, 3] }, 9],
    [{ active_org_id: 4 }, 4],
    [{ active_org_id: 9007199254740992, org_ids: [0, -2] }, undefined],
    [{ active_org_id: 88, org_ids: [2, 9] }, 9],
    [{ org_ids: [-1, 0, 1.5, "3", 4] }, 4],
    [{}, undefined],
    [{ active_org_id: -1, org_ids: [null, false, "bad"] }, undefined]
  ])("uses returned workspace claims and clears stale prior scope: %j", async (claims, expected) => {
    mocks.token = tokenFor(claims as Record<string, unknown>)
    await login(new TldwAuthService(), method)
    expect(mocks.snapshots).toHaveLength(1)
    expect(mocks.snapshots[0].orgId).toBe(expected)
  })

  it("clears the previous account workspace for unreadable token claims", async () => {
    mocks.token = "legacy-opaque-token"
    await login(new TldwAuthService(), method)
    expect(mocks.snapshots.map(config => config.orgId)).toEqual([undefined])
  })

  it("retains the hosted workspace lookup", async () => {
    mocks.hosted = true
    await login(new TldwAuthService(), method)
    expect(mocks.bgRequest).toHaveBeenCalledWith({ path: "/api/v1/orgs", method: "GET" })
    expect(mocks.snapshots.at(-1)?.orgId).toBe(2)
  })
})

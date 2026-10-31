import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { AuthConnectionAttempt } from "../auth-connection-target"
import type { TldwConfig } from "../TldwApiClient"
const mocks = vi.hoisted(() => ({
  hosted: false,
  initialize: vi.fn(),
  getCurrentUserProfile: vi.fn(),
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  bgRequest: vi.fn(),
  splash: vi.fn(),
}))
vi.mock("../TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    getCurrentUserProfile: mocks.getCurrentUserProfile,
    getConfig: mocks.getConfig,
    updateConfig: mocks.updateConfig,
  },
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/splash-events", () => ({
  emitSplashAfterLoginSuccess: mocks.splash,
}))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => mocks.hosted,
}))
import { TldwAuthService } from "../TldwAuth"
const target = {
  serverUrl: "https://auth.example.test/base",
  authMode: "multi-user" as const,
}
const tokens = {
  access_token: "new-access",
  refresh_token: "new-refresh",
  token_type: "bearer",
}
let saved: TldwConfig
const methods = [
  {
    name: "password",
    call: (auth: TldwAuthService, context: AuthConnectionAttempt) =>
      auth.login({ username: "alice", password: "synthetic" }, context),
    path: "/api/v1/auth/login",
  },
  {
    name: "request",
    call: (auth: TldwAuthService, context: AuthConnectionAttempt) =>
      auth.requestMagicLink("alice@example.test", context),
    path: "/api/v1/auth/magic-link/request",
  },
  {
    name: "verify",
    call: (auth: TldwAuthService, context: AuthConnectionAttempt) =>
      auth.verifyMagicLink("synthetic-token", context),
    path: "/api/v1/auth/magic-link/verify",
  },
]
beforeEach(() => {
  vi.clearAllMocks()
  mocks.hosted = false
  mocks.initialize.mockResolvedValue(undefined)
  mocks.getCurrentUserProfile.mockResolvedValue({})
  saved = { ...target }
  mocks.getConfig.mockImplementation(async () => saved)
  mocks.bgRequest.mockResolvedValue(tokens)
  mocks.updateConfig.mockImplementation(async (update, assertCurrent) => {
    assertCurrent?.(saved)
    saved = { ...saved, ...update }
  })
})
afterEach(() => vi.restoreAllMocks())
it.each(methods)(
  "rejects $name when the expected full target is not saved",
  async ({ call }) => {
    saved = { ...target, serverUrl: "https://auth.example.test/other" }
    await expect(call(new TldwAuthService(), { target })).rejects.toMatchObject(
      { status: 412 },
    )
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.updateConfig).not.toHaveBeenCalled()
  },
)
it.each(methods)(
  "pins the $name endpoint to the authorized base URL",
  async ({ call, path }) => {
    await call(new TldwAuthService(), { target })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({ path: target.serverUrl + path, noAuth: true }),
    )
  },
)
it.each(methods)(
  "cancels $name after an account A→B→A boundary during config read",
  async ({ call }) => {
    let release!: (config: TldwConfig) => void
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => {
        release = resolve
      }),
    )
    const pending = call(new TldwAuthService(), { target })
    const result = expect(pending).rejects.toMatchObject({ status: 412 })
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true },
      }),
    )
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true },
      }),
    )
    release(target)
    await result
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  },
)
it.each(methods)(
  "accepts $name after same-target readiness notifications",
  async ({ call }) => {
    mocks.getConfig.mockImplementationOnce(async () => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false },
        }),
      )
      return saved
    })
    await call(new TldwAuthService(), { target })
    expect(mocks.bgRequest).toHaveBeenCalledOnce()
  },
)
it.each(methods)(
  "rejects late $name results after lifetime cancellation",
  async ({ call }) => {
    const controller = new AbortController()
    mocks.bgRequest.mockImplementation(async () => {
      controller.abort()
      return tokens
    })
    await expect(
      call(new TldwAuthService(), { target, signal: controller.signal }),
    ).rejects.toMatchObject({ status: 412 })
    expect(mocks.updateConfig).not.toHaveBeenCalled()
    expect(mocks.splash).not.toHaveBeenCalled()
  },
)
it.each(methods.filter((method) => method.name !== "request"))(
  "rechecks $name inside the awaited token publication",
  async ({ call }) => {
    mocks.updateConfig.mockImplementation(async (_update, assertCurrent) => {
      await Promise.resolve()
      saved = { ...target, serverUrl: "https://other.example.test" }
      assertCurrent?.(saved)
    })
    await expect(call(new TldwAuthService(), { target })).rejects.toMatchObject(
      { status: 412 },
    )
    expect(mocks.splash).not.toHaveBeenCalled()
  },
)

it.each(methods)(
  "allows a real same-principal token refresh while $name waits",
  async ({ call }) => {
    const jwt = (nonce: number) =>
      `header.${btoa(JSON.stringify({ sub: "42", nonce }))}.signature`
    const previous = { ...target, accessToken: jwt(1) }
    saved = { ...target, accessToken: jwt(2) }
    mocks.getConfig.mockImplementationOnce(async () => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "tldwConfig",
          oldValue: JSON.stringify(previous),
          newValue: JSON.stringify(saved),
        }),
      )
      return saved
    })
    await call(new TldwAuthService(), { target })
    expect(mocks.bgRequest).toHaveBeenCalledOnce()
  },
)

it.each(methods)(
  "rejects $name after genuine storage A→B→A changes while waiting",
  async ({ call }) => {
    mocks.bgRequest.mockImplementationOnce(async () => {
      const foreign = { ...target, serverUrl: "https://other.example.test" }
      for (const [oldValue, newValue] of [
        [target, foreign],
        [foreign, target],
      ]) {
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: "tldwConfig",
            oldValue: JSON.stringify(oldValue),
            newValue: JSON.stringify(newValue),
          }),
        )
      }
      return tokens
    })
    await expect(call(new TldwAuthService(), { target })).rejects.toMatchObject(
      { status: 412 },
    )
    expect(mocks.updateConfig).not.toHaveBeenCalled()
    expect(mocks.splash).not.toHaveBeenCalled()
  },
)

it.each(methods)(
  "preserves the guarded hosted $name BFF route and cookie-only token storage",
  async ({ call, path }) => {
    mocks.hosted = true
    await call(new TldwAuthService(), { target })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: path.replace("/api/v1/auth/", "/api/auth/"),
        noAuth: true,
      }),
    )
    if (path.endsWith("/request"))
      expect(mocks.updateConfig).not.toHaveBeenCalled()
    else
      expect(saved).toMatchObject({
        authMode: "multi-user",
        accessToken: undefined,
        refreshToken: undefined,
      })
  },
)

it.each(methods)(
  "cancels guarded hosted $name after a target boundary while BFF is pending",
  async ({ call }) => {
    mocks.hosted = true
    mocks.bgRequest.mockImplementationOnce(async () => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true },
        }),
      )
      return tokens
    })
    await expect(call(new TldwAuthService(), { target })).rejects.toMatchObject(
      { status: 412 },
    )
    expect(mocks.updateConfig).not.toHaveBeenCalled()
  },
)

it.each([
  { lookup: "org", change: "foreign" },
  { lookup: "profile", change: "foreign" },
  { lookup: "org", change: "ABA" },
  { lookup: "profile", change: "ABA" },
])(
  "does not write late hosted $lookup metadata after $change replaces the login owner",
  async ({ lookup, change }) => {
    mocks.hosted = true
    const original = { ...target }
    const foreign = {
      ...target,
      serverUrl: "https://other.example.test",
      orgId: 700,
    }
    const replace = () => {
      saved = foreign
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true },
        }),
      )
      if (change === "ABA") {
        saved = original
        window.dispatchEvent(
          new CustomEvent("tldw:config-updated", {
            detail: { authorityChanged: true },
          }),
        )
      }
    }
    mocks.bgRequest.mockImplementation(async ({ path }) => {
      if (path === "/api/auth/login") return tokens
      if (lookup === "org") {
        replace()
        return { items: [{ id: 99 }] }
      }
      return { items: [] }
    })
    mocks.getCurrentUserProfile.mockImplementation(async () => {
      replace()
      return { active_org_id: 99 }
    })
    await expect(
      new TldwAuthService().login(
        { username: "alice", password: "synthetic" },
        { target },
      ),
    ).rejects.toMatchObject({ status: 412 })
    expect(mocks.updateConfig).toHaveBeenCalledTimes(1)
    expect(saved).toEqual(change === "ABA" ? original : foreign)
    expect(mocks.splash).not.toHaveBeenCalled()
  },
)

it("allows its own guarded hosted login and organization events", async () => {
  mocks.hosted = true
  let revision = 0
  mocks.updateConfig.mockImplementation(async (update, before, after) => {
    before?.(saved)
    saved = { ...saved, ...update }
    revision += 1
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true },
      }),
    )
    after?.(saved)
  })
  mocks.bgRequest.mockImplementation(async ({ path }) =>
    path === "/api/v1/orgs" ? { items: [{ id: 99 }] } : tokens,
  )
  await new TldwAuthService().login(
    { username: "alice", password: "synthetic" },
    {
      target,
      assertCurrent: () => {
        if (revision)
          throw Object.assign(new Error("caller account changed"), {
            status: 412,
          })
      },
    },
  )
  expect(saved).toMatchObject({
    ...target,
    orgId: 99,
    accessToken: undefined,
    refreshToken: undefined,
  })
  expect(mocks.splash).toHaveBeenCalledOnce()
})

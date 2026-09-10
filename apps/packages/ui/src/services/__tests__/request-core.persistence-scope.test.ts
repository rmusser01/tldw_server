import { beforeEach, describe, expect, it, vi } from "vitest";
import { tldwRequest } from "../tldw/request-core";
import { buildChatSurfaceScopeKeyFromConfig } from "../chat-surface-scope";

const state = vi.hoisted(() => ({ runtimeKey: "" }));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => state.runtimeKey,
}));
const config = (serverUrl = "https://a.test", sub = "alice", exp = 1) => ({
  serverUrl,
  authMode: "multi-user",
  accessToken: `h.${btoa(JSON.stringify({ sub, exp }))}.s`,
  refreshToken: "refresh",
});
const payload = {
  path: "/api/v1/prompt-studio/prompts/create" as const,
  method: "POST",
  capturePersistenceScope: true,
  requirePersistenceScope: true,
  body: {},
};
const response = (status: number) =>
  new Response("{}", {
    status,
    headers: { "content-type": "application/json" },
  });

describe("request dispatch scope capture", () => {
  it.each(["quickstart", "advanced"])(
    "reports the actual %s transport backend instead of the configured URL",
    async (mode) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = mode;
      const initial = config("https://a.test/ignored-prefix");
      const actualBase =
        mode === "quickstart" ? window.location.origin : "https://a.test";
      const fetchFn = vi
        .fn()
        .mockResolvedValueOnce(response(401))
        .mockResolvedValueOnce(response(200));
      const result = await tldwRequest(payload, {
        getConfig: async () => initial,
        fetchFn,
        refreshAuth: async () => {},
      });
      expect(fetchFn).toHaveBeenCalledTimes(2);
      expect(fetchFn.mock.calls[0][0]).toBe(
        mode === "quickstart" ? payload.path : `${actualBase}${payload.path}`,
      );
      expect(result.persistenceScope).toBe(
        buildChatSurfaceScopeKeyFromConfig({
          ...initial,
          serverUrl: actualBase,
        }),
      );
    },
  );
  beforeEach(() => {
    state.runtimeKey = "";
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
  });
  it.each([
    config("https://b.test"),
    config("https://a.test", "bob"),
    { ...config(), orgId: "other" },
  ])(
    "does not retry a captured write with a changed owner %j",
    async (changed) => {
      const initial = config();
      const fetchFn = vi.fn().mockImplementation(async () => response(401));
      const getConfig = vi
        .fn()
        .mockResolvedValueOnce(initial)
        .mockResolvedValue(changed);
      const result = await tldwRequest(payload, {
        getConfig,
        fetchFn,
        refreshAuth: async () => {},
      });
      expect(fetchFn).toHaveBeenCalledTimes(1);
      expect(result).toMatchObject({
        ok: false,
        status: 412,
        persistenceScope: buildChatSurfaceScopeKeyFromConfig(initial),
      });
    },
  );
  it("retains the same owner during token rotation", async () => {
    const initial = config();
    const updated = config("https://a.test", "alice", 2);
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(200));
    const result = await tldwRequest(payload, {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce(initial)
        .mockResolvedValue(updated),
      fetchFn,
      refreshAuth: async () => {},
    });
    expect(result).toMatchObject({
      ok: true,
      persistenceScope: buildChatSurfaceScopeKeyFromConfig(initial),
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(fetchFn.mock.calls[1][1].headers.Authorization).toBe(
      `Bearer ${updated.accessToken}`,
    );
  });
  it("captures the API key override actually dispatched instead of an unused bearer principal", async () => {
    state.runtimeKey = "real-runtime-key";
    const initial = config();
    const fetchFn = vi.fn().mockResolvedValue(response(200));
    const result = await tldwRequest(payload, {
      getConfig: async () => initial,
      fetchFn,
    });
    expect(result).toMatchObject({
      persistenceScope: buildChatSurfaceScopeKeyFromConfig({
        ...initial,
        authMode: "single-user",
        apiKey: state.runtimeKey,
      }),
    });
    expect(fetchFn.mock.calls[0][1].headers["X-API-KEY"]).toBe(
      state.runtimeKey,
    );
  });
  it.each([
    { ...config(), accessToken: "opaque" },
    { ...config(), accessToken: "opaque", userId: "stale-local-user" },
    { ...config(), serverUrl: "" },
  ])(
    "rejects missing trustworthy ownership before dispatch %j",
    async (initial) => {
      const fetchFn = vi.fn().mockResolvedValue(response(200));
      const result = await tldwRequest(payload, {
        getConfig: async () => initial,
        fetchFn,
      });
      expect(fetchFn).not.toHaveBeenCalled();
      expect(result).toMatchObject({
        ok: false,
        persistenceScope: null,
        requestDispatched: false,
      });
    },
  );
  it("does not infer hosted cookie identity from an unused local bearer", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted";
    const fetchFn = vi.fn().mockResolvedValue(response(200));
    const result = await tldwRequest(payload, {
      getConfig: async () => config(),
      fetchFn,
    });
    expect(fetchFn).not.toHaveBeenCalled();
    expect(result).toMatchObject({
      ok: false,
      persistenceScope: null,
      requestDispatched: false,
    });
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => {}),
    remove: vi.fn(async () => {})
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"
const target = "https://analysis.example";
const token = `fixture.${btoa(JSON.stringify({ sub: "7" }))}.signature`;
beforeEach(() => {
  vi.clearAllMocks();
  boundary.runtimeId = null;
  boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? { serverUrl: target, authMode: "multi-user", authSource: "manual", accessToken: token, refreshToken: "fixture-refresh" } : null);
  boundary.fetch.mockImplementation(async () => new Response(JSON.stringify({ media_id: 42, processing: { analysis: "Complete analysis" } }), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", boundary.fetch);
});
afterEach(() => vi.unstubAllGlobals());
describe("B09 analysis captured transport", () => {
  it("dispatches version creation with the captured principal and cancellation", async () => {
    const controller = new AbortController();
    const scope = { config: { serverUrl: target, authMode: "multi-user" as const, authSource: "manual" as const, expectedRefreshToken: "fixture-refresh" }, userId: 7 };
    await bgRequest({ path: "/api/v1/media/42/versions", method: "POST", ...requestScopeFields(scope), abortSignal: controller.signal, body: { content: "Original source", analysis_content: "Complete analysis" } });
    expect(boundary.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = boundary.fetch.mock.calls[0];
    expect(url).toBe(target + "/api/v1/media/42/versions");
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7");
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});

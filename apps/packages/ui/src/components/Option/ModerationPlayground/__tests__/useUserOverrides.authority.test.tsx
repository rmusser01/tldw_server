// @vitest-environment jsdom
import React from "react";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn(), user: 7 }));
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }));
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (v: unknown) => v,
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(),
    remove: vi.fn(),
  }),
}));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false,
}));
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => false }));
import { useUserOverrides } from "../hooks/useUserOverrides";
import { decideModerationReviewItem } from "@/services/moderation";
const config = (user = boundary.user) => ({
  serverUrl: "https://moderation.test",
  authMode: "multi-user" as const,
  accessToken:
    "fixture." + btoa(JSON.stringify({ sub: String(user) })) + ".signature",
});
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
let client: QueryClient;
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
);
beforeEach(() => {
  vi.clearAllMocks();
  boundary.user = 7;
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null,
  );
  boundary.fetch.mockImplementation(async () => json({ overrides: {} }));
  vi.stubGlobal("fetch", boundary.fetch);
});
afterEach(() => {
  cleanup();
  client.clear();
  vi.unstubAllGlobals();
});
describe("moderation mutation continuation ownership", () => {
  it("stops bulk deletion after its owning workspace unmounts", async () => {
    let release!: () => void;
    boundary.fetch.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(json({ status: "deleted" }));
        }),
    );
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    let pending!: Promise<string[]>;
    act(() => {
      pending = view.result.current.bulkDelete(["alpha", "beta"]);
    });
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledOnce());
    view.unmount();
    release();
    await act(async () => {
      await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    });
    expect(
      boundary.fetch.mock.calls.filter(([, init]) => init.method === "DELETE"),
    ).toHaveLength(1);
  });
  it("does not continue a failed bulk request into a replacement account", async () => {
    let release!: () => void;
    boundary.fetch.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(json({ status: "deleted" }));
        }),
    );
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    let pending!: Promise<string[]>;
    act(() => {
      pending = view.result.current.bulkDelete(["alpha", "beta"]);
    });
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledOnce());
    boundary.user = 8;
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    release();
    await act(async () => {
      await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    });
    const writes = boundary.fetch.mock.calls.filter(
      ([, init]) => init.method === "DELETE",
    );
    expect(writes).toHaveLength(1);
  });
  it("retains real same-owner partial-failure continuation", async () => {
    boundary.fetch.mockImplementationOnce(async () =>
      json({ detail: "first refused" }, 409),
    );
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    let failed!: string[];
    await act(async () => {
      failed = await view.result.current.bulkDelete(["alpha", "beta"]);
    });
    expect(failed).toEqual(["alpha"]);
    const writes = boundary.fetch.mock.calls.filter(
      ([, init]) => init.method === "DELETE",
    );
    expect(writes.map(([url]) => new URL(String(url)).pathname)).toEqual([
      "/api/v1/moderation/users/alpha",
      "/api/v1/moderation/users/beta",
    ]);
    expect(
      writes.every(
        ([, init]) =>
          new Headers(init.headers).get("Authorization") ===
          "Bearer " + config(7).accessToken,
      ),
    ).toBe(true);
  });
  it("refuses a principal change while the actual decision transport resolves credentials", async () => {
    let release!: (value: unknown) => void;
    boundary.get.mockImplementation((key: string) =>
      key === "tldwConfig"
        ? new Promise((resolve) => {
            release = resolve;
          })
        : Promise.resolve(null),
    );
    const pending = decideModerationReviewItem("owned-item", {
      action: "approve",
    });
    const rejected = expect(pending).rejects.toBeDefined();
    await waitFor(() => expect(release).toBeTypeOf("function"));
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    release(config(8));
    await rejected;
    expect(boundary.fetch).not.toHaveBeenCalled();
  });
  it("refuses a retained bulk callback after workspace retirement before any request", async () => {
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    const bulk = view.result.current.bulkDelete;
    view.unmount();
    await expect(bulk(["alpha"])).rejects.toMatchObject({ name: "AbortError" });
    expect(boundary.fetch).not.toHaveBeenCalled();
  });
  it("does not refetch after the final delete completes on a retired workspace", async () => {
    let release!: () => void;
    boundary.fetch.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(json({ status: "deleted" }));
        }),
    );
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    let pending!: Promise<string[]>;
    act(() => {
      pending = view.result.current.bulkDelete(["alpha"]);
    });
    await waitFor(() => expect(boundary.fetch).toHaveBeenCalledOnce());
    view.unmount();
    release();
    await act(async () => {
      await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    });
    expect(boundary.fetch).toHaveBeenCalledOnce();
  });
  it("rejects retirement during the final refetch before reporting completion", async () => {
    let finish!: () => void;
    boundary.fetch.mockImplementation(async (_url, init) => {
      if (init.method === "DELETE") return json({ status: "deleted" });
      return new Promise(resolve => { finish = () => resolve(json({ overrides: {} })); });
    });
    const view = renderHook(() => useUserOverrides(null), { wrapper });
    let pending!: Promise<string[]>;
    act(() => { pending = view.result.current.bulkDelete(["alpha"]); });
    await waitFor(() => expect(finish).toBeTypeOf("function"));
    view.unmount();
    finish();
    await act(async () => { await expect(pending).rejects.toMatchObject({ name: "AbortError" }); });
    expect(boundary.fetch.mock.calls.filter(([, init]) => init.method === "DELETE")).toHaveLength(1);
  });

});

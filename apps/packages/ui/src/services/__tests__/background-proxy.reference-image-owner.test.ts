import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null,
}));
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId;
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args),
    },
  },
}));
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value,
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined),
  }),
}));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false,
}));

import { TldwApiClient } from "@/services/tldw/TldwApiClient";
import { bgRequest } from "../background-proxy";
import { requestScopeFields } from "../tldw/domains/service-prompts";

const config = (user = 7, serverUrl = "https://images.example") => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`,
});
const requestScope = {
  config: {
    serverUrl: "https://images.example",
    authMode: "multi-user" as const,
  },
  userId: 7,
};
const response = {
  items: [
    {
      file_id: "17",
      title: "Owned reference.png",
      mime_type: "image/png",
      width: 640,
      height: 480,
      created_at: "2026-10-07T12:00:00Z",
    },
  ],
};
const jsonResponse = () =>
  new Response(JSON.stringify(response), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
const list = (signal = new AbortController().signal) => {
  const options = { requestScope, signal };
  return new TldwApiClient().listReferenceImageCandidates(options);
};

beforeEach(() => {
  vi.clearAllMocks();
  boundary.runtimeId = null;
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null,
  );
  boundary.fetch.mockImplementation(async () => jsonResponse());
  boundary.sendMessage.mockResolvedValue({
    ok: true,
    status: 200,
    data: response,
  });
  vi.stubGlobal("fetch", boundary.fetch);
});
afterEach(() => vi.unstubAllGlobals());

describe("managed reference-image catalog through its real media method and transport", () => {
  it("dispatches and normalizes the catalog for the captured principal", async () => {
    await expect(list()).resolves.toEqual({
      items: [
        {
          file_id: 17,
          title: "Owned reference.png",
          mime_type: "image/png",
          width: 640,
          height: 480,
          created_at: "2026-10-07T12:00:00Z",
        },
      ],
    });
    expect(boundary.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = boundary.fetch.mock.calls[0];
    expect(String(url)).toBe(
      "https://images.example/api/v1/files/reference-images",
    );
    expect(init.method).toBe("GET");
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7");
    expect(new Headers(init.headers).get("Authorization")).toBe(
      `Bearer ${config().accessToken}`,
    );
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(init.body).toBeUndefined();
  });

  it("preserves the optional unscoped caller contract", async () => {
    await expect(
      new TldwApiClient().listReferenceImageCandidates({
        signal: new AbortController().signal,
      }),
    ).resolves.toMatchObject({ items: [{ file_id: 17 }] });
    expect(boundary.fetch).toHaveBeenCalledTimes(1);
    expect(
      new Headers(boundary.fetch.mock.calls[0][1].headers).get(
        "X-TLDW-Expected-User-ID",
      ),
    ).toBeNull();
  });

  it("forwards captured scope and principal through extension messaging", async () => {
    boundary.runtimeId = "extension-fixture";
    await expect(list()).resolves.toMatchObject({ items: [{ file_id: 17 }] });
    expect(boundary.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:request",
        payload: expect.objectContaining({
          path: "/api/v1/files/reference-images",
          method: "GET",
          servicePromptConfig: {
            serverUrl: "https://images.example",
            authMode: "multi-user",
            expectedUserId: 7,
          },
          headers: { "X-TLDW-Expected-User-ID": "7" },
          requestId: expect.any(String),
        }),
      }),
    );
    expect(boundary.fetch).not.toHaveBeenCalled();
  });

  it.each(["principal", "server"])(
    "refuses a changed %s during credential loading without HTTP dispatch",
    async (change) => {
      let release!: (value: unknown) => void;
      const pendingConfig = new Promise((resolve) => {
        release = resolve;
      });
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? pendingConfig : null,
      );
      const pending = list();
      const rejected = expect(pending).rejects.toMatchObject({ status: 412 });
      release(
        change === "principal" ? config(8) : config(7, "https://other.example"),
      );
      await rejected;
      expect(boundary.fetch).not.toHaveBeenCalled();
    },
  );

  it("rechecks captured ownership at the request-core configuration boundary", async () => {
    let reads = 0;
    boundary.get.mockImplementation(async (key: string) => {
      if (key !== "tldwConfig") return null;
      reads += 1;
      return config(reads === 1 ? 7 : 8);
    });
    await expect(list()).rejects.toMatchObject({ status: 412 });
    expect(boundary.fetch).not.toHaveBeenCalled();
  });

  it("preserves cancellation before the catalog fetch begins", async () => {
    boundary.fetch.mockImplementation(
      async (_url: unknown, init: RequestInit) => {
        if (init.signal?.aborted)
          throw new DOMException("Canceled", "AbortError");
        return jsonResponse();
      },
    );
    const controller = new AbortController();
    controller.abort();
    await expect(list(controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(boundary.fetch).not.toHaveBeenCalled();
    expect(boundary.sendMessage).not.toHaveBeenCalled();
  });

  it("passes cancellation to an already dispatched catalog request", async () => {
    boundary.fetch.mockImplementation(
      (_url: unknown, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Canceled", "AbortError")),
            { once: true },
          );
        }),
    );
    const controller = new AbortController();
    const pending = list(controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1));
    controller.abort();
    await rejected;
    expect(boundary.fetch.mock.calls[0][1].signal.aborted).toBe(true);
  });

  it("admits only the canonical scoped catalog GET through the real proxy", async () => {
    await expect(
      bgRequest({
        ...requestScopeFields(requestScope),
        path: "/api/v1/files/reference-images" as never,
        method: "GET",
      }),
    ).resolves.toEqual(response);
    expect(boundary.fetch).toHaveBeenCalledTimes(1);
  });

  it.each(["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])(
    "refuses scoped catalog %s before transport dispatch",
    async (method) => {
      await expect(
        bgRequest({
          ...requestScopeFields(requestScope),
          path: "/api/v1/files/reference-images" as never,
          method: method as never,
        }),
      ).rejects.toThrow(/Service Prompt config/);
      expect(boundary.fetch).not.toHaveBeenCalled();
      expect(boundary.sendMessage).not.toHaveBeenCalled();
    },
  );

  it.each([
    "/api/v1/files",
    "/api/v1/files/reference-images/",
    "/api/v1/files/reference-images/17",
    "/api/v1/files/reference-images-other",
    "/api/v1/files//reference-images",
    "/api/v1/files/../files/reference-images",
    "/api/v1/files/%2e%2e/reference-images",
    "/api/v1/files%2freference-images",
    "/api/v1/files\\reference-images",
    "/api/v1/files/reference-images%zz",
    "https://other.example/api/v1/files/reference-images",
  ])("refuses malformed or expanded scoped catalog path %s", async (path) => {
    await expect(
      bgRequest({
        ...requestScopeFields(requestScope),
        path: path as never,
        method: "GET",
      }),
    ).rejects.toThrow(/Service Prompt config/);
    expect(boundary.fetch).not.toHaveBeenCalled();
    expect(boundary.sendMessage).not.toHaveBeenCalled();
  });
});

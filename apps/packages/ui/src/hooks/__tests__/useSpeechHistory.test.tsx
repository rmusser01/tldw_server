// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { Storage } from "../../../../../tldw-frontend/extension/shims/plasmo-storage";
import { useSpeechHistory } from "../useSpeechHistory";

const scope = vi.hoisted(() => ({
  user: "alice" as string | null,
  server: "http://server-a",
}));
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock(
  "@plasmohq/storage/hook",
  async () =>
    import("../../../../../tldw-frontend/extension/shims/plasmo-storage-hook"),
);
vi.mock("wxt/browser", () => ({
  browser: {
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } },
  },
}));
vi.mock("@/services/service-prompts", () => ({
  isServicePromptScopeUnresolvedError: () => false,
  resolveServicePromptScope: async () => {
    if (!scope.user)
      throw Object.assign(new Error("No active principal"), { status: 401 });
    return {
      config: {
        serverUrl: scope.server,
        authMode: "multi-user",
        authSource: "cookie-session",
      },
      userId: scope.user,
    };
  },
}));
const boundary = async (user: string | null, server = "http://server-a") => {
  await act(async () => {
    scope.user = user;
    scope.server = server;
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"));
  });
};
beforeEach(() => {
  localStorage.clear();
  scope.user = "alice";
  scope.server = "http://server-a";
});

it("B04 history hides another account's text and preserves each owner's history on return", async () => {
  const changed = vi.fn();
  const hook = renderHook(() => useSpeechHistory<string>(changed));
  await act(async () => {});
  await act(async () => {
    await hook.result.current[1](["Alice private text"]);
  });
  await waitFor(() =>
    expect(hook.result.current[0]).toEqual(["Alice private text"]),
  );
  const aliceWriter = hook.result.current[1];
  await boundary("bob");
  expect(hook.result.current[0]).toEqual([]);
  await act(async () => {
    await hook.result.current[1](["Bob private text"]);
  });
  await act(async () => {
    await aliceWriter(["Late cancelled Alice text"]);
  });
  expect(hook.result.current[0]).toEqual(["Bob private text"]);
  await boundary("alice");
  await waitFor(() =>
    expect(hook.result.current[0]).toEqual(["Alice private text"]),
  );
  expect(changed).toHaveBeenCalled();
  hook.unmount();
  const again = renderHook(() => useSpeechHistory<string>(vi.fn()));
  await waitFor(() =>
    expect(again.result.current[0]).toEqual(["Alice private text"]),
  );
});

it("B04 history keeps server identities separate and refuses unowned legacy rows", async () => {
  const storage = new Storage();
  await storage.set("speechPlaygroundHistory", ["Unowned legacy text"]);
  const hook = renderHook(() => useSpeechHistory<string>(vi.fn()));
  await act(async () => {});
  expect(hook.result.current[0]).toEqual([]);
  await act(async () => {
    await hook.result.current[1](["Server A text"]);
  });
  await boundary("alice", "http://server-b");
  expect(hook.result.current[0]).toEqual([]);
  await boundary("alice");
  await waitFor(() =>
    expect(hook.result.current[0]).toEqual(["Server A text"]),
  );
  expect(await storage.get("speechPlaygroundHistory")).toEqual([
    "Unowned legacy text",
  ]);
});

it("B04 history fails closed across logout and ignores a late owner writer", async () => {
  const hook = renderHook(() => useSpeechHistory<string>(vi.fn()));
  await act(async () => {});
  await act(async () => {
    await hook.result.current[1](["Private text"]);
  });
  const writer = hook.result.current[1];
  await boundary(null);
  expect(hook.result.current[0]).toEqual([]);
  await act(async () => {
    await writer(["Stale text"]);
  });
  expect(hook.result.current[0]).toEqual([]);
});

it("B04 history rejects malformed owned arrays without adopting them", async () => {
  const hook = renderHook(() => useSpeechHistory<string>(vi.fn()));
  await act(async () => {});
  await act(async () => {
    await hook.result.current[1](["Owned text"]);
  });
  const key = Object.keys(localStorage).find((key) =>
    key.startsWith("speechPlaygroundHistory:"),
  )!;
  const storage = new Storage();
  const previous = await storage.get<{ ownerKey: string }>(key);
  await act(async () => {
    await storage.set(key, { ...previous, items: "malformed" });
  });
  expect(hook.result.current[0]).toEqual([]);
  await act(async () => {
    await hook.result.current[1]((items) => [...items, "New text"]);
  });
  expect(hook.result.current[0]).toEqual(["New text"]);
});

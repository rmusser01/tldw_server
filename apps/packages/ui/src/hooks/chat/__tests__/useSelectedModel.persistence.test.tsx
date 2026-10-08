// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Storage } from "@web/extension/shims/plasmo-storage";
import { useStorage } from "@web/extension/shims/plasmo-storage-hook";
import { useStoreMessageOption } from "@/store/option";
import { useSelectedModel } from "../useSelectedModel";

// Two deployed model-hook owners, the option store, and the WebUI storage hook
// and Storage implementation are real. Only the localStorage write failure or
// the existing Storage.get wait is controlled. No native extension/browser I/O.
const oldModel = "anthropic:old-model";
const nextModel = "openai:new-model";
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const mount = () =>
  renderHook(() => ({
    first: useSelectedModel(),
    second: useSelectedModel(),
    observed: useStorage<string | null>("selectedModel", null),
  }));
const ready = async (view: ReturnType<typeof mount>) => {
  await waitFor(() =>
    expect([
      view.result.current.first.selectedModelIsLoading,
      view.result.current.second.selectedModelIsLoading,
      view.result.current.observed[2].isLoading,
    ]).toEqual([false, false, false]),
  );
};
const recordWrites = (fail?: () => boolean) => {
  const backend = window.localStorage;
  const prototype = Object.getPrototypeOf(backend) as globalThis.Storage;
  const original = prototype.setItem;
  const writes: string[] = [];
  const error = new Error("The model preference backend is unavailable");
  vi.spyOn(prototype, "setItem").mockImplementation(function (
    this: globalThis.Storage,
    key: string,
    value: string,
  ) {
    if (key === "selectedModel") {
      writes.push(value);
      if (fail?.()) throw error;
    }
    return original.call(this, key, value);
  });
  return { writes, error };
};
beforeEach(() => {
  window.localStorage.setItem("selectedModel", JSON.stringify(oldModel));
  useStoreMessageOption.getState().setSelectedModel(oldModel);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.localStorage.removeItem("selectedModel");
});

describe("explicit model persistence with multiple actual hook owners", () => {
  it("returns a failed explicit write without a sibling retry and permits a deliberate retry", async () => {
    const view = mount();
    await ready(view);
    let failed = true;
    const { writes, error } = recordWrites(() => failed);
    let rejected: unknown;
    await act(async () => {
      await view.result.current.first
        .setSelectedModel(nextModel)
        .catch((failure: unknown) => {
          rejected = failure;
        });
    });
    expect(rejected).toBe(error);
    expect(writes).toEqual([JSON.stringify(nextModel)]);
    expect(useStoreMessageOption.getState().selectedModel).toBe(nextModel);
    expect(view.result.current.second.selectedModel).toBe(nextModel);
    expect(window.localStorage.getItem("selectedModel")).toBe(
      JSON.stringify(oldModel),
    );
    failed = false;
    await act(async () => {
      await view.result.current.first.setSelectedModel(nextModel);
    });
    expect(writes).toEqual([
      JSON.stringify(nextModel),
      JSON.stringify(nextModel),
    ]);
    expect(window.localStorage.getItem("selectedModel")).toBe(
      JSON.stringify(nextModel),
    );
    expect(view.result.current.observed[0]).toBe(nextModel);
  });
  it("retries a failed device write without publishing another model selection", async () => {
    const view = mount();
    await ready(view);
    let failed = true;
    const { writes } = recordWrites(() => failed);
    await act(async () => { await view.result.current.first.setSelectedModel(nextModel).catch(() => undefined); });
    const revision = useStoreMessageOption.getState().selectedModelRevision;
    failed = false;
    await act(async () => { await view.result.current.first.retrySelectedModelSave(); });
    expect({ revision: useStoreMessageOption.getState().selectedModelRevision, writes, stored: window.localStorage.getItem("selectedModel") })
      .toEqual({ revision, writes: [JSON.stringify(nextModel), JSON.stringify(nextModel)], stored: JSON.stringify(nextModel) });
  });

  it("does not adopt a reentrant same-model selection's revision for a hidden retry", async () => {
    const view = mount();
    await ready(view);
    let attempts = 0;
    const { writes, error } = recordWrites(() => ++attempts === 1);
    let armed = true;
    let newer!: Promise<void>;
    const unsubscribe = useStoreMessageOption.subscribe(state => {
      if (armed && state.selectedModel === nextModel) {
        armed = false;
        newer = view.result.current.second.setSelectedModel(nextModel);
      }
    });
    try {
      let outcomes!: PromiseSettledResult<void>[];
      await act(async () => {
        const older = view.result.current.first.setSelectedModel(nextModel);
        outcomes = await Promise.allSettled([older, newer]);
      });
      expect({ writes, newer: outcomes[1], stored: window.localStorage.getItem("selectedModel") })
        .toEqual({ writes: [JSON.stringify(nextModel)], newer: { status: "rejected", reason: error }, stored: JSON.stringify(oldModel) });
    } finally { unsubscribe(); }
  });

  it("keeps the newer model durable when an older storage preparation resolves last", async () => {
    const view = mount();
    await ready(view);
    const held = deferred<string>();
    const originalGet = Storage.prototype.get;
    let reads = 0;
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (this: Storage, key: string) {
      if (key === "selectedModel" && ++reads === 1) return held.promise as Promise<never>;
      return originalGet.call(this, key) as Promise<never>;
    });
    let first!: Promise<void>;
    let newer!: Promise<void>;
    act(() => { first = view.result.current.first.setSelectedModel(nextModel); });
    await waitFor(() => expect(reads).toBe(1));
    act(() => { newer = view.result.current.second.setSelectedModel("openai:latest-model"); });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      held.resolve(oldModel);
      await Promise.all([first, newer]);
    });
    expect({ active: useStoreMessageOption.getState().selectedModel, stored: window.localStorage.getItem("selectedModel") })
      .toEqual({ active: "openai:latest-model", stored: JSON.stringify("openai:latest-model") });
  });

  it("does not start a sibling write while the explicit storage preparation is held", async () => {
    const view = mount();
    await ready(view);
    const { writes } = recordWrites();
    const held = deferred<string>();
    const originalGet = Storage.prototype.get;
    let heldReads = 0;
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (
      this: Storage,
      key: string,
    ) {
      if (key === "selectedModel") {
        heldReads += 1;
        return held.promise as Promise<never>;
      }
      return originalGet.call(this, key) as Promise<never>;
    });
    let completion!: Promise<void>;
    act(() => {
      completion = view.result.current.first.setSelectedModel(nextModel);
    });
    await waitFor(() => expect(heldReads).toBeGreaterThan(0));
    const readsBeforeRelease = heldReads;
    expect(writes).toEqual([]);
    expect(useStoreMessageOption.getState().selectedModel).toBe(nextModel);
    await act(async () => {
      held.resolve(oldModel);
      await completion;
    });
    expect(readsBeforeRelease).toBe(1);
    expect(writes).toEqual([JSON.stringify(nextModel)]);
    expect(view.result.current.observed[0]).toBe(nextModel);
  });
  it("preserves store-only hydration into storage across two readers", async () => {
    const view = mount();
    await ready(view);
    act(() => useStoreMessageOption.getState().setSelectedModel(nextModel));
    await waitFor(() =>
      expect(window.localStorage.getItem("selectedModel")).toBe(
        JSON.stringify(nextModel),
      ),
    );
    expect(view.result.current.first.selectedModel).toBe(nextModel);
    expect(view.result.current.second.selectedModel).toBe(nextModel);
    expect(view.result.current.observed[0]).toBe(nextModel);
  });

  it("hydrates a saved choice into two initially empty model owners", async () => {
    useStoreMessageOption.getState().setSelectedModel(null);
    const view = mount();
    await ready(view);
    expect(useStoreMessageOption.getState().selectedModel).toBe(oldModel);
    expect(view.result.current.first.selectedModel).toBe(oldModel);
    expect(view.result.current.second.selectedModel).toBe(oldModel);
    expect(view.result.current.observed[0]).toBe(oldModel);
  });
  it("delivers a changed external storage value through actual cross-instance watches to empty owners", async () => {
    window.localStorage.removeItem("selectedModel");
    useStoreMessageOption.getState().setSelectedModel(null);
    const view = mount();
    await ready(view);
    expect(view.result.current.first.selectedModel).toBeNull();
    expect(view.result.current.second.selectedModel).toBeNull();
    await act(async () => {
      await new Storage().set("selectedModel", nextModel);
    });
    expect(view.result.current.observed[0]).toBe(nextModel);
    expect(view.result.current.first.selectedModel).toBe(nextModel);
    expect(view.result.current.second.selectedModel).toBe(nextModel);
    expect(useStoreMessageOption.getState().selectedModel).toBe(nextModel);
  });
  it("permits an explicit same-value store-only retry after its failed background persistence", async () => {
    const view = mount();
    await ready(view);
    let failed = true;
    const { writes } = recordWrites(() => failed);
    await act(async () => {
      useStoreMessageOption.getState().setSelectedModel(nextModel);
      await Promise.resolve();
    });
    expect(writes).toEqual([JSON.stringify(nextModel)]);
    expect(window.localStorage.getItem("selectedModel")).toBe(
      JSON.stringify(oldModel),
    );
    failed = false;
    act(() => useStoreMessageOption.getState().setSelectedModel(nextModel));
    await waitFor(() =>
      expect(window.localStorage.getItem("selectedModel")).toBe(
        JSON.stringify(nextModel),
      ),
    );
    expect(writes).toEqual([
      JSON.stringify(nextModel),
      JSON.stringify(nextModel),
    ]);
    expect(view.result.current.observed[0]).toBe(nextModel);
    expect(view.result.current.first.selectedModel).toBe(nextModel);
    expect(view.result.current.second.selectedModel).toBe(nextModel);
  });
  it("keeps an explicit null across two owners while old backing storage is still visible", async () => {
    const view = mount();
    await ready(view);
    const held = deferred<string>();
    const originalGet = Storage.prototype.get;
    vi.spyOn(Storage.prototype, "get").mockImplementation(function (
      this: Storage,
      key: string,
    ) {
      return key === "selectedModel"
        ? (held.promise as Promise<never>)
        : (originalGet.call(this, key) as Promise<never>);
    });
    let completion!: Promise<void>;
    act(() => {
      completion = view.result.current.first.setSelectedModel(null);
    });
    expect(window.localStorage.getItem("selectedModel")).toBe(
      JSON.stringify(oldModel),
    );
    expect(view.result.current.first.selectedModel).toBeNull();
    expect(view.result.current.second.selectedModel).toBeNull();
    expect(useStoreMessageOption.getState().selectedModel).toBeNull();
    await act(async () => {
      held.resolve(oldModel);
      await completion;
    });
    expect(window.localStorage.getItem("selectedModel")).toBe("null");
    expect(view.result.current.first.selectedModel).toBeNull();
    expect(view.result.current.second.selectedModel).toBeNull();
    expect(view.result.current.observed[0]).toBeNull();
  });
  it("retains the handled store choice after a changed external watch without a hidden rewrite", async () => {
    const view = mount();
    await ready(view);
    const { writes } = recordWrites();
    await act(async () => {
      await view.result.current.first.setSelectedModel(nextModel);
    });
    const externalModel = "anthropic:another-window-model";
    await act(async () => {
      await new Storage().set("selectedModel", externalModel);
    });
    expect(writes).toEqual([
      JSON.stringify(nextModel),
      JSON.stringify(externalModel),
    ]);
    expect(window.localStorage.getItem("selectedModel")).toBe(
      JSON.stringify(externalModel),
    );
    expect(view.result.current.observed[0]).toBe(externalModel);
    expect(useStoreMessageOption.getState().selectedModel).toBe(nextModel);
    expect(view.result.current.first.selectedModel).toBe(nextModel);
    expect(view.result.current.second.selectedModel).toBe(nextModel);
  });
});

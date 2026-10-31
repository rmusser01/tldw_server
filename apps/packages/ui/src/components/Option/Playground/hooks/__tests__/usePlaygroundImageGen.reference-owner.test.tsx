// @vitest-environment jsdom
import React from "react";
import {
  onlineManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner";
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope";
import type { ServicePromptSnapshot } from "@/services/service-prompts";
import type { ReferenceImageCandidate } from "@/services/tldw/TldwApiClient";
import { usePlaygroundImageGen } from "../usePlaygroundImageGen";

// QueryClient and the image hook are real. Account/source predicates, snapshot
// resolution and reference-image list I/O are declared boundary doubles.
const io = vi.hoisted(() => ({ list: vi.fn(), snapshot: vi.fn() }));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listReferenceImageCandidates: (...args: unknown[]) => io.list(...args),
  },
}));
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => io.snapshot(...args),
}));
vi.mock("@/services/image-generation", async (original) => ({
  ...(await original<typeof import("@/services/image-generation")>()),
  getImageBackendConfigs: async () => ({}),
}));

const candidate = (title: string, fileId = 17): ReferenceImageCandidate => ({
  file_id: fileId,
  title,
  mime_type: "image/png",
  created_at: "2026-10-07T00:00:00Z",
});
const createOwner = (userId: string) => {
  const scope = {
    config: {
      serverUrl: "https://references.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
    },
    userId,
  };
  return { scope, key: chatDraftOwnerKey(scope) };
};
type Owner = ReturnType<typeof createOwner>;
let owner: Owner;
let source: object;
const leases: Array<ReturnType<typeof snapshotFor>> = [];
const clients: QueryClient[] = [];
const snapshotFor = (captured: Owner) => {
  const lease = {
    scopeKey: buildChatSurfaceScopeKeyFromConfig(captured.scope.config, {
      userId: captured.scope.userId,
    }),
    requestScope: captured.scope,
    capability: "unchecked" as const,
    definitions: {},
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: vi.fn(),
  } satisfies ServicePromptSnapshot;
  leases.push(lease);
  return lease;
};
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
// These two outputs are the planned public recovery contract. The intersection
// lets the failure-first suite run before production introduces the properties.
type CatalogResult = ReturnType<typeof usePlaygroundImageGen> & {
  referenceImageCandidatesFailed: boolean;
  retryReferenceImageCandidates: () => Promise<unknown> | void;
};
const mount = () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      },
    },
  });
  clients.push(client);
  const sendMessage = vi.fn(async () => ({ status: "submitted" as const }));
  const notificationError = vi.fn();
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const view = renderHook(
    () => {
      const renderedOwner = owner;
      const renderedSource = source;
      const isDraftOwnerCurrent = React.useCallback(
        () => owner === renderedOwner,
        [renderedOwner],
      );
      const captureCompletionGuard = React.useCallback(
        () => () => owner === renderedOwner && source === renderedSource,
        [renderedOwner, renderedSource],
      );
      return usePlaygroundImageGen({
        imageBackendDefaultTrimmed: "comfyui",
        imageBackendOptions: [{ value: "comfyui", label: "ComfyUI" }],
        imageEventSyncChatMode: "off",
        imageEventSyncGlobalDefault: "off",
        updateChatSettings: vi.fn(),
        setImageEventSyncGlobalDefault: vi.fn(),
        messages: [],
        selectedCharacterName: null,
        selectedModel: "chat-model",
        currentApiProvider: "custom",
        formMessage: "",
        sendMessage,
        textAreaFocus: vi.fn(),
        notificationApi: { error: notificationError },
        t: (_key: string, fallback?: string | { defaultValue?: string }) =>
          typeof fallback === "string"
            ? fallback
            : (fallback?.defaultValue ?? _key),
        setToolsPopoverOpen: vi.fn(),
        draftOwnerKey: renderedOwner.key,
        isDraftOwnerCurrent,
        captureCompletionGuard,
      }) as CatalogResult;
    },
    { wrapper },
  );
  act(() => {
    view.result.current.setImageGeneratePrompt("Keep this exact image prompt");
    view.result.current.openImageGenerateModal();
  });
  return { ...view, client, sendMessage, notificationError };
};
const expectCatalog = async (
  view: ReturnType<typeof mount>,
  titles: string[],
) => {
  await waitFor(() =>
    expect(
      view.result.current.referenceImageCandidates.map((item) => item.title),
    ).toEqual(titles),
  );
};

beforeEach(() => {
  io.list.mockReset();
  io.snapshot.mockReset();
  leases.length = 0;
  owner = createOwner("alice");
  source = {};
  io.snapshot.mockImplementation(async () => snapshotFor(owner));
  io.list.mockImplementation(async () => ({
    items: [candidate(`${owner.scope.userId} image`)],
  }));
});
afterEach(() => {
  cleanup();
  onlineManager.setOnline(true);
  for (const client of clients.splice(0)) client.clear();
});

describe("managed reference image catalog and selection ownership", () => {
  it("does not expose a warm Alice catalog or retained file id to Bob", async () => {
    const view = mount();
    await expectCatalog(view, ["alice image"]);
    act(() => {
      view.result.current.setImageGenerateReferenceFileId(17);
    });
    act(() => {
      owner = createOwner("bob");
      view.rerender();
    });
    expect(view.result.current.referenceImageCandidates).toEqual([]);
    expect(view.result.current.imageGenerateReferenceFileId).toBeUndefined();
    await expectCatalog(view, ["bob image"]);
    expect(io.list).toHaveBeenCalledTimes(2);
  });

  it("retires a held Alice list through Bob and an Alice return without adopting the old response", async () => {
    const oldRead = deferred<{ items: ReferenceImageCandidate[] }>();
    io.list.mockImplementationOnce(() => oldRead.promise);
    const view = mount();
    await waitFor(() => expect(io.list).toHaveBeenCalledTimes(1));
    act(() => {
      owner = createOwner("bob");
      view.rerender();
    });
    await expectCatalog(view, ["bob image"]);
    act(() => {
      owner = createOwner("alice");
      view.rerender();
    });
    await expectCatalog(view, ["alice image"]);
    await act(async () => {
      oldRead.resolve({ items: [candidate("retired Alice image")] });
      await oldRead.promise;
    });
    expect(
      view.result.current.referenceImageCandidates.map((item) => item.title),
    ).toEqual(["alice image"]);
    expect(leases.every((lease) => lease.release.mock.calls.length === 1)).toBe(
      true,
    );
    expect(io.list.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        requestScope: owner.scope,
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("does not dispatch from a held obsolete snapshot after the account changes", async () => {
    const oldSnapshot = deferred<ServicePromptSnapshot>();
    const alice = owner;
    io.snapshot.mockImplementationOnce(() => oldSnapshot.promise);
    const view = mount();
    await waitFor(() => expect(io.snapshot).toHaveBeenCalledTimes(1));
    act(() => {
      owner = createOwner("bob");
      view.rerender();
    });
    await expectCatalog(view, ["bob image"]);
    const lease = snapshotFor(alice);
    await act(async () => {
      oldSnapshot.resolve(lease);
      await oldSnapshot.promise;
    });
    expect(io.list).toHaveBeenCalledTimes(1);
    expect(lease.release).toHaveBeenCalledTimes(1);
    expect(
      view.result.current.referenceImageCandidates.map((item) => item.title),
    ).toEqual(["bob image"]);
  });

  it("starts one fresh read when the same key returns under a new owner incarnation in one render", async () => {
    const view = mount();
    await expectCatalog(view, ["alice image"]);
    const oldKey = owner.key;
    io.list.mockResolvedValueOnce({ items: [candidate("fresh Alice image")] });
    act(() => {
      // The real owner hook synchronously retires its current-owner ref, then
      // can publish a newly verified same-account object before React commits.
      owner = createOwner("between-account-checks");
      owner = createOwner("alice");
      view.rerender();
    });
    expect(owner.key).toBe(oldKey);
    await expectCatalog(view, ["fresh Alice image"]);
    expect(io.list).toHaveBeenCalledTimes(2);
    expect(io.snapshot).toHaveBeenCalledTimes(2);
  });

  it("starts a fresh same-key read when a pending owner retires before any catalog data exists", async () => {
    const retiredRead = deferred<{ items: ReferenceImageCandidate[] }>();
    io.list.mockImplementationOnce(() => retiredRead.promise);
    const view = mount();
    await waitFor(() => expect(io.list).toHaveBeenCalledTimes(1));
    const oldKey = owner.key;
    io.list.mockResolvedValueOnce({
      items: [candidate("fresh pending Alice image")],
    });
    act(() => {
      owner = createOwner("alice");
      view.rerender();
    });
    expect(owner.key).toBe(oldKey);
    await expectCatalog(view, ["fresh pending Alice image"]);
    await act(async () => {
      retiredRead.resolve({
        items: [candidate("retired pending Alice image")],
      });
      await retiredRead.promise;
    });
    expect(
      view.result.current.referenceImageCandidates.map((item) => item.title),
    ).toEqual(["fresh pending Alice image"]);
    expect(io.list).toHaveBeenCalledTimes(2);
  });

  it.each(["modal", "source"] as const)(
    "rejects an old %s selection callback without overwriting or submitting the obsolete choice",
    async (retirement) => {
      io.list.mockResolvedValue({
        items: [candidate("first image", 17), candidate("second image", 23)],
      });
      const view = mount();
      await expectCatalog(view, ["first image", "second image"]);
      act(() => {
        view.result.current.setImageGenerateReferenceFileId(17);
      });
      const oldSelection = view.result.current.setImageGenerateReferenceFileId;
      act(() => {
        if (retirement === "modal") {
          view.result.current.closeImageGenerateModal();
          view.result.current.openImageGenerateModal();
        } else {
          source = {};
          view.rerender();
        }
      });
      expect(view.result.current.imageGenerateReferenceFileId).toBeUndefined();
      await expectCatalog(view, ["first image", "second image"]);
      act(() => {
        view.result.current.setImageGenerateReferenceFileId(23);
        oldSelection(17);
      });
      expect(view.result.current.imageGenerateReferenceFileId).toBe(23);
      await act(async () => {
        await view.result.current.submitImageGenerateModal();
      });
      expect(view.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          imageGenerationRequest: expect.objectContaining({
            referenceFileId: 23,
          }),
        }),
      );
    },
  );

  it("allows the same account to select its catalog image again in another chat", async () => {
    const view = mount();
    await expectCatalog(view, ["alice image"]);
    act(() => {
      source = {};
      view.rerender();
    });
    await expectCatalog(view, ["alice image"]);
    act(() => {
      view.result.current.setImageGenerateReferenceFileId(17);
    });
    await act(async () => {
      await view.result.current.submitImageGenerateModal();
    });
    expect(view.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        imageGenerationRequest: expect.objectContaining({
          referenceFileId: 17,
        }),
      }),
    );
  });

  it("reports the first read failure and retries without losing the prompt", async () => {
    io.list.mockRejectedValueOnce(new Error("Reference storage unavailable"));
    const view = mount();
    await waitFor(() =>
      expect(view.client.getQueryCache().getAll()[0]?.state.status).toBe(
        "error",
      ),
    );
    expect(view.result.current.referenceImageCandidatesFailed).toBe(true);
    expect(view.result.current.referenceImageCandidates).toEqual([]);
    expect(view.result.current.imageGeneratePrompt).toBe(
      "Keep this exact image prompt",
    );
    expect(view.result.current.retryReferenceImageCandidates).toBeTypeOf(
      "function",
    );
    await act(async () => {
      await view.result.current.retryReferenceImageCandidates();
    });
    await expectCatalog(view, ["alice image"]);
    expect(view.result.current.referenceImageCandidatesFailed).toBe(false);
    expect(view.result.current.imageGeneratePrompt).toBe(
      "Keep this exact image prompt",
    );
    const oldModalRetry = view.result.current.retryReferenceImageCandidates;
    act(() => {
      view.result.current.closeImageGenerateModal();
      view.result.current.openImageGenerateModal();
    });
    await expectCatalog(view, ["alice image"]);
    const modalReads = io.list.mock.calls.length;
    await act(async () => {
      await oldModalRetry();
    });
    expect(io.list).toHaveBeenCalledTimes(modalReads);
    const oldAccountRetry = view.result.current.retryReferenceImageCandidates;
    act(() => {
      owner = createOwner("bob");
      view.rerender();
    });
    await expectCatalog(view, ["bob image"]);
    const accountReads = io.list.mock.calls.length;
    await act(async () => {
      await oldAccountRetry();
    });
    expect(io.list).toHaveBeenCalledTimes(accountReads);
  });

  it("shows recovery for a paused refetch instead of exposing rows that cannot be selected", async () => {
    const view = mount();
    await expectCatalog(view, ["alice image"]);
    act(() => {
      onlineManager.setOnline(false);
    });
    await act(async () => {
      await view.client.refetchQueries({
        queryKey: ["reference-image-candidates"],
      });
    });
    await waitFor(() =>
      expect(view.client.getQueryCache().getAll()[0]?.state.fetchStatus).toBe(
        "paused",
      ),
    );
    expect(view.result.current.referenceImageCandidates).toEqual([]);
    expect(view.result.current.referenceImageCandidatesFailed).toBe(true);
    expect(view.result.current.imageGeneratePrompt).toBe(
      "Keep this exact image prompt",
    );
    act(() => {
      view.result.current.setImageGenerateReferenceFileId(17);
    });
    expect(view.result.current.imageGenerateReferenceFileId).toBeUndefined();
    expect(io.list).toHaveBeenCalledTimes(1);
    act(() => {
      onlineManager.setOnline(true);
    });
    await expectCatalog(view, ["alice image"]);
    expect(view.result.current.referenceImageCandidatesFailed).toBe(false);
    act(() => {
      view.result.current.setImageGenerateReferenceFileId(17);
    });
    await act(async () => {
      await view.result.current.submitImageGenerateModal();
    });
    expect(view.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        imageGenerationRequest: expect.objectContaining({
          referenceFileId: 17,
        }),
      }),
    );
  });

  it("hides a failed refetch and blocks an unverified selected reference until Retry succeeds", async () => {
    const view = mount();
    await expectCatalog(view, ["alice image"]);
    act(() => {
      view.result.current.setImageGenerateReferenceFileId(17);
    });
    io.list.mockRejectedValueOnce(new Error("Reference storage unavailable"));
    await act(async () => {
      await view.client.refetchQueries({
        queryKey: ["reference-image-candidates"],
      });
    });
    await waitFor(() =>
      expect(view.client.getQueryCache().getAll()[0]?.state.status).toBe(
        "error",
      ),
    );
    await act(async () => {
      await view.result.current.submitImageGenerateModal();
    });
    expect(view.sendMessage).not.toHaveBeenCalled();
    expect(view.result.current.referenceImageCandidates).toEqual([]);
    expect(view.result.current.referenceImageCandidatesFailed).toBe(true);
    expect(view.result.current.imageGenerateModalOpen).toBe(true);
    expect(view.result.current.imageGeneratePrompt).toBe(
      "Keep this exact image prompt",
    );
    await act(async () => {
      await view.result.current.retryReferenceImageCandidates();
    });
    await expectCatalog(view, ["alice image"]);
    await act(async () => {
      await view.result.current.submitImageGenerateModal();
    });
    expect(view.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        imageGenerationRequest: expect.objectContaining({
          referenceFileId: 17,
        }),
      }),
    );
  });
});

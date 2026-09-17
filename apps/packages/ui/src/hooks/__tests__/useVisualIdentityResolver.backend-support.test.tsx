import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VisualIdentityResolveResponse } from "@/types/visual-identities";
import {
  clearVisualIdentityResolverCaches,
  useVisualIdentityResolver,
} from "../useVisualIdentityResolver";

const unavailable: VisualIdentityResolveResponse = {
  actor_kind: "character",
  actor_id: 42,
  pack_id: null,
  pack_version_id: null,
  expression_key: null,
  requested_expression_key: "neutral",
  asset_id: null,
  storage_relpath: null,
  fallback_reason: "metadata_backend_unsupported",
  is_animated: false,
  content_type: null,
  asset_url: null,
  preview_url: null,
  role_id: null,
  role_label: null,
  resolution_source: "placeholder",
};

const options = { actorKind: "character" as const, actorId: 42 };

// These are compatibility controls for the unchanged real hook, not a frontend fix.
describe("optional unsupported visual metadata", () => {
  beforeEach(clearVisualIdentityResolverCaches);
  afterEach(cleanup);

  it("coalesces and caches a successful unavailable result without an asset or error", async () => {
    let finish!: (value: VisualIdentityResolveResponse) => void;
    const client = {
      resolveVisualIdentityBinding: vi.fn(
        () =>
          new Promise<VisualIdentityResolveResponse>((resolve) => {
            finish = resolve;
          }),
      ),
    };
    const first = renderHook(() =>
      useVisualIdentityResolver({ ...options, client }),
    );
    const second = renderHook(() =>
      useVisualIdentityResolver({ ...options, client }),
    );
    expect(first.result.current.isLoading).toBe(true);
    expect(second.result.current.isLoading).toBe(true);
    await act(async () => {
      finish(unavailable);
    });
    for (const hook of [first, second]) {
      expect(hook.result.current.resolution?.fallback_reason).toBe(
        "metadata_backend_unsupported",
      );
      expect(hook.result.current.resolution?.asset_url).toBeNull();
      expect(hook.result.current.error).toBeNull();
      expect(hook.result.current.isLoading).toBe(false);
      hook.unmount();
    }
    const remounted = renderHook(() =>
      useVisualIdentityResolver({ ...options, client }),
    );
    expect(remounted.result.current.resolution?.fallback_reason).toBe(
      "metadata_backend_unsupported",
    );
    expect(client.resolveVisualIdentityBinding).toHaveBeenCalledTimes(1);
  });

  it("explicit refresh replaces an unavailable cached result after support changes", async () => {
    const client = {
      resolveVisualIdentityBinding: vi
        .fn()
        .mockResolvedValueOnce(unavailable)
        .mockResolvedValueOnce({
          ...unavailable,
          pack_id: 7,
          pack_version_id: 8,
          asset_id: 9,
          asset_url: "/api/v1/visual-identities/packs/7/assets/9/content",
          expression_key: "neutral",
          storage_relpath: "fixture/neutral.png",
          content_type: "image/png",
          fallback_reason: null,
          resolution_source: "binding",
        }),
    };
    const { result } = renderHook(() =>
      useVisualIdentityResolver({ ...options, client }),
    );
    await waitFor(() =>
      expect(result.current.resolution?.fallback_reason).toBe(
        "metadata_backend_unsupported",
      ),
    );
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.resolution?.asset_id).toBe(9));
    expect(result.current.resolution?.asset_url).toBe(
      "/api/v1/visual-identities/packs/7/assets/9/content",
    );
    expect(result.current.error).toBeNull();
    expect(client.resolveVisualIdentityBinding).toHaveBeenCalledTimes(2);
  });

  it.each([401, 500])(
    "keeps a real HTTP%s failure visible and permits a fresh request",
    async (status) => {
      const error = Object.assign(new Error("Resolution request failed"), {
        status,
      });
      const client = {
        resolveVisualIdentityBinding: vi
          .fn()
          .mockRejectedValueOnce(error)
          .mockResolvedValueOnce(unavailable),
      };
      const first = renderHook(() =>
        useVisualIdentityResolver({ ...options, client }),
      );
      await waitFor(() => expect(first.result.current.error).toBe(error));
      expect(first.result.current.resolution).toBeNull();
      expect(first.result.current.isLoading).toBe(false);
      first.unmount();
      const second = renderHook(() =>
        useVisualIdentityResolver({ ...options, client }),
      );
      await waitFor(() =>
        expect(second.result.current.resolution?.fallback_reason).toBe(
          "metadata_backend_unsupported",
        ),
      );
      expect(second.result.current.error).toBeNull();
      expect(client.resolveVisualIdentityBinding).toHaveBeenCalledTimes(2);
    },
  );
});

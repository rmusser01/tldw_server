import React from "react";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { SpriteFrameRenderer } from "../SpriteFrameRenderer";
const fetchWithAuth = vi.hoisted(() => vi.fn());
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { fetchWithAuth },
}));
const path =
  "/api/v1/persona/profiles/migu/visual-packs/pack/assets/frame/content";
const assets = {
  frame: {
    id: "frame",
    url: path,
    mime_type: "image/png",
    asset_role: "frame" as const,
  },
};
const manifest = {
  manifest_version: 1 as const,
  renderer_type: "sprite_frames" as const,
  states: {
    idle: { animation_id: "idle" },
    thinking: { animation_id: "idle" },
  },
  animations: { idle: { frames: [{ asset_id: "frame", duration_ms: 50 }] } },
};
const props = {
  manifest,
  assets,
  fallbackLabel: "Migu",
  state: "idle" as const,
};
beforeEach(() => {
  fetchWithAuth
    .mockReset()
    .mockResolvedValue({ ok: true, data: new Uint8Array([1, 2, 3]).buffer });
  URL.createObjectURL = vi.fn(() => "blob:migu");
  URL.revokeObjectURL = vi.fn();
});
afterEach(cleanup);
it("loads protected frames through auth once and releases object URLs", async () => {
  const view = render(<SpriteFrameRenderer {...props} />);
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:migu",
    ),
  );
  expect(fetchWithAuth).toHaveBeenCalledWith(
    path,
    expect.objectContaining({
      responseType: "arrayBuffer",
      signal: expect.any(AbortSignal),
    }),
  );
  view.rerender(<SpriteFrameRenderer {...props} state="thinking" />);
  expect(fetchWithAuth).toHaveBeenCalledTimes(1);
  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:migu");
});
it("retains failed assets across animation state changes without retrying", async () => {
  fetchWithAuth.mockResolvedValue({ ok: false, status: 403 });
  const onRenderError = vi.fn();
  const view = render(
    <SpriteFrameRenderer {...props} onRenderError={onRenderError} />,
  );
  await waitFor(() =>
    expect(onRenderError).toHaveBeenCalledWith("missing_asset"),
  );
  expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument();
  view.rerender(
    <SpriteFrameRenderer
      {...props}
      state="thinking"
      onRenderError={onRenderError}
    />,
  );
  await act(async () => {});
  expect(fetchWithAuth).toHaveBeenCalledTimes(1);
});
it("aborts in-flight loading and does not publish a late URL after unmount", async () => {
  let finish!: (value: unknown) => void;
  fetchWithAuth.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const view = render(<SpriteFrameRenderer {...props} />);
  await waitFor(() => expect(fetchWithAuth).toHaveBeenCalledTimes(1));
  const signal = fetchWithAuth.mock.calls[0][1].signal;
  view.unmount();
  expect(signal.aborted).toBe(true);
  await act(async () => finish({ ok: true, data: new Uint8Array([1]).buffer }));
  expect(URL.createObjectURL).not.toHaveBeenCalled();
});

it("does not send credentials to an external asset origin", () => {
  const url =
    "https://images.example.test/api/v1/persona/profiles/migu/assets/frame/content";
  render(
    <SpriteFrameRenderer
      {...props}
      assets={{ frame: { ...assets.frame, url } }}
    />,
  );
  expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
    "src",
    url,
  );
  expect(fetchWithAuth).not.toHaveBeenCalled();
});

it("releases the previous source when the pack changes", async () => {
  const view = render(<SpriteFrameRenderer {...props} />);
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:migu",
    ),
  );
  URL.createObjectURL = vi.fn(() => "blob:replacement");
  view.rerender(
    <SpriteFrameRenderer
      {...props}
      assets={{
        frame: {
          ...assets.frame,
          url: path.replace("/pack/", "/replacement/"),
        },
      }}
    />,
  );
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:replacement",
    ),
  );
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:migu");
  expect(fetchWithAuth).toHaveBeenCalledTimes(2);
});

it("loads only the displayed frame in a large protected pack", async () => {
  const pack = Object.fromEntries(
    Array.from({ length: 256 }, (_, index) => {
      const id = index === 0 ? "frame" : `unused-${index}`;
      return [
        id,
        { ...assets.frame, id, url: path.replace("/frame/", `/${id}/`) },
      ];
    }),
  );
  render(<SpriteFrameRenderer {...props} assets={pack} />);
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toBeInTheDocument(),
  );
  expect(fetchWithAuth).toHaveBeenCalledTimes(1);
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
});

it("evicts old blobs as different large frames are displayed", async () => {
  const pack = {
    ...assets,
    next: {
      ...assets.frame,
      id: "next",
      url: path.replace("/frame/", "/next/"),
    },
  };
  const states = {
    ...manifest,
    states: {
      idle: { animation_id: "idle" },
      thinking: { animation_id: "next" },
    },
    animations: {
      ...manifest.animations,
      next: { frames: [{ asset_id: "next" }] },
    },
  };
  fetchWithAuth.mockResolvedValue({
    ok: true,
    data: new ArrayBuffer(10 * 1024 * 1024),
  });
  URL.createObjectURL = vi
    .fn()
    .mockReturnValueOnce("blob:first")
    .mockReturnValueOnce("blob:next");
  const view = render(
    <SpriteFrameRenderer {...props} assets={pack} manifest={states} />,
  );
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:first",
    ),
  );
  expect(fetchWithAuth).toHaveBeenCalledTimes(1);
  view.rerender(
    <SpriteFrameRenderer
      {...props}
      assets={pack}
      manifest={states}
      state="thinking"
    />,
  );
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:next",
    ),
  );
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:first");
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith("blob:next");
  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:next");
});

it("reuses recent frames and releases older frames in a long animation", async () => {
  const pack = Object.fromEntries(
    Array.from({ length: 12 }, (_, index) => {
      const id = `frame-${index}`;
      return [
        id,
        { ...assets.frame, id, url: path.replace("/frame/", `/${id}/`) },
      ];
    }),
  );
  let nextUrl = 0;
  URL.createObjectURL = vi.fn(() => `blob:${nextUrl++}`);
  const withFrame = (id: string) => ({
    ...manifest,
    animations: { idle: { frames: [{ asset_id: id }] } },
  });
  const view = render(
    <SpriteFrameRenderer
      {...props}
      assets={pack}
      manifest={withFrame("frame-0")}
    />,
  );
  for (let index = 0; index < 12; index++) {
    view.rerender(
      <SpriteFrameRenderer
        {...props}
        assets={pack}
        manifest={withFrame(`frame-${index}`)}
      />,
    );
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "src",
        `blob:${index}`,
      ),
    );
  }
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(4);
  view.rerender(
    <SpriteFrameRenderer
      {...props}
      assets={pack}
      manifest={withFrame("frame-10")}
    />,
  );
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:10",
    ),
  );
  expect(fetchWithAuth).toHaveBeenCalledTimes(12);
  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(12);
});

it("aborts an abandoned frame request and ignores its late completion", async () => {
  let finish!: (value: unknown) => void;
  fetchWithAuth.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const pack = {
    ...assets,
    next: {
      ...assets.frame,
      id: "next",
      url: path.replace("/frame/", "/next/"),
    },
  };
  const states = {
    ...manifest,
    states: {
      idle: { animation_id: "idle" },
      thinking: { animation_id: "next" },
    },
    animations: {
      ...manifest.animations,
      next: { frames: [{ asset_id: "next" }] },
    },
  };
  const view = render(
    <SpriteFrameRenderer {...props} assets={pack} manifest={states} />,
  );
  const firstSignal = fetchWithAuth.mock.calls[0][1].signal;
  view.rerender(
    <SpriteFrameRenderer
      {...props}
      assets={pack}
      manifest={states}
      state="thinking"
    />,
  );
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      "blob:migu",
    ),
  );
  expect(firstSignal.aborted).toBe(true);
  await act(async () => finish({ ok: true, data: new ArrayBuffer(1) }));
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
});

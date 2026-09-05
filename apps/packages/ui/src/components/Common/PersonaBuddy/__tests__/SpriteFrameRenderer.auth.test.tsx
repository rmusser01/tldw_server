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

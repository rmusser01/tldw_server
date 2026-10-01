import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  useIngestResults,
  type UseIngestResultsDeps,
} from "../hooks/useIngestResults"
import * as mediaHandoff from "@/services/tldw/media-chat-handoff"

const runtime = vi.hoisted(() => ({
  extension: false,
  owner: "server:alice" as string | null,
  navigate: vi.fn(),
  createTab: vi.fn(),
}))
vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => runtime.extension,
}))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => runtime.owner,
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => runtime.navigate }))
vi.mock("wxt/browser", () => ({
  browser: {
    tabs: { create: runtime.createTab },
    runtime: { getURL: (path: string) => `chrome-extension://test${path}` },
  },
}))
vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
)

const error = vi.fn()
const deps = {
  open: true,
  running: false,
  messageApi: { error },
  qi: (_key: string, fallback: string) => fallback,
  t: (key: string) => key,
  onClose: vi.fn(),
  reviewBeforeStorage: false,
  storeRemote: true,
  processOnly: false,
  common: {
    perform_analysis: false,
    perform_chunking: true,
    overwrite_existing: false,
  },
  advancedValues: {},
  rows: [],
  formatBytes: String,
  plannedRunContextRef: { current: null },
  setRows: vi.fn(),
  setQueuedFiles: vi.fn(),
  setLocalFiles: vi.fn(),
  buildRowEntry: vi.fn(),
  createDefaultsSnapshot: vi.fn(),
  setReviewBeforeStorage: vi.fn(),
  setStoreRemote: vi.fn(),
} as unknown as UseIngestResultsDeps

describe("Quick Ingest destination-owned media handoff", () => {
  beforeEach(() => {
    runtime.extension = false
    runtime.owner = "server:alice"
    runtime.navigate.mockReset()
    runtime.createTab.mockReset().mockResolvedValue({ id: 42 })
    error.mockReset()
    sessionStorage.clear()
    localStorage.clear()
    window.history.replaceState(null, "", "/sidepanel.html")
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: {
          value: { request: (_key: string, work: () => unknown) => work() },
        },
      }),
    )
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    window.history.replaceState(null, "", "/")
  })

  it.each([false, true])(
    "keeps discussion in the current WebUI/options tab (extension=%s)",
    async (extension) => {
      runtime.extension = extension
      if (extension) window.history.replaceState(null, "", "/options.html")
      const { result } = renderHook(() => useIngestResults(deps))
      await act(async () => {
        await result.current.discussInChat({
          id: "result",
          status: "ok",
          type: "pdf",
          data: { media_id: 7 },
        })
      })
      const route = runtime.navigate.mock.calls[0][0]
      const token = new URL(route, "http://localhost").searchParams.get(
        mediaHandoff.MEDIA_CHAT_HANDOFF_PARAM,
      )!
      expect(token).toMatch(/^tab-/)
      expect(
        await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
      ).toEqual({ ownerScope: "server:alice", mediaId: "7", mode: "rag_media" })
      expect(runtime.createTab).not.toHaveBeenCalled()
    },
  )

  it("opens the extension options destination with only its opaque transfer token", async () => {
    runtime.extension = true
    const { result } = renderHook(() => useIngestResults(deps))
    await act(async () => {
      await result.current.discussInChat({
        id: "result",
        status: "ok",
        type: "pdf",
        data: { media_id: 7, url: "https://private-source.test/document" },
      })
    })
    const url = runtime.createTab.mock.calls[0][0].url
    expect(url).toMatch(
      /^chrome-extension:\/\/test\/options.html#\/chat\?media_handoff=extension-[a-f0-9-]+$/,
    )
    expect(runtime.navigate).not.toHaveBeenCalled()
    const token = new URLSearchParams(url.split("?")[1]).get(
      mediaHandoff.MEDIA_CHAT_HANDOFF_PARAM,
    )!
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:bob"),
    ).toBeNull()
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toEqual({
      ownerScope: "server:alice",
      mediaId: "7",
      url: "https://private-source.test/document",
      mode: "rag_media",
    })
  })

  it.each(["unmount", "owner round trip", "account boundary before render"])(
    "does not open a delayed extension transfer after %s",
    async (change) => {
      runtime.extension = true
      const create = mediaHandoff.createMediaChatHandoff
      let release!: () => void
      let token!: string
      vi.spyOn(mediaHandoff, "createMediaChatHandoff").mockImplementationOnce(
        async (...args) => {
          token = await create(...args)
          await new Promise<void>((resolve) => {
            release = resolve
          })
          return token
        },
      )
      const view = renderHook(() => useIngestResults(deps))
      let transfer!: Promise<void>
      await act(async () => {
        transfer = view.result.current.discussInChat({
          id: "result",
          status: "ok",
          type: "pdf",
          data: { media_id: 7 },
        })
        await Promise.resolve()
      })
      if (change === "unmount") view.unmount()
      else if (change === "account boundary before render")
        window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      else {
        runtime.owner = "server:bob"
        view.rerender()
        runtime.owner = "server:alice"
        view.rerender()
      }
      await act(async () => {
        release()
        await transfer
      })
      expect(runtime.createTab).not.toHaveBeenCalled()
      expect(runtime.navigate).not.toHaveBeenCalled()
      expect(
        await mediaHandoff.readMediaChatHandoff(token, "server:alice"),
      ).toBeNull()
    },
  )

  it("removes a transfer when opening the extension destination fails", async () => {
    runtime.extension = true
    runtime.createTab.mockRejectedValueOnce(new Error("Tab unavailable"))
    const { result } = renderHook(() => useIngestResults(deps))
    await act(async () => {
      await result.current.discussInChat({
        id: "result",
        status: "ok",
        type: "pdf",
        data: { media_id: 7 },
      })
    })
    const token = new URLSearchParams(
      runtime.createTab.mock.calls[0][0].url.split("?")[1],
    ).get(mediaHandoff.MEDIA_CHAT_HANDOFF_PARAM)!
    expect(
      await mediaHandoff.readMediaChatHandoff(token, "server:alice"),
    ).toBeNull()
    expect(error).toHaveBeenCalledWith(
      expect.stringContaining("Please try again"),
    )
  })
})

describe("Quick Ingest content-review handoff (UAT396)", () => {
  it.each([
    { extension: false, path: "/media" },
    { extension: true, path: "/options.html" },
    { extension: true, path: "/sidepanel.html" },
  ])(
    "opens the produced review batch from $path (extension=$extension)",
    async ({ extension, path }) => {
      runtime.extension = extension;
      runtime.navigate.mockReset();
      runtime.createTab.mockReset().mockResolvedValue({ id: 42 });
      window.history.replaceState(null, "", path);
      const { result } = renderHook(() => useIngestResults(deps));
      let opened = false;
      await act(async () => {
        opened = await result.current.openContentReview("uat396-batch");
      });
      expect(opened).toBe(true);
      if (extension && path === "/sidepanel.html") {
        expect(runtime.createTab).toHaveBeenCalledWith({
          url: "chrome-extension://test/options.html#/content-review?batch=uat396-batch",
        });
        expect(runtime.navigate).not.toHaveBeenCalled();
      } else {
        // The shipped WebUI page is pages/content-review.tsx; it is not options.html.
        expect(runtime.navigate).toHaveBeenCalledWith(
          "/content-review?batch=uat396-batch",
        );
        expect(runtime.createTab).not.toHaveBeenCalled();
      }
    },
  );
  afterEach(() => {
    window.history.replaceState(null, "", "/");
  });
});

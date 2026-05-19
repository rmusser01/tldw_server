import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getScreenshotFromCurrentTab: vi.fn()
}))

vi.mock("@/libs/get-screenshot", () => ({
  getScreenshotFromCurrentTab: (...args: unknown[]) =>
    mocks.getScreenshotFromCurrentTab(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    i18n: {
      getMessage: (key: string) =>
        (
          {
            contextSaveToClipperRestrictedPage:
              "Localized restricted clipper message."
          } as Record<string, string>
        )[key] || key
    }
  }
}))

import {
  captureScreenshotClip,
  isRestrictedClipperPage
} from "@/services/web-clipper/content-extract"
import { buildClipDraft, normalizeClipDraft } from "@/services/web-clipper/draft-builder"

describe("buildClipDraft", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("normalizes article captures into one draft and records the fallback path", () => {
    const draft = buildClipDraft({
      requestedType: "article",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      extracted: {
        articleText: "",
        fullPageText: "Fallback body"
      }
    })

    expect(draft.clipType).toBe("article")
    expect(draft.captureMetadata.fallbackPath).toEqual([
      "article",
      "full_page"
    ])
    expect(draft.visibleBody).toContain("Fallback body")
  })

  it("normalizes bookmark and selection captures into the same draft shape", () => {
    const bookmarkDraft = buildClipDraft({
      requestedType: "bookmark",
      pageUrl: "https://example.com/bookmark",
      pageTitle: "Bookmark",
      extracted: {}
    })

    const selectionDraft = buildClipDraft({
      requestedType: "selection",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      extracted: {
        selectionText: "Selected excerpt"
      }
    })

    expect(bookmarkDraft.clipType).toBe("bookmark")
    expect(bookmarkDraft.captureMetadata.fallbackPath).toEqual(["bookmark"])
    expect(selectionDraft.clipType).toBe("selection")
    expect(selectionDraft.captureMetadata.fallbackPath).toEqual(["selection"])
    expect(selectionDraft.visibleBody).toContain("Selected excerpt")
  })

  it("marks restricted browser pages with a user-visible explanation", () => {
    const draft = buildClipDraft({
      requestedType: "full_page",
      pageUrl: "chrome://extensions",
      pageTitle: "Extensions",
      extracted: {
        fullPageText: ""
      }
    })

    expect(isRestrictedClipperPage("chrome://extensions")).toBe(true)
    expect(draft.userVisibleError).toBe("Localized restricted clipper message.")
    expect(draft.captureMetadata.fallbackPath).toEqual([
      "full_page",
      "blocked"
    ])
  })

  it("keeps error-state screenshot drafts even when the visible body is empty", () => {
    const normalized = normalizeClipDraft({
      clipId: "clip-error",
      requestedType: "screenshot",
      clipType: "screenshot",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      visibleBody: "",
      captureMetadata: {
        clipType: "screenshot",
        actualType: "screenshot",
        fallbackPath: ["screenshot"]
      },
      capturedAt: "2026-04-03T00:00:00.000Z",
      userVisibleError: "Screenshot capture failed."
    })

    expect(normalized).toMatchObject({
      clipId: "clip-error",
      clipType: "screenshot",
      visibleBody: "",
      userVisibleError: "Screenshot capture failed."
    })
  })
})

describe("captureScreenshotClip", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses the visible-tab screenshot helper for screenshot captures", async () => {
    mocks.getScreenshotFromCurrentTab.mockResolvedValue({
      success: true,
      screenshot: "data:image/png;base64,clip",
      error: null
    })

    const draft = await captureScreenshotClip({
      pageUrl: "https://example.com/story",
      pageTitle: "Story"
    })

    expect(mocks.getScreenshotFromCurrentTab).toHaveBeenCalledTimes(1)
    expect(draft.clipType).toBe("screenshot")
    expect(draft.captureMetadata.fallbackPath).toEqual(["screenshot"])
    expect(draft.captureMetadata.screenshotDataUrl).toBe(
      "data:image/png;base64,clip"
    )
  })
})

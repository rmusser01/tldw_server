import { afterEach, describe, expect, it, vi } from "vitest"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import {
  clearPendingClipDraft,
  readPendingClipDraft,
  writePendingClipDraft
} from "@/services/web-clipper/pending-draft"

const createDraft = () =>
  buildClipDraft({
    requestedType: "article",
    pageUrl: "https://example.com/story",
    pageTitle: "Story",
    extracted: {
      articleText: "Example article body"
    }
  })

describe("pending clip draft storage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.sessionStorage.clear()
    clearPendingClipDraft()
  })

  it("falls back to in-memory storage when sessionStorage rejects the write", () => {
    const draft = createDraft()

    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota exceeded")
    })

    writePendingClipDraft(draft)

    expect(readPendingClipDraft()).toMatchObject({
      clipId: draft.clipId,
      pageUrl: "https://example.com/story",
      visibleBody: "Example article body"
    })
  })

  it("prefers the newest in-memory draft over stale session storage after a write failure", () => {
    const staleDraft = createDraft()
    writePendingClipDraft(staleDraft)

    const freshDraft = buildClipDraft({
      requestedType: "selection",
      pageUrl: "https://example.com/fresh",
      pageTitle: "Fresh Story",
      extracted: {
        selectionText: "Fresh selection"
      }
    })

    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota exceeded")
    })

    writePendingClipDraft(freshDraft)

    expect(readPendingClipDraft()).toMatchObject({
      clipId: freshDraft.clipId,
      pageUrl: "https://example.com/fresh",
      visibleBody: "Fresh selection"
    })
  })
})

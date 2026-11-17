import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getScreenshotFromCurrentTab: vi.fn(),
  defaultExtractContent: vi.fn()
}))

vi.mock("@/libs/get-screenshot", () => ({
  getScreenshotFromCurrentTab: (...args: unknown[]) =>
    mocks.getScreenshotFromCurrentTab(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    i18n: {
      getMessage: (key: string) => key
    }
  }
}))

vi.mock("@/parser/default", () => ({
  defaultExtractContent: (...args: unknown[]) =>
    mocks.defaultExtractContent(...args)
}))

import { extractClipPageTextFromDocument } from "@/services/web-clipper/content-extract"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"

type SpyDocument = {
  doc: Document
  outerHTML: ReturnType<typeof vi.fn>
  innerText: ReturnType<typeof vi.fn>
  textContent: ReturnType<typeof vi.fn>
}

const stubSelection = (text: string) =>
  ({ toString: () => text }) as unknown as Selection

const buildSpyDocument = (html: string, bodyText: string): SpyDocument => {
  const outerHTML = vi.fn(() => html)
  const innerText = vi.fn(() => bodyText)
  const textContent = vi.fn(() => bodyText)
  const doc = {
    documentElement: {
      get outerHTML() {
        return outerHTML()
      }
    },
    body: {
      get innerText() {
        return innerText()
      },
      get textContent() {
        return textContent()
      }
    }
  } as unknown as Document
  return { doc, outerHTML, innerText, textContent }
}

const PAGE_URL = "https://example.com/story"
const PAGE_TITLE = "Story"

const buildDraftFromExtraction = (
  requestedType: "selection" | "article" | "full_page",
  extracted: Awaited<ReturnType<typeof extractClipPageTextFromDocument>>
) =>
  buildClipDraft({
    requestedType,
    pageUrl: PAGE_URL,
    pageTitle: PAGE_TITLE,
    extracted
  })

describe("extractClipPageTextFromDocument tiered capture (WP-56/WP-57)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.defaultExtractContent.mockReturnValue("")
  })

  it("characterization: selection present -> the selection is the whole capture", async () => {
    vi.spyOn(window, "getSelection").mockReturnValue(stubSelection("Selected excerpt"))
    const spyDoc = buildSpyDocument(
      "<html><body><p>Huge page body</p></body></html>",
      "Huge page body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "selection")

    expect(extracted).toEqual({ selectionText: "Selected excerpt" })

    // Mechanism pin (W3 convention): the selection path must never pay for
    // the full-page serialize, the heavy parser pass, or the forced
    // innerText layout.
    expect(spyDoc.outerHTML).not.toHaveBeenCalled()
    expect(spyDoc.innerText).not.toHaveBeenCalled()
    expect(spyDoc.textContent).not.toHaveBeenCalled()
    expect(mocks.defaultExtractContent).not.toHaveBeenCalled()

    const draft = buildDraftFromExtraction("selection", extracted)
    expect(draft.clipType).toBe("selection")
    expect(draft.visibleBody).toBe("Selected excerpt")
    expect(draft.captureMetadata).toMatchObject({
      clipType: "selection",
      actualType: "selection",
      fallbackPath: ["selection"]
    })
    expect(draft.fullExtract).toBe("Selected excerpt")
  })

  it("characterization: selection absent -> falls back to the article tier", async () => {
    vi.spyOn(window, "getSelection").mockReturnValue(stubSelection(""))
    mocks.defaultExtractContent.mockReturnValue("Article body")
    const spyDoc = buildSpyDocument(
      "<html><body><main><p>Article body</p></main></body></html>",
      "Full page body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "selection")

    expect(extracted.selectionText).toBeUndefined()
    expect(extracted.articleText).toBe("Article body")
    // fullPage stays the final fallback tier: not computed while the
    // article tier produced content.
    expect(extracted.fullPageText).toBeUndefined()
    expect(spyDoc.outerHTML).toHaveBeenCalled()
    expect(spyDoc.innerText).not.toHaveBeenCalled()

    const draft = buildDraftFromExtraction("selection", extracted)
    expect(draft.clipType).toBe("selection")
    expect(draft.visibleBody).toBe("Article body")
    expect(draft.captureMetadata).toMatchObject({
      clipType: "selection",
      actualType: "article",
      fallbackPath: ["selection", "article"]
    })
  })

  it("characterization: article empty -> falls back to the full-page tier", async () => {
    vi.spyOn(window, "getSelection").mockReturnValue(stubSelection(""))
    mocks.defaultExtractContent.mockReturnValue("")
    const spyDoc = buildSpyDocument(
      "<html><head></head><body></body></html>",
      "Full page fallback body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "selection")

    expect(extracted.articleText).toBeUndefined()
    expect(extracted.fullPageText).toBe("Full page fallback body")
    // Both lower tiers were attempted in order before the fallback won.
    expect(spyDoc.outerHTML).toHaveBeenCalled()
    expect(spyDoc.innerText).toHaveBeenCalled()

    const draft = buildDraftFromExtraction("selection", extracted)
    expect(draft.clipType).toBe("selection")
    expect(draft.visibleBody).toBe("Full page fallback body")
    expect(draft.captureMetadata).toMatchObject({
      clipType: "selection",
      actualType: "full_page",
      fallbackPath: ["selection", "full_page"]
    })
  })

  it("article requests keep the general flow's article tier without the full-page fallback", async () => {
    vi.spyOn(window, "getSelection").mockReturnValue(stubSelection("Stray selection"))
    mocks.defaultExtractContent.mockReturnValue("Article body")
    const spyDoc = buildSpyDocument(
      "<html><body><main><p>Article body</p></main></body></html>",
      "Full page body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "article")

    expect(extracted.selectionText).toBe("Stray selection")
    expect(extracted.articleText).toBe("Article body")
    expect(extracted.fullPageText).toBeUndefined()
    expect(spyDoc.innerText).not.toHaveBeenCalled()

    const draft = buildDraftFromExtraction("article", extracted)
    expect(draft.visibleBody).toBe("Article body")
    expect(draft.captureMetadata).toMatchObject({
      actualType: "article",
      fallbackPath: ["article"]
    })
  })

  it("article requests fall back to the full-page tier when the article is empty", async () => {
    vi.spyOn(window, "getSelection").mockReturnValue(stubSelection(""))
    mocks.defaultExtractContent.mockReturnValue("")
    const spyDoc = buildSpyDocument(
      "<html><head></head><body></body></html>",
      "Full page fallback body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "article")

    expect(extracted.fullPageText).toBe("Full page fallback body")
    expect(spyDoc.innerText).toHaveBeenCalled()

    const draft = buildDraftFromExtraction("article", extracted)
    expect(draft.visibleBody).toBe("Full page fallback body")
    expect(draft.captureMetadata).toMatchObject({
      actualType: "full_page",
      fallbackPath: ["article", "full_page"]
    })
  })

  it("non-tiered request types keep the existing eager capture behavior", async () => {
    mocks.defaultExtractContent.mockReturnValue("Article body")
    const spyDoc = buildSpyDocument(
      "<html><body><main><p>Article body</p></main></body></html>",
      "Full page body"
    )

    const extracted = await extractClipPageTextFromDocument(spyDoc.doc, "full_page")

    expect(extracted.articleText).toBe("Article body")
    expect(extracted.fullPageText).toBe("Full page body")
    expect(spyDoc.outerHTML).toHaveBeenCalled()
    expect(spyDoc.innerText).toHaveBeenCalled()

    const draft = buildDraftFromExtraction("full_page", extracted)
    expect(draft.visibleBody).toBe("Full page body")
    expect(draft.fullExtract).toBe("Full page body")
    expect(draft.captureMetadata).toMatchObject({
      actualType: "full_page",
      fallbackPath: ["full_page"]
    })

    // The legacy no-type call site keeps the same eager general flow.
    const untyped = await extractClipPageTextFromDocument(spyDoc.doc)
    expect(untyped.articleText).toBe("Article body")
    expect(untyped.fullPageText).toBe("Full page body")
  })
})

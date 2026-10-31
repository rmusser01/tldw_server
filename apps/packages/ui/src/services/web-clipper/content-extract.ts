import { getScreenshotFromCurrentTab } from "@/libs/get-screenshot"
import { browser } from "wxt/browser"

type DefaultExtractContent = (html: string) => string

// The default parser drags cheerio + Readability + Turndown with it. In
// extension contexts it is loaded on demand from a web-accessible chunk so
// the every-page web-clipper content script (an IIFE build that inlines any
// bundler-analyzable import) stays small.
const PARSER_CHUNK_RESOURCE = "parser-main.js"
const PARSER_GLOBAL = "__tldwParserDefaultExtractContent"

let parserLoad: Promise<DefaultExtractContent> | null = null

const extensionRuntime = (): { getURL?: (path: string) => string } | null => {
  const scope = globalThis as Record<string, any> | undefined
  if (!scope) return null
  if (scope.browser?.runtime?.id && scope.browser.runtime.getURL) {
    return scope.browser.runtime
  }
  if (scope.chrome?.runtime?.id && scope.chrome.runtime.getURL) {
    return scope.chrome.runtime
  }
  return null
}

const loadDefaultExtractContent = async (): Promise<DefaultExtractContent> => {
  const runtime = extensionRuntime()
  if (runtime?.getURL) {
    const chunkUrl = runtime.getURL(PARSER_CHUNK_RESOURCE)
    await import(/* @vite-ignore */ chunkUrl)
    const handle = (globalThis as Record<string, unknown>)[PARSER_GLOBAL]
    if (typeof handle === "function") {
      return handle as DefaultExtractContent
    }
  }
  // Non-extension contexts (tests, web app): a variable specifier keeps IIFE
  // content-script builds from inlining the parser while ESM/vitest runtimes
  // still resolve the alias at runtime.
  const specifier = "@/parser/default"
  const mod = (await import(/* @vite-ignore */ specifier)) as {
    defaultExtractContent?: DefaultExtractContent
  }
  if (typeof mod?.defaultExtractContent !== "function") {
    throw new Error("Failed to load the default page parser.")
  }
  return mod.defaultExtractContent
}

const getDefaultExtractContent = (): Promise<DefaultExtractContent> => {
  if (!parserLoad) {
    parserLoad = loadDefaultExtractContent().catch((error) => {
      parserLoad = null
      throw error
    })
  }
  return parserLoad
}

export type ClipCaptureType =
  | "bookmark"
  | "article"
  | "full_page"
  | "selection"
  | "screenshot"

export type ClipCaptureFallbackStep = ClipCaptureType | "blocked"

export type ClipCaptureInput = {
  requestedType: ClipCaptureType
  pageUrl: string
  pageTitle: string
  selectionText?: string
  articleText?: string
  fullPageText?: string
  screenshotDataUrl?: string
}

export type ClipCaptureResolution = {
  clipType: ClipCaptureType
  visibleBody: string
  fallbackPath: ClipCaptureFallbackStep[]
  actualType: ClipCaptureType
  screenshotDataUrl?: string
}

export type ClipCaptureResult = ClipCaptureInput & {
  clipType: ClipCaptureType
  visibleBody: string
  captureMetadata: ClipCaptureResolution
  userVisibleError?: string
}

const RESTRICTED_PAGE_PREFIXES = [
  "about:",
  "chrome:",
  "edge:",
  "moz-extension:",
  "opera:",
  "view-source:",
  "file:"
]

const trimText = (value: unknown): string => String(value || "").trim()

const buildBookmarkBody = (pageTitle: string, pageUrl: string): string =>
  [pageTitle, pageUrl].filter(Boolean).join("\n")

const getRestrictedPageMessage = (): string =>
  (browser.i18n.getMessage as (key: string) => string)(
    "contextSaveToClipperRestrictedPage"
  ) ||
  "This page is restricted, so the clipper cannot capture it."

export const isRestrictedClipperPage = (pageUrl: string): boolean => {
  const normalized = trimText(pageUrl).toLowerCase()
  if (!normalized) return true
  if (!/^https?:\/\//.test(normalized)) return true
  return RESTRICTED_PAGE_PREFIXES.some((prefix) => normalized.startsWith(prefix))
}

export const extractClipPageTextFromDocument = async (
  doc: Document = document
): Promise<Pick<ClipCaptureInput, "selectionText" | "articleText" | "fullPageText">> => {
  const selectionText = trimText(window.getSelection()?.toString())
  const html = doc.documentElement?.outerHTML || ""
  let articleText = ""
  try {
    const defaultExtractContent = await getDefaultExtractContent()
    articleText = trimText(defaultExtractContent(html))
  } catch {
    // Parser unavailable: the capture flow degrades to full-page text or a
    // bookmark fallback, so never block the clip on this.
  }
  const fullPageText = trimText(doc.body?.innerText || doc.body?.textContent)
  return {
    selectionText: selectionText || undefined,
    articleText: articleText || undefined,
    fullPageText: fullPageText || undefined
  }
}

export const resolveClipCaptureResolution = (
  input: ClipCaptureInput
): ClipCaptureResolution & { userVisibleError?: string } => {
  const requestedType = input.requestedType
  const pageUrl = trimText(input.pageUrl)
  const pageTitle = trimText(input.pageTitle)

  if (isRestrictedClipperPage(pageUrl)) {
    return {
      clipType: requestedType,
      visibleBody: "",
      fallbackPath: [requestedType, "blocked"],
      actualType: requestedType,
      userVisibleError: getRestrictedPageMessage()
    }
  }

  const selectionText = trimText(input.selectionText)
  const articleText = trimText(input.articleText)
  const fullPageText = trimText(input.fullPageText)
  const screenshotDataUrl = trimText(input.screenshotDataUrl) || undefined

  if (requestedType === "selection") {
    if (selectionText) {
      return {
        clipType: "selection",
        visibleBody: selectionText,
        fallbackPath: ["selection"],
        actualType: "selection"
      }
    }
    if (articleText) {
      return {
        clipType: "selection",
        visibleBody: articleText,
        fallbackPath: ["selection", "article"],
        actualType: "article"
      }
    }
    if (fullPageText) {
      return {
        clipType: "selection",
        visibleBody: fullPageText,
        fallbackPath: ["selection", "full_page"],
        actualType: "full_page"
      }
    }
    return {
      clipType: "selection",
      visibleBody: buildBookmarkBody(pageTitle, pageUrl),
      fallbackPath: ["selection", "bookmark"],
      actualType: "bookmark"
    }
  }

  if (requestedType === "article") {
    if (articleText) {
      return {
        clipType: "article",
        visibleBody: articleText,
        fallbackPath: ["article"],
        actualType: "article"
      }
    }
    if (fullPageText) {
      return {
        clipType: "article",
        visibleBody: fullPageText,
        fallbackPath: ["article", "full_page"],
        actualType: "full_page"
      }
    }
    return {
      clipType: "article",
      visibleBody: buildBookmarkBody(pageTitle, pageUrl),
      fallbackPath: ["article", "bookmark"],
      actualType: "bookmark"
    }
  }

  if (requestedType === "full_page") {
    if (fullPageText) {
      return {
        clipType: "full_page",
        visibleBody: fullPageText,
        fallbackPath: ["full_page"],
        actualType: "full_page"
      }
    }
    return {
      clipType: "full_page",
      visibleBody: buildBookmarkBody(pageTitle, pageUrl),
      fallbackPath: ["full_page", "bookmark"],
      actualType: "bookmark"
    }
  }

  if (requestedType === "screenshot") {
    return {
      clipType: "screenshot",
      visibleBody: screenshotDataUrl ? "[screenshot captured]" : "",
      fallbackPath: ["screenshot"],
      actualType: "screenshot",
      screenshotDataUrl
    }
  }

  return {
    clipType: "bookmark",
    visibleBody: buildBookmarkBody(pageTitle, pageUrl),
    fallbackPath: ["bookmark"],
    actualType: "bookmark"
  }
}

export const captureScreenshotClip = async (
  input: Omit<ClipCaptureInput, "requestedType"> & {
    requestedType?: "screenshot"
  }
): Promise<ClipCaptureResult> => {
  const screenshot = await getScreenshotFromCurrentTab()
  const result = resolveClipCaptureResolution({
    requestedType: "screenshot",
    pageUrl: input.pageUrl,
    pageTitle: input.pageTitle,
    selectionText: input.selectionText,
    articleText: input.articleText,
    fullPageText: input.fullPageText,
    screenshotDataUrl: screenshot.success ? screenshot.screenshot || undefined : undefined
  })

  return {
    ...input,
    requestedType: "screenshot",
    clipType: result.clipType,
    visibleBody: result.visibleBody,
    captureMetadata: result,
    userVisibleError: screenshot.success
      ? undefined
      : screenshot.error || "Failed to capture screenshot from the visible tab."
  }
}

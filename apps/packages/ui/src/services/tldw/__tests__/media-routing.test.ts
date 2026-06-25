import { describe, expect, it } from "vitest"
import {
  inferIngestTypeFromFilename,
  inferIngestTypeFromUrl,
  inferUploadMediaTypeFromFile
} from "../media-routing"

describe("media-routing upload inference", () => {
  it("routes Quick Ingest document-like uploads by filename when browser MIME is missing", () => {
    for (const filename of [
      "book.docx",
      "notes.rtf",
      "plain.txt",
      "markdown.md",
      "markdown.markdown",
      "page.html",
      "page.htm",
      "page.xhtml",
      "feed.xml",
      "data.json"
    ]) {
      expect(inferUploadMediaTypeFromFile(filename, "")).toBe("document")
    }
  })

  it("routes Ogg uploads as audio for common browser MIME detections", () => {
    expect(inferUploadMediaTypeFromFile("clip.ogg", "application/ogg")).toBe("audio")
    expect(inferUploadMediaTypeFromFile("clip", "application/ogg")).toBe("audio")
  })

  it("routes AVI uploads as video", () => {
    expect(inferUploadMediaTypeFromFile("movie.avi", "video/avi")).toBe("video")
  })

  it("keeps URL HTML routes on web scraping while file HTML uploads go through documents", () => {
    expect(inferIngestTypeFromUrl("https://example.test/page.html")).toBe("html")
    expect(inferIngestTypeFromFilename("page.html")).toBe("document")
  })
})

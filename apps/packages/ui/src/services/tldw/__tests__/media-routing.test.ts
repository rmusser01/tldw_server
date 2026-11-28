import { describe, expect, it } from "vitest"
import {
  inferIngestTypeFromFilename,
  inferIngestTypeFromUrl,
  inferUploadMediaTypeFromFile
} from "../media-routing"

describe("media-routing upload inference", () => {
  it.each([
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
  ])(
    "routes Quick Ingest document-like upload %s by filename when browser MIME is missing",
    (filename) => {
      expect(inferUploadMediaTypeFromFile(filename, "")).toBe("document")
    }
  )

  it("routes Ogg uploads as audio for common browser MIME detections", () => {
    expect(inferUploadMediaTypeFromFile("clip.ogg", "application/ogg")).toBe("audio")
    expect(inferUploadMediaTypeFromFile("clip", "application/ogg")).toBe("audio")
  })

  it("routes AVI uploads as video", () => {
    expect(inferUploadMediaTypeFromFile("movie.avi", "video/avi")).toBe("video")
  })

  it("routes OGM uploads and direct links as video", () => {
    expect(inferUploadMediaTypeFromFile("movie.OGM", "")).toBe("video")
    expect(inferUploadMediaTypeFromFile("movie.ogm", "application/ogg")).toBe("video")
    expect(inferIngestTypeFromUrl("https://example.test/movie.ogm?token=123")).toBe("video")
  })

  it("keeps URL HTML routes on web scraping while file HTML uploads go through documents", () => {
    expect(inferIngestTypeFromUrl("https://example.test/page.html")).toBe("html")
    expect(inferIngestTypeFromFilename("page.html")).toBe("document")
  })
})

// @vitest-environment jsdom
import { describe, expect, it } from "vitest"
import { shouldKeepOriginalFile } from "../media-routing"

describe("shouldKeepOriginalFile", () => {
  it("returns true for pdf media type", () => {
    expect(shouldKeepOriginalFile("pdf")).toBe(true)
  })

  it("returns true for ebook media type", () => {
    expect(shouldKeepOriginalFile("ebook")).toBe(true)
  })

  it("returns true for document media type", () => {
    expect(shouldKeepOriginalFile("document")).toBe(true)
  })

  it("returns false for audio media type", () => {
    expect(shouldKeepOriginalFile("audio")).toBe(false)
  })

  it("returns false for video media type", () => {
    expect(shouldKeepOriginalFile("video")).toBe(false)
  })

  it("returns false for html media type", () => {
    expect(shouldKeepOriginalFile("html")).toBe(false)
  })

  it("returns false for unknown types", () => {
    expect(shouldKeepOriginalFile("auto")).toBe(false)
  })

  it("normalizes uppercase input", () => {
    expect(shouldKeepOriginalFile("PDF")).toBe(true)
  })

  it("returns true for epub alias", () => {
    expect(shouldKeepOriginalFile("epub")).toBe(true)
  })
})

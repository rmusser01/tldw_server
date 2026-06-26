import { describe, expect, it } from "vitest"

import {
  ANNOTATION_CONTEXT_MAX_CHARS,
  buildSceneRangeAnnotationInput,
  captureAnnotationContext,
  codePointOffsetToUtf16Offset,
  utf16OffsetToCodePointOffset,
  validateSelectedRange
} from "../writing-annotation-anchor-utils"

describe("writing annotation anchor utilities", () => {
  it("clamps captured prefix and suffix context to 240 characters", () => {
    const prefix = "p".repeat(300)
    const suffix = "s".repeat(300)
    const context = captureAnnotationContext(`${prefix}selected${suffix}`, {
      start: prefix.length,
      end: prefix.length + "selected".length
    })

    expect(context.prefix).toHaveLength(ANNOTATION_CONTEXT_MAX_CHARS)
    expect(context.suffix).toHaveLength(ANNOTATION_CONTEXT_MAX_CHARS)
    expect(context.prefix).toBe("p".repeat(ANNOTATION_CONTEXT_MAX_CHARS))
    expect(context.suffix).toBe("s".repeat(ANNOTATION_CONTEXT_MAX_CHARS))
  })

  it("converts UTF-16 browser positions to Unicode code-point offsets with astral symbols", () => {
    const text = "A😀BC"

    expect(text.slice(0, 3)).toBe("A😀")
    expect(utf16OffsetToCodePointOffset(text, 0)).toBe(0)
    expect(utf16OffsetToCodePointOffset(text, 1)).toBe(1)
    expect(utf16OffsetToCodePointOffset(text, 3)).toBe(2)
    expect(utf16OffsetToCodePointOffset(text, text.length)).toBe(4)
  })

  it("converts Unicode code-point offsets back to UTF-16 browser positions", () => {
    const text = "A😀BC"

    expect(codePointOffsetToUtf16Offset(text, 0)).toBe(0)
    expect(codePointOffsetToUtf16Offset(text, 1)).toBe(1)
    expect(codePointOffsetToUtf16Offset(text, 2)).toBe(3)
    expect(codePointOffsetToUtf16Offset(text, 4)).toBe(text.length)
  })

  it("rejects empty selections", () => {
    const result = validateSelectedRange({
      documentText: "Scene text",
      selection: { start: 2, end: 2 },
      selectedText: ""
    })

    expect(result).toEqual({ ok: false, reason: "empty" })
  })

  it("rejects stale selections whose selected text no longer matches the document", () => {
    const result = validateSelectedRange({
      documentText: "Fresh scene text",
      selection: { start: 0, end: 5 },
      selectedText: "Stale"
    })

    expect(result).toEqual({ ok: false, reason: "stale" })
  })

  it("builds scene range input with code-point offsets, selected text, context, and fingerprint", () => {
    const input = buildSceneRangeAnnotationInput({
      canCreateRangeAnnotation: true,
      sceneId: "scene-1",
      sceneVersion: 7,
      documentText: "A😀 selected text",
      selection: { start: 1, end: 12 },
      category: "clarity",
      body: "Clarify this phrasing."
    })

    expect(input).toEqual(
      expect.objectContaining({
        target_type: "scene",
        target_id: "scene-1",
        category: "clarity",
        body: "Clarify this phrasing.",
        scene_version: 7,
        start: 1,
        end: 11,
        selected_text: "😀 selected"
      })
    )
    expect(input.metadata).toEqual(
      expect.objectContaining({
        anchor_prefix: "A",
        anchor_suffix: " text",
        anchor_fingerprint: expect.any(String)
      })
    )
  })

  it("throws when range input is requested without a saved scene binding", () => {
    expect(() =>
      buildSceneRangeAnnotationInput({
        canCreateRangeAnnotation: false,
        sceneId: "scene-1",
        sceneVersion: 7,
        documentText: "Scene text",
        selection: { start: 0, end: 5 },
        category: "other",
        body: "Note"
      })
    ).toThrow(/saved scene/i)
  })
})

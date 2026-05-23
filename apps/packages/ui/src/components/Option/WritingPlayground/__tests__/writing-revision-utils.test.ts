import { describe, expect, it } from "vitest"
import {
  buildInsertionAnchor,
  confirmRevisionTarget,
  countWords,
  createDocumentFingerprint,
  findParagraphRange,
  planRevisionApply,
  resolveRevisionTarget
} from "../writing-revision-utils"

const makeReplacementProposal = ({
  start,
  end,
  beforeText,
  replacementText,
  documentText = beforeText
}: {
  start: number
  end: number
  beforeText: string
  replacementText?: string
  documentText?: string
}) => ({
  id: "revision-1",
  sessionId: "session-1",
  action: "rewrite" as const,
  operation: "replace" as const,
  instruction: "Rewrite this text.",
  target: {
    mode: "paragraph" as const,
    start,
    end,
    beforeText,
    anchor: buildInsertionAnchor(documentText, start),
    label: "current paragraph",
    requiresConfirmation: false
  },
  ...(typeof replacementText === "string" ? { replacementText } : {}),
  createdAt: "2026-05-22T00:00:00.000Z",
  status: "pending" as const
})

const makeInsertionProposal = ({
  start,
  end,
  beforeText,
  replacementText,
  anchor
}: {
  start: number
  end: number
  beforeText: string
  replacementText: string
  anchor: ReturnType<typeof buildInsertionAnchor>
}) => ({
  id: "revision-2",
  sessionId: "session-1",
  action: "continue" as const,
  operation: "insert" as const,
  instruction: "Continue this text.",
  target: {
    mode: "cursor" as const,
    start,
    end,
    beforeText,
    anchor,
    label: "cursor",
    requiresConfirmation: false
  },
  replacementText,
  createdAt: "2026-05-22T00:00:00.000Z",
  status: "pending" as const
})

describe("writing revision utilities", () => {
  it("counts words and selected words deterministically", () => {
    expect(countWords("One two\nthree.")).toBe(3)
    expect(countWords("   ")).toBe(0)
  })

  it("resolves the current paragraph around a cursor", () => {
    const text = "Alpha one.\n\nBeta two.\nBeta three."
    expect(findParagraphRange(text, 14)).toEqual({ start: 12, end: 33 })
  })

  it("resolves leading blank paragraphs to the first content paragraph", () => {
    expect(findParagraphRange("\n\nAlpha", 0)).toEqual({ start: 2, end: 7 })
  })

  it("keeps a cursor at paragraph end on the preceding paragraph", () => {
    expect(findParagraphRange("Alpha\n\nBeta", 5)).toEqual({
      start: 0,
      end: 5
    })
  })

  it("keeps a cursor inside a paragraph delimiter on the preceding paragraph", () => {
    expect(findParagraphRange("Alpha\n\nBeta", 6)).toEqual({
      start: 0,
      end: 5
    })
  })

  it("plans a direct replacement when beforeText still matches", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega",
      documentText: "Alpha beta"
    })
    expect(planRevisionApply("Alpha beta", proposal)).toEqual({
      type: "apply",
      start: 0,
      end: 5,
      nextText: "Omega beta"
    })
  })

  it("noops when a replacement proposal has no replacement text", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      documentText: "Alpha beta"
    })
    expect(planRevisionApply("Alpha beta", proposal)).toMatchObject({
      type: "noop"
    })
  })

  it("conflicts when replacement target drift is ambiguous", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega",
      documentText: "Alpha beta"
    })
    expect(planRevisionApply("Intro Alpha beta Alpha", proposal).type).toBe(
      "conflict"
    )
  })

  it("retargets a zero-length insertion by unique prefix and suffix anchor", () => {
    const original = "Alpha beta"
    const anchor = buildInsertionAnchor(original, 5)
    const proposal = makeInsertionProposal({
      start: 5,
      end: 5,
      beforeText: "",
      replacementText: " brave",
      anchor
    })
    expect(planRevisionApply("Intro. Alpha beta", proposal)).toEqual({
      type: "retarget",
      start: 12,
      end: 12,
      nextText: "Intro. Alpha brave beta"
    })
  })

  it("conflicts when an insert proposal has a non-zero target", () => {
    const proposal = makeInsertionProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: " brave",
      anchor: buildInsertionAnchor("Alpha beta", 0)
    })
    expect(planRevisionApply("Alpha beta", proposal)).toMatchObject({
      type: "conflict",
      reason: expect.stringContaining("Insert")
    })
  })

  it("does not treat empty beforeText as a safe insertion match after drift", () => {
    const proposal = makeInsertionProposal({
      start: 5,
      end: 5,
      beforeText: "",
      replacementText: " brave",
      anchor: {
        documentFingerprint: createDocumentFingerprint("Alpha beta"),
        prefix: "Alpha",
        suffix: " beta"
      }
    })
    expect(planRevisionApply("Completely different", proposal).type).toBe(
      "conflict"
    )
  })

  it("targets the whole document for advisory outline requests", () => {
    const text = "First paragraph.\n\nSecond paragraph."
    const target = resolveRevisionTarget({
      text,
      action: "outline",
      operation: "advisory",
      cursor: 3
    })
    expect(target).toMatchObject({
      mode: "document",
      start: 0,
      end: text.length,
      requiresConfirmation: false
    })
  })

  it("conflicts when a text-changing target still requires confirmation", () => {
    const text = "First paragraph.\n\nSecond paragraph."
    const target = resolveRevisionTarget({
      text,
      action: "rewrite",
      operation: "replace",
      cursor: 3,
      preferredTargetMode: "document"
    })
    const proposal = {
      ...makeReplacementProposal({
        start: target.start,
        end: target.end,
        beforeText: target.beforeText,
        replacementText: "Updated document.",
        documentText: text
      }),
      target
    }
    expect(planRevisionApply(text, proposal)).toMatchObject({
      type: "conflict"
    })
  })

  it("requires confirmation before large text-changing document targets", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "rewrite",
      operation: "replace",
      cursor: 3,
      preferredTargetMode: "document"
    })
    expect(target).toMatchObject({
      mode: "document",
      requiresConfirmation: true
    })
  })

  it("allows apply after a whole-document text-changing target is confirmed", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "rewrite",
      operation: "replace",
      cursor: 3,
      preferredTargetMode: "document"
    })
    const confirmed = confirmRevisionTarget(target)
    expect(confirmed.requiresConfirmation).toBe(false)
    expect(confirmed.confirmationReason).toBeUndefined()
  })

  it("surfaces the resolved target for custom requests before generation", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "custom",
      operation: "replace",
      cursor: 20
    })
    expect(target.mode).toBe("paragraph")
    expect(target.label).toContain("paragraph")
  })
})

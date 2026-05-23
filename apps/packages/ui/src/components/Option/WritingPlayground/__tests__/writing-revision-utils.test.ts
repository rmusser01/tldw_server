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
  replacementText
}: {
  start: number
  end: number
  beforeText: string
  replacementText: string
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
    anchor: {
      documentFingerprint: createDocumentFingerprint(beforeText),
      prefix: "",
      suffix: ""
    },
    label: "current paragraph",
    requiresConfirmation: false
  },
  replacementText,
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

  it("plans a direct replacement when beforeText still matches", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega"
    })
    expect(planRevisionApply("Alpha beta", proposal)).toEqual({
      type: "apply",
      start: 0,
      end: 5,
      nextText: "Omega beta"
    })
  })

  it("conflicts when replacement target drift is ambiguous", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega"
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
    expect(planRevisionApply("Intro. Alpha beta", proposal)).toMatchObject({
      type: "retarget"
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

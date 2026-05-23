import type { JSONContent } from "@tiptap/react"
import { describe, expect, it } from "vitest"
import {
  DEFAULT_SETTINGS,
  getRevisionPayloadSignature,
  getRevisionPresetIdFromPayload,
  getRevisionsFromPayload,
  getPromptRichFromPayload,
  mergePendingPayloadIntoSession,
  mergeRevisionsIntoPayload,
  mergePayloadIntoSession
} from "../hooks/utils"
import type { WritingRevisionProposal } from "../writing-revision-types"

const RICH_DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "Rich draft" }]
    }
  ]
}

const buildRevision = (
  overrides: Partial<WritingRevisionProposal> = {}
): WritingRevisionProposal => ({
  id: "revision-1",
  sessionId: "session-1",
  action: "rewrite",
  operation: "replace",
  presetId: "polish_prose",
  presetInstruction: "Polish without changing meaning.",
  instruction: "Make this clearer.",
  target: {
    mode: "selection",
    start: 0,
    end: 11,
    beforeText: "Hello world",
    anchor: {
      documentFingerprint: "fingerprint-1",
      prefix: "",
      suffix: ""
    },
    label: "Selection",
    requiresConfirmation: false
  },
  replacementText: "Hello there",
  createdAt: "2026-05-22T12:00:00.000Z",
  status: "pending",
  ...overrides
})

describe("writing session payload utils", () => {
  it("stores prompt_rich when rich content is supplied", () => {
    const payload = mergePayloadIntoSession({}, "Rich draft", DEFAULT_SETTINGS, null, null, false, {
      promptRich: RICH_DOC
    })

    expect(payload.prompt).toBe("Rich draft")
    expect(payload.prompt_rich).toEqual(RICH_DOC)
  })

  it("clears prompt_rich on plain-text prompt updates", () => {
    const payload = mergePayloadIntoSession(
      { prompt: "old", prompt_rich: RICH_DOC },
      "Plain replacement",
      DEFAULT_SETTINGS,
      null,
      null,
      false,
      { promptRich: null }
    )

    expect(payload.prompt).toBe("Plain replacement")
    expect(payload).not.toHaveProperty("prompt_rich")
  })

  it("returns null for malformed prompt_rich payloads", () => {
    expect(getPromptRichFromPayload({ prompt_rich: "bad" })).toBeNull()
  })

  it("preserves prompt and settings when merging revisions into the payload", () => {
    const revision = buildRevision()
    const payload = mergeRevisionsIntoPayload(
      {
        prompt: "Existing draft",
        settings: DEFAULT_SETTINGS,
        template_name: "story",
        chat_mode: false
      },
      [revision]
    )

    expect(payload.prompt).toBe("Existing draft")
    expect(payload.settings).toBe(DEFAULT_SETTINGS)
    expect(payload.revisions?.items).toEqual([revision])
  })

  it("stores revisions as a schema-versioned object instead of a bare array", () => {
    const revision = buildRevision()
    const payload = mergeRevisionsIntoPayload({}, [revision])

    expect(payload.revisions).toEqual({
      schemaVersion: 1,
      items: [revision]
    })
    expect(Array.isArray(payload.revisions)).toBe(false)
  })

  it("ignores malformed revisions in session payloads", () => {
    const revision = buildRevision()

    expect(getRevisionsFromPayload({ revisions: [revision] })).toEqual([])
    expect(
      getRevisionsFromPayload({
        revisions: { schemaVersion: 2, items: [revision] }
      })
    ).toEqual([])
    expect(
      getRevisionsFromPayload({
        revisions: { schemaVersion: 1, items: "bad" }
      })
    ).toEqual([])
    expect(
      getRevisionsFromPayload({
        revisions: {
          schemaVersion: 1,
          items: [revision, { ...revision, target: null }]
        }
      })
    ).toEqual([revision])
  })

  it("ignores revisions with malformed target offsets", () => {
    const target = buildRevision().target
    const malformedTargets = [
      { ...target, start: -1, end: 11 },
      { ...target, start: 0.5, end: 11 },
      { ...target, start: 0, end: 10.5 },
      { ...target, start: 11, end: 0 }
    ]

    for (const malformedTarget of malformedTargets) {
      expect(
        getRevisionsFromPayload({
          revisions: {
            schemaVersion: 1,
            items: [buildRevision({ target: malformedTarget })]
          }
        })
      ).toEqual([])
    }
  })

  it("preserves a known revision workflow preset id and rejects unknown ids", () => {
    const payload = mergePayloadIntoSession(
      { revision_preset_id: "preserve_voice" },
      "Draft",
      DEFAULT_SETTINGS,
      null,
      null,
      false
    )

    expect(getRevisionPresetIdFromPayload(payload)).toBe("preserve_voice")
    expect(
      getRevisionPresetIdFromPayload({ revision_preset_id: "unknown_preset" })
    ).toBeNull()
  })

  it("changes revision payload signatures when proposal status changes", () => {
    const pendingSignature = getRevisionPayloadSignature({
      revisions: { schemaVersion: 1, items: [buildRevision({ status: "pending" })] }
    })
    const appliedSignature = getRevisionPayloadSignature({
      revisions: { schemaVersion: 1, items: [buildRevision({ status: "applied" })] }
    })

    expect(pendingSignature).not.toBe(appliedSignature)
  })

  it("removes the revisions field when all revisions are cleared", () => {
    const payload = mergeRevisionsIntoPayload(
      {
        prompt: "Existing draft",
        revisions: { schemaVersion: 1, items: [buildRevision()] }
      },
      []
    )

    expect(payload.prompt).toBe("Existing draft")
    expect(payload).not.toHaveProperty("revisions")
    expect(getRevisionsFromPayload(payload)).toEqual([])
  })

  it("uses pending payload state when merging prompt edits into a session payload", () => {
    const revision = buildRevision()
    const activePayload = {
      prompt: "Server draft",
      settings: DEFAULT_SETTINGS,
      template_name: null,
      theme_name: null,
      chat_mode: false
    }
    const pendingPayload = mergeRevisionsIntoPayload(activePayload, [revision])

    const payload = mergePendingPayloadIntoSession(
      activePayload,
      pendingPayload,
      "Typed draft",
      DEFAULT_SETTINGS,
      null,
      null,
      false,
      { promptRich: null }
    )

    expect(payload.prompt).toBe("Typed draft")
    expect(payload.revisions?.items).toEqual([revision])
  })
})

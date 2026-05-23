import { act, renderHook } from "@testing-library/react"
import { JSDOM } from "jsdom"
import { afterAll, describe, expect, it, vi } from "vitest"
import { useWritingRevisions } from "../hooks/useWritingRevisions"
import type { WritingSessionPayload } from "../hooks/utils"
import { mergeRevisionsIntoPayload } from "../hooks/utils"
import type { WritingRevisionProposal } from "../writing-revision-types"

const dom = new JSDOM("<!doctype html><html><body></body></html>")

Object.defineProperties(globalThis, {
  window: { value: dom.window, configurable: true },
  document: { value: dom.window.document, configurable: true },
  navigator: { value: dom.window.navigator, configurable: true },
  HTMLElement: { value: dom.window.HTMLElement, configurable: true },
  MutationObserver: {
    value: dom.window.MutationObserver,
    configurable: true
  },
  Node: { value: dom.window.Node, configurable: true }
})

afterAll(() => {
  dom.window.close()
})

const target: WritingRevisionProposal["target"] = {
  mode: "selection",
  start: 0,
  end: 5,
  beforeText: "Alpha",
  anchor: {
    documentFingerprint: "fingerprint-1",
    prefix: "",
    suffix: " beta"
  },
  label: "selection",
  requiresConfirmation: false
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
  target,
  replacementText: "Omega",
  createdAt: "2026-05-22T12:00:00.000Z",
  status: "pending",
  ...overrides
})

const setup = (
  overrides: Partial<Parameters<typeof useWritingRevisions>[0]> = {}
) => {
  const applyEditorText = vi.fn(() => ({ applied: true as const }))
  const applySessionPayloadPatch = vi.fn()

  const result = renderHook(
    (props: Parameters<typeof useWritingRevisions>[0]) =>
      useWritingRevisions(props),
    {
      initialProps: {
        activeSessionId: "session-1",
        activeSessionPayload: null,
        editorText: "Alpha beta",
        applyEditorText,
        applySessionPayloadPatch,
        ...overrides
      }
    }
  )

  return {
    ...result,
    applyEditorText,
    applySessionPayloadPatch
  }
}

const latestPersistedRevisions = (
  applySessionPayloadPatch: ReturnType<typeof vi.fn>
): WritingRevisionProposal[] => {
  const patcher = applySessionPayloadPatch.mock.calls.at(-1)?.[0] as
    | ((payload: WritingSessionPayload) => WritingSessionPayload)
    | undefined

  if (!patcher) return []
  return patcher({ prompt: "Existing draft" }).revisions?.items ?? []
}

const createDeferred = <T,>() => {
  let resolve: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return {
    promise,
    resolve: resolve!
  }
}

describe("useWritingRevisions", () => {
  it("loads revisions from the active session payload", () => {
    const revision = buildRevision()
    const payload = mergeRevisionsIntoPayload({ prompt: "Alpha beta" }, [revision])

    const { result, rerender } = setup({
      activeSessionId: "session-1",
      activeSessionPayload: payload
    })

    expect(result.current.revisions).toEqual([revision])

    const nextRevision = buildRevision({ id: "revision-2" })
    rerender({
      activeSessionId: "session-2",
      activeSessionPayload: mergeRevisionsIntoPayload({}, [nextRevision]),
      editorText: "Alpha beta",
      applyEditorText: vi.fn(() => ({ applied: true as const })),
      applySessionPayloadPatch: vi.fn()
    })

    expect(result.current.revisions).toEqual([nextRevision])
  })

  it("rejects a proposal without changing text", () => {
    const revision = buildRevision()
    const { result, applyEditorText, applySessionPayloadPatch } = setup()

    act(() => {
      result.current.addRevision(revision)
      result.current.rejectRevision(revision.id)
    })

    expect(applyEditorText).not.toHaveBeenCalled()
    expect(result.current.revisions[0]).toMatchObject({ status: "rejected" })
    expect(latestPersistedRevisions(applySessionPayloadPatch)[0]).toMatchObject({
      status: "rejected"
    })
  })

  it("applies a plain replacement through the provided text callback", () => {
    const revision = buildRevision()
    const { result, applyEditorText, applySessionPayloadPatch } = setup()

    act(() => {
      result.current.addRevision(revision)
      result.current.applyRevision(revision.id)
    })

    expect(applyEditorText).toHaveBeenCalledWith("Omega beta")
    expect(result.current.revisions[0]).toMatchObject({ status: "applied" })
    expect(latestPersistedRevisions(applySessionPayloadPatch)[0]).toMatchObject({
      status: "applied"
    })
  })

  it("marks conflict without mutating text", () => {
    const revision = buildRevision({
      target: {
        ...target,
        start: 0,
        end: 5,
        beforeText: "Alpha"
      }
    })
    const { result, applyEditorText, applySessionPayloadPatch } = setup({
      editorText: "Intro Alpha beta Alpha"
    })

    act(() => {
      result.current.addRevision(revision)
      result.current.applyRevision(revision.id)
    })

    expect(applyEditorText).not.toHaveBeenCalled()
    expect(result.current.revisions[0]).toMatchObject({ status: "conflict" })
    expect(latestPersistedRevisions(applySessionPayloadPatch)[0]).toMatchObject({
      status: "conflict"
    })
  })

  it("does not call the text callback for advisory proposals", () => {
    const revision = buildRevision({
      operation: "advisory",
      replacementText: undefined,
      rawText: "Consider making the stakes clearer."
    })
    const { result, applyEditorText } = setup()

    act(() => {
      result.current.addRevision(revision)
      result.current.applyRevision(revision.id)
    })

    expect(applyEditorText).not.toHaveBeenCalled()
    expect(result.current.revisions[0]).toMatchObject({ status: "advisory" })
  })

  it("passes proposal state to applySessionPayloadPatch", () => {
    const revision = buildRevision()
    const { result, applySessionPayloadPatch } = setup()

    act(() => {
      result.current.addRevision(revision)
      result.current.rejectRevision(revision.id)
    })

    const patcher = applySessionPayloadPatch.mock.calls.at(-1)?.[0] as (
      payload: WritingSessionPayload
    ) => WritingSessionPayload
    const patched = patcher({ prompt: "Existing draft" })

    expect(patched.prompt).toBe("Existing draft")
    expect(patched.revisions?.items).toEqual([
      expect.objectContaining({ id: revision.id, status: "rejected" })
    ])
  })

  it("regenerates by rejecting the source and appending a pending replacement", async () => {
    const source = buildRevision({
      id: "source",
      presetId: "preserve_voice",
      presetInstruction: "Preserve voice.",
      instruction: "Keep the rhythm."
    })
    const replacement = buildRevision({
      id: "replacement",
      target: {
        ...target,
        beforeText: "Different"
      },
      instruction: "Different instruction",
      presetId: "make_concise",
      presetInstruction: "Different preset.",
      status: "applied"
    })
    const createReplacement = vi.fn(async () => replacement)
    const { result, applySessionPayloadPatch } = setup()

    await act(async () => {
      result.current.addRevision(source)
      await result.current.regenerateRevision(source.id, createReplacement)
    })

    expect(createReplacement).toHaveBeenCalledWith(source)
    expect(result.current.revisions).toEqual([
      expect.objectContaining({ id: source.id, status: "rejected" }),
      expect.objectContaining({
        id: replacement.id,
        regeneratedFromId: source.id,
        target: source.target,
        instruction: source.instruction,
        presetId: source.presetId,
        presetInstruction: source.presetInstruction,
        status: "pending"
      })
    ])
    expect(latestPersistedRevisions(applySessionPayloadPatch)).toHaveLength(2)
  })

  it("keeps a pending regeneration after a same-payload save echo", async () => {
    const source = buildRevision({ id: "source" })
    const replacement = buildRevision({ id: "replacement" })
    const deferred = createDeferred<WritingRevisionProposal>()
    const createReplacement = vi.fn(() => deferred.promise)
    const payload = mergeRevisionsIntoPayload({}, [source])
    const { result, rerender, applyEditorText, applySessionPayloadPatch } = setup({
      activeSessionPayload: payload
    })

    await act(async () => {
      void result.current.regenerateRevision(source.id, createReplacement)
      await Promise.resolve()
    })

    rerender({
      activeSessionId: "session-1",
      activeSessionPayload: mergeRevisionsIntoPayload({}, [{ ...source }]),
      editorText: "Alpha beta",
      applyEditorText,
      applySessionPayloadPatch
    })

    await act(async () => {
      deferred.resolve(replacement)
      await deferred.promise
    })

    expect(result.current.revisions).toEqual([
      expect.objectContaining({ id: source.id, status: "rejected" }),
      expect.objectContaining({
        id: replacement.id,
        regeneratedFromId: source.id,
        status: "pending"
      })
    ])
    expect(latestPersistedRevisions(applySessionPayloadPatch)).toHaveLength(2)
  })

  it("ignores a pending regeneration when the active session changes before completion", async () => {
    const source = buildRevision({ id: "source" })
    const replacement = buildRevision({
      id: "replacement",
      sessionId: "session-1"
    })
    const nextSessionRevision = buildRevision({
      id: "next-session",
      sessionId: "session-2"
    })
    const deferred = createDeferred<WritingRevisionProposal>()
    const createReplacement = vi.fn(() => deferred.promise)
    const applyEditorText = vi.fn(() => ({ applied: true as const }))
    const applySessionPayloadPatch = vi.fn()
    const nextSessionPayloadPatch = vi.fn()
    const { result, rerender } = setup({
      applyEditorText,
      applySessionPayloadPatch
    })

    await act(async () => {
      result.current.addRevision(source)
    })
    const callsBeforeRegenerationCompletes =
      applySessionPayloadPatch.mock.calls.length

    await act(async () => {
      void result.current.regenerateRevision(source.id, createReplacement)
      await Promise.resolve()
    })

    rerender({
      activeSessionId: "session-2",
      activeSessionPayload: mergeRevisionsIntoPayload({}, [nextSessionRevision]),
      editorText: "Session two text",
      applyEditorText,
      applySessionPayloadPatch: nextSessionPayloadPatch
    })

    await act(async () => {
      deferred.resolve(replacement)
      await deferred.promise
    })

    expect(result.current.revisions).toEqual([nextSessionRevision])
    expect(applySessionPayloadPatch).toHaveBeenCalledTimes(
      callsBeforeRegenerationCompletes
    )
    expect(nextSessionPayloadPatch).not.toHaveBeenCalled()
  })

  it("ignores a pending regeneration when the source is no longer pending", async () => {
    const source = buildRevision({ id: "source" })
    const replacement = buildRevision({ id: "replacement" })
    const deferred = createDeferred<WritingRevisionProposal>()
    const createReplacement = vi.fn(() => deferred.promise)
    const { result, applySessionPayloadPatch } = setup()

    await act(async () => {
      result.current.addRevision(source)
    })

    await act(async () => {
      void result.current.regenerateRevision(source.id, createReplacement)
      await Promise.resolve()
    })

    await act(async () => {
      result.current.rejectRevision(source.id)
    })
    const callsBeforeRegenerationCompletes =
      applySessionPayloadPatch.mock.calls.length

    await act(async () => {
      deferred.resolve(replacement)
      await deferred.promise
    })

    expect(result.current.revisions).toEqual([
      expect.objectContaining({ id: source.id, status: "rejected" })
    ])
    expect(latestPersistedRevisions(applySessionPayloadPatch)).toEqual([
      expect.objectContaining({ id: source.id, status: "rejected" })
    ])
    expect(applySessionPayloadPatch).toHaveBeenCalledTimes(
      callsBeforeRegenerationCompletes
    )
  })

  it("marks unsupported rich-editor apply responses as conflict for manual apply", () => {
    const revision = buildRevision()
    const applyEditorText = vi.fn(() => ({
      applied: false as const,
      reason: "Rich editor patching is not supported."
    }))
    const { result } = setup({ applyEditorText })

    act(() => {
      result.current.addRevision(revision)
      result.current.applyRevision(revision.id)
    })

    expect(applyEditorText).toHaveBeenCalledWith("Omega beta")
    expect(result.current.revisions[0]).toMatchObject({
      status: "conflict",
      notes: expect.arrayContaining([
        expect.stringContaining("Rich editor patching is not supported.")
      ])
    })
  })
})

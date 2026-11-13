/**
 * NL-07: opening a note from a search scrolls its editor to the first hit.
 */
import React from "react"
import { renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { findFirstSearchHit, useRevealFirstSearchHit } from "../useRevealFirstSearchHit"

const mocks = vi.hoisted(() => ({ reveal: vi.fn() }))

vi.mock("../../notes-manager-utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../notes-manager-utils")>()
  return { ...actual, revealTextareaOffset: mocks.reveal }
})

type Props = {
  selectedId: string | null
  content: string
  query: string
  enabled: boolean
}

const BODY = "Intro line.\nMore text here.\nLater we compare FAISS and others."

describe("useRevealFirstSearchHit (NL-07)", () => {
  let textarea: HTMLTextAreaElement
  let textareaRef: React.MutableRefObject<HTMLTextAreaElement | null>

  beforeEach(() => {
    mocks.reveal.mockReset()
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      callback(0)
      return 0
    })
    textarea = document.createElement("textarea")
    document.body.append(textarea)
    textareaRef = { current: textarea }
  })

  afterEach(() => {
    textarea.remove()
    vi.restoreAllMocks()
  })

  const renderReveal = (initial: Props) =>
    renderHook((props: Props) => useRevealFirstSearchHit({ ...props, textareaRef }), {
      initialProps: initial
    })

  it("scrolls to the first hit when a note opens during a search", () => {
    const { rerender } = renderReveal({ selectedId: null, content: "", query: "faiss", enabled: true })
    rerender({ selectedId: "n1", content: BODY, query: "faiss", enabled: true })

    expect(mocks.reveal).toHaveBeenCalledTimes(1)
    expect(mocks.reveal).toHaveBeenCalledWith(textarea, BODY.indexOf("FAISS"))
  })

  it("does nothing without a query, without a hit, or when disabled", () => {
    const { rerender } = renderReveal({ selectedId: null, content: "", query: "", enabled: true })
    rerender({ selectedId: "n1", content: BODY, query: "", enabled: true })
    rerender({ selectedId: "n2", content: BODY, query: "pgvector", enabled: true })
    rerender({ selectedId: "n3", content: BODY, query: "faiss", enabled: false })

    expect(mocks.reveal).not.toHaveBeenCalled()
  })

  it("leaves the editor alone while the user types in it", () => {
    textarea.focus()
    const { rerender } = renderReveal({ selectedId: null, content: BODY, query: "faiss", enabled: true })
    // Saving a new draft gives it an id; that is not opening a note.
    rerender({ selectedId: "new-note", content: BODY, query: "faiss", enabled: true })

    expect(mocks.reveal).not.toHaveBeenCalled()
  })

  it("does not scroll again while the same note stays open", () => {
    const { rerender } = renderReveal({ selectedId: "n1", content: BODY, query: "faiss", enabled: true })
    expect(mocks.reveal).toHaveBeenCalledTimes(1)
    rerender({ selectedId: "n1", content: `${BODY} faiss`, query: "faiss", enabled: true })

    expect(mocks.reveal).toHaveBeenCalledTimes(1)
  })
})

describe("findFirstSearchHit", () => {
  it("finds the whole query first, ignoring case", () => {
    expect(findFirstSearchHit("a model; the Open Source model", "open source")).toBe(13)
  })

  it("falls back to the earliest single term", () => {
    expect(findFirstSearchHit("source first, then open", "open source")).toBe(0)
    expect(findFirstSearchHit("nothing here", "open source")).toBe(-1)
  })

  it("reads a quoted phrase as one term", () => {
    expect(findFirstSearchHit('x "open source" y', '"open source"')).toBe(3)
  })
})

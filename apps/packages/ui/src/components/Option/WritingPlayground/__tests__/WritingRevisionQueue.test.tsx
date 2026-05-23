import React from "react"
import { cleanup, fireEvent, render, within } from "@testing-library/react"
import { JSDOM } from "jsdom"
import { afterAll, afterEach, describe, expect, it, vi } from "vitest"
import { WritingRevisionQueue } from "../WritingRevisionQueue"
import type { WritingRevisionProposal } from "../writing-revision-types"

const dom = new JSDOM("<!doctype html><html><body></body></html>")
const requestAnimationFrame = (callback: FrameRequestCallback) =>
  dom.window.setTimeout(() => callback(Date.now()), 0)
const cancelAnimationFrame = (id: number) => {
  dom.window.clearTimeout(id)
}

Object.defineProperties(globalThis, {
  window: { value: dom.window, configurable: true },
  document: { value: dom.window.document, configurable: true },
  navigator: { value: dom.window.navigator, configurable: true },
  HTMLElement: { value: dom.window.HTMLElement, configurable: true },
  SVGElement: { value: dom.window.SVGElement, configurable: true },
  Element: { value: dom.window.Element, configurable: true },
  ShadowRoot: { value: dom.window.ShadowRoot, configurable: true },
  Node: { value: dom.window.Node, configurable: true },
  MutationObserver: {
    value: dom.window.MutationObserver,
    configurable: true
  },
  requestAnimationFrame: {
    value: requestAnimationFrame,
    configurable: true
  },
  cancelAnimationFrame: {
    value: cancelAnimationFrame,
    configurable: true
  },
  getComputedStyle: {
    value: dom.window.getComputedStyle.bind(dom.window),
    configurable: true
  }
})

Object.assign(dom.window, {
  requestAnimationFrame,
  cancelAnimationFrame
})

afterAll(() => {
  dom.window.close()
})

afterEach(() => {
  cleanup()
})

const target: WritingRevisionProposal["target"] = {
  mode: "selection",
  start: 0,
  end: 11,
  beforeText: "Old wording",
  anchor: {
    documentFingerprint: "fingerprint-1",
    prefix: "",
    suffix: ""
  },
  label: "Selection: 11 chars",
  requiresConfirmation: false
}

const buildProposal = (
  overrides: Partial<WritingRevisionProposal> = {}
): WritingRevisionProposal => ({
  id: "revision-1",
  sessionId: "session-1",
  action: "rewrite",
  operation: "replace",
  presetId: "polish_prose",
  presetInstruction: "Improve flow.",
  instruction: "Rewrite this.",
  target,
  replacementText: "New wording",
  rationale: "Clearer.",
  title: "Rewrite selection",
  createdAt: "2026-05-22T12:00:00.000Z",
  status: "pending",
  ...overrides
})

describe("WritingRevisionQueue", () => {
  it("renders pending proposal diff and Apply/Reject/Copy/Regenerate", () => {
    const view = render(
      <WritingRevisionQueue
        proposals={[buildProposal()]}
        onApply={vi.fn()}
        onReject={vi.fn()}
        onCopy={vi.fn()}
        onRegenerate={vi.fn()}
      />
    )

    const queue = view.getByTestId("writing-revision-queue")
    expect(within(queue).getByText("Old wording")).toBeTruthy()
    expect(within(queue).getByText("New wording")).toBeTruthy()
    expect(within(queue).getByRole("button", { name: /apply/i })).toBeTruthy()
    expect(within(queue).getByRole("button", { name: /reject/i })).toBeTruthy()
    expect(within(queue).getByRole("button", { name: /copy/i })).toBeTruthy()
    expect(
      within(queue).getByRole("button", { name: /regenerate/i })
    ).toBeTruthy()
    expect(
      within(queue).getByRole("button", { name: /apply/i }).querySelector("svg")
    ).toBeTruthy()
    expect(
      within(queue).getByRole("button", { name: /copy/i }).querySelector("svg")
    ).toBeTruthy()
  })

  it("hides Apply for advisory proposals", () => {
    const view = render(
      <WritingRevisionQueue
        proposals={[
          buildProposal({
            operation: "advisory",
            status: "advisory",
            replacementText: undefined,
            rawText: "Consider strengthening the stakes."
          })
        ]}
        onApply={vi.fn()}
        onReject={vi.fn()}
        onCopy={vi.fn()}
        onRegenerate={vi.fn()}
      />
    )

    expect(view.queryByRole("button", { name: /apply/i })).not.toBeTruthy()
    expect(view.getByText("Consider strengthening the stakes.")).toBeTruthy()
  })

  it("shows copy/manual-apply guidance for conflict state", () => {
    const view = render(
      <WritingRevisionQueue
        proposals={[
          buildProposal({
            status: "conflict",
            notes: ["Anchor moved after the draft changed."]
          })
        ]}
        onApply={vi.fn()}
        onReject={vi.fn()}
        onCopy={vi.fn()}
        onRegenerate={vi.fn()}
      />
    )

    expect(view.getByText(/copy the suggestion and apply it manually/i)).toBeTruthy()
    expect(view.getByText("Anchor moved after the draft changed.")).toBeTruthy()
  })

  it("shows raw text and Copy only for raw suggestion state", () => {
    const onCopy = vi.fn()

    const view = render(
      <WritingRevisionQueue
        proposals={[
          buildProposal({
            status: "raw_suggestion",
            operation: "advisory",
            replacementText: undefined,
            rawText: "Raw model output"
          })
        ]}
        onApply={vi.fn()}
        onReject={vi.fn()}
        onCopy={onCopy}
        onRegenerate={vi.fn()}
      />
    )

    expect(view.getByText("Raw model output")).toBeTruthy()
    expect(view.getByRole("button", { name: /copy/i })).toBeTruthy()
    expect(view.queryByRole("button", { name: /apply/i })).not.toBeTruthy()
    expect(view.queryByRole("button", { name: /reject/i })).not.toBeTruthy()
    expect(view.queryByRole("button", { name: /regenerate/i })).not.toBeTruthy()

    fireEvent.click(view.getByRole("button", { name: /copy/i }))

    expect(onCopy).toHaveBeenCalledWith(expect.objectContaining({ id: "revision-1" }))
  })
})

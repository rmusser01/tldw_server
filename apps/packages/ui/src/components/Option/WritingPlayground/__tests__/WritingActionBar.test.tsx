import React from "react"
import { cleanup, fireEvent, render } from "@testing-library/react"
import { JSDOM } from "jsdom"
import { afterAll, afterEach, describe, expect, it, vi } from "vitest"
import { WritingActionBar } from "../WritingActionBar"
import { WRITING_REVISION_PRESETS } from "../writing-revision-presets"
import type {
  WritingRevisionAction,
  WritingRevisionTarget
} from "../writing-revision-types"

const dom = new JSDOM("<!doctype html><html><body></body></html>")

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
  ResizeObserver: {
    value: class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
    configurable: true
  },
  getComputedStyle: {
    value: dom.window.getComputedStyle.bind(dom.window),
    configurable: true
  }
})

afterAll(() => {
  dom.window.close()
})

afterEach(() => {
  cleanup()
})

const selectionTarget: WritingRevisionTarget = {
  mode: "selection",
  start: 0,
  end: 16,
  beforeText: "The old sentence",
  anchor: {
    documentFingerprint: "fingerprint-1",
    prefix: "",
    suffix: ""
  },
  label: "Selection: 16 chars",
  requiresConfirmation: false
}

const documentTarget: WritingRevisionTarget = {
  ...selectionTarget,
  mode: "document",
  start: 0,
  end: 1200,
  beforeText: "Whole document",
  label: "Whole document: 1,200 chars",
  requiresConfirmation: true,
  confirmationReason: "This will rewrite the full draft."
}

describe("WritingActionBar", () => {
  it("disables actions when generation is unavailable", () => {
    const onRequest = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable={false}
        target={selectionTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(view.getByRole("button", { name: /rewrite/i }))

    expect(onRequest).not.toHaveBeenCalled()
    expect(
      (view.getByRole("button", { name: /rewrite/i }) as HTMLButtonElement)
        .disabled
    ).toBe(true)
    expect(view.getByText(/generation unavailable/i)).toBeTruthy()
  })

  it("renders the six workflow presets and shows the selected preset instruction", () => {
    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        onRequest={vi.fn()}
      />
    )

    for (const preset of WRITING_REVISION_PRESETS) {
      expect(view.getByText(preset.label)).toBeTruthy()
    }

    fireEvent.click(view.getByRole("radio", { name: /make concise/i }))

    expect(
      view.getByText(WRITING_REVISION_PRESETS[4].instruction)
    ).toBeTruthy()
  })

  it("shows the resolved target summary before sending Custom requests", () => {
    const onRequest = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        onRequest={onRequest}
      />
    )

    expect(view.getByText("Selection: 16 chars")).toBeTruthy()

    fireEvent.click(view.getByRole("button", { name: /custom/i }))
    fireEvent.change(view.getByRole("textbox", { name: /custom instruction/i }), {
      target: { value: "Make this more direct." }
    })
    fireEvent.click(view.getByRole("button", { name: /send custom/i }))

    expect(onRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "custom",
        instruction: "Make this more direct.",
        target: selectionTarget
      })
    )
  })

  it("requires explicit confirmation before sending whole-document text-changing requests", () => {
    const onRequest = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable
        target={documentTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(view.getByRole("button", { name: /rewrite/i }))
    expect(onRequest).not.toHaveBeenCalled()
    expect(view.getAllByText(/this will rewrite the full draft/i).length).toBe(2)

    fireEvent.click(
      view.getByLabelText(/confirm whole-document text change/i)
    )
    fireEvent.click(view.getByRole("button", { name: /rewrite/i }))

    expect(onRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "rewrite",
        target: documentTarget
      })
    )
  })

  it("exposes a direction/custom instruction input for Tone", () => {
    const onRequest = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(view.getByRole("button", { name: /tone/i }))
    const toneInput = view.getByRole("textbox", { name: /tone direction/i })
    fireEvent.change(toneInput, {
      target: { value: "warmer and more confident" }
    })
    fireEvent.input(toneInput, {
      target: { value: "warmer and more confident" }
    })
    fireEvent.click(view.getByRole("button", { name: /send tone/i }))

    expect(onRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "tone" satisfies WritingRevisionAction,
        instruction: "warmer and more confident"
      })
    )
  })

  it("defaults Outline to advisory copy", () => {
    const onRequest = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(view.getByRole("button", { name: /outline/i }))

    expect(onRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "outline",
        operation: "advisory",
        instruction: expect.stringMatching(/outline/i)
      })
    )
  })
})

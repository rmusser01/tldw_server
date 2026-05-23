import React from "react"
import { cleanup, fireEvent, render } from "@testing-library/react"
import { JSDOM } from "jsdom"
import { afterAll, afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/design-system", () => ({
  READY_STATE_LABEL: "Registry Ready"
}))

import { WritingActionBar } from "../WritingActionBar"
import { WRITING_REVISION_PRESETS } from "../writing-revision-presets"
import type {
  WritingRevisionAction,
  WritingRevisionTarget
} from "../writing-revision-types"

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
  it("renders the available status label from the design-system registry", () => {
    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        onRequest={vi.fn()}
      />
    )

    expect(view.getByText("Registry Ready")).toBeTruthy()
  })

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
    expect(
      view.getByRole("button", { name: /rewrite/i }).querySelector("svg")
    ).toBeTruthy()
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

  it("can be controlled by a persisted workflow preset", () => {
    const onPresetChange = vi.fn()

    const view = render(
      <WritingActionBar
        generationAvailable
        target={selectionTarget}
        selectedPresetId="preserve_voice"
        onPresetChange={onPresetChange}
        onRequest={vi.fn()}
      />
    )

    expect(
      view.getByText(
        "Keep the author's diction, cadence, point of view, and stylistic fingerprints."
      )
    ).toBeTruthy()

    fireEvent.click(view.getByRole("radio", { name: /make concise/i }))

    expect(onPresetChange).toHaveBeenCalledWith("make_concise")
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
    const warnings = view.getAllByText(/this will rewrite the full draft/i)
    expect(warnings.length).toBe(2)
    expect(
      warnings.some((warning) =>
        warning.closest('[data-ds-component="Alert"]')
      )
    ).toBe(true)

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

  it("requires fresh confirmation when the broad target changes", () => {
    const onRequest = vi.fn()
    const nextDocumentTarget: WritingRevisionTarget = {
      ...documentTarget,
      start: 20,
      end: 1400,
      beforeText: "Different full document",
      anchor: {
        ...documentTarget.anchor,
        documentFingerprint: "fingerprint-2"
      }
    }

    const view = render(
      <WritingActionBar
        generationAvailable
        target={documentTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(
      view.getByLabelText(/confirm whole-document text change/i)
    )
    fireEvent.click(view.getByRole("button", { name: /rewrite/i }))
    expect(onRequest).toHaveBeenCalledTimes(1)

    view.rerender(
      <WritingActionBar
        generationAvailable
        target={nextDocumentTarget}
        onRequest={onRequest}
      />
    )

    fireEvent.click(view.getByRole("button", { name: /rewrite/i }))
    expect(onRequest).toHaveBeenCalledTimes(1)
    expect(view.getAllByText(/this will rewrite the full draft/i).length).toBe(2)
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

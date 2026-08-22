import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const monaco = vi.hoisted(() => ({
  props: null as Record<string, any> | null,
  editor: {
    getDomNode: vi.fn(),
    getModel: vi.fn(),
    dispose: vi.fn(),
    updateOptions: vi.fn()
  },
  model: { dispose: vi.fn() },
  providerRegistration: vi.fn(),
  renderFailure: false
}))

vi.mock("@monaco-editor/react", async () => {
  const ReactModule = await import("react")
  const FakeMonaco = (props: Record<string, any>) => {
    if (monaco.renderFailure) throw new Error("Monaco unavailable")
    monaco.props = props
    ReactModule.useEffect(() => {
      monaco.editor.getModel.mockReturnValue(monaco.model)
      monaco.editor.getDomNode.mockReturnValue(document.querySelector("[data-fake-monaco-root]"))
      props.onMount?.(monaco.editor, {
        languages: {
          registerDocumentLinkProvider: monaco.providerRegistration,
          registerHoverProvider: monaco.providerRegistration
        }
      })
    }, [props])
    return (
      <div data-fake-monaco-root {...props.wrapperProps}>
        <textarea aria-label="HTML source" value={props.value} onChange={(event) => props.onChange?.(event.target.value)} />
      </div>
    )
  }
  return { default: FakeMonaco }
})

const loadSource = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-source"].join("/"))

const loadEditor = () =>
  vi.importActual<Record<string, any>>(["..", "StandaloneHtmlSourceEditor"].join("/"))

describe("standalone HTML scalar source boundary", () => {
  const RealTextEncoder = globalThis.TextEncoder

  afterEach(() => {
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: RealTextEncoder
    })
  })

  it.each([
    ["U+0000", "before\u0000after"],
    ["a lone high surrogate", "before\ud800after"],
    ["a terminal lone high surrogate", "before\ud800"],
    ["a lone low surrogate", "before\udc00after"],
    ["a high surrogate followed by a non-low surrogate", "before\ud800xafter"],
    ["more than exactly 1 MiB", "😀".repeat(262_145)]
  ])("rejects %s before encoding or digest work", async (_case, candidate) => {
    const subject = await loadSource()
    const encoder = vi.fn(() => {
      throw new Error("TextEncoder must not run for rejected scalar input")
    })
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: encoder
    })

    const result = await subject.validateStandaloneHtmlSource(candidate)

    expect(result).toEqual(expect.objectContaining({ ok: false }))
    expect(String(result.message).length).toBeGreaterThan(0)
    expect(String(result.message).length).toBeLessThanOrEqual(160)
    expect(encoder).not.toHaveBeenCalled()
  })

  it("supports an explicit nonblank policy without changing editor-draft semantics", async () => {
    const subject = await loadSource()

    await expect(subject.validateStandaloneHtmlSource("", { allowEmpty: false })).resolves.toEqual(
      expect.objectContaining({ ok: false, code: "source_required" })
    )
    await expect(subject.validateStandaloneHtmlSource("", { allowEmpty: true })).resolves.toEqual(
      expect.objectContaining({ ok: true, byteLength: 0, scalarCount: 0 })
    )
  })

  it("returns exact scalar count, UTF-8 bytes, SHA-256 digest, and byte-for-byte round trip", async () => {
    const subject = await loadSource()

    const result = await subject.validateStandaloneHtmlSource("A😀é")

    expect(result).toEqual(
      expect.objectContaining({
        ok: true,
        source: "A😀é",
        scalarCount: 3,
        byteLength: 7,
        digest: "17ff00477d1bd497bfd8f730aa1eed978adee5c8c1c72701988ee6faff6023e4"
      })
    )
    expect(Array.from(result.bytes)).toEqual([65, 240, 159, 152, 128, 195, 169])
    expect(new TextDecoder("utf-8", { fatal: true }).decode(result.bytes)).toBe("A😀é")
  })
})

describe("StandaloneHtmlSourceEditor", () => {
  beforeEach(() => {
    monaco.props = null
    monaco.editor.dispose.mockReset()
    monaco.editor.getModel.mockReset()
    monaco.editor.getDomNode.mockReset()
    monaco.editor.updateOptions.mockReset()
    monaco.model.dispose.mockReset()
    monaco.providerRegistration.mockReset()
    monaco.renderFailure = false
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("provides a labelled inert textarea fallback that browser forms cannot persist", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const accepted = vi.fn()
    const opened = vi.spyOn(window, "open")

    render(
      <form>
        <StandaloneHtmlSourceEditor
          value={'<a href="https://attacker.invalid">https://attacker.invalid</a>'}
          onAcceptedChange={accepted}
          forceFallback
        />
      </form>
    )

    const input = screen.getByLabelText("HTML source") as HTMLTextAreaElement
    expect(screen.getByText("HTML source")).toBeVisible()
    expect(input).not.toHaveAttribute("name")
    expect(input).toHaveAttribute("spellcheck", "false")
    expect(input).toHaveAttribute("autocorrect", "off")
    expect(input).toHaveAttribute("autocapitalize", "off")
    expect(input).toHaveAttribute("autocomplete", "off")
    expect(input).toHaveAttribute("data-1p-ignore", "true")
    expect(input).toHaveAttribute("data-lpignore", "true")

    fireEvent.click(input, { ctrlKey: true })
    fireEvent.keyDown(input, { key: "Enter", metaKey: true })
    expect(opened).not.toHaveBeenCalled()

    fireEvent.change(input, { target: { value: "<!doctype html><title>Accepted</title>" } })
    await waitFor(() => expect(accepted).toHaveBeenCalledTimes(1))
    expect(accepted.mock.calls[0][0]).toEqual(
      expect.objectContaining({ ok: true, source: "<!doctype html><title>Accepted</title>" })
    )
  })

  it("preserves the last accepted fallback buffer and announces a bounded invalid-edit reason", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()

    const Harness = () => {
      const [accepted, setAccepted] = React.useState("safe source")
      return (
        <>
          <StandaloneHtmlSourceEditor
            value={accepted}
            forceFallback
            onAcceptedChange={(next: { source: string }) => setAccepted(next.source)}
          />
          <output data-testid="accepted-buffer">{accepted}</output>
        </>
      )
    }

    render(<Harness />)
    fireEvent.change(screen.getByLabelText("HTML source"), {
      target: { value: "must-not-stick\ud800" }
    })

    await waitFor(() => expect(screen.getByRole("alert")).toBeVisible())
    expect(screen.getByTestId("accepted-buffer")).toHaveTextContent("safe source")
    expect(screen.getByLabelText("HTML source")).toHaveValue("safe source")
    expect(screen.getByRole("alert").textContent?.length).toBeLessThanOrEqual(160)
  })

  it("configures lazy Monaco as inert plain text with a rejecting scoped opener", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const opened = vi.spyOn(window, "open")

    render(
      <StandaloneHtmlSourceEditor
        value={'href="https://one.invalid"; src="https://two.invalid"; url(https://three.invalid)'}
        onAcceptedChange={() => undefined}
      />
    )

    await waitFor(() => expect(monaco.props).not.toBeNull())
    expect(monaco.props?.language).toBe("plaintext")
    expect(monaco.props?.defaultLanguage).toBe("plaintext")
    expect(monaco.props?.options).toEqual(
      expect.objectContaining({
        links: false,
        hover: expect.objectContaining({ enabled: false })
      })
    )
    expect(monaco.props?.options).not.toHaveProperty("opener")
    expect(monaco.providerRegistration).not.toHaveBeenCalled()
    const label = screen.getByText("HTML source")
    const root = document.querySelector("[data-fake-monaco-root]")
    const input = root?.querySelector("textarea")
    expect(label.id).not.toBe("")
    expect(root).toHaveAttribute("aria-labelledby", label.id)
    expect(input).toHaveAttribute("aria-labelledby", label.id)
    expect(input).not.toHaveAttribute("name")
    expect(input).toHaveAttribute("spellcheck", "false")
    expect(input).toHaveAttribute("autocorrect", "off")
    expect(input).toHaveAttribute("autocapitalize", "off")
    expect(input).toHaveAttribute("autocomplete", "off")
    expect(input).toHaveAttribute("data-1p-ignore", "true")
    expect(input).toHaveAttribute("data-lpignore", "true")

    for (const resource of [
      "https://one.invalid",
      "https://two.invalid/image.png",
      "https://three.invalid/style.css",
      "mailto:attacker@example.invalid"
    ]) {
      await expect(
        monaco.props?.overrideServices.openerService.open({ resource })
      ).resolves.toBe(false)
    }
    expect(opened).not.toHaveBeenCalled()
  })

  it("applies read-only parity to Monaco without changing its inert service boundary", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()

    render(
      <StandaloneHtmlSourceEditor
        value="read-only source"
        readOnly
        onAcceptedChange={() => undefined}
      />
    )

    await waitFor(() => expect(monaco.props).not.toBeNull())
    expect(monaco.props?.options).toEqual(expect.objectContaining({ readOnly: true }))
    expect(monaco.props?.overrideServices.openerService).toBeDefined()
  })

  it("falls back to the labelled inert textarea when Monaco cannot render", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    monaco.renderFailure = true
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    const suppressExpectedError = (event: ErrorEvent) => event.preventDefault()
    window.addEventListener("error", suppressExpectedError)

    try {
      render(
        <StandaloneHtmlSourceEditor
          value="preserved source"
          onAcceptedChange={() => undefined}
        />
      )

      const fallback = await screen.findByLabelText("HTML source")
      expect(fallback.tagName).toBe("TEXTAREA")
      expect(fallback).toHaveValue("preserved source")
      expect(fallback).not.toHaveAttribute("name")
      expect(fallback).toHaveAttribute("autocomplete", "off")
    } finally {
      window.removeEventListener("error", suppressExpectedError)
    }
  })

  it("disposes only its editor model and editor without touching global providers", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()

    const view = render(
      <StandaloneHtmlSourceEditor value="private source" onAcceptedChange={() => undefined} />
    )
    await waitFor(() => expect(monaco.editor.getModel).toHaveBeenCalled())

    view.unmount()

    expect(monaco.model.dispose).toHaveBeenCalledTimes(1)
    expect(monaco.editor.dispose).toHaveBeenCalledTimes(1)
    expect(monaco.providerRegistration).not.toHaveBeenCalled()
  })
})

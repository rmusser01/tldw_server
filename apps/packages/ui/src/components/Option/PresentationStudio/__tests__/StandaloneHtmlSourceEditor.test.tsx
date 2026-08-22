import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const monaco = vi.hoisted(() => ({
  props: null as Record<string, any> | null,
  editor: {
    getDomNode: vi.fn(),
    getModel: vi.fn(),
    getValue: vi.fn(),
    setValue: vi.fn(),
    dispose: vi.fn(),
    updateOptions: vi.fn()
  },
  model: { dispose: vi.fn(), getValue: vi.fn(), setValue: vi.fn() },
  providerRegistration: vi.fn(),
  renderFailure: false,
  suspense: null as Promise<void> | null
}))

vi.mock("@monaco-editor/react", async () => {
  const ReactModule = await import("react")
  const FakeMonaco = (props: Record<string, any>) => {
    if (monaco.suspense) throw monaco.suspense
    if (monaco.renderFailure) throw new Error("Monaco unavailable")
    monaco.props = props
    ReactModule.useEffect(() => {
      monaco.editor.getModel.mockReturnValue(monaco.model)
      monaco.editor.getValue.mockImplementation(() => monaco.props?.value ?? "")
      monaco.editor.setValue.mockImplementation((next: string) => {
        monaco.props?.onChange?.(next)
      })
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
    monaco.editor.getValue.mockReset()
    monaco.editor.setValue.mockReset()
    monaco.editor.updateOptions.mockReset()
    monaco.model.dispose.mockReset()
    monaco.model.getValue.mockReset()
    monaco.model.setValue.mockReset()
    monaco.providerRegistration.mockReset()
    monaco.renderFailure = false
    monaco.suspense = null
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

  it.each([
    ["U+0000", "rejected\u0000candidate"],
    ["a lone surrogate", "rejected\ud800candidate"],
    ["more than 1 MiB", "x".repeat(1_048_577)]
  ])("never enqueues rejected %s input in React draft state", async (_case, candidate) => {
    const stateWrites: unknown[] = []
    const realUseState = React.useState
    vi.spyOn(React, "useState").mockImplementation(((initial: unknown) => {
      const [current, setCurrent] = realUseState(initial)
      return [
        current,
        (next: unknown) => {
          stateWrites.push(next)
          setCurrent(next as never)
        }
      ]
    }) as typeof React.useState)
    const { StandaloneHtmlSourceEditor } = await loadEditor()

    render(
      <StandaloneHtmlSourceEditor
        value="safe source"
        forceFallback
        onAcceptedChange={() => undefined}
      />
    )
    fireEvent.change(screen.getByLabelText("HTML source"), {
      target: { value: candidate }
    })

    await waitFor(() => expect(screen.getByRole("alert")).toBeVisible())
    expect(stateWrites).not.toContain(candidate)
    expect(screen.getByLabelText("HTML source")).toHaveValue("safe source")
  })

  it("keeps a preflight-valid fallback candidate visible while its digest is pending", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const candidate = "candidate awaiting digest"
    const realDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(candidate)
    )
    const digestResolvers: Array<(value: ArrayBuffer) => void> = []
    vi.spyOn(crypto.subtle, "digest").mockImplementation(
      () => new Promise<ArrayBuffer>((resolve) => digestResolvers.push(resolve))
    )

    const Harness = () => {
      const [accepted, setAccepted] = React.useState("")
      const [pending, setPending] = React.useState<string | null>(null)
      return (
        <>
          <StandaloneHtmlSourceEditor
            value={accepted}
            forceFallback
            onAcceptedChange={(next: { source: string }) => setAccepted(next.source)}
            onPendingChange={setPending}
          />
          <output data-testid="pending-buffer">{pending}</output>
        </>
      )
    }

    render(<Harness />)
    const user = userEvent.setup()
    const fallback = screen.getByLabelText("HTML source")
    await user.type(fallback, candidate)

    expect(screen.getByTestId("pending-buffer")).toHaveTextContent(candidate)
    expect(fallback).toHaveValue(candidate)

    digestResolvers.at(-1)?.(realDigest)
    await waitFor(() => expect(fallback).toHaveValue(candidate))
  })

  it("retires Monaco on a post-mount failure without losing its pending draft validation", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const candidate = "pending Monaco candidate"
    const realDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(candidate)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => {
        resolveDigest = resolve
      })
    )
    const suppressExpectedError = (event: ErrorEvent) => event.preventDefault()
    window.addEventListener("error", suppressExpectedError)
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    const acceptedChange = vi.fn()
    let adoptExternal: ((source: string) => void) | null = null

    const Harness = () => {
      const [accepted, setAccepted] = React.useState("accepted source")
      const [pending, setPending] = React.useState<string | null>(null)
      adoptExternal = setAccepted
      return (
        <>
          <StandaloneHtmlSourceEditor
            value={accepted}
            onAcceptedChange={(next: { source: string }) => {
              acceptedChange(next.source)
              setAccepted(next.source)
            }}
            onPendingChange={setPending}
          />
          <output data-testid="pending-buffer">{pending}</output>
        </>
      )
    }

    try {
      render(<Harness />)
      await waitFor(() => expect(monaco.props).not.toBeNull())
      const monacoRoot = document.querySelector("[data-fake-monaco-root]") as HTMLElement
      monaco.renderFailure = true

      act(() => monaco.props?.onChange?.(candidate))

      const fallback = await screen.findByLabelText("HTML source")
      expect(screen.getByTestId("pending-buffer")).toHaveTextContent(candidate)
      expect(fallback).toHaveValue(candidate)
      expect(monaco.model.dispose).toHaveBeenCalledTimes(1)
      expect(monaco.editor.dispose).toHaveBeenCalledTimes(1)
      const retiredNavigation = new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        ctrlKey: true
      })
      monacoRoot.dispatchEvent(retiredNavigation)
      expect(retiredNavigation.defaultPrevented).toBe(false)

      resolveDigest?.(realDigest)
      await waitFor(() => expect(acceptedChange).toHaveBeenCalledWith(candidate))
      expect(screen.getByTestId("pending-buffer")).toBeEmptyDOMElement()
      expect(fallback).toHaveValue(candidate)

      act(() => adoptExternal?.("externally adopted source"))
      await waitFor(() => expect(fallback).toHaveValue("externally adopted source"))
      fireEvent.change(fallback, { target: { value: "invalid\u0000candidate" } })
      await waitFor(() => expect(screen.getByRole("alert")).toBeVisible())
      expect(fallback).toHaveValue("externally adopted source")
      expect(monaco.editor.setValue).not.toHaveBeenCalled()
      expect(monaco.model.dispose).toHaveBeenCalledTimes(1)
      expect(monaco.editor.dispose).toHaveBeenCalledTimes(1)
    } finally {
      monaco.renderFailure = false
      resolveDigest?.(realDigest)
      window.removeEventListener("error", suppressExpectedError)
    }
  })

  it("keeps a pending Suspense fallback candidate when Monaco finishes loading", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const candidate = "pending Suspense candidate"
    const realDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(candidate)
    )
    const digestResolvers: Array<(value: ArrayBuffer) => void> = []
    vi.spyOn(crypto.subtle, "digest").mockImplementation(
      () => new Promise<ArrayBuffer>((resolve) => digestResolvers.push(resolve))
    )
    let releaseMonaco: (() => void) | null = null
    monaco.suspense = new Promise<void>((resolve) => {
      releaseMonaco = resolve
    })

    const Harness = () => {
      const [accepted, setAccepted] = React.useState("")
      const [pending, setPending] = React.useState<string | null>(null)
      return (
        <>
          <StandaloneHtmlSourceEditor
            value={accepted}
            onAcceptedChange={(next: { source: string }) => setAccepted(next.source)}
            onPendingChange={setPending}
          />
          <output data-testid="pending-buffer">{pending}</output>
        </>
      )
    }

    render(<Harness />)
    const user = userEvent.setup()
    const fallback = screen.getByLabelText("HTML source")
    await user.type(fallback, candidate)
    expect(fallback).toHaveValue(candidate)

    await act(async () => {
      monaco.suspense = null
      releaseMonaco?.()
      await Promise.resolve()
    })

    const monacoInput = await waitFor(() => {
      const input = document.querySelector("[data-fake-monaco-root] textarea")
      expect(input).not.toBeNull()
      return input as HTMLTextAreaElement
    })
    expect(screen.getByTestId("pending-buffer")).toHaveTextContent(candidate)
    expect(monacoInput).toHaveValue(candidate)

    digestResolvers.at(-1)?.(realDigest)
    await waitFor(() => expect(monacoInput).toHaveValue(candidate))
  })

  it("configures lazy Monaco as inert plain text with editor-scoped navigation prevention", async () => {
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
        hover: expect.objectContaining({ enabled: false }),
        contextmenu: false,
        mouseMiddleClickAction: "default"
      })
    )
    expect(monaco.props?.options).not.toHaveProperty("opener")
    expect(monaco.props?.overrideServices).toBeUndefined()
    expect(monaco.props?.saveViewState).toBe(false)
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

    for (const event of [
      new MouseEvent("click", { bubbles: true, cancelable: true, ctrlKey: true }),
      new MouseEvent("mousedown", { bubbles: true, cancelable: true, button: 1 }),
      new MouseEvent("mouseup", { bubbles: true, cancelable: true, button: 1 }),
      new MouseEvent("pointerdown", { bubbles: true, cancelable: true, button: 1 }),
      new MouseEvent("pointerup", { bubbles: true, cancelable: true, button: 1 }),
      new MouseEvent("auxclick", { bubbles: true, cancelable: true, button: 1 }),
      new MouseEvent("contextmenu", { bubbles: true, cancelable: true }),
      new KeyboardEvent("keydown", { bubbles: true, cancelable: true, metaKey: true, key: "Enter" }),
      new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "F12" })
    ]) {
      root?.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(true)
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
    expect(monaco.props?.overrideServices).toBeUndefined()
  })

  it("rolls an invalid Monaco model edit back to the last accepted value without recursive acceptance", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const accepted = vi.fn()

    render(
      <StandaloneHtmlSourceEditor value="last accepted" onAcceptedChange={accepted} />
    )
    await waitFor(() => expect(monaco.props).not.toBeNull())
    monaco.editor.getValue.mockReturnValue("invalid\u0000candidate")

    act(() => monaco.props?.onChange?.("invalid\u0000candidate"))

    await waitFor(() => expect(screen.getByRole("alert")).toBeVisible())
    expect(monaco.editor.setValue).toHaveBeenCalledTimes(1)
    expect(monaco.editor.setValue).toHaveBeenCalledWith("last accepted")
    expect(accepted).not.toHaveBeenCalled()
  })

  it("fences an async edit validation when an external value is adopted", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const accepted = vi.fn()
    const realDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode("candidate awaiting digest")
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => {
        resolveDigest = resolve
      })
    )
    const view = render(
      <StandaloneHtmlSourceEditor value="server A" onAcceptedChange={accepted} />
    )
    await waitFor(() => expect(monaco.props).not.toBeNull())

    act(() => monaco.props?.onChange?.("candidate awaiting digest"))
    view.rerender(
      <StandaloneHtmlSourceEditor value="server C" onAcceptedChange={accepted} />
    )
    resolveDigest?.(realDigest)
    await act(async () => Promise.resolve())

    expect(accepted).not.toHaveBeenCalled()
    expect(
      screen.getAllByLabelText("HTML source").find((node) => node.tagName === "TEXTAREA")
    ).toHaveValue("server C")
  })

  it("reports a valid pending candidate synchronously and clears it on invalid rollback", async () => {
    const { StandaloneHtmlSourceEditor } = await loadEditor()
    const accepted = vi.fn()
    const pending = vi.fn()
    const candidate = "candidate awaiting digest"
    const realDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(candidate)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    render(
      <StandaloneHtmlSourceEditor
        value="accepted source"
        onAcceptedChange={accepted}
        onPendingChange={pending}
      />
    )
    await waitFor(() => expect(monaco.props).not.toBeNull())

    act(() => monaco.props?.onChange?.(candidate))
    expect(pending).toHaveBeenLastCalledWith(candidate)
    expect(accepted).not.toHaveBeenCalled()

    act(() => monaco.props?.onChange?.("invalid\u0000candidate"))
    expect(pending).toHaveBeenLastCalledWith(null)
    resolveDigest?.(realDigest)
    await act(async () => Promise.resolve())
    expect(accepted).not.toHaveBeenCalled()
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

  it.each(["model", "editor"] as const)(
    "continues fail-closed editor cleanup when %s disposal throws",
    async (throwingOwner) => {
      const { StandaloneHtmlSourceEditor } = await loadEditor()
      const throwingDispose = throwingOwner === "model"
        ? monaco.model.dispose
        : monaco.editor.dispose
      throwingDispose.mockImplementationOnce(() => {
        throw new Error(`${throwingOwner} cleanup unavailable`)
      })
      const view = render(
        <StandaloneHtmlSourceEditor value="private source" onAcceptedChange={() => undefined} />
      )
      await waitFor(() => expect(monaco.editor.getModel).toHaveBeenCalled())

      expect(() => view.unmount()).not.toThrow()
      expect(monaco.model.dispose).toHaveBeenCalledTimes(1)
      expect(monaco.editor.dispose).toHaveBeenCalledTimes(1)
    }
  )
})

describe("StandaloneHtmlSourceEditor with pinned real Monaco 0.55.1", () => {
  beforeEach(() => {
    if (typeof document.queryCommandSupported !== "function") {
      Object.defineProperty(document, "queryCommandSupported", {
        configurable: true,
        value: () => false
      })
    }
  })

  it.each(["control first", "standalone first"])(
    "keeps opener behavior and options editor-scoped when initialized %s",
    async (order) => {
      vi.resetModules()
      const actualMonaco = await vi.importActual<Record<string, any>>(
        "../../../../../../../node_modules/.bun/monaco-editor@0.55.1/node_modules/monaco-editor/esm/vs/editor/editor.main.js"
      )
      const { installStandaloneHtmlMonacoGuards } = await loadEditor()
      expect(typeof installStandaloneHtmlMonacoGuards).toBe("function")
      const standaloneRoot = document.createElement("div")
      const controlRoot = document.createElement("div")
      standaloneRoot.style.cssText = "width:640px;height:320px"
      controlRoot.style.cssText = "width:640px;height:320px"
      document.body.append(standaloneRoot, controlRoot)
      const makeStandalone = () => {
        const editor = actualMonaco.editor.create(standaloneRoot, {
          value: "https://standalone.invalid",
          language: "plaintext",
          links: true,
          contextmenu: true,
          occurrencesHighlight: "off",
          selectionHighlight: false
        })
        const guard = installStandaloneHtmlMonacoGuards(editor)
        return { editor, guard }
      }
      const makeControl = () =>
        actualMonaco.editor.create(controlRoot, {
          value: "https://control.invalid",
          language: "plaintext",
          links: true,
          contextmenu: true,
          occurrencesHighlight: "off",
          selectionHighlight: false
        })
      const first = order === "standalone first" ? makeStandalone() : makeControl()
      const second = order === "standalone first" ? makeControl() : makeStandalone()
      const standalone = (order === "standalone first" ? first : second) as {
        editor: any
        guard: { dispose: () => void }
      }
      const control = (order === "standalone first" ? second : first) as any

      expect(
        standalone.editor.getOption(actualMonaco.editor.EditorOption.links)
      ).toBe(false)
      expect(
        standalone.editor.getOption(actualMonaco.editor.EditorOption.contextmenu)
      ).toBe(false)
      expect(
        standalone.editor.getOption(actualMonaco.editor.EditorOption.mouseMiddleClickAction)
      ).toBe("default")
      expect(control.getOption(actualMonaco.editor.EditorOption.links)).toBe(true)
      expect(control.getOption(actualMonaco.editor.EditorOption.contextmenu)).toBe(true)

      for (const event of [
        new MouseEvent("click", { bubbles: true, cancelable: true, metaKey: true }),
        new MouseEvent("mousedown", { bubbles: true, cancelable: true, button: 1 }),
        new MouseEvent("mouseup", { bubbles: true, cancelable: true, button: 1 }),
        new MouseEvent("pointerdown", { bubbles: true, cancelable: true, button: 1 }),
        new MouseEvent("pointerup", { bubbles: true, cancelable: true, button: 1 }),
        new MouseEvent("auxclick", { bubbles: true, cancelable: true, button: 1 }),
        new MouseEvent("contextmenu", { bubbles: true, cancelable: true }),
        new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ctrlKey: true, key: "Enter" }),
        new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "F12" })
      ]) {
        standalone.editor.getDomNode()?.dispatchEvent(event)
        expect(event.defaultPrevented).toBe(true)
      }
      const controlContextMenu = new MouseEvent("contextmenu", {
        bubbles: true,
        cancelable: true
      })
      controlRoot.dispatchEvent(controlContextMenu)
      expect(controlContextMenu.defaultPrevented).toBe(false)

      const open = vi.fn(async () => true)
      const opener = actualMonaco.editor.registerLinkOpener({ open })
      const provider = actualMonaco.languages.registerLinkProvider("plaintext", {
        provideLinks: (model) => ({
          links: [
            {
              range: new actualMonaco.Range(1, 1, 1, model.getLineMaxColumn(1)),
              url: model.getValue()
            }
          ]
        })
      })
      control.setPosition({ lineNumber: 1, column: 5 })
      await waitFor(
        async () => {
          await control.getAction("editor.action.openLink")?.run()
          expect(open).toHaveBeenCalled()
        },
        { timeout: 2_000 }
      )
      open.mockClear()
      const standaloneTarget =
        standalone.editor.getDomNode()?.querySelector(".view-lines") ??
        standalone.editor.getDomNode()
      for (const type of ["pointerdown", "mousedown", "mouseup", "pointerup", "auxclick"]) {
        const middleEvent = new MouseEvent(type, {
          bubbles: true,
          cancelable: true,
          button: 1
        })
        standaloneTarget?.dispatchEvent(middleEvent)
        expect(middleEvent.defaultPrevented).toBe(true)
        await Promise.resolve()
        expect(open).not.toHaveBeenCalled()
      }
      for (const [type, modifier] of [
        ["pointerdown", "ctrlKey"],
        ["mousedown", "metaKey"],
        ["mouseup", "ctrlKey"],
        ["pointerup", "metaKey"]
      ] as const) {
        const modifierEvent = new MouseEvent(type, {
          bubbles: true,
          cancelable: true,
          button: 0,
          [modifier]: true
        })
        standaloneTarget?.dispatchEvent(modifierEvent)
        expect(modifierEvent.defaultPrevented).toBe(true)
        await Promise.resolve()
        expect(open).not.toHaveBeenCalled()
      }
      standalone.editor.setPosition({ lineNumber: 1, column: 5 })
      await standalone.editor.getAction("editor.action.openLink")?.run()
      expect(open).not.toHaveBeenCalled()
      await new Promise((resolve) => setTimeout(resolve, 400))

      provider.dispose()
      opener.dispose()
      standalone.guard.dispose()
      standalone.editor.dispose()
      control.dispose()
      standaloneRoot.remove()
      controlRoot.remove()
    }
  )
})

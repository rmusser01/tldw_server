import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import useDynamicTextareaSize from "../useDynamicTextareaSize"
import { ChatComposer } from "@/components/Chat/composer/ChatComposer"

// Real hook/DOM refs; geometry, observer delivery and frames are platform doubles.
let frames: Map<number, FrameRequestCallback>
let frameId: number
let observers: ControlledResizeObserver[]

class ControlledResizeObserver {
  disconnected = false
  target: Element | null = null
  constructor(private callback: ResizeObserverCallback) {
    observers.push(this)
  }
  observe(target: Element) {
    this.target = target
  }
  disconnect() {
    this.disconnected = true
  }
  unobserve() {}
  notify(width: number, height = 44) {
    this.callback(
      [
        { target: this.target, contentRect: { width, height } }
      ] as ResizeObserverEntry[],
      this as unknown as ResizeObserver
    )
  }
}

const draft = "Keep this unchanged draft while the composer narrows."
function Harness({
  maxHeight = 120,
  text = draft,
  nodeKey = "A",
  delayed = false,
  objectRef
}: {
  maxHeight?: number
  text?: string
  nodeKey?: string
  delayed?: boolean
  objectRef?: React.RefObject<HTMLTextAreaElement>
}) {
  const localRef = React.useRef<HTMLTextAreaElement>(null)
  const textareaRef = objectRef ?? localRef
  const attachTextarea = useDynamicTextareaSize(textareaRef, text, maxHeight)
  // Fallback keeps the checkpoint's void API a valid causal control.
  const nativeRef = attachTextarea || textareaRef
  return delayed ? (
    <DelayedSlot nativeRef={nativeRef} text={text} />
  ) : (
    <textarea key={nodeKey} ref={nativeRef} value={text} readOnly />
  )
}

function DelayedSlot({
  nativeRef,
  text
}: {
  nativeRef: React.Ref<HTMLTextAreaElement>
  text: string
}) {
  const [visible, setVisible] = React.useState(false)
  return (
    <>
      <button onClick={() => setVisible(true)}>Attach textarea</button>
      {visible && <textarea ref={nativeRef} value={text} readOnly />}
    </>
  )
}

function geometry(node: HTMLTextAreaElement, width = 400) {
  const layout = { width, height: 44 }
  Object.defineProperties(node, {
    clientWidth: { configurable: true, get: () => Math.round(layout.width) },
    scrollHeight: { configurable: true, get: () => layout.height }
  })
  return layout
}
function flushFrames() {
  act(() => {
    const scheduled = [...frames.values()]
    frames.clear()
    scheduled.forEach((callback) => callback(0))
  })
}
function currentObserver(node: Element) {
  const observer = observers.findLast(
    (entry) => entry.target === node && !entry.disconnected
  )
  expect(
    observer,
    "the mounted textarea must have a width observer"
  ).toBeDefined()
  return observer!
}
beforeEach(() => {
  frames = new Map()
  frameId = 0
  observers = []
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((callback: FrameRequestCallback) => {
      const id = ++frameId
      frames.set(id, callback)
      return id
    })
  )
  vi.stubGlobal(
    "cancelAnimationFrame",
    vi.fn((id: number) => frames.delete(id))
  )
  vi.stubGlobal("ResizeObserver", ControlledResizeObserver)
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("useDynamicTextareaSize mounted layout", () => {
  it("grows and shrinks unchanged text after width-only reflow", () => {
    render(<Harness />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    expect(node.style.height).toBe("44px")
    const observer = currentObserver(node)
    layout.width = 230
    layout.height = 64
    act(() => observer.notify(layout.width))
    flushFrames()
    expect(node.style.height).toBe("64px")
    expect(node.style.overflowY).toBe("hidden")
    layout.width = 400
    layout.height = 44
    act(() => observer.notify(layout.width))
    flushFrames()
    expect(node.style.height).toBe("44px")
    expect(node.value).toBe(draft)
  })
  it("does not round away a fractional content-width wrap", () => {
    render(<Harness />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node, 300.4)
    flushFrames()
    const observer = currentObserver(node)
    act(() => observer.notify(300.4))
    flushFrames()
    layout.width = 300.2
    layout.height = 64
    act(() => observer.notify(300.2))
    flushFrames()
    expect(node.style.height).toBe("64px")
  })
  it("keeps capped reflow scrollable and retires an older text/cap frame", () => {
    const view = render(<Harness />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    const observer = currentObserver(node)
    layout.height = 240
    act(() => observer.notify(220))
    flushFrames()
    expect(node.style.height).toBe("120px")
    expect(node.style.overflowY).toBe("scroll")
    act(() => observer.notify(200))
    const retiredFrame = [...frames.values()][0]
    view.rerender(<Harness maxHeight={80} text="A changed display value" />)
    flushFrames()
    act(() => retiredFrame(0))
    expect(node.style.height).toBe("80px")
    expect(node.style.maxHeight).toBe("80px")
    layout.height = 44
    act(() => currentObserver(node).notify(400))
    flushFrames()
    expect(node.style.height).toBe("44px")
    expect(node.style.overflowY).toBe("hidden")
  })
  it("sizes a child attached after the parent effect with unchanged input", () => {
    const objectRef = React.createRef<HTMLTextAreaElement>()
    render(<Harness delayed objectRef={objectRef} />)
    flushFrames()
    fireEvent.click(screen.getByRole("button", { name: "Attach textarea" }))
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    geometry(node, 230).height = 64
    flushFrames()
    expect(node.style.height).toBe("64px")
    expect(objectRef.current).toBe(node)
    objectRef.current!.focus()
    expect(document.activeElement).toBe(node)
  })
  it("rebinds replacement nodes and ignores retired observer/frame delivery", () => {
    const objectRef = React.createRef<HTMLTextAreaElement>()
    const view = render(<Harness objectRef={objectRef} />)
    const oldNode = screen.getByRole("textbox") as HTMLTextAreaElement
    const oldLayout = geometry(oldNode)
    flushFrames()
    const oldObserver = currentObserver(oldNode)
    act(() => oldObserver.notify(220))
    const retiredFrame = [...frames.values()][0]
    view.rerender(<Harness objectRef={objectRef} nodeKey="B" />)
    const newNode = screen.getByRole("textbox") as HTMLTextAreaElement
    geometry(newNode).height = 64
    flushFrames()
    oldLayout.height = 200
    act(() => {
      oldObserver.notify(100)
      retiredFrame(0)
    })
    flushFrames()
    expect(oldObserver.disconnected).toBe(true)
    expect(oldNode.style.height).toBe("44px")
    expect(newNode.style.height).toBe("64px")
    expect(objectRef.current).toBe(newNode)
  })
  it("ignores height-only feedback and coalesces width bursts into one frame", () => {
    render(<Harness />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    geometry(node)
    flushFrames()
    const observer = currentObserver(node)
    act(() => observer.notify(400))
    flushFrames()
    act(() => observer.notify(400, 64))
    expect(frames.size).toBe(0)
    act(() => {
      observer.notify(320)
      observer.notify(310)
      observer.notify(300)
    })
    expect(frames.size).toBe(1)
  })
  it("continues sizing through StrictMode effect replay", () => {
    const objectRef = React.createRef<HTMLTextAreaElement>()
    render(
      <React.StrictMode>
        <Harness objectRef={objectRef} />
      </React.StrictMode>
    )
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    layout.height = 64
    act(() => currentObserver(node).notify(220))
    flushFrames()
    expect(node.style.height).toBe("64px")
    expect(objectRef.current).toBe(node)
    expect(observers.filter((observer) => !observer.disconnected)).toHaveLength(
      1
    )
  })
  it("retires all queued measurement work on unmount", () => {
    const objectRef = React.createRef<HTMLTextAreaElement>()
    const view = render(<Harness objectRef={objectRef} />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    geometry(node).height = 64
    const retiredFrame = [...frames.values()][0]
    view.unmount()
    act(() => retiredFrame(0))
    expect(node.style.height).toBe("")
    expect(objectRef.current).toBeNull()
    expect(frames.size).toBe(0)
    expect(observers.every((observer) => observer.disconnected)).toBe(true)
  })
  it("rebinds a replacement external ref while retaining the same DOM node", () => {
    const originalRef = React.createRef<HTMLTextAreaElement>()
    const replacementRef = React.createRef<HTMLTextAreaElement>()
    const view = render(<Harness objectRef={originalRef} />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    view.rerender(<Harness objectRef={replacementRef} />)
    flushFrames()
    layout.height = 64
    act(() => currentObserver(node).notify(220))
    flushFrames()
    expect(originalRef.current).toBeNull()
    expect(replacementRef.current).toBe(node)
    expect(node.style.height).toBe("64px")
  })
  it("uses window resize when ResizeObserver is unavailable", () => {
    vi.stubGlobal("ResizeObserver", undefined)
    const view = render(<Harness />)
    const node = screen.getByRole("textbox") as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    layout.width = 220
    layout.height = 64
    fireEvent(window, new Event("resize"))
    flushFrames()
    expect(node.style.height).toBe("64px")
    view.unmount()
    fireEvent(window, new Event("resize"))
    expect(frames.size).toBe(0)
  })
})

function VariantHarness({
  variant,
  objectRef
}: {
  variant: "v1" | "v3" | "v5"
  objectRef: React.RefObject<HTMLTextAreaElement>
}) {
  const attachTextarea = useDynamicTextareaSize(objectRef, draft, 120)
  const props = {
    message: draft,
    onMessageChange: () => {},
    onSend: () => {},
    textareaSlot: (
      <textarea ref={attachTextarea || objectRef} value={draft} readOnly />
    )
  }
  return variant === "v3" ? (
    <ChatComposer variant="v3" {...props} briefSections={[]} />
  ) : variant === "v5" ? (
    <ChatComposer variant="v5" {...props} />
  ) : (
    <ChatComposer variant="v1" {...props} />
  )
}

it("sizes actual lazy composer slots across V1, V3 and V5 without editing the draft", async () => {
  const objectRef = React.createRef<HTMLTextAreaElement>()
  const view = render(<VariantHarness variant="v1" objectRef={objectRef} />)
  let oldNode: HTMLTextAreaElement | null = null
  for (const variant of ["v1", "v3", "v5"] as const) {
    if (variant !== "v1")
      view.rerender(<VariantHarness variant={variant} objectRef={objectRef} />)
    const node = (await screen.findByRole("textbox")) as HTMLTextAreaElement
    const layout = geometry(node)
    flushFrames()
    expect(node.style.height).toBe("44px")
    if (oldNode) expect(node).not.toBe(oldNode)
    layout.height = 64
    act(() => currentObserver(node).notify(230))
    flushFrames()
    expect(node.style.height).toBe("64px")
    expect(node.value).toBe(draft)
    expect(objectRef.current).toBe(node)
    oldNode = node
  }
})

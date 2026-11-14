// @vitest-environment jsdom
import {
  act,
  cleanup,
  render,
  renderHook,
  screen
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { StrictMode } from "react"
import { useComposerDockHeight } from "../useComposerDockHeight"

// DOM layout and observer delivery are platform doubles, not browser geometry proof.
let observers: ControlledObserver[]
class ControlledObserver {
  target: Element | null = null
  box: ResizeObserverBoxOptions = "content-box"
  disconnect = vi.fn()
  constructor(readonly callback: ResizeObserverCallback) {
    observers.push(this)
  }
  observe(target: Element, options?: ResizeObserverOptions) {
    this.target = target
    this.box = options?.box ?? "content-box"
  }
  notify() {
    this.callback([], this as unknown as ResizeObserver)
  }
}

function measuredDock(initialHeight: number) {
  const node = document.createElement("div")
  let height = initialHeight
  node.getBoundingClientRect = () => ({ height }) as DOMRect
  return {
    node,
    resize: (value: number) => {
      height = value
    }
  }
}

describe("occupied composer dock height", () => {
  beforeEach(() => {
    observers = []
    vi.stubGlobal("ResizeObserver", ControlledObserver)
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("measures the capped border box again when the dock resizes without a Form resize", () => {
    const dock = measuredDock(216.4)
    const { result } = renderHook(() => useComposerDockHeight())
    act(() => result.current.ref(dock.node))
    expect(result.current.height).toBe(217)
    dock.resize(144)
    act(() => observers[0].notify())
    expect(result.current.height).toBe(144)
  })

  it("updates occupied height when only border or safe-area padding changes", () => {
    const dock = measuredDock(200)
    const { result } = renderHook(() => useComposerDockHeight())
    act(() => result.current.ref(dock.node))
    dock.resize(224)
    // A native content-box observer does not deliver this border-only change.
    act(() => {
      if (observers[0].box === "border-box") observers[0].notify()
    })
    expect(result.current.height).toBe(224)
  })

  it("keeps observing the mounted dock through Strict Mode effect restoration", () => {
    let height = 200
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      () => ({ height }) as DOMRect
    )
    function MountedDock() {
      const dock = useComposerDockHeight()
      return (
        <div ref={dock.ref} data-testid="occupied-height">
          {dock.height}
        </div>
      )
    }
    render(
      <StrictMode>
        <MountedDock />
      </StrictMode>
    )
    height = 140
    act(() => observers.at(-1)!.notify())
    expect(screen.getByTestId("occupied-height")).toHaveTextContent("140")
  })

  it("retires the old observer when a restoring or replaced dock detaches", () => {
    const oldDock = measuredDock(300)
    const replacement = measuredDock(120)
    const { result, unmount } = renderHook(() => useComposerDockHeight())
    act(() => result.current.ref(oldDock.node))
    const oldObserver = observers[0]
    act(() => result.current.ref(null))
    expect(result.current.height).toBe(0)
    act(() => result.current.ref(replacement.node))
    oldDock.resize(900)
    act(() => oldObserver.notify())
    expect(result.current.height).toBe(120)
    expect(oldObserver.disconnect).toHaveBeenCalled()
    unmount()
    expect(observers[1].disconnect).toHaveBeenCalled()
  })
})

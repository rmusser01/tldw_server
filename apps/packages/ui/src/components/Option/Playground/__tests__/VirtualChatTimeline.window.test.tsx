// Stage 3 (WP-03) windowed rendering for the web Playground transcript:
// PlaygroundChat renders through VirtualChatTimeline (already virtualized on
// dev; see TASK-13511). This suite pins the Stage 3 success criterion at the
// engine level with the REAL @tanstack/react-virtual instance: a
// 1,000-block transcript mounts a bounded window, and the streaming last
// block is reachable at the window end.
import { fireEvent, render } from "@testing-library/react"
import React from "react"
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest"
import { VirtualChatTimeline } from "../VirtualChatTimeline"


const scrollDescriptor = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "scrollTo"
)

beforeAll(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
  Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    configurable: true,
    value(this: HTMLElement, options?: ScrollToOptions | number) {
      if (typeof options === "object") this.scrollTop = options.top ?? 0
    }
  })
})

// vitest's restoreMocks (config) restores and detaches every spy right
// before each test body runs, i.e. AFTER user beforeEach hooks: create the
// layout spies afresh inside each test that needs a measurable viewport.
const applyLayoutStubs = () => {
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(640)
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(800)
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    () => ({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 800,
      bottom: 160,
      width: 800,
      height: 160,
      toJSON() {
        return {}
      }
    })
  )
}

afterAll(() => {
  if (scrollDescriptor) {
    Object.defineProperty(HTMLElement.prototype, "scrollTo", scrollDescriptor)
  } else {
    delete (HTMLElement.prototype as Partial<HTMLElement>).scrollTo
  }
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const MESSAGE_COUNT = 1000

type Harness = {
  parent: HTMLElement
  renderedIndexes: Set<number>
}

const renderTimeline = (): Harness => {
  applyLayoutStubs()
  const renderedIndexes = new Set<number>()
  function Case() {
    const parent = React.useRef<HTMLDivElement>(null)
    const blocks = React.useMemo(
      () => Array.from({ length: MESSAGE_COUNT }, (_unused, index) => `row-${index}`),
      []
    )
    const indexes = React.useMemo(
      () => new Map(blocks.map((_block, index) => [index, index])),
      [blocks]
    )
    return (
      <div ref={parent} data-testid="timeline-parent">
        <VirtualChatTimeline
          blocks={blocks}
          getKey={(index) => blocks[index]}
          messageBlocks={indexes}
          scrollParentRef={parent}
          renderBlock={(block, index) => {
            renderedIndexes.add(index)
            return <div>{block}</div>
          }}
        />
      </div>
    )
  }
  const page = render(<Case />)
  // The scroll parent's ref belongs to an ancestor fiber: it attaches after
  // the timeline's first layout effects, so the virtualizer observes the
  // element only from the next render on (in the app, the parent re-renders
  // on every store update; mirror that with one rerender).
  page.rerender(<Case />)
  return {
    parent: page.getByTestId("timeline-parent"),
    renderedIndexes
  }
}

describe("VirtualChatTimeline windowed rendering (Stage 3, WP-03)", () => {
  it("mounts a bounded window for a 1,000-block transcript", async () => {
    const { parent, renderedIndexes } = renderTimeline()

    await vi.waitFor(() => {
      expect(parent.textContent).toContain("row-0")
    })
    // Plan success criterion: <= ~20 shells; e2e bound < 60.
    expect(renderedIndexes.size).toBeLessThan(60)
    expect(renderedIndexes.size).toBeGreaterThan(0)
    // The un-mounted majority must not be in the DOM.
    expect(parent.textContent).not.toContain("row-999")
  })

  it("mounts the last (streaming) block when the window is pinned to the end", async () => {
    const { parent, renderedIndexes } = renderTimeline()

    parent.scrollTop = 10_000_000
    fireEvent.scroll(parent)

    await vi.waitFor(() => {
      expect(parent.textContent).toContain(`row-${MESSAGE_COUNT - 1}`)
    })
    // The end window stays bounded too.
    expect(renderedIndexes.size).toBeLessThan(60)
    expect(renderedIndexes.has(MESSAGE_COUNT - 1)).toBe(true)
  })
})

import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { useVirtualizer } from "@tanstack/react-virtual"
import {
  VirtualChatTimeline,
  type ChatTimelineNavigation
} from "../VirtualChatTimeline"

it.each([
  "before first frame",
  "after first frame",
  "later explicit scroll",
  "timeline owner replacement"
] as const)(
  "stops installed measured navigation when React replaces its owner (scenario: %s)",
  async (scenario) => {
    const frames = new Map<number, FrameRequestCallback>()
    let frameId = 0
    const raf = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        frames.set(++frameId, callback)
        return frameId
      })
    const cancel = vi
      .spyOn(window, "cancelAnimationFrame")
      .mockImplementation((id) => {
        frames.delete(id)
      })
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    )
    const height = vi
      .spyOn(HTMLElement.prototype, "offsetHeight", "get")
      .mockReturnValue(160)
    const width = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(800)
    const rect = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(() => ({
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
      }))
    const scrolls: HTMLElement[] = []
    const scrollDescriptor = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "scrollTo"
    )
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      value() {}
    })
    const scrollTo = vi
      .spyOn(HTMLElement.prototype, "scrollTo")
      .mockImplementation(function (
        this: HTMLElement,
        options?: ScrollToOptions | number
      ) {
        scrolls.push(this)
        if (typeof options === "object") this.scrollTop = options.top ?? 0
      })
    const navigation = { current: null as ChatTimelineNavigation | null }
    const blocks = Array.from({ length: 201 }, (_, index) => `row-${index}`)
    const indexes = new Map(blocks.map((_, index) => [index, index]))
    const nativeLoop =
      scenario === "before first frame" || scenario === "after first frame"
    // Existing sidepanel consumers still use native index navigation. Qualify
    // its supported dependency cleanup separately from the coarse H1 timeline.
    function NativeTimeline({
      parent
    }: {
      parent: React.RefObject<HTMLDivElement>
    }) {
      const virtualizer = useVirtualizer({
        count: blocks.length,
        getScrollElement: () => parent.current,
        estimateSize: () => 160
      })
      React.useLayoutEffect(() => {
        navigation.current = {
          reveal: async (index) => {
            virtualizer.scrollToIndex(index, { align: "center" })
            return true
          }
        }
        return () => {
          navigation.current = null
        }
      }, [virtualizer])
      return (
        <div style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((row) => (
            <div
              key={row.key}
              data-index={row.index}
              ref={virtualizer.measureElement}
            >
              {blocks[row.index]}
            </div>
          ))}
        </div>
      )
    }
    function Owner({ owner }: { owner: string }) {
      const parent = React.useRef<HTMLDivElement>(null)
      const [mounted, setMounted] = React.useState(false)
      React.useLayoutEffect(() => {
        setMounted(true)
      }, [])
      return (
        <div ref={parent} data-testid={owner}>
          {mounted &&
            (nativeLoop ? (
              <NativeTimeline parent={parent} />
            ) : (
              <VirtualChatTimeline
                blocks={blocks}
                getKey={(index) => blocks[index]}
                messageBlocks={indexes}
                scrollParentRef={parent}
                navigationRef={navigation}
                renderBlock={(row) => <div>{row}</div>}
              />
            ))}
        </div>
      )
    }
    const page = render(<Owner key="A" owner="A" />)
    try {
      expect(screen.getByText("row-1")).toBeInTheDocument()
      frames.clear()
      await act(async () => {
        expect(await navigation.current!.reveal(1)).toBe(true)
      })
      if (nativeLoop) expect(frames.size).toBeGreaterThan(0)
      if (scenario === "after first frame")
        await act(async () => {
          const [id, callback] = frames.entries().next().value!
          frames.delete(id)
          callback(0)
        })
      if (scenario === "later explicit scroll") {
        const parent = screen.getByTestId("A")
        parent.scrollTop = 5000
        fireEvent.scroll(parent)
      } else {
        page.rerender(<Owner key="B" owner="B" />)
        expect(screen.queryByTestId("A")).not.toBeInTheDocument()
      }
      scrolls.length = 0
      await act(async () => {
        while (frames.size) {
          const [id, callback] = frames.entries().next().value!
          frames.delete(id)
          callback(0)
        }
      })
      expect(scrolls).toEqual([])
      expect(navigation.current).not.toBeNull()
    } finally {
      page.unmount()
      scrollTo.mockRestore()
      if (scrollDescriptor)
        Object.defineProperty(
          HTMLElement.prototype,
          "scrollTo",
          scrollDescriptor
        )
      else delete (HTMLElement.prototype as Partial<HTMLElement>).scrollTo
      rect.mockRestore()
      width.mockRestore()
      height.mockRestore()
      raf.mockRestore()
      cancel.mockRestore()
      vi.unstubAllGlobals()
    }
  }
)

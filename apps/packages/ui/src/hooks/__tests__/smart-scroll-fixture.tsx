import React from "react"
import { act, fireEvent, render } from "@testing-library/react"
import { vi } from "vitest"
import { useSmartScroll } from "../useSmartScroll"

export type FixtureMessage = { id: string; isBot: boolean; message?: string }

export type ScrollMetrics = {
  scrollTop: number
  scrollHeight: number
  clientHeight: number
}

type Api = ReturnType<typeof useSmartScroll>

/**
 * Mounts useSmartScroll on a real <div> whose scroll metrics are driven by
 * the test, with requestAnimationFrame and ResizeObserver under test control.
 * Call `vi.useFakeTimers()` before using it.
 */
export const renderSmartScroll = ({
  messages,
  streaming = false,
  metrics: initialMetrics
}: {
  messages: FixtureMessage[]
  streaming?: boolean
  metrics: ScrollMetrics
}) => {
  const frames: FrameRequestCallback[] = []
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames.push(callback)
    return frames.length
  })
  const resizeCallbacks: Array<() => void> = []
  vi.stubGlobal(
    "ResizeObserver",
    class {
      private readonly callback: () => void
      constructor(callback: () => void) {
        this.callback = callback
        resizeCallbacks.push(callback)
      }
      observe() {}
      unobserve() {}
      disconnect() {
        const index = resizeCallbacks.indexOf(this.callback)
        if (index >= 0) resizeCallbacks.splice(index, 1)
      }
    }
  )

  const metrics = { ...initialMetrics }
  let element: HTMLDivElement | null = null
  // Like a browser, a programmatic scroll that moves the scroller fires a
  // scroll event.
  const scrollTo = vi.fn(({ top }: { top: number; behavior?: string }) => {
    const next = Math.max(
      0,
      Math.min(Number(top), metrics.scrollHeight - metrics.clientHeight)
    )
    if (next === metrics.scrollTop) return
    metrics.scrollTop = next
    if (element) fireEvent.scroll(element)
  })
  const fake = (element: HTMLDivElement) => {
    Object.defineProperties(element, {
      scrollTop: {
        configurable: true,
        get: () => metrics.scrollTop,
        set: (value: number) => {
          metrics.scrollTop = Number(value)
        }
      },
      scrollHeight: { configurable: true, get: () => metrics.scrollHeight },
      clientHeight: { configurable: true, get: () => metrics.clientHeight },
      scrollTo: { configurable: true, value: scrollTo }
    })
  }

  const current: { api: Api | null } = { api: null }

  const Probe = (props: { messages: FixtureMessage[]; streaming: boolean }) => {
    const api = useSmartScroll(props.messages, props.streaming, 100)
    current.api = api
    const attach = React.useCallback(
      (node: HTMLDivElement | null) => {
        if (node) {
          fake(node)
          element = node
        }
        ;(api.containerRef as React.MutableRefObject<HTMLDivElement | null>).current =
          node
      },
      [api.containerRef]
    )
    return <div ref={attach} />
  }

  const view = render(<Probe messages={messages} streaming={streaming} />)

  /** Runs queued animation frames and pending timers (scroll debounce). */
  const settle = () => {
    for (let round = 0; round < 3; round += 1) {
      act(() => {
        frames.splice(0).forEach((callback) => callback(0))
      })
      act(() => {
        vi.advanceTimersByTime(400)
      })
    }
  }

  return {
    metrics,
    scrollTo,
    settle,
    get api(): Api {
      return current.api as Api
    },
    rerender: (next: { messages: FixtureMessage[]; streaming?: boolean }) => {
      view.rerender(
        <Probe messages={next.messages} streaming={next.streaming ?? false} />
      )
    },
    /**
     * The reader moves the scroller to `top`. Like a real wheel or drag
     * gesture it emits more than one scroll event: a small step, then the
     * destination.
     */
    userScrollTo: (top: number) => {
      const start = metrics.scrollTop
      if (top !== start) {
        metrics.scrollTop = start + Math.sign(top - start) * 10
        fireEvent.scroll(element as HTMLDivElement)
      }
      metrics.scrollTop = top
      fireEvent.scroll(element as HTMLDivElement)
    },
    /** The browser reports a scroll after layout changed under the reader. */
    fireScroll: () => {
      fireEvent.scroll(element as HTMLDivElement)
    },
    resize: () => {
      act(() => {
        resizeCallbacks.forEach((callback) => callback())
      })
    }
  }
}

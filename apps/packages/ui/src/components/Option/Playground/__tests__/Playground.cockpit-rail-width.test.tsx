// @vitest-environment jsdom
// CO-04 (#3113): the cockpit rails are width-aware. Below a threshold only
// one rail is open at a time, and a narrow shell overlays the rail on the
// transcript. Neither changes the saved rail preferences.
import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

let reportedWidth = 0
const observers: Array<{
  callback: ResizeObserverCallback
  target: Element | null
}> = []
const OriginalResizeObserver = window.ResizeObserver

class WidthReportingObserver {
  private entry: { callback: ResizeObserverCallback; target: Element | null }
  constructor(callback: ResizeObserverCallback) {
    this.entry = { callback, target: null }
    observers.push(this.entry)
  }
  observe(target: Element) {
    this.entry.target = target
    this.entry.callback(
      [{ target, contentRect: { width: reportedWidth } } as ResizeObserverEntry],
      this as unknown as ResizeObserver
    )
  }
  unobserve() {}
  disconnect() {
    this.entry.target = null
  }
}

const resizeShellTo = (width: number) => {
  reportedWidth = width
  act(() => {
    for (const observer of observers) {
      if (!observer.target) continue
      observer.callback(
        [
          {
            target: observer.target,
            contentRect: { width }
          } as ResizeObserverEntry
        ],
        {} as ResizeObserver
      )
    }
  })
}

const renderShell = (width: number) => {
  reportedWidth = width
  const onLeftRailVisibleChange = vi.fn()
  const onRightRailVisibleChange = vi.fn()
  render(
    <PlaygroundCockpitShell
      mode="cockpit"
      leftRailVisible
      rightRailVisible
      onLeftRailVisibleChange={onLeftRailVisibleChange}
      onRightRailVisibleChange={onRightRailVisibleChange}
      leftRail={<div>Context tools</div>}
      rightRail={<div>Runtime tools</div>}
    >
      <div>Chat transcript</div>
    </PlaygroundCockpitShell>
  )
  return { onLeftRailVisibleChange, onRightRailVisibleChange }
}

const gridColumns = () =>
  screen
    .getByTestId("playground-cockpit-main")
    .parentElement?.style.getPropertyValue("--cockpit-grid-columns")

describe("Playground cockpit rails by available width", () => {
  beforeEach(() => {
    // The last-opened rail is kept for the session; start each test fresh.
    window.sessionStorage.clear()
    observers.length = 0
    ;(window as any).ResizeObserver = WidthReportingObserver
    ;(globalThis as any).ResizeObserver = WidthReportingObserver
  })
  afterEach(() => {
    ;(window as any).ResizeObserver = OriginalResizeObserver
    ;(globalThis as any).ResizeObserver = OriginalResizeObserver
  })

  it("keeps both rails inline on a wide shell", () => {
    renderShell(1440)

    expect(screen.getByTestId("playground-cockpit-left-rail")).toBeInTheDocument()
    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
    expect(gridColumns()).toBe(
      "minmax(220px,280px) minmax(0,1fr) minmax(240px,300px)"
    )
  })

  it("opens one rail at a time at 1024px and swaps without touching the saved preferences", () => {
    const { onLeftRailVisibleChange, onRightRailVisibleChange } =
      renderShell(1024)

    expect(screen.getByTestId("playground-cockpit-left-rail")).toBeInTheDocument()
    expect(
      screen.queryByTestId("playground-cockpit-right-rail")
    ).not.toBeInTheDocument()
    expect(gridColumns()).toBe("minmax(220px,280px) minmax(0,1fr)")
    const shell = screen.getByTestId("playground-cockpit-shell")
    expect(shell).toHaveAttribute("data-right-rail", "hidden")

    fireEvent.click(screen.getByTestId("playground-cockpit-right-rail-restore"))

    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
    expect(
      screen.queryByTestId("playground-cockpit-left-rail")
    ).not.toBeInTheDocument()
    expect(gridColumns()).toBe("minmax(0,1fr) minmax(240px,300px)")
    expect(onLeftRailVisibleChange).not.toHaveBeenCalled()
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true)
  })

  it("brings both rails back when the shell widens again", () => {
    renderShell(1024)
    resizeShellTo(1440)

    expect(screen.getByTestId("playground-cockpit-left-rail")).toBeInTheDocument()
    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
  })

  it("overlays the rail on the transcript when the shell is narrow", () => {
    renderShell(800)

    const rail = screen.getByTestId("playground-cockpit-left-rail")
    expect(rail).toHaveAttribute("data-placement", "overlay")
    expect(rail.className).toContain("absolute")
    expect(gridColumns()).toBe("minmax(0,1fr)")
  })

  it("names the restore tabs Context and Runtime, without 'rail'", () => {
    renderShell(1024)

    expect(
      screen.getByTestId("playground-cockpit-right-rail-restore")
    ).toHaveTextContent(/^Runtime$/)
  })

  // AX-04 review: the overlay stays off the composer, closing it doesn't open
  // the other rail as a second overlay, and keyboard focus follows it.
  const StatefulShell = ({
    width,
    overlayBottomInset,
    onLeft = vi.fn(),
    onRight = vi.fn(),
    leftRail = <div>Context tools</div>
  }: {
    width: number
    overlayBottomInset?: number
    onLeft?: (visible: boolean) => void
    onRight?: (visible: boolean) => void
    leftRail?: React.ReactNode
  }) => {
    reportedWidth = width
    const [left, setLeft] = React.useState(true)
    const [right, setRight] = React.useState(true)
    return (
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={left}
        rightRailVisible={right}
        overlayBottomInset={overlayBottomInset}
        onLeftRailVisibleChange={(visible) => {
          onLeft(visible)
          setLeft(visible)
        }}
        onRightRailVisibleChange={(visible) => {
          onRight(visible)
          setRight(visible)
        }}
        leftRail={leftRail}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )
  }

  it("keeps the overlay above the composer dock", () => {
    render(<StatefulShell width={800} overlayBottomInset={120} />)

    expect(screen.getByTestId("playground-cockpit-left-rail").style.bottom).toBe(
      "120px"
    )
  })

  it("does not open the other rail as an overlay when the shown one closes", () => {
    render(<StatefulShell width={800} />)
    fireEvent.click(screen.getByRole("button", { name: "Hide context" }))

    expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull()
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull()
    expect(
      screen.getByTestId("playground-cockpit-right-rail-restore")
    ).toBeInTheDocument()
  })

  it("moves focus into an overlay it opens, and Escape closes it back to its tab", () => {
    const onRight = vi.fn()
    const windowKeys = vi.fn()
    window.addEventListener("keydown", windowKeys)
    render(<StatefulShell width={800} onRight={onRight} />)
    fireEvent.click(screen.getByRole("button", { name: "Hide context" }))

    fireEvent.click(screen.getByTestId("playground-cockpit-right-rail-restore"))
    const overlay = screen.getByTestId("playground-cockpit-right-rail")
    expect(overlay).toHaveAttribute("data-placement", "overlay")
    expect(overlay).toContainElement(document.activeElement as HTMLElement)

    fireEvent.keyDown(document.activeElement as HTMLElement, { key: "Escape" })

    expect(onRight).toHaveBeenLastCalledWith(false)
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull()
    expect(screen.getByTestId("playground-cockpit-right-rail-restore")).toHaveFocus()
    expect(windowKeys).not.toHaveBeenCalled()
    window.removeEventListener("keydown", windowKeys)
  })

  // Minor 7: no one-frame flash of the wrong layout.
  it("measures the shell before the first paint", () => {
    observers.length = 0
    ;(window as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    const original = HTMLElement.prototype.getBoundingClientRect
    HTMLElement.prototype.getBoundingClientRect = function () {
      return { width: 1024, height: 600, top: 0, left: 0, right: 1024, bottom: 600, x: 0, y: 0, toJSON() {} } as DOMRect
    }
    try {
      renderShell(0)
      expect(
        screen.queryByTestId("playground-cockpit-right-rail")
      ).not.toBeInTheDocument()
    } finally {
      HTMLElement.prototype.getBoundingClientRect = original
    }
  })

  it("shows a newly opened rail in the same render, without first showing the other", () => {
    // Counts commits of the Context rail's desktop column (the mobile tab
    // panel renders the same content too, so count only inside the aside).
    const contextRenders = vi.fn()
    const Probe = () => {
      const ref = React.useRef<HTMLDivElement>(null)
      React.useLayoutEffect(() => {
        if (ref.current?.closest("#playground-cockpit-left-rail")) contextRenders()
      })
      return <div ref={ref}>Context tools</div>
    }
    reportedWidth = 1024
    const { rerender } = render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible={false}
        leftRail={<Probe />}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )
    contextRenders.mockClear()

    rerender(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        leftRail={<Probe />}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )

    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
    expect(contextRenders).not.toHaveBeenCalled()
  })

  // Minor 8: the last-opened rail survives a remount within the session.
  it("remembers the last opened rail for the session", () => {
    window.sessionStorage.clear()
    const { unmount } = render(<StatefulShell width={1024} />)
    fireEvent.click(screen.getByTestId("playground-cockpit-right-rail-restore"))
    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
    unmount()

    render(<StatefulShell width={1024} />)

    expect(screen.getByTestId("playground-cockpit-right-rail")).toBeInTheDocument()
    expect(screen.queryByTestId("playground-cockpit-left-rail")).toBeNull()
    window.sessionStorage.clear()
  })

  // Saved rail preferences load after the first render, so both rails turn on
  // together; that is not the user opening Runtime, so Context (or the rail
  // opened last this session) stays.
  it("keeps the preferred rail when both saved rails load together", () => {
    reportedWidth = 1024
    const { rerender } = render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={false}
        rightRailVisible={false}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )
    rerender(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>
    )

    expect(screen.getByTestId("playground-cockpit-left-rail")).toBeInTheDocument()
    expect(screen.queryByTestId("playground-cockpit-right-rail")).toBeNull()
  })
})

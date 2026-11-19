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
})

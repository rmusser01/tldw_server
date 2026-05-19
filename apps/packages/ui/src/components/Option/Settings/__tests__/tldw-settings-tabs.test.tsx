// @vitest-environment jsdom

import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TldwSettingsTabs } from "../tldw-settings-tabs"

const tabsRenderSpy = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("antd", () => ({
  Tabs: ({
    activeKey,
    className,
    items,
    onChange,
    onTabClick
  }: {
    activeKey?: string
    className?: string
    items?: Array<{ key: string; label: React.ReactNode }>
    onChange?: (key: string) => void
    onTabClick?: (key: string) => void
  }) => {
    tabsRenderSpy({ activeKey, className, items })

    return (
      <div data-testid="tabs-root" data-active-key={activeKey} className={className}>
        {items?.map((item) => (
          <button
            key={item.key}
            type="button"
            data-testid={`tab-${item.key}`}
            data-active={activeKey === item.key ? "true" : "false"}
            onClick={() => {
              onTabClick?.(item.key)
              if (item.key !== activeKey) {
                onChange?.(item.key)
              }
            }}>
            {item.label}
          </button>
        ))}
      </div>
    )
  }
}))

type MockIntersectionObserverEntry = Pick<
  IntersectionObserverEntry,
  "target" | "isIntersecting" | "intersectionRatio"
>

const observerState = vi.hoisted(() => ({
  instances: [] as Array<{
    callback: IntersectionObserverCallback
    observe: ReturnType<typeof vi.fn>
    disconnect: ReturnType<typeof vi.fn>
  }>
}))

class IntersectionObserverMock {
  callback: IntersectionObserverCallback
  observe = vi.fn()
  disconnect = vi.fn()

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback
    observerState.instances.push({
      callback,
      observe: this.observe,
      disconnect: this.disconnect
    })
  }
}

const installSectionTargets = () => {
  document.body.innerHTML = `
    <div id="tldw-settings-connection"></div>
    <div id="tldw-settings-timeouts"></div>
    <div id="tldw-settings-billing"></div>
  `
}

const emitIntersection = (entries: MockIntersectionObserverEntry[]) => {
  const instance = observerState.instances.at(-1)
  if (!instance) {
    throw new Error("No IntersectionObserver instance registered")
  }

  instance.callback(entries as IntersectionObserverEntry[], {} as IntersectionObserver)
}

describe("TldwSettingsTabs", () => {
  const originalIntersectionObserver = globalThis.IntersectionObserver

  beforeEach(() => {
    tabsRenderSpy.mockClear()
    observerState.instances.length = 0
    installSectionTargets()
    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: IntersectionObserverMock
    })
  })

  afterEach(() => {
    cleanup()
    document.body.innerHTML = ""
    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: originalIntersectionObserver
    })
  })

  it("keeps the navigation sticky and shows billing only for logged-in multi-user mode", () => {
    render(<TldwSettingsTabs authMode="multi-user" isLoggedIn />)

    expect(screen.getByTestId("tabs-root")).toHaveClass("sticky", "top-0")
    expect(screen.getByTestId("tab-billing")).toBeInTheDocument()
  })

  it("scrolls to the target section every time a tab is clicked, even if it is already active", () => {
    const scrollIntoView = vi.fn()
    Object.defineProperty(
      document.getElementById("tldw-settings-timeouts") as HTMLElement,
      "scrollIntoView",
      {
        configurable: true,
        value: scrollIntoView
      }
    )

    render(<TldwSettingsTabs authMode="single-user" isLoggedIn={false} />)

    const timeoutsTab = screen.getByTestId("tab-timeouts")
    fireEvent.click(timeoutsTab)
    fireEvent.click(timeoutsTab)

    expect(scrollIntoView).toHaveBeenCalledTimes(2)
    expect(scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "start"
    })
  })

  it("updates the active tab when a different section becomes visible", () => {
    render(<TldwSettingsTabs authMode="single-user" isLoggedIn={false} />)

    act(() => {
      emitIntersection([
        {
          target: document.getElementById("tldw-settings-timeouts") as Element,
          isIntersecting: true,
          intersectionRatio: 0.8
        }
      ])
    })

    expect(screen.getByTestId("tabs-root")).toHaveAttribute(
      "data-active-key",
      "timeouts"
    )
    expect(screen.getByTestId("tab-timeouts")).toHaveAttribute(
      "data-active",
      "true"
    )
  })

  it("disconnects the observer on unmount", () => {
    const { unmount } = render(
      <TldwSettingsTabs authMode="single-user" isLoggedIn={false} />
    )

    const instance = observerState.instances.at(-1)
    if (!instance) {
      throw new Error("No IntersectionObserver instance registered")
    }

    unmount()

    expect(instance.disconnect).toHaveBeenCalledTimes(1)
  })
})

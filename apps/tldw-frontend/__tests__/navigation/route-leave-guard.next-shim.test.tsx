import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

// The web build resolves react-router-dom to the Next shim. This checks that
// the shared RouteLeaveGuard holds a Next navigation (router.push, Link, the
// rail) until the Notes leave flush finishes (#3102 NS-01).

const { mockRouter, routerEventHandlers } = vi.hoisted(() => {
  const handlers = new Map<string, Set<(...args: unknown[]) => void>>()
  const events = {
    on: (name: string, handler: (...args: unknown[]) => void) => {
      const set = handlers.get(name) ?? new Set()
      set.add(handler)
      handlers.set(name, set)
    },
    off: (name: string, handler: (...args: unknown[]) => void) => {
      handlers.get(name)?.delete(handler)
    },
    emit: (name: string, ...args: unknown[]) => {
      for (const handler of [...(handlers.get(name) ?? [])]) handler(...args)
    }
  }
  const router = {
    asPath: "/notes",
    pathname: "/notes",
    query: {},
    events,
    beforePopState: (_handler: unknown) => undefined,
    push: (href: string): Promise<boolean> => {
      try {
        events.emit("routeChangeStart", href, { shallow: false })
      } catch (error) {
        return Promise.reject(error)
      }
      router.asPath = href
      return Promise.resolve(true)
    },
    replace: (href: string): Promise<boolean> => router.push(href)
  }
  return { mockRouter: router, routerEventHandlers: handlers }
})

vi.mock("next/router", () => ({ useRouter: () => mockRouter }))
vi.mock("react-router-dom", async () => await import("@web/extension/shims/react-router-dom"))

import { RouteLeaveGuard } from "@/entries/shared/route-leave-guard"

describe("RouteLeaveGuard on the Next router shim", () => {
  beforeEach(() => {
    routerEventHandlers.clear()
    mockRouter.asPath = "/notes"
  })

  it("holds router.push until the leave work allows it, then navigates once", async () => {
    let allow!: (value: boolean) => void
    const onLeave = vi.fn(() => new Promise<boolean>((resolve) => { allow = resolve }))
    const pushSpy = vi.spyOn(mockRouter, "push")
    render(<RouteLeaveGuard when onLeave={onLeave} />)

    await act(async () => {
      await mockRouter.push("/chat").catch(() => undefined)
    })

    await waitFor(() => expect(onLeave).toHaveBeenCalledTimes(1))
    expect(mockRouter.asPath).toBe("/notes")

    await act(async () => {
      allow(true)
    })

    await waitFor(() => expect(mockRouter.asPath).toBe("/chat"))
    expect(pushSpy).toHaveBeenCalledTimes(2)
    expect(onLeave).toHaveBeenCalledTimes(1)
  })

  it("keeps the page when the leave work says no", async () => {
    const onLeave = vi.fn(async () => false)
    render(<RouteLeaveGuard when onLeave={onLeave} />)

    await act(async () => {
      await mockRouter.push("/chat").catch(() => undefined)
    })

    await waitFor(() => expect(onLeave).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(mockRouter.asPath).toBe("/notes")
  })

  it("does nothing while there is nothing to protect", async () => {
    const onLeave = vi.fn(async () => true)
    render(<RouteLeaveGuard when={false} onLeave={onLeave} />)

    await act(async () => {
      await mockRouter.push("/chat")
    })

    expect(mockRouter.asPath).toBe("/chat")
    expect(onLeave).not.toHaveBeenCalled()
  })
})

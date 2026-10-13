import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useBlocker } from "@web/extension/shims/react-router-dom"

// The Next shim's useBlocker lets shared code hold an in-app navigation while
// it finishes async work (the Notes leave flush, #3102 NS-01), like
// react-router's data-router blocker does in the extension.

const routerEventHandlers = new Map<string, Set<(...args: unknown[]) => void>>()
const mockRouterEvents = {
  on: vi.fn((name: string, handler: (...args: unknown[]) => void) => {
    const handlers = routerEventHandlers.get(name) ?? new Set()
    handlers.add(handler)
    routerEventHandlers.set(name, handlers)
  }),
  off: vi.fn((name: string, handler: (...args: unknown[]) => void) => {
    routerEventHandlers.get(name)?.delete(handler)
  }),
  emit: vi.fn((name: string, ...args: unknown[]) => {
    for (const handler of [...(routerEventHandlers.get(name) ?? [])]) handler(...args)
  })
}

let beforePopStateHandler: (state: { url: string; as: string; options: Record<string, unknown> }) => boolean =
  () => true

const mockRouter = {
  asPath: "/notes",
  pathname: "/notes",
  query: {} as Record<string, string | string[] | undefined>,
  push: vi.fn(),
  replace: vi.fn(),
  beforePopState: vi.fn((handler: typeof beforePopStateHandler) => {
    beforePopStateHandler = handler
  }),
  events: mockRouterEvents
}

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

/** Mimics Next: emit routeChangeStart, reject when a listener cancels. */
const navigate = (method: "push" | "replace", href: string): Promise<boolean> => {
  try {
    mockRouterEvents.emit("routeChangeStart", href, { shallow: false })
  } catch (error) {
    return Promise.reject(error)
  }
  mockRouter.asPath = href
  mockRouter.pathname = href.split("?")[0]
  return Promise.resolve(true)
}

/** Mimics a browser Back press: Next asks beforePopState, then changes route. */
const popTo = (href: string): boolean => {
  const allowed = beforePopStateHandler({ url: href, as: href, options: {} })
  if (allowed) void navigate("replace", href)
  return allowed
}

const leavingPathOnly = ({
  currentLocation,
  nextLocation
}: {
  currentLocation: { pathname: string }
  nextLocation: { pathname: string }
}) => currentLocation.pathname !== nextLocation.pathname

describe("Next shim useBlocker", () => {
  beforeEach(() => {
    routerEventHandlers.clear()
    mockRouterEvents.on.mockClear()
    mockRouterEvents.off.mockClear()
    mockRouterEvents.emit.mockClear()
    mockRouter.asPath = "/notes"
    mockRouter.pathname = "/notes"
    mockRouter.push.mockReset()
    mockRouter.replace.mockReset()
    mockRouter.push.mockImplementation((href: string) => navigate("push", href))
    mockRouter.replace.mockImplementation((href: string) => navigate("replace", href))
    beforePopStateHandler = () => true
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("advertises that it can block Next router navigations", () => {
    expect((useBlocker as unknown as { tldwNextRouterShim?: boolean }).tldwNextRouterShim).toBe(true)
  })

  it("stays inert when told never to block, leaving other guards' handlers alone", async () => {
    const { result } = renderHook(() => useBlocker(false))
    expect(result.current.state).toBe("unblocked")
    expect(mockRouterEvents.on).not.toHaveBeenCalled()
    expect(mockRouter.beforePopState).not.toHaveBeenCalled()
    await expect(mockRouter.push("/chat")).resolves.toBe(true)
  })

  it("lets navigations through while the blocker function declines", async () => {
    const { result } = renderHook(() => useBlocker(() => false))
    await expect(mockRouter.push("/chat")).resolves.toBe(true)
    expect(result.current.state).toBe("unblocked")
  })

  it("holds a push to another page, then proceeds to it exactly once", async () => {
    const { result } = renderHook(() => useBlocker(leavingPathOnly))

    let rejected: unknown = null
    await act(async () => {
      await mockRouter.push("/chat?tab=1").catch((error: unknown) => {
        rejected = error
      })
    })

    expect(rejected).toMatchObject({ cancelled: true })
    expect(mockRouterEvents.emit).toHaveBeenCalledWith(
      "routeChangeError",
      expect.objectContaining({ cancelled: true }),
      "/chat?tab=1",
      { shallow: false }
    )
    expect(mockRouter.asPath).toBe("/notes")
    expect(result.current.state).toBe("blocked")
    expect(result.current.location).toMatchObject({ pathname: "/chat", search: "?tab=1" })

    mockRouter.push.mockClear()
    await act(async () => {
      result.current.proceed()
    })

    expect(mockRouter.push).toHaveBeenCalledTimes(1)
    expect(mockRouter.push).toHaveBeenCalledWith("/chat?tab=1")
    expect(mockRouter.asPath).toBe("/chat?tab=1")
    expect(result.current.state).toBe("unblocked")
  })

  it("passes the current and next locations to the blocker function", async () => {
    const shouldBlock = vi.fn(leavingPathOnly)
    renderHook(() => useBlocker(shouldBlock))

    await expect(mockRouter.push("/notes?note=7")).resolves.toBe(true)

    expect(shouldBlock).toHaveBeenCalledWith(
      expect.objectContaining({
        currentLocation: expect.objectContaining({ pathname: "/notes" }),
        nextLocation: expect.objectContaining({ pathname: "/notes", search: "?note=7" }),
        historyAction: "PUSH"
      })
    )
  })

  it("stays on the page when reset", async () => {
    const { result } = renderHook(() => useBlocker(true))
    await act(async () => {
      await mockRouter.push("/chat").catch(() => undefined)
    })
    mockRouter.push.mockClear()

    act(() => {
      result.current.reset()
    })

    expect(result.current.state).toBe("unblocked")
    expect(mockRouter.push).not.toHaveBeenCalled()
    expect(mockRouter.asPath).toBe("/notes")
  })

  it("holds a browser Back press and finishes it with a replace", async () => {
    const { result } = renderHook(() => useBlocker(leavingPathOnly))

    let allowed = true
    act(() => {
      allowed = popTo("/chat")
    })

    expect(allowed).toBe(false)
    expect(result.current.state).toBe("blocked")
    expect(result.current.location).toMatchObject({ pathname: "/chat" })

    await act(async () => {
      result.current.proceed()
    })

    expect(mockRouter.replace).toHaveBeenCalledWith("/chat")
    expect(mockRouter.asPath).toBe("/chat")
    expect(result.current.state).toBe("unblocked")
  })

  it("restores the guarded page when a held Back press is reset", async () => {
    const { result } = renderHook(() => useBlocker(true))
    act(() => {
      popTo("/chat")
    })

    await act(async () => {
      result.current.reset()
    })

    expect(mockRouter.push).toHaveBeenCalledWith("/notes")
    expect(mockRouter.asPath).toBe("/notes")
    expect(result.current.state).toBe("unblocked")
  })

  it("removes its listeners and releases Back on unmount", () => {
    const { unmount } = renderHook(() => useBlocker(true))
    expect(routerEventHandlers.get("routeChangeStart")?.size ?? 0).toBe(1)

    unmount()

    expect(routerEventHandlers.get("routeChangeStart")?.size ?? 0).toBe(0)
    expect(routerEventHandlers.get("hashChangeStart")?.size ?? 0).toBe(0)
    expect(beforePopStateHandler({ url: "/chat", as: "/chat", options: {} })).toBe(true)
  })
})

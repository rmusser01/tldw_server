import React from "react"
import { render, renderHook, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import * as routerShim from "@web/extension/shims/react-router-dom"
import {
  Navigate,
  UNSAFE_DataRouterContext,
  useNavigate,
  useParams,
  useSearchParams
} from "@web/extension/shims/react-router-dom"

const mockPush = vi.fn()
const mockReplace = vi.fn()
const mockBack = vi.fn()
const mockBeforePopState = vi.fn()
const mockPopAccepted = vi.fn()
const routerEventHandlers = new Map<string, Set<(...args: any[]) => void>>()
const mockRouterEvents = {
  on: vi.fn((name: string, handler: (...args: any[]) => void) => {
    const handlers = routerEventHandlers.get(name) ?? new Set()
    handlers.add(handler)
    routerEventHandlers.set(name, handlers)
  }),
  off: vi.fn((name: string, handler: (...args: any[]) => void) => {
    routerEventHandlers.get(name)?.delete(handler)
  }),
  emit: vi.fn((name: string, ...args: any[]) => {
    for (const handler of [...(routerEventHandlers.get(name) ?? [])]) handler(...args)
  })
}

let beforePopStateHandler: (state: {
  url: string
  as: string
  options: Record<string, unknown>
}) => boolean = () => true

const mockRouter = {
  asPath: "/current?tab=one",
  pathname: "/current",
  query: {} as Record<string, string | string[] | undefined>,
  push: mockPush,
  replace: mockReplace,
  back: mockBack,
  beforePopState: mockBeforePopState,
  events: mockRouterEvents
}

let mockCurrentRouter = mockRouter

vi.mock("next/router", () => ({
  useRouter: () => mockCurrentRouter
}))

const NavigateButton = ({
  to,
  replace = false,
  flushSync = false
}: {
  to: string | number
  replace?: boolean
  flushSync?: boolean
}) => {
  const navigate = useNavigate()
  return (
    <button
      type="button"
      onClick={() =>
        navigate(
          to,
          replace || flushSync
            ? { replace: replace || undefined, flushSync: flushSync || undefined }
            : undefined
        )
      }
    >
      navigate
    </button>
  )
}

const SearchParamsButton = () => {
  const [, setSearchParams] = useSearchParams()
  return (
    <button
      type="button"
      onClick={() => setSearchParams({ q: "updated" })}
    >
      search
    </button>
  )
}

const ParamsReader = () => {
  const params = useParams<{ sourceId?: string }>()
  return <span>{params.sourceId ?? "missing"}</span>
}

const RouterContextReader = () => (
  <span>{routerShim.useInRouterContext?.() ? "in-router" : "out-of-router"}</span>
)

const useShimPrompt =
  (routerShim as typeof routerShim & {
    unstable_usePrompt?: (options: { when: boolean; message: string }) => void
  }).unstable_usePrompt ?? (() => undefined)

const LEAVE_MESSAGE = "Leave without saving? Your local draft is preserved only in this tab."

const GuardedDraft = () => {
  const [draft, setDraft] = React.useState("")
  const navigate = useNavigate()
  useShimPrompt({ when: draft.length > 0, message: LEAVE_MESSAGE })
  return (
    <div>
      <label htmlFor="next-guarded-draft">Draft</label>
      <input
        id="next-guarded-draft"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <routerShim.Link to="/destination">Destination Link</routerShim.Link>
      <button type="button" onClick={() => navigate("/replacement", { replace: true })}>
        replace
      </button>
      <button type="button" onClick={() => navigate("/current#details")}>hash</button>
      <button type="button" onClick={() => navigate(-1)}>back</button>
    </div>
  )
}

describe("react-router-dom Next.js shim transitions", () => {
  let startTransitionSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    mockPush.mockReset()
    mockReplace.mockReset()
    mockBack.mockReset()
    mockBeforePopState.mockReset()
    mockPopAccepted.mockReset()
    mockRouterEvents.on.mockClear()
    mockRouterEvents.off.mockClear()
    mockRouterEvents.emit.mockClear()
    routerEventHandlers.clear()
    mockRouter.asPath = "/current?tab=one"
    mockRouter.pathname = "/current"
    mockRouter.query = {}
    mockCurrentRouter = mockRouter
    beforePopStateHandler = () => true
    mockBeforePopState.mockImplementation((handler) => {
      beforePopStateHandler = handler
    })
    const runRouteStart = (href: unknown) => {
      const target =
        typeof href === "string"
          ? href
          : String((href as { pathname?: unknown } | null)?.pathname ?? href)
      const eventName = target.includes("#") ? "hashChangeStart" : "routeChangeStart"
      mockRouterEvents.emit(eventName, target, { shallow: false })
      mockRouter.asPath = target
      return true
    }
    mockPush.mockImplementation((href) => {
      try {
        return Promise.resolve(runRouteStart(href))
      } catch (error) {
        return Promise.reject(error)
      }
    })
    mockReplace.mockImplementation((href) => {
      try {
        return Promise.resolve(runRouteStart(href))
      } catch (error) {
        return Promise.reject(error)
      }
    })
    mockBack.mockImplementation(() => {
      const allowed = beforePopStateHandler({
        url: "/previous",
        as: "/previous",
        options: {}
      })
      if (!allowed) return
      mockRouterEvents.emit("routeChangeStart", "/previous", { shallow: false })
      mockRouter.asPath = "/previous"
      mockPopAccepted()
    })
    startTransitionSpy = vi.spyOn(React, "startTransition")
  })

  afterEach(() => {
    startTransitionSpy.mockRestore()
    vi.restoreAllMocks()
  })

  it.each([
    ["/media?id=1", "/chat?media_handoff=owned-token#composer"],
    ["/chat?media_handoff=owned-token#composer", "/media?id=1"],
    ["/sources/source%2F42?q=a%3Fb#part?1", "/media?id=1"]
  ])("keeps one location snapshot while router %s and browser %s differ", (route, browserUrl) => {
    const previousUrl = window.location.href
    mockRouter.asPath = route
    window.history.replaceState(null, "", browserUrl)
    const LocationReader = () => {
      const location = routerShim.useLocation()
      return <output aria-label="Current route">{location.pathname + location.search + location.hash}</output>
    }
    try {
      render(<LocationReader />)
      expect(screen.getByLabelText("Current route").textContent).toBe(route)
    } finally {
      window.history.replaceState(null, "", previousUrl)
    }
  })

  it("does not replay pending permalink synchronization after a Chat handoff", async () => {
    const user = userEvent.setup()
    mockRouter.asPath = "/media"
    mockRouter.pathname = "/media"
    const pending = new Promise<boolean>(() => undefined)
    // Next leaves asPath on the source while destination modules are loading.
    // A second replace during that interval cancels the user's pending push.
    mockReplace.mockReturnValue(pending)
    mockPush.mockReturnValue(pending)
    const destinations: string[] = []
    mockReplace.mockImplementation((href: string) => {
      destinations.push(href)
      return pending
    })
    mockPush.mockImplementation((href: string) => {
      destinations.push(href)
      return pending
    })
    const MediaPermalink = () => {
      const navigate = useNavigate()
      const location = routerShim.useLocation()
      const [notice, setNotice] = React.useState("")
      React.useEffect(() => {
        if (!location.search) navigate("/media?id=7", { replace: true })
      }, [location.search, navigate])
      return <>
        <button onClick={() => {
          navigate("/chat?media_handoff=owned-token")
          setNotice("Prepared source")
        }}>Chat with source</button>
        <output>{notice}</output>
      </>
    }
    render(<MediaPermalink />)
    await user.click(screen.getByRole("button", { name: "Chat with source" }))
    expect(screen.getByText("Prepared source")).toBeVisible()
    expect(destinations).toEqual(["/media?id=7", "/chat?media_handoff=owned-token"])
  })

  it("keeps a pending handoff when Next republishes its public router", async () => {
    const user = userEvent.setup()
    mockRouter.asPath = "/media"
    mockRouter.pathname = "/media"
    const pending = new Promise<boolean>(() => undefined)
    mockPush.mockReturnValue(pending)
    mockReplace.mockReturnValue(pending)
    const Source = () => {
      const navigate = useNavigate()
      const location = routerShim.useLocation()
      React.useEffect(() => {
        if (!location.search) navigate("/media?id=7", { replace: true })
      }, [location.search, navigate])
      return <button onClick={() => navigate("/chat?media_handoff=owned-token")}>Handoff</button>
    }
    const view = render(<Source />)
    await user.click(screen.getByRole("button", { name: "Handoff" }))
    // Next's AppContainer republishes its public router on root renders,
    // including hydration/Fast Refresh while a destination is still loading.
    mockCurrentRouter = { ...mockRouter }
    view.rerender(<Source />)
    expect(mockPush).toHaveBeenCalledExactlyOnceWith("/chat?media_handoff=owned-token")
    expect(mockReplace).toHaveBeenCalledExactlyOnceWith("/media?id=7")
  })

  it("uses the latest published router in a retained navigation callback", () => {
    const { result, rerender } = renderHook(() => useNavigate())
    const retainedNavigate = result.current
    const latestPush = vi.fn().mockResolvedValue(true)
    mockCurrentRouter = { ...mockRouter, asPath: "/latest?tab=two", push: latestPush }
    rerender()
    retainedNavigate({})
    expect(latestPush).toHaveBeenCalledExactlyOnceWith("/latest?tab=two")
    expect(mockPush).not.toHaveBeenCalled()
  })

  it("wraps useNavigate push updates in startTransition", async () => {
    const user = userEvent.setup()
    render(<NavigateButton to="/destination" />)

    await user.click(screen.getByRole("button", { name: "navigate" }))

    expect(startTransitionSpy).toHaveBeenCalled()
    expect(mockPush).toHaveBeenCalledWith("/destination")
  })

  it("runs flushSync useNavigate pushes without startTransition", async () => {
    const user = userEvent.setup()
    render(<NavigateButton to="/destination" flushSync />)

    await user.click(screen.getByRole("button", { name: "navigate" }))

    expect(startTransitionSpy).not.toHaveBeenCalled()
    expect(mockPush).toHaveBeenCalledWith("/destination")
  })

  it("wraps useNavigate back updates in startTransition", async () => {
    const user = userEvent.setup()
    render(<NavigateButton to={-1} />)

    await user.click(screen.getByRole("button", { name: "navigate" }))

    expect(startTransitionSpy).toHaveBeenCalled()
    expect(mockBack).toHaveBeenCalledTimes(1)
  })

  it("wraps useSearchParams updates in startTransition", async () => {
    const user = userEvent.setup()
    render(<SearchParamsButton />)

    await user.click(screen.getByRole("button", { name: "search" }))

    expect(startTransitionSpy).toHaveBeenCalled()
    expect(mockPush).toHaveBeenCalledWith("/current?q=updated")
  })

  it("wraps Navigate redirects in startTransition", async () => {
    render(<Navigate to="/redirected" replace />)

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith("/redirected")
    })
    expect(startTransitionSpy).toHaveBeenCalled()
  })

  it("exposes Next router query params through useParams", () => {
    mockRouter.query = { sourceId: "source-123" }

    render(<ParamsReader />)

    expect(screen.getByText("source-123")).toBeInTheDocument()
  })

  it("exports UNSAFE_DataRouterContext for shared route modules", () => {
    expect(UNSAFE_DataRouterContext).toBeDefined()
  })

  it("reports router context availability for shared components", () => {
    render(<RouterContextReader />)

    expect(screen.getByText("in-router")).toBeInTheDocument()
  })

  it("denies then allows a dirty Next Link without losing the local draft or hard-falling back", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    render(<GuardedDraft />)
    await user.type(screen.getByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("link", { name: "Destination Link" }))
    await waitFor(() => expect(confirm).toHaveBeenCalledTimes(1))
    expect(mockRouter.asPath).toBe("/current?tab=one")
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")
    expect(consoleError).not.toHaveBeenCalled()

    await user.click(screen.getByRole("link", { name: "Destination Link" }))
    await waitFor(() => expect(mockRouter.asPath).toBe("/destination"))
    expect(confirm).toHaveBeenCalledTimes(2)
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")
  })

  it.each([
    ["replace", "replace", "/replacement"],
    ["hash", "hash", "/current#details"]
  ])("denies then allows dirty Next %s navigation once", async (_case, buttonName, destination) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    render(<GuardedDraft />)
    await user.type(screen.getByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("button", { name: buttonName }))
    await waitFor(() => expect(confirm).toHaveBeenCalledTimes(1))
    expect(mockRouter.asPath).toBe("/current?tab=one")

    await user.click(screen.getByRole("button", { name: buttonName }))
    await waitFor(() => expect(mockRouter.asPath).toBe(destination))
    expect(confirm).toHaveBeenCalledTimes(2)
  })

  it("denies then allows browser POP with one prompt and one-shot route-event bypass", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    render(<GuardedDraft />)
    await user.type(screen.getByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("button", { name: "back" }))
    expect(mockPopAccepted).not.toHaveBeenCalled()
    expect(mockRouter.asPath).toBe("/current?tab=one")
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")

    await user.click(screen.getByRole("button", { name: "back" }))
    expect(mockPopAccepted).toHaveBeenCalledTimes(1)
    expect(mockRouter.asPath).toBe("/previous")
    expect(confirm).toHaveBeenCalledTimes(2)
  })

  it("leaves clean navigation unprompted and removes all guards after StrictMode cleanup", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm")
    const view = render(
      <React.StrictMode>
        <GuardedDraft />
      </React.StrictMode>
    )

    await user.click(screen.getByRole("link", { name: "Destination Link" }))
    await waitFor(() => expect(mockRouter.asPath).toBe("/destination"))
    expect(confirm).not.toHaveBeenCalled()
    expect(routerEventHandlers.get("routeChangeStart")?.size ?? 0).toBe(1)

    view.unmount()
    expect(routerEventHandlers.get("routeChangeStart")?.size ?? 0).toBe(0)
    mockRouter.asPath = "/current"
    await expect(mockPush("/after-unmount")).resolves.toBe(true)
    expect(mockRouter.asPath).toBe("/after-unmount")
    expect(confirm).not.toHaveBeenCalled()
  })
})

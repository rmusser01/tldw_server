import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
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

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
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

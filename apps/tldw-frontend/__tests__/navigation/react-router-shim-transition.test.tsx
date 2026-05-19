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

const mockRouter = {
  asPath: "/current?tab=one",
  pathname: "/current",
  query: {} as Record<string, string | string[] | undefined>,
  push: mockPush,
  replace: mockReplace,
  back: mockBack
}

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

const NavigateButton = ({
  to,
  replace = false
}: {
  to: string | number
  replace?: boolean
}) => {
  const navigate = useNavigate()
  return (
    <button
      type="button"
      onClick={() => navigate(to, replace ? { replace: true } : undefined)}
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

describe("react-router-dom Next.js shim transitions", () => {
  let startTransitionSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    mockPush.mockReset()
    mockReplace.mockReset()
    mockBack.mockReset()
    mockRouter.asPath = "/current?tab=one"
    mockRouter.pathname = "/current"
    mockRouter.query = {}
    startTransitionSpy = vi.spyOn(React, "startTransition")
  })

  afterEach(() => {
    startTransitionSpy.mockRestore()
  })

  it("wraps useNavigate push updates in startTransition", async () => {
    const user = userEvent.setup()
    render(<NavigateButton to="/destination" />)

    await user.click(screen.getByRole("button", { name: "navigate" }))

    expect(startTransitionSpy).toHaveBeenCalled()
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
})

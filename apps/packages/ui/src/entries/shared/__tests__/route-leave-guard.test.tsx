import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  MemoryRouter,
  RouterProvider,
  createMemoryRouter,
  useLocation,
  useNavigate
} from "react-router-dom"
import { RouteLeaveGuard } from "../route-leave-guard"

// RouteLeaveGuard holds an in-app navigation away from a page until async
// work (the Notes leave flush, #3102 NS-01) says whether to continue.

const deferred = () => {
  let resolve!: (value: boolean) => void
  const promise = new Promise<boolean>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const Where = () => <span data-testid="where">{useLocation().pathname + useLocation().search}</span>

const GuardedPage = ({ when, onLeave }: { when: boolean; onLeave: () => Promise<boolean> }) => {
  const navigate = useNavigate()
  return (
    <div>
      <RouteLeaveGuard when={when} onLeave={onLeave} />
      <Where />
      <button type="button" onClick={() => navigate("/chat")}>
        go to chat
      </button>
      <button type="button" onClick={() => navigate("/notes?note=2")}>
        same page
      </button>
    </div>
  )
}

const renderInDataRouter = (when: boolean, onLeave: () => Promise<boolean>) => {
  const router = createMemoryRouter(
    [
      { path: "/notes", element: <GuardedPage when={when} onLeave={onLeave} /> },
      { path: "/chat", element: <Where /> }
    ],
    { initialEntries: ["/notes"] }
  )
  render(<RouterProvider router={router} />)
  return router
}

describe("RouteLeaveGuard", () => {
  it("holds navigation to another page until the leave work allows it", async () => {
    const leave = deferred()
    const onLeave = vi.fn(() => leave.promise)
    renderInDataRouter(true, onLeave)

    fireEvent.click(screen.getByText("go to chat"))

    await waitFor(() => expect(onLeave).toHaveBeenCalledTimes(1))
    expect(screen.getByTestId("where")).toHaveTextContent("/notes")

    await act(async () => {
      leave.resolve(true)
    })

    await waitFor(() => expect(screen.getByTestId("where")).toHaveTextContent("/chat"))
    expect(onLeave).toHaveBeenCalledTimes(1)
  })

  it("stays on the page when the leave work says no", async () => {
    const onLeave = vi.fn(async () => false)
    renderInDataRouter(true, onLeave)

    fireEvent.click(screen.getByText("go to chat"))

    await waitFor(() => expect(onLeave).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(screen.getByTestId("where")).toHaveTextContent("/notes")
  })

  it("stays on the page when the leave work throws", async () => {
    const onLeave = vi.fn(async () => {
      throw new Error("flush failed")
    })
    renderInDataRouter(true, onLeave)

    fireEvent.click(screen.getByText("go to chat"))

    await waitFor(() => expect(onLeave).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(screen.getByTestId("where")).toHaveTextContent("/notes")
  })

  it("does not hold navigation when there is nothing to protect", async () => {
    const onLeave = vi.fn(async () => true)
    renderInDataRouter(false, onLeave)

    fireEvent.click(screen.getByText("go to chat"))

    await waitFor(() => expect(screen.getByTestId("where")).toHaveTextContent("/chat"))
    expect(onLeave).not.toHaveBeenCalled()
  })

  it("lets same-page URL updates through", async () => {
    const onLeave = vi.fn(async () => true)
    renderInDataRouter(true, onLeave)

    fireEvent.click(screen.getByText("same page"))

    await waitFor(() => expect(screen.getByTestId("where")).toHaveTextContent("/notes?note=2"))
    expect(onLeave).not.toHaveBeenCalled()
  })

  it("renders nothing outside a router that can block", () => {
    const onLeave = vi.fn(async () => true)
    expect(() =>
      render(
        <MemoryRouter initialEntries={["/notes"]}>
          <RouteLeaveGuard when onLeave={onLeave} />
          <Where />
        </MemoryRouter>
      )
    ).not.toThrow()
    expect(screen.getByTestId("where")).toHaveTextContent("/notes")
  })
})

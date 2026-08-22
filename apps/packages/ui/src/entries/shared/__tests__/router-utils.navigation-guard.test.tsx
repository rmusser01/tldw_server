import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  Link,
  UNSAFE_DataRouterContext,
  useLocation,
  useNavigate
} from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  HashRouterWithFuture,
  MemoryRouterWithFuture,
  RouteLeavePrompt
} from "../router-utils"

const LEAVE_MESSAGE = "Leave without saving? Your local draft is preserved only in this tab."

const GuardedNavigationHarness: React.FC = () => {
  const [draft, setDraft] = React.useState("")
  const location = useLocation()
  const navigate = useNavigate()

  return (
    <div>
      <RouteLeavePrompt when={draft.length > 0} message={LEAVE_MESSAGE} />
      <label htmlFor="guarded-draft">Draft</label>
      <input
        id="guarded-draft"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <output aria-label="Current route">{location.pathname}</output>
      <Link to="/linked">Linked route</Link>
      <button type="button" onClick={() => navigate("/programmatic")}>Programmatic route</button>
      <button type="button" onClick={() => navigate("/replacement", { replace: true })}>
        Replacement route
      </button>
      <button type="button" onClick={() => navigate(-1)}>Browser Back</button>
    </div>
  )
}

const InlineConfirmedBackHarness: React.FC = () => {
  const [approved, setApproved] = React.useState(false)
  const [showConfirmation, setShowConfirmation] = React.useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  React.useEffect(() => {
    if (!approved) return
    const timer = window.setTimeout(() => navigate("/presentations"), 0)
    return () => window.clearTimeout(timer)
  }, [approved, navigate])
  return (
    <div>
      <RouteLeavePrompt when={!approved} message={LEAVE_MESSAGE} />
      <output aria-label="Current route">{location.pathname}</output>
      <button type="button" onClick={() => setShowConfirmation(true)}>Back to presentations</button>
      {showConfirmation ? (
        <button type="button" onClick={() => setApproved(true)}>Leave presentation</button>
      ) : null}
    </div>
  )
}

const routers = [
  ["hash", HashRouterWithFuture],
  ["memory", MemoryRouterWithFuture]
] as const

describe("guardable shared router wrappers", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/#/")
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it.each(routers)("balances the %s router's global listeners under StrictMode", async (_mode, Router) => {
    const addEventListener = vi.spyOn(window, "addEventListener")
    const removeEventListener = vi.spyOn(window, "removeEventListener")
    const view = render(
      <React.StrictMode>
        <Router><span>Strict router child</span></Router>
      </React.StrictMode>
    )

    expect(await screen.findByText("Strict router child")).toBeVisible()
    view.unmount()
    await new Promise((resolve) => window.setTimeout(resolve, 10))

    const trackedTypes = new Set(["pagehide", "popstate"])
    const additions = addEventListener.mock.calls.filter(([type]) => trackedTypes.has(type))
    const removals = removeEventListener.mock.calls.filter(([type]) => trackedTypes.has(type))
    expect(additions.length).toBeGreaterThan(0)
    expect(removals).toHaveLength(additions.length)
    for (const [type, listener] of additions) {
      expect(removals.some(([removedType, removedListener]) => (
        removedType === type && removedListener === listener
      ))).toBe(true)
    }
  })

  it.each(routers)("gives the %s wrapper a data-router context without freezing child updates", async (_mode, Router) => {
    const Probe: React.FC<{ revision: string }> = ({ revision }) => {
      const dataRouter = React.useContext(UNSAFE_DataRouterContext)
      return <span>{`${dataRouter ? "data" : "declarative"}:${revision}`}</span>
    }
    const view = render(<Router><Probe revision="one" /></Router>)

    expect(await screen.findByText("data:one")).toBeVisible()
    view.rerender(<Router><Probe revision="two" /></Router>)
    expect(await screen.findByText("data:two")).toBeVisible()
  })

  it.each(routers)("lets clean %s programmatic navigation proceed without prompting", async (_mode, Router) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm")
    render(<Router><GuardedNavigationHarness /></Router>)

    await user.click(await screen.findByRole("button", { name: "Programmatic route" }))

    expect(await screen.findByRole("status", { name: "Current route" })).toHaveTextContent(
      "/programmatic"
    )
    expect(confirm).not.toHaveBeenCalled()
  })

  it.each(routers)("denies and then allows a dirty %s Link while preserving the draft", async (_mode, Router) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    render(<Router><GuardedNavigationHarness /></Router>)
    await user.type(await screen.findByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("link", { name: "Linked route" }))
    expect(confirm).toHaveBeenNthCalledWith(1, LEAVE_MESSAGE)
    expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent("/")
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")

    await user.click(screen.getByRole("link", { name: "Linked route" }))
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent("/linked")
    )
    expect(confirm).toHaveBeenCalledTimes(2)
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")
  })

  it.each(routers)("denies and then allows dirty %s programmatic replace navigation", async (_mode, Router) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true)
    render(<Router><GuardedNavigationHarness /></Router>)
    await user.type(await screen.findByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("button", { name: "Replacement route" }))
    expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent("/")
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")

    await user.click(screen.getByRole("button", { name: "Replacement route" }))
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent(
        "/replacement"
      )
    )
    expect(confirm).toHaveBeenCalledTimes(2)
  })

  it.each(routers)("denies dirty %s numeric Back navigation", async (mode, Router) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<Router><GuardedNavigationHarness /></Router>)
    await user.click(await screen.findByRole("button", { name: "Programmatic route" }))
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent(
        "/programmatic"
      )
    )
    await user.type(screen.getByLabelText("Draft"), "local source")

    await user.click(screen.getByRole("button", { name: "Browser Back" }))
    expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent(
      "/programmatic"
    )
    expect(screen.getByLabelText("Draft")).toHaveValue("local source")
    if (mode === "hash") {
      await waitFor(() => expect(window.location.hash).toBe("#/programmatic"))
    }
    expect(confirm).toHaveBeenCalledTimes(1)
  })

  it.each(routers)("allows dirty %s numeric Back navigation after confirmation", async (_mode, Router) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
    const errors: unknown[] = []
    const handleError = (event: ErrorEvent) => {
      errors.push(event.error)
      event.preventDefault()
    }
    window.addEventListener("error", handleError)
    try {
      const view = render(<Router><GuardedNavigationHarness /></Router>)
      await user.click(await screen.findByRole("button", { name: "Programmatic route" }))
      await waitFor(() =>
        expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent(
          "/programmatic"
        )
      )
      await user.type(screen.getByLabelText("Draft"), "local source")

      await user.click(screen.getByRole("button", { name: "Browser Back" }))
      await waitFor(() =>
        expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent("/")
      )
      view.unmount()
      await new Promise((resolve) => window.setTimeout(resolve, 10))
      expect(confirm).toHaveBeenCalledTimes(1)
      expect(errors).toEqual([])
    } finally {
      window.removeEventListener("error", handleError)
    }
  })

  it.each(routers)("uses one inline confirmation and one navigation for dirty %s dedicated Back", async (_mode, Router) => {
    const user = userEvent.setup()
    const browserConfirm = vi.spyOn(window, "confirm")
    render(<Router><InlineConfirmedBackHarness /></Router>)

    await user.click(await screen.findByRole("button", { name: "Back to presentations" }))
    expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent("/")
    await user.click(screen.getByRole("button", { name: "Leave presentation" }))

    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Current route" })).toHaveTextContent(
        "/presentations"
      )
    )
    expect(browserConfirm).not.toHaveBeenCalled()
    expect(screen.getAllByRole("button", { name: "Leave presentation" })).toHaveLength(1)
  })
})

// @vitest-environment jsdom
/**
 * G18 follow-ups (#3119): the chat address under a slow router.
 *
 * The WebUI's router (the Next.js shim of react-router-dom) lands a
 * navigation a little later, and a navigation started meanwhile cancels the
 * one still on its way. The stand-in router below does the same, so these
 * tests see the history entries the browser would record.
 */
import {
  clickHeaderNewChat,
  createFakeTldwServer,
  HARNESS_WAIT,
  HeaderNewChatButton,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  waitFor
} from "./harness/playground-harness"
import { useLocation, useNavigate } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  const React = await import("react")
  let inFlight: ReturnType<typeof setTimeout> | null = null
  return {
    ...actual,
    useNavigate: () => {
      const navigate = actual.useNavigate()
      return React.useCallback(
        (to: Parameters<typeof navigate>[0], options?: Parameters<typeof navigate>[1]) => {
          if (typeof to === "number") {
            navigate(to)
            return
          }
          // A new navigation cancels the one still on its way, as Next.js does.
          if (inFlight !== null) clearTimeout(inFlight)
          inFlight = setTimeout(() => {
            inFlight = null
            navigate(to, options)
          }, 20)
        },
        [navigate]
      )
    }
  }
})

/** Shows the router's address and offers the browser's Back button. */
const BrowserChrome = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <>
      <output data-testid="harness-address">{`${location.pathname}${location.search}`}</output>
      <button type="button" onClick={() => navigate(-1)}>Harness browser: Back</button>
    </>
  )
}

const address = () => screen.getByTestId("harness-address").textContent

const seedChat = () => {
  const server = createFakeTldwServer()
  server.seedChat({
    id: "chat-a",
    title: "Alpha topic",
    turns: [
      { role: "user", content: "Alpha question" },
      { role: "assistant", content: "Alpha answer" }
    ]
  })
  return server
}

const chrome = (
  <>
    <BrowserChrome />
    <HeaderNewChatButton />
    <SidebarServerChatButton chatId="chat-a" title="Alpha topic" />
  </>
)

describe("Playground chat history entries under a slow router (#3119)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("Back after New chat returns to the chat that was open", async () => {
    const view = await renderPlayground({ server: seedChat(), extras: chrome })
    await openSavedServerChat(view, { title: "Alpha topic", lastMessage: "Alpha answer" })
    await waitFor(() => expect(address()).toBe("/chat?chat=chat-a"), HARNESS_WAIT)

    await clickHeaderNewChat(view)
    await waitFor(() => expect(address()).toBe("/chat"), HARNESS_WAIT)
    expect(screen.queryByText("Alpha answer")).not.toBeInTheDocument()

    await view.user.click(screen.getByRole("button", { name: "Harness browser: Back" }))

    await waitFor(() => {
      expect(address()).toBe("/chat?chat=chat-a")
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-a")
      expect(screen.getByText("Alpha answer")).toBeInTheDocument()
    }, HARNESS_WAIT)
  })
})

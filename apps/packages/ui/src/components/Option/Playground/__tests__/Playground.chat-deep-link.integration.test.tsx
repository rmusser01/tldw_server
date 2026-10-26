// @vitest-environment jsdom
/**
 * XP-05 (#3101, G18): a saved chat has an address. Opening one puts
 * `/chat?chat=<server chat id>` in the URL, Back and Forward move between the
 * chats the user opened, a `/chat?chat=<id>` link opens that chat, and New
 * chat drops the id again.
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
  sendFromComposer,
  SidebarServerChatButton,
  waitFor
} from "./harness/playground-harness"
import { useLocation, useNavigate } from "react-router-dom"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

/** Shows the router's address and offers the browser's Back and Forward. */
const BrowserChrome = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <>
      <output data-testid="harness-address">{`${location.pathname}${location.search}`}</output>
      <button type="button" onClick={() => navigate(-1)}>Harness browser: Back</button>
      <button type="button" onClick={() => navigate(1)}>Harness browser: Forward</button>
    </>
  )
}

const address = () => screen.getByTestId("harness-address").textContent

const seedTwoChats = () => {
  const server = createFakeTldwServer()
  server.seedChat({
    id: "chat-a",
    title: "Alpha topic",
    turns: [
      { role: "user", content: "Alpha question" },
      { role: "assistant", content: "Alpha answer" }
    ]
  })
  server.seedChat({
    id: "chat-b",
    title: "Beta topic",
    turns: [
      { role: "user", content: "Beta question" },
      { role: "assistant", content: "Beta answer" }
    ]
  })
  return server
}

const chrome = (
  <>
    <BrowserChrome />
    <HeaderNewChatButton />
    <SidebarServerChatButton chatId="chat-a" title="Alpha topic" />
    <SidebarServerChatButton chatId="chat-b" title="Beta topic" />
  </>
)

describe("Playground chat addresses (XP-05)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("puts the opened saved chat in the address", async () => {
    const view = await renderPlayground({ server: seedTwoChats(), extras: chrome })

    await openSavedServerChat(view, { title: "Alpha topic", lastMessage: "Alpha answer" })

    await waitFor(() => expect(address()).toBe("/chat?chat=chat-a"), HARNESS_WAIT)
  })

  it("Back and Forward move between the chats the user opened", async () => {
    const view = await renderPlayground({ server: seedTwoChats(), extras: chrome })
    await openSavedServerChat(view, { title: "Alpha topic", lastMessage: "Alpha answer" })
    await openSavedServerChat(view, { title: "Beta topic", lastMessage: "Beta answer" })
    await waitFor(() => expect(address()).toBe("/chat?chat=chat-b"), HARNESS_WAIT)

    await view.user.click(screen.getByRole("button", { name: "Harness browser: Back" }))

    await waitFor(() => {
      expect(address()).toBe("/chat?chat=chat-a")
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-a")
      expect(screen.getByText("Alpha answer")).toBeInTheDocument()
    }, HARNESS_WAIT)
    expect(screen.queryByText("Beta answer")).not.toBeInTheDocument()

    await view.user.click(screen.getByRole("button", { name: "Harness browser: Forward" }))

    await waitFor(() => {
      expect(address()).toBe("/chat?chat=chat-b")
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-b")
      expect(screen.getByText("Beta answer")).toBeInTheDocument()
    }, HARNESS_WAIT)
    expect(screen.queryByText("Alpha answer")).not.toBeInTheDocument()
  })

  it("opens the chat a /chat?chat=<id> link names", async () => {
    await renderPlayground({
      server: seedTwoChats(),
      extras: chrome,
      initialPath: "/chat?chat=chat-b"
    })

    await waitFor(() => {
      expect(screen.getByText("Beta answer")).toBeInTheDocument()
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-b")
    }, HARNESS_WAIT)
    expect(address()).toBe("/chat?chat=chat-b")
  })

  it("drops the chat from the address on New chat", async () => {
    const view = await renderPlayground({ server: seedTwoChats(), extras: chrome })
    await openSavedServerChat(view, { title: "Alpha topic", lastMessage: "Alpha answer" })
    await waitFor(() => expect(address()).toBe("/chat?chat=chat-a"), HARNESS_WAIT)

    await clickHeaderNewChat(view)

    await waitFor(() => expect(address()).toBe("/chat"), HARNESS_WAIT)
    expect(screen.queryByText("Alpha answer")).not.toBeInTheDocument()
  })

  it("names a new chat in the address once its first turn is saved on the server", async () => {
    const view = await renderPlayground({ server: seedTwoChats(), extras: chrome })

    await sendFromComposer(view, "A brand new question")

    const chatId = useStoreMessageOption.getState().serverChatId
    expect(chatId).toBeTruthy()
    await waitFor(
      () => expect(address()).toBe(`/chat?chat=${encodeURIComponent(chatId as string)}`),
      HARNESS_WAIT
    )
  })
})

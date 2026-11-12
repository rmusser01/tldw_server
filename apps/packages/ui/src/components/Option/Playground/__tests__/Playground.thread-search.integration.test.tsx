// @vitest-environment jsdom
/**
 * CS-11 (#3101, G07): in-thread search has a visible button, highlights the
 * phrase the user typed (not each word on its own), and leaves no
 * highlights behind once closed, returning focus to where the user was.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  waitFor
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

const transcript = () => screen.getByRole("log", { name: "Chat messages" })
const marks = () => Array.from(transcript().querySelectorAll("mark")).map((mark) => mark.textContent)

const openPlanChat = async () => {
  const server = createFakeTldwServer()
  server.seedChat({
    id: "plan-chat",
    title: "Release plan",
    turns: [
      { role: "user", content: "Summarise Part 17 of the plan" },
      { role: "assistant", content: "Part 2 covers testing; Part 17 covers the rollout." }
    ]
  })
  const view = await renderPlayground({
    server,
    extras: <SidebarServerChatButton chatId="plan-chat" title="Release plan" />
  })
  await openSavedServerChat(view, {
    title: "Release plan",
    lastMessage: "Part 2 covers testing; Part 17 covers the rollout."
  })
  return view
}

describe("Playground in-thread search (CS-11)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("opens from a Search button in the chat header, which names its shortcut", async () => {
    const view = await openPlanChat()

    const button = screen.getByRole("button", { name: "Search this conversation" })
    expect(button.getAttribute("title")).toMatch(/Ctrl\+F|⌘F/)
    await view.user.click(button)

    const input = await screen.findByPlaceholderText("Search messages in this conversation")
    await waitFor(() => expect(input).toHaveFocus())
    expect(button).toHaveAttribute("aria-expanded", "true")
  })

  it("highlights the phrase that was searched for, not each word on its own", async () => {
    const view = await openPlanChat()
    await view.user.click(screen.getByRole("button", { name: "Search this conversation" }))

    await view.user.type(
      await screen.findByPlaceholderText("Search messages in this conversation"),
      "Part 17"
    )

    await waitFor(() => expect(marks().length).toBeGreaterThan(0), HARNESS_WAIT)
    expect(new Set(marks())).toEqual(new Set(["Part 17"]))
  })

  it("clears the highlights on Escape and returns focus to the composer", async () => {
    const view = await openPlanChat()
    await view.user.click(view.composer)
    await view.user.keyboard("{Control>}f{/Control}")
    const input = await screen.findByPlaceholderText("Search messages in this conversation")
    await view.user.type(input, "rollout")
    await waitFor(() => expect(marks()).toContain("rollout"), HARNESS_WAIT)

    await view.user.keyboard("{Escape}")

    await waitFor(() => {
      expect(screen.queryByPlaceholderText("Search messages in this conversation")).toBeNull()
      expect(marks()).toEqual([])
    })
    expect(view.composer).toHaveFocus()
  })
})

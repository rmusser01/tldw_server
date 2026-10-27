// @vitest-environment jsdom
/**
 * CS-06 (#3108): a saved chat whose messages fail to load put the raw request
 * line ("Internal Server Error (GET /api/v1/chats/…/messages?…)") in an error
 * toast. The empty-transcript case (panel with Retry) is covered in
 * PlaygroundChat.server-load-state.test.tsx.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  visibleNotifications,
  waitFor,
  within
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const MESSAGES_PATH = /^\/api\/v1\/chats\/saved-chat\/messages$/

describe("Playground saved chat that fails to load (CS-06)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("reports the failure in plain words, without the raw request line", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Explain LoRA fine-tuning",
      turns: [
        { role: "user", content: "What is LoRA?" },
        { role: "assistant", content: "Low-rank adaptation." }
      ]
    })
    const flakyServer = {
      ...server,
      fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = new URL(
          typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
          server.url
        )
        const method = String(init?.method ?? "GET").toUpperCase()
        if (method === "GET" && MESSAGES_PATH.test(url.pathname)) {
          return new Response(JSON.stringify({ detail: "Internal Server Error" }), {
            status: 500,
            headers: { "content-type": "application/json" }
          })
        }
        return server.fetch(input, init)
      }
    }
    const view = await renderPlayground({
      server: flakyServer,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Explain LoRA fine-tuning" />
    })

    await view.user.click(
      screen.getByRole("button", { name: "Harness sidebar: open Explain LoRA fine-tuning" })
    )

    await waitFor(
      () => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("failed"),
      HARNESS_WAIT
    )
    await waitFor(
      () => expect(visibleNotifications().join(" ")).toContain("Couldn't load this conversation."),
      HARNESS_WAIT
    )
    expect(visibleNotifications().join(" ")).not.toMatch(/GET |\/api\/v1\//)
    // The raw line stays available as the technical detail.
    expect(useStoreMessageOption.getState().serverChatLoadError).toContain(
      "/api/v1/chats/saved-chat/messages"
    )
  })

  it("loads the chat when the user presses Retry after the server recovers", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Explain LoRA fine-tuning",
      turns: [
        { role: "user", content: "What is LoRA?" },
        { role: "assistant", content: "Low-rank adaptation." }
      ]
    })
    let failing = true
    const flakyServer = {
      ...server,
      fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = new URL(
          typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
          server.url
        )
        const method = String(init?.method ?? "GET").toUpperCase()
        if (failing && method === "GET" && MESSAGES_PATH.test(url.pathname)) {
          return new Response(JSON.stringify({ detail: "Internal Server Error" }), {
            status: 500,
            headers: { "content-type": "application/json" }
          })
        }
        return server.fetch(input, init)
      }
    }
    const view = await renderPlayground({
      server: flakyServer,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Explain LoRA fine-tuning" />
    })
    await view.user.click(
      screen.getByRole("button", { name: "Harness sidebar: open Explain LoRA fine-tuning" })
    )
    await waitFor(
      () => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("failed"),
      HARNESS_WAIT
    )
    // The history read can fill the transcript even when the message list
    // request fails; clear it so the empty-transcript panel is what shows.
    useStoreMessageOption.getState().setMessages([])
    const panel = await screen.findByTestId("server-chat-load-failure", {}, HARNESS_WAIT)

    failing = false
    await view.user.click(within(panel).getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded")
      expect(screen.getByText("Low-rank adaptation.")).toBeInTheDocument()
    }, HARNESS_WAIT)
    expect(screen.queryByTestId("server-chat-load-failure")).not.toBeInTheDocument()
  })
})

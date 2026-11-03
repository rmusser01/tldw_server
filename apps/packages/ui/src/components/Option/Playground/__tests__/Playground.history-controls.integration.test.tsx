// @vitest-environment jsdom
/**
 * Red-first reproduction for #3106 (UX G05, CC-04): the history-path controls
 * ("Use no prior messages" / "Review conversation history") must not appear
 * for a fresh chat that has no history to review. The actual shared Playground sends its first turn while the provider is held.
 */
import { beforeEach, describe, expect, it } from "vitest"

import {
  HARNESS_WAIT,
  act,
  createFakeTldwServer,
  deferred,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  waitFor,
  waitForChatIdle,
  within
} from "./harness/playground-harness"
import { useStoreMessageOption } from "@/store/option"

describe(
  "Playground history-path controls (#3106)",
  { timeout: 60_000 },
  () => {
    beforeEach(async () => {
      await resetPlaygroundHarness()
    })

    const openSavedChat = async (
      turns: Array<{ role: "user" | "assistant"; content: string }>
    ) => {
      const server = createFakeTldwServer()
      server.seedChat({ id: "saved-chat", title: "Saved topic", turns })
      const view = await renderPlayground({
        server,
        extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
      })
      await openSavedServerChat(view, {
        title: "Saved topic",
        lastMessage: turns.at(-1)!.content
      })
      return { server, view }
    }
    const transcriptTexts = () =>
      useStoreMessageOption.getState().messages.map((message) => message.message)

    // CC-04 (owner decision 2026-10-27): no one-click "Use no prior messages"
    // above a chat; a single exchange has nothing to choose between.
    it("CC-04: a saved chat with one exchange shows no history controls", async () => {
      await openSavedChat([
        { role: "user", content: "Only question" },
        { role: "assistant", content: "Only answer" }
      ])
      expect(screen.queryByRole("button", { name: "Review conversation history" })).toBeNull()
      expect(screen.queryByRole("button", { name: "Use no prior messages" })).toBeNull()
    })

    it("CC-04: Start fresh hides the earlier messages from the AI, says so, and Restore all brings them back", async () => {
      const { view } = await openSavedChat([
        { role: "user", content: "First question" },
        { role: "assistant", content: "First answer" },
        { role: "user", content: "Second question" },
        { role: "assistant", content: "Second answer" }
      ])
      expect(screen.queryByRole("button", { name: "Use no prior messages" })).toBeNull()

      await view.user.click(screen.getByRole("button", { name: "Review conversation history" }))
      await view.user.click(screen.getByRole("button", { name: "Start fresh" }))

      const restore = await screen.findByRole("button", { name: "Restore all" }, HARNESS_WAIT)
      await waitFor(() => expect(restore).toHaveFocus(), HARNESS_WAIT)
      expect(screen.getByText("AI sees 0 of 4 messages")).toBeInTheDocument()
      const hidden = screen.getByRole("list", { name: "Earlier messages, hidden from the AI" })
      expect(within(hidden).getAllByRole("listitem").map((item) => item.textContent)).toEqual([
        "You · First question",
        "Assistant · First answer",
        "You · Second question",
        "Assistant · Second answer"
      ])
      await waitFor(() => expect(transcriptTexts()).toEqual([]), HARNESS_WAIT)

      await view.user.click(restore)
      await waitFor(
        () =>
          expect(transcriptTexts()).toEqual([
            "First question",
            "First answer",
            "Second question",
            "Second answer"
          ]),
        HARNESS_WAIT
      )
      expect(screen.queryByText(/AI sees/)).toBeNull()
    })

    // CC-04 #3106 — HistorySelectionReview.tsx:326-333 renders "Use no prior messages" whenever the selection status is
    // "ready", even for the empty capture a fresh chat's first send installs (normalChatMode.ts:644-669).
    it("CC-04 (#3106): a fresh chat with no reviewable history shows no history-path controls while its first message is sent", async () => {
      const server = createFakeTldwServer()
      const view = await renderPlayground({ server })
      expect(
        screen.queryByRole("button", { name: "Use no prior messages" })
      ).not.toBeInTheDocument()

      // Hold the reply so the first turn stays in flight while the controls are inspected.
      const reply = deferred()
      server.planCompletion({ gate: reply.promise })
      try {
        await view.user.click(view.composer)
        await view.user.type(view.composer, "Fresh question")
        await view.user.keyboard("{Enter}")
        await waitFor(
          () => expect(server.completionRequests()).toHaveLength(1),
          HARNESS_WAIT
        )
        const creation = server.requests.find(
          (request) =>
            request.method === "POST" && request.path === "/api/v1/chats/"
        )
        expect([...server.chats.keys()]).toEqual([creation?.body?.id])
        await act(async () => {})

        expect(
          screen.queryByRole("button", { name: "Use no prior messages" })
        ).not.toBeInTheDocument()
        expect(
          screen.queryByRole("button", { name: "Review conversation history" })
        ).not.toBeInTheDocument()
      } finally {
        reply.resolve()
        await waitForChatIdle()
      }
    })
  }
)

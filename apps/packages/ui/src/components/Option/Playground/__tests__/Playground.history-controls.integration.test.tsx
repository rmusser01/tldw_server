// @vitest-environment jsdom
/**
 * History-path controls on /chat (#3106, UX G05: CC-04).
 *
 * The controls that choose what the AI sees from earlier messages must not
 * crowd a fresh or one-turn chat, must never offer a one-click way to blank
 * the context, and must never blank the visible transcript: earlier messages
 * stay readable, with a persistent "AI sees N of M" note and a way back.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

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

const transcriptText = () =>
  useStoreMessageOption.getState().messages.map((message) => message.message)

describe(
  "Playground history-path controls (#3106)",
  { timeout: 90_000 },
  () => {
    beforeEach(async () => {
      await resetPlaygroundHarness()
    })
    afterEach(() => {
      vi.restoreAllMocks()
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
          screen.queryByRole("button", { name: "Choose what the AI remembers" })
        ).not.toBeInTheDocument()
      } finally {
        reply.resolve()
        await waitForChatIdle()
      }
    })

    it("CC-04 (#3106): a saved chat with one turn shows no history-path controls", async () => {
      const server = createFakeTldwServer()
      server.seedChat({
        id: "one-turn",
        title: "One turn",
        turns: [
          { role: "user", content: "Only question" },
          { role: "assistant", content: "Only answer" }
        ]
      })
      const view = await renderPlayground({
        server,
        extras: <SidebarServerChatButton chatId="one-turn" title="One turn" />
      })
      await openSavedServerChat(view, { title: "One turn", lastMessage: "Only answer" })

      expect(
        screen.queryByRole("button", { name: "Choose what the AI remembers" })
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Use no prior messages" })
      ).not.toBeInTheDocument()
    })

    it("CC-04 (#3106): starting fresh keeps earlier messages readable, says what the AI sees, and Restore all brings them back", async () => {
      const server = createFakeTldwServer()
      server.seedChat({
        id: "two-turns",
        title: "Two turns",
        turns: [
          { role: "user", content: "First **bold** question" },
          { role: "assistant", content: "First answer" },
          { role: "user", content: "Second question" },
          { role: "assistant", content: "Second answer" }
        ]
      })
      const view = await renderPlayground({
        server,
        extras: <SidebarServerChatButton chatId="two-turns" title="Two turns" />
      })
      await openSavedServerChat(view, { title: "Two turns", lastMessage: "Second answer" })

      // jsdom has no layout; give the virtualized review list a viewport.
      vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(480)
      vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(320)
      // No one-click way to blank what the AI sees.
      expect(
        screen.queryByRole("button", { name: "Use no prior messages" })
      ).not.toBeInTheDocument()
      await view.user.click(
        await screen.findByRole("button", { name: "Choose what the AI remembers" }, HARNESS_WAIT)
      )
      // The review lists messages in conversation order, as plain text.
      const rows = within(screen.getByRole("list", { name: "Messages in this chat" }))
        .getAllByRole("listitem")
        .map((row) => row.textContent ?? "")
      expect(rows[0]).toContain("You · First bold question")
      expect(rows[1]).toContain("Assistant · First answer")
      expect(screen.queryByText(/Path position/)).not.toBeInTheDocument()
      expect(screen.queryByText(/saved interpretation/)).not.toBeInTheDocument()

      await view.user.click(
        screen.getByRole("button", { name: "Start fresh: the AI sees no earlier messages" })
      )
      await waitFor(() => expect(transcriptText()).toEqual([]), HARNESS_WAIT)
      await waitForChatIdle()

      // The thread is not blanked: earlier messages stay readable, marked as hidden from the AI.
      const coverage = await screen.findByRole("status", { name: "AI sees 0 of 4 messages" }, HARNESS_WAIT)
      expect(within(coverage).getByText("AI sees 0 of 4 messages")).toBeInTheDocument()
      const earlier = screen.getByRole("list", { name: "Earlier messages, hidden from the AI" })
      expect(
        within(earlier).getAllByRole("listitem").map((row) => row.textContent)
      ).toEqual([
        "You · First bold question",
        "Assistant · First answer",
        "You · Second question",
        "Assistant · Second answer"
      ])

      await view.user.click(within(coverage).getByRole("button", { name: "Restore all" }))
      await waitFor(
        () =>
          expect(transcriptText()).toEqual([
            "First **bold** question",
            "First answer",
            "Second question",
            "Second answer"
          ]),
        HARNESS_WAIT
      )
      await waitFor(
        () =>
          expect(
            screen.queryByRole("status", { name: /AI sees \d+ of \d+ messages/ })
          ).not.toBeInTheDocument(),
        HARNESS_WAIT
      )
    })
  }
)

// @vitest-environment jsdom
/**
 * Decision [4] (#3101): in the web composer "@" picks a note and attaches it
 * to the next message. Every server here has Notes (capabilities are cached
 * per test file, so this lives apart from the slash-command tests).
 */
import {
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitForChatIdle
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

describe("Playground note mentions", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
    useStoreMessageOption.setState({ ragPinnedResults: [], fileRetrievalEnabled: false })
  })

  // Decision [4]: '@' picks a note; its content goes out with the message as
  // pinned context does, and it stays as a removable chip linking to the note.
  it("attaches a note picked with @ and sends its content with the message", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    expect(view.composer).toHaveValue("")
    expect(screen.getByRole("link", { name: "Rebase notes" })).toHaveAttribute("href", "/notes?note=7")
    await view.user.type(view.composer, "Summarize this")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    const sent = String(completionMessages(completions[0]).at(-1)?.content)
    expect(sent).toContain("Summarize this")
    expect(sent).toContain("Use git rebase -i to squash.")
  })

  // Review fix: an attached note belongs to one message. After it is sent the
  // chip clears, and the next message goes without it.
  it("sends an attached note with one message only", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    await screen.findByRole("link", { name: "Rebase notes" }, HARNESS_WAIT)
    await view.user.type(view.composer, "First question")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    expect(screen.queryByRole("link", { name: "Rebase notes" })).toBeNull()
    await view.user.type(view.composer, "Second question")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(2)
    expect(String(completionMessages(completions[0]).at(-1)?.content)).toContain("Use git rebase -i to squash.")
    expect(String(completionMessages(completions[1]).at(-1)?.content)).toContain("Second question")
    expect(String(completionMessages(completions[1]).at(-1)?.content)).not.toContain("Use git rebase -i to squash.")
  })

  // Review fix: a note is an explicit attachment, so file retrieval does not drop it.
  it("sends an attached note with file retrieval on", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    useStoreMessageOption.getState().setFileRetrievalEnabled(true)
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    await view.user.type(view.composer, "Summarize this")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    expect(String(completionMessages(completions[0]).at(-1)?.content)).toContain("Use git rebase -i to squash.")
  })
})

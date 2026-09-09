import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { BuddyInteraction } from "../BuddyInteraction"
const mocks = vi.hoisted(() => ({
  getBuddyAttachment: vi.fn(),
  listBuddyTurns: vi.fn(),
  listBuddyActivity: vi.fn(),
  readBuddyConversation: vi.fn(),
  acceptBuddyTurn: vi.fn(),
  stopBuddyTurn: vi.fn(),
  acknowledgeBuddyResult: vi.fn(),
  start: vi.fn(),
  stop: vi.fn(),
  speak: vi.fn(),
  cancel: vi.fn()
}))
vi.mock("@/services/buddies", () => mocks)
vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: mocks.speak,
    cancel: mocks.cancel,
    isSpeaking: false
  })
}))
vi.mock("@/hooks/useSttSettings", () => ({ useSttSettings: () => ({}) }))
vi.mock("@/hooks/useServerDictation", () => ({
  useServerDictation: () => ({
    startServerDictation: mocks.start,
    stopServerDictation: mocks.stop,
    isServerDictating: false
  })
}))
const first = {
  id: "one",
  title: "Research one",
  scope_type: "workspace",
  workspace_id: "ws"
} as any
const second = {
  id: "two",
  title: "Research two",
  scope_type: "workspace",
  workspace_id: "ws"
} as any
const attachment = {
  buddy_id: "duck",
  scope_type: "workspace",
  scope_id: "ws"
} as const
const props = {
  attachment,
  attachmentVersion: 1,
  conversation: first,
  conversations: [first, second],
  visible: true
}
beforeEach(() => {
  vi.clearAllMocks()
  mocks.getBuddyAttachment.mockResolvedValue({ version: 1, attachment })
  mocks.listBuddyTurns.mockResolvedValue({ turns: [] })
  mocks.listBuddyActivity.mockResolvedValue({
    items: [{ conversation_id: "one", result: null, acknowledged: false }]
  })
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: first,
    replySettings: { provider: "custom-openai-api", model: "saved-model" },
    messages: [
      {
        id: "result",
        role: "assistant",
        content: "Only belongs to one",
        created_at: "2026-09-08"
      }
    ]
  })
})
afterEach(cleanup)
it("distinguishes duplicate result titles while replies and acknowledgements retain exact IDs", async () => {
  const duplicateOne = { ...first, title: "Research", created_at: "" }
  const duplicateTwo = { ...second, title: "Research", created_at: "" }
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: duplicateTwo,
    messages: [],
    replySettings: { provider: "custom-openai-api", model: "saved-model" }
  })
  mocks.listBuddyActivity.mockResolvedValue({
    items: [
      {
        conversation_id: "one",
        title: "Research",
        result: { id: "r1", content: "First result" },
        acknowledged: false
      },
      {
        conversation_id: "two",
        title: "Research",
        result: { id: "r2", content: "Second result" },
        acknowledged: false
      }
    ]
  })
  mocks.acknowledgeBuddyResult.mockResolvedValue({})
  mocks.acceptBuddyTurn.mockResolvedValue({
    id: "new-turn",
    conversation_id: "two",
    status: "queued"
  })
  render(
    <BuddyInteraction
      {...props}
      conversation={duplicateTwo}
      conversations={[duplicateOne, duplicateTwo]}
    />
  )
  const results = await screen.findAllByRole("button", { name: /New response/ })
  expect(new Set(results.map((result) => result.textContent)).size).toBe(2)
  const reply = screen.getByRole("textbox", { name: /Reply to/ })
  expect(reply).toHaveAccessibleName(/two.*Research/)
  fireEvent.click(screen.getAllByRole("button", { name: "Mark read" })[1])
  await waitFor(() =>
    expect(mocks.acknowledgeBuddyResult).toHaveBeenCalledWith({
      conversation_id: "two",
      result_message_id: "r2"
    })
  )
  fireEvent.change(reply, { target: { value: "Reply only to this target" } })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() =>
    expect(mocks.acceptBuddyTurn).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_id: "two",
        text: "Reply only to this target"
      })
    )
  )
})
it("preserves an encoded error quoted in a user message verbatim", async () => {
  const content =
    '__tldw_error__:{"summary":"Quoted summary","hint":"Quoted hint","detail":"Quoted diagnostic"}'
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: first,
    replySettings: { provider: "custom-openai-api", model: "saved-model" },
    messages: [
      { id: "quoted-error", role: "user", content, created_at: "2026-09-08" }
    ]
  })
  render(<BuddyInteraction {...props} />)
  expect(await screen.findByText(content)).toHaveTextContent(content)
})
it("presents a saved chat failure without its encoded envelope or diagnostic details", async () => {
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: first,
    replySettings: { provider: "custom-openai-api", model: "saved-model" },
    messages: [
      {
        id: "failure",
        role: "assistant",
        content:
          '__tldw_error__:{"summary":"The reply timed out.","hint":"Try again in a moment.","detail":"Internal upstream timeout diagnostic"}',
        created_at: "2026-09-08"
      },
      {
        id: "ordinary",
        role: "user",
        content: "Keep this ordinary message.\nIncluding its second line.",
        created_at: "2026-09-08"
      }
    ]
  })
  render(<BuddyInteraction {...props} />)
  await screen.findByText("The reply timed out. Try again in a moment.")
  const history = screen.getByRole("log", {
    name: "Conversation history: Research one"
  })
  expect(history).not.toHaveTextContent("__tldw_error__:")
  expect(history).not.toHaveTextContent("Internal upstream timeout diagnostic")
  expect(
    screen.getByText("Keep this ordinary message. Including its second line.")
      .textContent
  ).toBe("Keep this ordinary message.\nIncluding its second line.")
})
it("never relabels old transcript content as a newly selected conversation", async () => {
  const { rerender } = render(<BuddyInteraction {...props} />)
  await screen.findByText("Only belongs to one")
  mocks.readBuddyConversation.mockReturnValue(new Promise(() => {}))
  rerender(<BuddyInteraction {...props} conversation={second} />)
  expect(screen.queryByText("Only belongs to one")).not.toBeInTheDocument()
  expect(screen.getByLabelText("Reply to Research two")).toHaveValue("")
  expect(
    screen.queryByRole("button", { name: "Dictate a reply" })
  ).not.toBeInTheDocument()
})
it("preserves an ambiguous send's request identity on retry", async () => {
  mocks.acceptBuddyTurn
    .mockRejectedValueOnce(new Error("Acceptance response lost"))
    .mockResolvedValueOnce({
      id: "turn",
      status: "queued",
      conversation_id: "one",
      conversation_title: "Research one"
    })
  render(<BuddyInteraction {...props} />)
  await screen.findByText("Only belongs to one")
  fireEvent.change(screen.getByLabelText("Reply to Research one"), {
    target: { value: "Explain this" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await screen.findByText("Acceptance response lost")
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(mocks.acceptBuddyTurn).toHaveBeenCalledTimes(2))
  expect(mocks.acceptBuddyTurn.mock.calls[0][0]).toEqual(
    mocks.acceptBuddyTurn.mock.calls[1][0]
  )
  expect(mocks.acceptBuddyTurn.mock.calls[0][0].conversation_id).toBe("one")
})
it("hides stale content on access failure and clears the fetch error after recovery", async () => {
  const { rerender } = render(<BuddyInteraction {...props} />)
  await screen.findByText("Only belongs to one")
  mocks.getBuddyAttachment.mockRejectedValue(new Error("Access unavailable"))
  rerender(<BuddyInteraction {...props} conversations={[first, second]} />)
  await screen.findByText("Access unavailable")
  expect(screen.queryByText("Only belongs to one")).not.toBeInTheDocument()
  mocks.getBuddyAttachment.mockResolvedValue({ version: 1, attachment })
  rerender(<BuddyInteraction {...props} conversations={[first, second]} />)
  await screen.findByText("Only belongs to one")
  expect(screen.queryByText("Access unavailable")).not.toBeInTheDocument()
})

it("reads a newly arrived assistant failure with its conversation title and friendly text only", async () => {
  const { rerender } = render(<BuddyInteraction {...props} visible={false} />)
  await waitFor(() => expect(mocks.listBuddyActivity).toHaveBeenCalled())
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Read new responses aloud" })
  )
  await waitFor(() => expect(mocks.listBuddyActivity).toHaveBeenCalledTimes(2))
  const content =
    '__tldw_error__:{"summary":"The reply timed out.","hint":"Try again in a moment.","detail":"Internal upstream timeout diagnostic"}'
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: first,
    replySettings: { provider: "custom-openai-api", model: "saved-model" },
    messages: [
      { id: "late-error", role: "assistant", content, created_at: "2026-09-08" }
    ]
  })
  mocks.listBuddyActivity.mockResolvedValue({
    items: [
      {
        conversation_id: "one",
        title: "Research one",
        result: { id: "late-error", content },
        acknowledged: false
      }
    ]
  })
  rerender(
    <BuddyInteraction
      {...props}
      visible={false}
      conversations={[first, second]}
    />
  )
  await waitFor(() =>
    expect(mocks.speak).toHaveBeenCalledWith({
      utterance: "Research one. The reply timed out.\n\nTry again in a moment.",
      saveClip: false
    })
  )
  const { utterance } = mocks.speak.mock.calls[0][0]
  expect(utterance).not.toContain("__tldw_error__:")
  expect(utterance).not.toContain("Internal upstream timeout diagnostic")
})

it("does not start speech after unmount while its authorized transcript read is pending", async () => {
  const { rerender, unmount } = render(
    <BuddyInteraction {...props} visible={false} />
  )
  await waitFor(() => expect(mocks.listBuddyActivity).toHaveBeenCalled())
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Read new responses aloud" })
  )
  await waitFor(() => expect(mocks.listBuddyActivity).toHaveBeenCalledTimes(2))
  let resolve!: (value: any) => void
  mocks.readBuddyConversation.mockReturnValue(
    new Promise((r) => {
      resolve = r
    })
  )
  mocks.listBuddyActivity.mockResolvedValue({
    items: [
      {
        conversation_id: "one",
        title: "Research one",
        result: { id: "late", content: "Private reply" },
        acknowledged: false
      }
    ]
  })
  rerender(
    <BuddyInteraction
      {...props}
      visible={false}
      conversations={[first, second]}
    />
  )
  await waitFor(() => expect(mocks.readBuddyConversation).toHaveBeenCalled())
  unmount()
  await act(async () =>
    resolve({
      conversation: first,
      messages: [{ id: "late", content: "Private reply" }]
    })
  )
  expect(mocks.speak).not.toHaveBeenCalled()
})

it("shows missing reply settings before Send and preserves the draft while recovering", async () => {
  mocks.readBuddyConversation.mockResolvedValue({
    conversation: first,
    messages: [],
    replySettings: { model: null, provider: null }
  })
  mocks.acceptBuddyTurn.mockResolvedValue({
    id: "turn",
    status: "queued",
    conversation_id: "one"
  })
  render(<BuddyInteraction {...props} />)
  const reply = screen.getByLabelText("Reply to Research one")
  fireEvent.change(reply, { target: { value: "Keep this draft" } })
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
  await screen.findByText(/Choose a provider and model below before sending/)
  const model = screen.getByLabelText("Model (required)")
  const provider = screen.getByLabelText("Provider (required)")
  expect(model.closest("details")).toHaveAttribute("open")
  fireEvent.change(model, { target: { value: "recovery-model" } })
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
  fireEvent.change(provider, { target: { value: "custom-openai-api" } })
  expect(reply).toHaveValue("Keep this draft")
  expect(screen.getByRole("button", { name: "Send" })).toBeEnabled()
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() =>
    expect(mocks.acceptBuddyTurn).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_id: "one",
        text: "Keep this draft",
        model: "recovery-model",
        provider: "custom-openai-api"
      })
    )
  )
})

it("uses saved conversation settings without sending overrides and blocks a newly selected target until checked", async () => {
  const { rerender } = render(<BuddyInteraction {...props} />)
  await screen.findByText(/custom-openai-api.*saved-model/)
  fireEvent.change(screen.getByLabelText("Reply to Research one"), {
    target: { value: "Use my saved model" }
  })
  expect(screen.getByRole("button", { name: "Send" })).toBeEnabled()
  mocks.acceptBuddyTurn.mockResolvedValue({
    id: "turn",
    status: "queued",
    conversation_id: "one"
  })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(mocks.acceptBuddyTurn).toHaveBeenCalledTimes(1))
  expect(mocks.acceptBuddyTurn.mock.calls[0][0]).not.toHaveProperty("model")
  expect(mocks.acceptBuddyTurn.mock.calls[0][0]).not.toHaveProperty("provider")
  mocks.readBuddyConversation.mockReturnValue(new Promise(() => {}))
  rerender(<BuddyInteraction {...props} conversation={second} />)
  fireEvent.change(screen.getByLabelText("Reply to Research two"), {
    target: { value: "Different target" }
  })
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
  expect(
    screen.queryByText(/custom-openai-api.*saved-model/)
  ).not.toBeInTheDocument()
})

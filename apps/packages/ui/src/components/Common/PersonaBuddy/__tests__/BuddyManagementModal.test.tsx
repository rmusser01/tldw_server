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
import { BuddyManagementModal } from "../BuddyManagementModal"
import type { BuddyConversationSummary, BuddyProfile } from "@/services/buddies"
const mocks = vi.hoisted(() => ({
  locale: "en-US",
  t: (_key: string, options: { defaultValue: string }) => options.defaultValue,
  createBuddy: vi.fn(),
  putBuddyAttachment: vi.fn(),
  listChats: vi.fn(),
  listWorkspaces: vi.fn(),
  fetchWithAuth: vi.fn(),
  getWorkspace: vi.fn(),
  patchWorkspace: vi.fn(),
  resolveBuddyConversationTarget: vi.fn(),
  catalog: vi.fn()
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.t,
    i18n: { resolvedLanguage: mocks.locale }
  })
}))
vi.mock("@/services/buddies", () => ({
  createBuddy: mocks.createBuddy,
  putBuddyAttachment: mocks.putBuddyAttachment,
  resolveBuddyConversationTarget: mocks.resolveBuddyConversationTarget,
  buddyAssets: () => ({})
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: mocks }))
vi.mock("@/services/persona-visuals", () => ({
  listPersonaVisualStarterPacks: mocks.catalog
}))
vi.mock("../SpriteFrameRenderer", () => ({
  SpriteFrameRenderer: () => <span>Artwork preview</span>
}))
vi.mock("@/components/PersonaGarden/BuddyStarterArtwork", () => ({
  BuddyStarterArtwork: ({
    onReadyChange
  }: {
    onReadyChange: (ready: boolean) => void
  }) => <button onClick={() => onReadyChange(true)}>Load preview</button>
}))
beforeEach(() => {
  vi.clearAllMocks()
  mocks.locale = "en-US"
  mocks.catalog.mockResolvedValue({ starter_packs: [] })
  mocks.listChats.mockResolvedValue([{ id: "chat", title: "Research" }])
  mocks.listWorkspaces.mockResolvedValue({ items: [] })
  mocks.fetchWithAuth.mockResolvedValue({
    ok: true,
    json: async () => ({ personas: [] })
  })
})
afterEach(cleanup)
const profile: BuddyProfile = {
  id: "duck",
  name: "Duck",
  display_mode: "static",
  version: 1,
  manifest: {
    manifest_version: 1,
    renderer_type: "sprite_sheet",
    states: {},
    animations: {}
  },
  assets: [],
  optional_persona_id: null,
  optional_persona_available: false,
  attribution: {}
}
const props = {
  profiles: [profile],
  attachment: { client_slot: "default", version: 0, attachment: null },
  target: null,
  onClose: vi.fn(),
  onApplied: vi.fn()
}
it("Cancel does not attach the selected Buddy or change a Persona", async () => {
  render(<BuddyManagementModal {...props} />)
  await screen.findByRole("option", { name: "Research" })
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
  expect(mocks.createBuddy).not.toHaveBeenCalled()
  expect(mocks.patchWorkspace).not.toHaveBeenCalled()
})
it("attaches existing artwork to a named conversation without requiring a Persona", async () => {
  mocks.putBuddyAttachment.mockResolvedValue({
    version: 1,
    attachment: {
      buddy_id: "duck",
      scope_type: "conversation",
      scope_id: "chat"
    }
  })
  render(<BuddyManagementModal {...props} />)
  await screen.findByRole("option", { name: "Research" })
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  fireEvent.change(screen.getByLabelText("Conversation"), {
    target: { value: "chat" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() =>
    expect(mocks.putBuddyAttachment).toHaveBeenCalledWith({
      expected_version: 0,
      buddy_id: "duck",
      scope_type: "conversation",
      scope_id: "chat"
    })
  )
  expect(mocks.createBuddy).not.toHaveBeenCalled()
})

it("resolves a composer conversation into its workspace before allowing Apply", async () => {
  let resolve!: (value: BuddyConversationSummary) => void
  mocks.resolveBuddyConversationTarget.mockReturnValue(
    new Promise((r) => {
      resolve = r
    })
  )
  mocks.listWorkspaces.mockResolvedValue({
    items: [{ id: "ws", name: "Reading room" }]
  })
  render(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "chat" }}
    />
  )
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled()
  resolve({ id: "chat", title: "Research", workspace_id: "ws" })
  await waitFor(() =>
    expect(
      screen.getByRole("combobox", {
        name: "Conversation location"
      })
    ).toHaveValue("ws")
  )
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled()
  )
})
it("waits for preview and prevents follow-up attachment writes after unmount", async () => {
  mocks.catalog.mockResolvedValue({
    starter_packs: [
      { id: "lens", title: "Lens", production_status: "art_ready" }
    ]
  })
  let resolve!: (value: BuddyProfile) => void
  mocks.createBuddy.mockReturnValue(
    new Promise((r) => {
      resolve = r
    })
  )
  const { unmount } = render(<BuddyManagementModal {...props} profiles={[]} />)
  const use = await screen.findByRole("button", { name: "Use this Buddy" })
  expect(use).toBeDisabled()
  fireEvent.click(screen.getByRole("button", { name: "Load preview" }))
  fireEvent.click(use)
  fireEvent.change(screen.getByLabelText("Conversation"), {
    target: { value: "chat" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.createBuddy).toHaveBeenCalledTimes(1))
  unmount()
  resolve({ ...profile, id: "created" })
  await new Promise((r) => setTimeout(r, 0))
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
})

it("applies the new entry-point target when the open modal is retargeted", async () => {
  mocks.listChats.mockResolvedValue([
    { id: "chat", title: "Research" },
    { id: "next", title: "Next conversation" }
  ])
  mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({
    id,
    title: id === "next" ? "Next conversation" : "Research",
    workspace_id: null
  }))
  mocks.putBuddyAttachment.mockResolvedValue({ version: 1 })
  const view = render(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "chat" }}
    />
  )
  await screen.findByRole("option", { name: "Research" })
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  view.rerender(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "next" }}
    />
  )
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled()
  )
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() =>
    expect(mocks.putBuddyAttachment).toHaveBeenCalledWith({
      expected_version: 0,
      buddy_id: "duck",
      scope_type: "conversation",
      scope_id: "next"
    })
  )
})

it("preserves staged target edits when the same entry-point target is supplied again", async () => {
  mocks.listChats.mockResolvedValue([
    { id: "chat", title: "Research" },
    { id: "next", title: "Next conversation" }
  ])
  mocks.resolveBuddyConversationTarget.mockResolvedValue({
    id: "chat",
    title: "Research",
    workspace_id: null
  })
  const view = render(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "chat" }}
    />
  )
  await screen.findByRole("option", { name: "Next conversation" })
  fireEvent.change(screen.getByLabelText("Conversation"), {
    target: { value: "next" }
  })
  view.rerender(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "chat" }}
    />
  )
  expect(screen.getByLabelText("Conversation")).toHaveValue("next")
})

it("retains saved artwork without attaching or replacing the new draft after retargeting during creation", async () => {
  mocks.catalog.mockResolvedValue({
    starter_packs: [
      { id: "lens", title: "Lens", production_status: "art_ready" }
    ]
  })
  mocks.listChats.mockResolvedValue([
    { id: "chat", title: "Research" },
    { id: "next", title: "Next conversation" }
  ])
  mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({
    id,
    title: id,
    workspace_id: null
  }))
  let finish!: (value: BuddyProfile) => void
  const pending = new Promise<BuddyProfile>((resolve) => {
    finish = resolve
  })
  mocks.createBuddy.mockReturnValue(pending)
  const view = render(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "chat" }}
    />
  )
  fireEvent.click(
    await screen.findByRole("button", { name: "Load preview", hidden: true })
  )
  fireEvent.click(
    screen.getByRole("button", { name: "Use this Buddy", hidden: true })
  )
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled()
  )
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.createBuddy).toHaveBeenCalledTimes(1))
  view.rerender(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "conversation", scope_id: "next" }}
    />
  )
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  await act(async () => {
    finish({ ...profile, id: "lens-copy", name: "Saved Lens" })
    await pending
  })
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
  expect(props.onApplied).not.toHaveBeenCalled()
  expect(props.onClose).not.toHaveBeenCalled()
  expect(screen.getByRole("button", { name: "Use Duck" })).toHaveAttribute(
    "aria-pressed",
    "true"
  )
  expect(
    screen.getByRole("button", { name: "Use Saved Lens" })
  ).toBeInTheDocument()
  expect(screen.getByLabelText("Conversation")).toHaveValue("next")
  expect(screen.getByRole("status")).toHaveTextContent("Buddy artwork saved")
  fireEvent.click(
    screen.getByRole("button", { name: "Use this Buddy", hidden: true })
  )
  mocks.putBuddyAttachment.mockResolvedValue({ version: 1 })
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() =>
    expect(mocks.putBuddyAttachment).toHaveBeenCalledWith({
      expected_version: 0,
      buddy_id: "lens-copy",
      scope_type: "conversation",
      scope_id: "next"
    })
  )
  expect(mocks.createBuddy).toHaveBeenCalledTimes(1)
})

it("reports the saved old workspace default without attaching or overwriting a retargeted workspace draft", async () => {
  mocks.listWorkspaces.mockResolvedValue({
    items: [
      { id: "old", name: "Old workspace" },
      { id: "next", name: "New workspace" }
    ]
  })
  mocks.getWorkspace.mockImplementation(async (id) => ({
    id,
    name: id,
    version: 1,
    assistant_defaults: null
  }))
  mocks.fetchWithAuth.mockResolvedValue({
    ok: true,
    json: async () => [{ id: "guide", name: "Guide" }]
  })
  let finish!: (value: {
    id: string
    version: number
    assistant_defaults: { assistant_id: string }
  }) => void
  const pending = new Promise((resolve) => {
    finish = resolve
  })
  mocks.patchWorkspace.mockReturnValue(pending)
  const view = render(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "workspace", scope_id: "old" }}
    />
  )
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  fireEvent.change(
    await screen.findByLabelText("Default Persona for new conversations"),
    { target: { value: "guide" } }
  )
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.patchWorkspace).toHaveBeenCalledTimes(1))
  view.rerender(
    <BuddyManagementModal
      {...props}
      target={{ scope_type: "workspace", scope_id: "next" }}
    />
  )
  await waitFor(() =>
    expect(
      screen.getByLabelText("Default Persona for new conversations")
    ).toHaveValue("")
  )
  await act(async () => {
    finish({
      id: "old",
      version: 2,
      assistant_defaults: { assistant_id: "guide" }
    })
    await pending
  })
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
  expect(props.onApplied).not.toHaveBeenCalled()
  expect(props.onClose).not.toHaveBeenCalled()
  expect(
    screen.getByLabelText("Workspace", { selector: "select" })
  ).toHaveValue("next")
  expect(
    screen.getByLabelText("Default Persona for new conversations")
  ).toHaveValue("")
  expect(screen.getByRole("status")).toHaveTextContent(
    "previous workspace default was saved"
  )
})

it.each(["attachment", "refresh"])(
  "preserves the new editor after retargeting during %s completion",
  async (stage) => {
    mocks.listChats.mockResolvedValue([
      { id: "chat", title: "Research" },
      { id: "next", title: "Next conversation" }
    ])
    mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({
      id,
      title: id,
      workspace_id: null
    }))
    let finish!: (value: { version: number }) => void
    const pending = new Promise((resolve) => {
      finish = resolve
    })
    const callbacks = { onApplied: vi.fn(), onClose: vi.fn() }
    mocks.putBuddyAttachment.mockReturnValue(
      stage === "attachment" ? pending : Promise.resolve({ version: 1 })
    )
    callbacks.onApplied.mockReturnValue(
      stage === "refresh" ? pending : Promise.resolve()
    )
    const view = render(
      <BuddyManagementModal
        {...props}
        {...callbacks}
        target={{ scope_type: "conversation", scope_id: "chat" }}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled()
    )
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() =>
      expect(
        stage === "attachment" ? mocks.putBuddyAttachment : callbacks.onApplied
      ).toHaveBeenCalledTimes(1)
    )
    view.rerender(
      <BuddyManagementModal
        {...props}
        {...callbacks}
        target={{ scope_type: "conversation", scope_id: "next" }}
      />
    )
    await act(async () => {
      finish({ version: 1 })
      await pending
    })
    expect(mocks.putBuddyAttachment).toHaveBeenCalledTimes(1)
    expect(callbacks.onApplied).toHaveBeenCalledTimes(1)
    expect(callbacks.onClose).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Conversation")).toHaveValue("next")
    expect(screen.getByRole("status")).toHaveTextContent(
      "previous Buddy attachment was saved"
    )
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled()
  }
)

it.each([
  ["distinct dates", "2026-09-08T12:00:00Z", "2026-09-09T12:00:00Z"],
  ["missing dates", null, undefined],
  ["colliding dates", "2026-09-08T12:00:00Z", "2026-09-08T12:00:00Z"]
])(
  "distinguishes management choices with %s while attaching their exact IDs",
  async (_case, firstDate, secondDate) => {
    const chats = [
      { id: "sharedid-one", title: "Research", created_at: firstDate },
      { id: "sharedid-two", title: "Research", created_at: secondDate },
      { id: "ordinary", title: "A different conversation" }
    ]
    mocks.listChats.mockResolvedValue(chats)
    mocks.putBuddyAttachment.mockResolvedValue({ version: 1 })
    render(<BuddyManagementModal {...props} />)
    await screen.findByRole("option", { name: "A different conversation" })
    const choices = screen.getAllByRole("option", { name: /Research/ })
    expect(choices).toHaveLength(2)
    expect(new Set(choices.map((choice) => choice.textContent)).size).toBe(2)
    if (!firstDate || firstDate === secondDate) {
      expect(choices[0]).toHaveAccessibleName(/sharedid-one.*Research/)
      expect(choices[1]).toHaveAccessibleName(/sharedid-two.*Research/)
    }
    fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
    for (const [index, choice] of choices.entries()) {
      expect(choice).toHaveValue(chats[index].id)
      fireEvent.change(screen.getByLabelText("Conversation"), {
        target: { value: choice.getAttribute("value") }
      })
      fireEvent.click(screen.getByRole("button", { name: "Apply" }))
      await waitFor(() =>
        expect(mocks.putBuddyAttachment).toHaveBeenNthCalledWith(index + 1, {
          expected_version: 0,
          buddy_id: "duck",
          scope_type: "conversation",
          scope_id: chats[index].id
        })
      )
      await waitFor(() =>
        expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled()
      )
    }
    expect(chats.map((chat) => chat.title)).toEqual([
      "Research",
      "Research",
      "A different conversation"
    ])
  }
)

it("updates management date labels for the active locale without changing the selected ID", async () => {
  const createdAt = "2026-09-08T12:00:00Z"
  mocks.listChats.mockResolvedValue([
    { id: "one", title: "Research", created_at: createdAt },
    { id: "two", title: "Research", created_at: "2026-09-09T12:00:00Z" }
  ])
  const { rerender } = render(<BuddyManagementModal {...props} />)
  await screen.findAllByRole("option", { name: /Research/ })
  fireEvent.change(screen.getByLabelText("Conversation"), {
    target: { value: "one" }
  })
  const dateLabel = (locale: string) =>
    new Intl.DateTimeFormat(locale, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    }).format(new Date(createdAt))
  expect(
    screen.getByRole("option", { name: `${dateLabel("en-US")} — Research` })
  ).toHaveValue("one")
  mocks.locale = "de-DE"
  rerender(<BuddyManagementModal {...props} />)
  expect(
    screen.getByRole("option", { name: `${dateLabel("de-DE")} — Research` })
  ).toHaveValue("one")
  expect(screen.getByLabelText("Conversation")).toHaveValue("one")
})

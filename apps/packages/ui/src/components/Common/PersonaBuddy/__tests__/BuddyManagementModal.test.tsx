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
const mocks = vi.hoisted(() => ({
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
  BuddyStarterArtwork: ({ onReadyChange }: any) => (
    <button onClick={() => onReadyChange(true)}>Load preview</button>
  )
}))
beforeEach(() => {
  vi.clearAllMocks()
  mocks.catalog.mockResolvedValue({ starter_packs: [] })
  mocks.listChats.mockResolvedValue([{ id: "chat", title: "Research" }])
  mocks.listWorkspaces.mockResolvedValue({ items: [] })
  mocks.fetchWithAuth.mockResolvedValue({
    ok: true,
    json: async () => ({ personas: [] })
  })
})
afterEach(cleanup)
const profile = {
  id: "duck",
  name: "Duck",
  display_mode: "static",
  version: 1,
  manifest: {},
  assets: [],
  optional_persona_id: null
} as any
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
  let resolve!: (value: any) => void
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
        name: "Conversation location",
        exact: true
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
  let resolve!: (value: any) => void
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
  resolve({ id: "created" })
  await new Promise((r) => setTimeout(r, 0))
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
})

it("applies the new entry-point target when the open modal is retargeted", async () => {
  mocks.listChats.mockResolvedValue([
    { id: "chat", title: "Research" },
    { id: "next", title: "Next conversation" }
  ])
  mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({
    id, title: id === "next" ? "Next conversation" : "Research", workspace_id: null
  }))
  mocks.putBuddyAttachment.mockResolvedValue({ version: 1 })
  const view = render(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "chat" }} />)
  await screen.findByRole("option", { name: "Research" })
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  view.rerender(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "next" }} />)
  await waitFor(() => expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled())
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.putBuddyAttachment).toHaveBeenCalledWith({
    expected_version: 0, buddy_id: "duck", scope_type: "conversation", scope_id: "next"
  }))
})

it("preserves staged target edits when the same entry-point target is supplied again", async () => {
  mocks.listChats.mockResolvedValue([{ id: "chat", title: "Research" }, { id: "next", title: "Next conversation" }])
  mocks.resolveBuddyConversationTarget.mockResolvedValue({ id: "chat", title: "Research", workspace_id: null })
  const view = render(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "chat" }} />)
  await screen.findByRole("option", { name: "Next conversation" })
  fireEvent.change(screen.getByLabelText("Conversation"), { target: { value: "next" } })
  view.rerender(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "chat" }} />)
  expect(screen.getByLabelText("Conversation")).toHaveValue("next")
})

it("retains saved artwork without attaching or replacing the new draft after retargeting during creation", async () => {
  mocks.catalog.mockResolvedValue({ starter_packs: [{ id: "lens", title: "Lens", production_status: "art_ready" }] })
  mocks.listChats.mockResolvedValue([{ id: "chat", title: "Research" }, { id: "next", title: "Next conversation" }])
  mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({ id, title: id, workspace_id: null }))
  let finish!: (value: any) => void
  const pending = new Promise((resolve) => { finish = resolve })
  mocks.createBuddy.mockReturnValue(pending)
  const view = render(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "chat" }} />)
  fireEvent.click(await screen.findByRole("button", { name: "Load preview", hidden: true }))
  fireEvent.click(screen.getByRole("button", { name: "Use this Buddy", hidden: true }))
  await waitFor(() => expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled())
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.createBuddy).toHaveBeenCalledTimes(1))
  view.rerender(<BuddyManagementModal {...props} target={{ scope_type: "conversation", scope_id: "next" }} />)
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  await act(async () => { finish({ ...profile, id: "lens-copy", name: "Saved Lens" }); await pending })
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
  expect(props.onApplied).not.toHaveBeenCalled()
  expect(props.onClose).not.toHaveBeenCalled()
  expect(screen.getByRole("button", { name: "Use Duck" })).toHaveAttribute("aria-pressed", "true")
  expect(screen.getByRole("button", { name: "Use Saved Lens" })).toBeInTheDocument()
  expect(screen.getByLabelText("Conversation")).toHaveValue("next")
  expect(screen.getByRole("status")).toHaveTextContent("Buddy artwork saved")
  fireEvent.click(screen.getByRole("button", { name: "Use this Buddy", hidden: true }))
  mocks.putBuddyAttachment.mockResolvedValue({ version: 1 })
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.putBuddyAttachment).toHaveBeenCalledWith({ expected_version: 0, buddy_id: "lens-copy", scope_type: "conversation", scope_id: "next" }))
  expect(mocks.createBuddy).toHaveBeenCalledTimes(1)
})

it("reports the saved old workspace default without attaching or overwriting a retargeted workspace draft", async () => {
  mocks.listWorkspaces.mockResolvedValue({ items: [{ id: "old", name: "Old workspace" }, { id: "next", name: "New workspace" }] })
  mocks.getWorkspace.mockImplementation(async (id) => ({ id, name: id, version: 1, assistant_defaults: null }))
  mocks.fetchWithAuth.mockResolvedValue({ ok: true, json: async () => [{ id: "guide", name: "Guide" }] })
  let finish!: (value: any) => void
  const pending = new Promise((resolve) => { finish = resolve })
  mocks.patchWorkspace.mockReturnValue(pending)
  const view = render(<BuddyManagementModal {...props} target={{ scope_type: "workspace", scope_id: "old" }} />)
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  fireEvent.change(await screen.findByLabelText("Default Persona for new conversations"), { target: { value: "guide" } })
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(mocks.patchWorkspace).toHaveBeenCalledTimes(1))
  view.rerender(<BuddyManagementModal {...props} target={{ scope_type: "workspace", scope_id: "next" }} />)
  await waitFor(() => expect(screen.getByLabelText("Default Persona for new conversations")).toHaveValue(""))
  await act(async () => { finish({ id: "old", version: 2, assistant_defaults: { assistant_id: "guide" } }); await pending })
  expect(mocks.putBuddyAttachment).not.toHaveBeenCalled()
  expect(props.onApplied).not.toHaveBeenCalled()
  expect(props.onClose).not.toHaveBeenCalled()
  expect(screen.getByLabelText("Workspace", { selector: "select" })).toHaveValue("next")
  expect(screen.getByLabelText("Default Persona for new conversations")).toHaveValue("")
  expect(screen.getByRole("status")).toHaveTextContent("previous workspace default was saved")
})

it.each(["attachment", "refresh"])("preserves the new editor after retargeting during %s completion", async (stage) => {
  mocks.listChats.mockResolvedValue([{ id: "chat", title: "Research" }, { id: "next", title: "Next conversation" }])
  mocks.resolveBuddyConversationTarget.mockImplementation(async (id) => ({ id, title: id, workspace_id: null }))
  let finish!: (value: any) => void
  const pending = new Promise((resolve) => { finish = resolve })
  const callbacks = { onApplied: vi.fn(), onClose: vi.fn() }
  mocks.putBuddyAttachment.mockReturnValue(stage === "attachment" ? pending : Promise.resolve({ version: 1 }))
  callbacks.onApplied.mockReturnValue(stage === "refresh" ? pending : Promise.resolve())
  const view = render(<BuddyManagementModal {...props} {...callbacks} target={{ scope_type: "conversation", scope_id: "chat" }} />)
  fireEvent.click(screen.getByRole("button", { name: "Use Duck" }))
  await waitFor(() => expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled())
  fireEvent.click(screen.getByRole("button", { name: "Apply" }))
  await waitFor(() => expect(stage === "attachment" ? mocks.putBuddyAttachment : callbacks.onApplied).toHaveBeenCalledTimes(1))
  view.rerender(<BuddyManagementModal {...props} {...callbacks} target={{ scope_type: "conversation", scope_id: "next" }} />)
  await act(async () => { finish({ version: 1 }); await pending })
  expect(mocks.putBuddyAttachment).toHaveBeenCalledTimes(1)
  expect(callbacks.onApplied).toHaveBeenCalledTimes(1)
  expect(callbacks.onClose).not.toHaveBeenCalled()
  expect(screen.getByLabelText("Conversation")).toHaveValue("next")
  expect(screen.getByRole("status")).toHaveTextContent("previous Buddy attachment was saved")
  expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled()
})

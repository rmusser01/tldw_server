import React from "react"
import {
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

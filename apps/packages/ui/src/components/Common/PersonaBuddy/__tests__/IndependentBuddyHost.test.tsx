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
import {
  IndependentBuddyHost,
  IndependentBuddySession
} from "../IndependentBuddyHost"
import { useBuddyManagementStore } from "@/store/buddy-management"
import { usePersonaBuddyShellStore } from "@/store/persona-buddy-shell"
const mocks = vi.hoisted(() => ({
  demoEnabled: false,
  connection: {
    serverUrl: "https://server.invalid",
    accessToken: "token",
    authMode: "multi-user",
    authSource: "browser",
    orgId: 1
  },
  listBuddies: vi.fn(),
  getBuddy: vi.fn(),
  getBuddyAttachment: vi.fn(),
  listBuddyConversations: vi.fn(),
  readBuddyConversation: vi.fn(),
  detachBuddy: vi.fn(),
  updateBuddy: vi.fn(),
  buddyAssets: vi.fn((_profile: unknown) => ({})),
  renderArtwork: vi.fn()
}))
vi.mock("@/services/buddies", () => ({
  ...mocks,
  BUDDY_PAGE_SIZE: 100
}))
vi.mock("../SpriteFrameRenderer", () => ({
  SpriteFrameRenderer: (props: unknown) => {
    mocks.renderArtwork(props)
    return <span>Duck artwork</span>
  }
}))
vi.mock("../BuddyInteraction", () => ({
  BuddyInteraction: ({ conversation, draftState }: any) => (
    <div>
      Reply to {conversation?.title}
      <input
        aria-label="Draft reply"
        value={draftState?.drafts[conversation?.id] ?? ""}
        onChange={(e) =>
          draftState?.setDrafts((previous: any) => ({
            ...previous,
            [conversation.id]: e.target.value
          }))
        }
      />
    </div>
  )
}))
vi.mock("../BuddyManagementModal", () => ({
  BuddyManagementModal: ({
    profiles,
    hasMoreProfiles,
    onLoadMoreProfiles,
    onApplied
  }: any) => (
    <div>
      Manage {profiles.map((p: any) => p.name).join(",")}
      {hasMoreProfiles ? (
        <button onClick={onLoadMoreProfiles}>More Buddies</button>
      ) : null}
      <button onClick={onApplied}>Refresh collections</button>
    </div>
  )
}))
vi.mock("@/context/demo-mode", () => ({
  useSafeDemoMode: () => ({ demoEnabled: mocks.demoEnabled })
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    loading: false,
    config: mocks.connection
  })
}))
beforeEach(() => {
  vi.clearAllMocks()
  mocks.demoEnabled = false
  mocks.connection = {
    serverUrl: "https://server.invalid",
    accessToken: "token",
    authMode: "multi-user",
    authSource: "browser",
    orgId: 1
  }
  useBuddyManagementStore.getState().close()
  useBuddyManagementStore.getState().setAttached(false)
  usePersonaBuddyShellStore
    .getState()
    .setPosition("web-desktop", { x: 30, y: 40 })
  usePersonaBuddyShellStore
    .getState()
    .setPosition("sidepanel-desktop", { x: 70, y: 80 })
  mocks.listBuddies.mockResolvedValue({
    buddies: [
      {
        id: "duck",
        name: "Duck",
        manifest: {},
        assets: [],
        display_mode: "static"
      }
    ]
  })
  mocks.getBuddyAttachment.mockResolvedValue({
    version: 1,
    attachment: {
      buddy_id: "duck",
      scope_type: "conversation",
      scope_id: "chat"
    },
    target: { title: "Research" }
  })
  mocks.listBuddyConversations.mockResolvedValue({
    conversations: [{ id: "chat", title: "Research" }]
  })
})
it("loads an attached Buddy directly when it is outside the first page", async () => {
  mocks.listBuddies.mockResolvedValue({ buddies: [] })
  mocks.getBuddy.mockResolvedValue({
    id: "duck",
    name: "Duck",
    manifest: {},
    assets: [],
    display_mode: "static"
  })
  render(<IndependentBuddySession />)
  await screen.findByRole("button", { name: "Open Duck — Research" })
  expect(mocks.getBuddy).toHaveBeenCalledWith("duck")
})
it("exposes an initial failure and retries the collections", async () => {
  mocks.listBuddies.mockRejectedValueOnce(new Error("Cannot load Buddies"))
  useBuddyManagementStore.getState().show()
  render(<IndependentBuddySession />)
  await screen.findByText("Cannot load Buddies")
  fireEvent.click(screen.getByRole("button", { name: "Retry" }))
  await screen.findByText("Manage Duck")
})
it("retains conversation drafts when the attached artwork changes", async () => {
  mocks.listBuddies.mockResolvedValue({
    buddies: [
      {
        id: "duck",
        name: "Duck",
        manifest: {},
        assets: [],
        display_mode: "static"
      },
      {
        id: "cat",
        name: "Cat",
        manifest: {},
        assets: [],
        display_mode: "static"
      }
    ]
  })
  render(<IndependentBuddySession />)
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Duck — Research" })
  )
  fireEvent.change(screen.getByLabelText("Draft reply"), {
    target: { value: "Keep this unsent draft" }
  })
  mocks.getBuddyAttachment.mockResolvedValue({
    version: 2,
    attachment: {
      buddy_id: "cat",
      scope_type: "conversation",
      scope_id: "chat"
    },
    target: { title: "Research" }
  })
  fireEvent.click(
    screen.getByRole("button", { name: "Manage Buddy & Persona" })
  )
  fireEvent.click(
    await screen.findByRole("button", { name: "Refresh collections" })
  )
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Cat — Research" })
  )
  expect(screen.getByLabelText("Draft reply")).toHaveValue(
    "Keep this unsent draft"
  )
})
it("keeps fetched profile pages through periodic refresh", async () => {
  const firstPage = Array.from({ length: 100 }, (_, i) => ({
    id: `b${i}`,
    name: `Buddy${i}`,
    manifest: {},
    assets: [],
    display_mode: "static"
  }))
  mocks.listBuddies.mockImplementation(async ({ offset }: any) => ({
    buddies:
      offset === 100
        ? [
            {
              id: "duck",
              name: "Duck",
              manifest: {},
              assets: [],
              display_mode: "static"
            }
          ]
        : firstPage
  }))
  mocks.getBuddy.mockResolvedValue({
    id: "duck",
    name: "Duck",
    manifest: {},
    assets: [],
    display_mode: "static"
  })
  useBuddyManagementStore.getState().show()
  render(<IndependentBuddySession />)
  fireEvent.click(await screen.findByRole("button", { name: "More Buddies" }))
  await waitFor(() =>
    expect(mocks.listBuddies).toHaveBeenCalledWith({ limit: 100, offset: 100 })
  )
  fireEvent.click(screen.getByRole("button", { name: "Refresh collections" }))
  await waitFor(() =>
    expect(
      mocks.listBuddies.mock.calls.filter(([page]) => page.offset === 100)
    ).toHaveLength(2)
  )
})
it("uses the sidepanel bucket for pointer and keyboard movement", async () => {
  vi.stubGlobal("PointerEvent", MouseEvent)
  render(<IndependentBuddySession root="sidepanel" />)
  const move = await screen.findByRole("button", {
    name: "Move Buddy with arrow keys"
  })
  fireEvent.keyDown(move, { key: "ArrowRight" })
  expect(
    usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
  ).toEqual({ x: 80, y: 80 })
  fireEvent.pointerDown(move, { clientX: 10, clientY: 10, button: 0 })
  fireEvent.pointerMove(move, { clientX: 40, clientY: 50 })
  fireEvent.pointerUp(move)
  expect(
    usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
  ).toEqual({ x: 110, y: 120 })
  expect(usePersonaBuddyShellStore.getState().positions["web-desktop"]).toEqual(
    { x: 30, y: 40 }
  )
  vi.unstubAllGlobals()
  vi.useRealTimers()
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})
it("keeps one attached Buddy and named reply target as page content changes", async () => {
  const { rerender } = render(
    <>
      <main>Console</main>
      <IndependentBuddySession />
    </>
  )
  await screen.findByRole("button", { name: "Open Duck — Research" })
  rerender(
    <>
      <main>Watchlists</main>
      <IndependentBuddySession />
    </>
  )
  fireEvent.click(screen.getByRole("button", { name: "Open Duck — Research" }))
  await screen.findByText("Reply to Research")
  expect(screen.getAllByText("Duck artwork")).toHaveLength(1)
  expect(mocks.detachBuddy).not.toHaveBeenCalled()
})
it("detaching removes the attachment without invoking turn cancellation", async () => {
  mocks.detachBuddy.mockResolvedValue({ version: 2, attachment: null })
  render(<IndependentBuddySession />)
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Duck — Research" })
  )
  fireEvent.click(screen.getByRole("button", { name: "Detach Buddy" }))
  await waitFor(() => expect(mocks.detachBuddy).toHaveBeenCalledWith(1))
})

it.each([
  ["authMode", "single-user"],
  ["authSource", "managed"],
  ["orgId", 2]
])(
  "clears private drafts when %s changes with the same credentials",
  async (field, value) => {
    const { rerender } = render(<IndependentBuddyHost />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open Duck — Research" })
    )
    fireEvent.change(screen.getByLabelText("Draft reply"), {
      target: { value: "Private draft" }
    })
    mocks.connection = { ...mocks.connection, [field]: value }
    rerender(<IndependentBuddyHost />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open Duck — Research" })
    )
    expect(screen.getByLabelText("Draft reply")).toHaveValue("")
  }
)
it("offers repair and suppresses the legacy shell when the attached profile cannot load", async () => {
  mocks.listBuddies.mockResolvedValue({ buddies: [] })
  mocks.getBuddy.mockRejectedValue(new Error("Attached artwork is unavailable"))
  render(<IndependentBuddySession />)
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Buddy attachment needs attention"
    })
  )
  await screen.findByText("Attached artwork is unavailable")
  expect(useBuddyManagementStore.getState().attached).toBe(true)
  expect(screen.queryByTestId("independent-buddy")).not.toBeInTheDocument()
  mocks.getBuddy.mockResolvedValue({
    id: "duck",
    name: "Duck",
    manifest: {},
    assets: [],
    display_mode: "static"
  })
  fireEvent.click(screen.getByRole("button", { name: "Retry" }))
  await screen.findByRole("button", { name: "Open Duck — Research" })
})
it("keeps later workspace conversation choices and drafts through refresh and artwork changes", async () => {
  const firstPage = Array.from({ length: 100 }, (_, i) => ({
    id: `c${i}`,
    title: `Conversation ${i}`
  }))
  mocks.listBuddyConversations.mockImplementation(async ({ offset }: any) => ({
    conversations:
      offset === 100
        ? [{ id: "later", title: "Later conversation" }]
        : firstPage
  }))
  mocks.listBuddies.mockResolvedValue({
    buddies: [
      {
        id: "duck",
        name: "Duck",
        manifest: {},
        assets: [],
        display_mode: "static"
      },
      {
        id: "cat",
        name: "Cat",
        manifest: {},
        assets: [],
        display_mode: "static"
      }
    ]
  })
  mocks.getBuddyAttachment.mockResolvedValue({
    version: 1,
    attachment: {
      buddy_id: "duck",
      scope_type: "workspace",
      scope_id: "research"
    },
    target: { title: "Research" }
  })
  render(<IndependentBuddySession />)
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Duck — Research" })
  )
  fireEvent.click(
    screen.getByRole("button", { name: "Load more conversations" })
  )
  await screen.findByRole("option", { name: "Later conversation" })
  fireEvent.change(
    screen.getByLabelText("Reply to conversation", { exact: true }),
    { target: { value: "later" } }
  )
  fireEvent.change(screen.getByLabelText("Draft reply"), {
    target: { value: "Later draft" }
  })
  mocks.getBuddyAttachment.mockResolvedValue({
    version: 2,
    attachment: {
      buddy_id: "cat",
      scope_type: "workspace",
      scope_id: "research"
    },
    target: { title: "Research" }
  })
  fireEvent.click(
    screen.getByRole("button", { name: "Manage Buddy & Persona" })
  )
  fireEvent.click(
    await screen.findByRole("button", { name: "Refresh collections" })
  )
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Cat — Research" })
  )
  expect(
    screen.getByLabelText("Reply to conversation", { exact: true })
  ).toHaveValue("later")
  expect(screen.getByLabelText("Draft reply")).toHaveValue("Later draft")
  expect(screen.getByLabelText("Expressions", { exact: true })).toHaveValue(
    "static"
  )
  expect(
    mocks.listBuddyConversations.mock.calls.filter(
      ([page]) => page.offset === 100
    )
  ).toHaveLength(2)
})

it("preserves animation props through unchanged polls and observes profile updates", async () => {
  vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] })
  let version = 1
  let optionalPersonaAvailable = true
  mocks.listBuddies.mockImplementation(async () => ({
    buddies: [
      {
        id: "duck",
        name: "Duck",
        version,
        optional_persona_available: optionalPersonaAvailable,
        manifest: { animations: {} },
        assets: [],
        display_mode: "dynamic"
      }
    ]
  }))
  render(<IndependentBuddySession />)
  await screen.findByRole("button", { name: "Open Duck — Research" })
  const before = mocks.renderArtwork.mock.calls.at(-1)![0]
  const initialReads = mocks.listBuddies.mock.calls.length
  await act(async () => {
    await vi.advanceTimersByTimeAsync(5000)
  })
  expect(mocks.listBuddies.mock.calls.length).toBeGreaterThan(initialReads)
  const unchanged = mocks.renderArtwork.mock.calls.at(-1)![0]
  expect(unchanged.manifest).toBe(before.manifest)
  expect(unchanged.assets).toBe(before.assets)
  version = 2
  await act(async () => {
    await vi.advanceTimersByTimeAsync(5000)
  })
  const updated = mocks.renderArtwork.mock.calls.at(-1)![0]
  expect(updated.manifest).not.toBe(before.manifest)
  expect(updated.assets).not.toBe(before.assets)
  expect(mocks.buddyAssets.mock.calls.at(-1)![0]).toMatchObject({
    version: 2,
    optional_persona_available: true
  })
  optionalPersonaAvailable = false
  await act(async () => {
    await vi.advanceTimersByTimeAsync(5000)
  })
  expect(mocks.buddyAssets.mock.calls.at(-1)![0]).toMatchObject({
    version: 2,
    optional_persona_available: false
  })
})

it("clears private Buddy state while demo mode is enabled", async () => {
  const { rerender } = render(<IndependentBuddyHost />)
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Duck — Research" })
  )
  fireEvent.change(screen.getByLabelText("Draft reply"), {
    target: { value: "Private draft" }
  })
  mocks.demoEnabled = true
  rerender(<IndependentBuddyHost />)
  expect(screen.queryByTestId("independent-buddy")).not.toBeInTheDocument()
  mocks.demoEnabled = false
  rerender(<IndependentBuddyHost />)
  fireEvent.click(
    await screen.findByRole("button", { name: "Open Duck — Research" })
  )
  expect(screen.getByLabelText("Draft reply")).toHaveValue("")
})

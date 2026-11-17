import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listBuddies: vi.fn(),
  getBuddy: vi.fn(),
  getBuddyAttachment: vi.fn(),
  listBuddyConversations: vi.fn(),
  updateBuddy: vi.fn(),
  detachBuddy: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      return defaultValueOrOptions?.defaultValue ?? ""
    },
    i18n: { resolvedLanguage: "en" }
  })
}))

vi.mock("@/services/buddies", () => ({
  BUDDY_PAGE_SIZE: 100,
  buddyAssets: () => ({}),
  listBuddies: mocks.listBuddies,
  getBuddy: mocks.getBuddy,
  getBuddyAttachment: mocks.getBuddyAttachment,
  listBuddyConversations: mocks.listBuddyConversations,
  updateBuddy: mocks.updateBuddy,
  detachBuddy: mocks.detachBuddy
}))

vi.mock("../BuddyManagementModal", () => ({
  BuddyManagementModal: () => <div data-testid="buddy-management-modal" />
}))

vi.mock("../BuddyInteraction", () => ({
  BuddyInteraction: () => <div data-testid="buddy-interaction" />
}))

vi.mock("../SpriteFrameRenderer", () => ({
  SpriteFrameRenderer: () => <div data-testid="sprite-frame-renderer" />
}))

import { IndependentBuddySession } from "../IndependentBuddyHost"
import { useBuddyManagementStore } from "@/store/buddy-management"
import { BUDDY_POLL_MAX_IDLE_MS } from "../buddy-poll-schedule"

// WP-39: the globally mounted Buddy host must not keep a request baseline
// going while its panel is closed or the page is hidden. Mounting performs
// two reads: the initial one, plus one more when the loaded scope is pinned
// (the polling effect restarts on that dependency change).

const setVisibility = (state: DocumentVisibilityState) => {
  vi.spyOn(document, "visibilityState", "get").mockReturnValue(state)
  document.dispatchEvent(new Event("visibilitychange"))
}

const attachmentState = (version: number) => ({
  client_slot: "default",
  version,
  attachment: {
    buddy_id: "buddy-1",
    scope_type: "workspace" as const,
    scope_id: "workspace-1"
  },
  unavailable_reason: null,
  target: { title: "Workspace", workspace_id: "workspace-1" }
})

const buddyProfile = () => ({
  id: "buddy-1",
  name: "Buddy",
  optional_persona_id: null,
  optional_persona_available: false,
  display_mode: "dynamic" as const,
  version: 1,
  manifest: null,
  assets: [],
  attribution: {}
})

describe("IndependentBuddySession poll gating (WP-39)", () => {
  beforeEach(() => {
    vi.useFakeTimers({
      toFake: ["setInterval", "clearInterval", "setTimeout", "clearTimeout"]
    })
    vi.clearAllMocks()
    useBuddyManagementStore.setState({
      open: false,
      target: null,
      attached: false,
      conversationSettings: null
    })
    mocks.listBuddies.mockResolvedValue({ buddies: [] })
    mocks.getBuddy.mockResolvedValue(buddyProfile())
    mocks.getBuddyAttachment.mockResolvedValue(attachmentState(1))
    mocks.listBuddyConversations.mockResolvedValue({ conversations: [] })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    cleanup()
  })

  it("issues no 5 s polls while the buddy panel is closed", async () => {
    await act(async () => {
      render(<IndependentBuddySession />)
    })
    const mountReads = mocks.getBuddyAttachment.mock.calls.length
    expect(mountReads).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    // Idle closed panel: no 5 s polls; the only repeat read is the 60 s backoff.
    expect(mocks.getBuddyAttachment).toHaveBeenCalledTimes(mountReads)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(BUDDY_POLL_MAX_IDLE_MS)
    })
    expect(mocks.getBuddyAttachment).toHaveBeenCalledTimes(mountReads + 1)
  })

  it("keeps the 5 s cadence while the panel is open and data keeps changing", async () => {
    useBuddyManagementStore.setState({ open: true })
    let version = 0
    mocks.getBuddyAttachment.mockImplementation(async () =>
      attachmentState(++version)
    )

    await act(async () => {
      render(<IndependentBuddySession />)
    })
    const mountReads = mocks.getBuddyAttachment.mock.calls.length
    expect(mountReads).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000)
    })
    expect(mocks.getBuddyAttachment).toHaveBeenCalledTimes(mountReads + 3)
  })

  it("produces no polls while the document is hidden and catches up on show", async () => {
    await act(async () => {
      render(<IndependentBuddySession />)
    })
    const mountReads = mocks.getBuddyAttachment.mock.calls.length
    expect(mountReads).toBe(2)

    setVisibility("hidden")
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000)
    })
    expect(mocks.getBuddyAttachment).toHaveBeenCalledTimes(mountReads)

    setVisibility("visible")
    expect(mocks.getBuddyAttachment).toHaveBeenCalledTimes(mountReads + 1)
  })
})

// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const mocks = vi.hoisted(() => ({
  hasResumableSidepanelChat: vi.fn<() => Promise<boolean>>(),
  getServerCapabilities: vi.fn<() => Promise<{ hasPersonalization?: boolean } | null>>(),
  fetchPersonalizationProfile: vi.fn<() => Promise<{ enabled?: boolean } | null>>()
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: () => mocks.getServerCapabilities()
}))

vi.mock("@/services/companion", () => ({
  fetchPersonalizationProfile: () => mocks.fetchPersonalizationProfile()
}))

vi.mock("../sidepanel-chat", () => ({
  __esModule: true,
  default: () => <div data-testid="sidepanel-chat-root">chat</div>
}))

vi.mock("../sidepanel-chat-resume", () => ({
  hasResumableSidepanelChat: () => mocks.hasResumableSidepanelChat()
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: ({ label }: { label?: string }) => (
    <div data-testid="page-assist-loader">{label || "loading"}</div>
  )
}))

vi.mock("../sidepanel-companion", () => ({
  __esModule: true,
  default: () => <div data-testid="sidepanel-companion-root">companion</div>
}))

import SidepanelHomeResolver from "../sidepanel-home-resolver"

const renderResolver = (initialEntries?: string[]) =>
  render(
    <MemoryRouter initialEntries={initialEntries}>
      <SidepanelHomeResolver />
    </MemoryRouter>
  )

describe("SidepanelHomeResolver", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getServerCapabilities.mockResolvedValue({ hasPersonalization: true })
    mocks.fetchPersonalizationProfile.mockResolvedValue({ enabled: true })
  })

  it("renders chat from sidepanel / when resumable chat exists", async () => {
    mocks.hasResumableSidepanelChat.mockResolvedValueOnce(true)

    render(
      <MemoryRouter>
        <SidepanelHomeResolver />
      </MemoryRouter>
    )

    // First use of the lazy chat chunk: the dynamic import can exceed the 1s
    // default wait when the machine is under load, so give it headroom.
    expect(
      await screen.findByTestId("sidepanel-chat-root", undefined, {
        timeout: 15_000
      })
    ).toBeInTheDocument()
  })

  it("XS-17: opens chat, not a Companion of setup cards, when personalization is unavailable", async () => {
    mocks.hasResumableSidepanelChat.mockResolvedValueOnce(false)
    mocks.getServerCapabilities.mockResolvedValueOnce({ hasPersonalization: false })

    renderResolver()

    expect(
      await screen.findByTestId("sidepanel-chat-root", undefined, { timeout: 15_000 })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("sidepanel-companion-root")).toBeNull()
    expect(mocks.fetchPersonalizationProfile).not.toHaveBeenCalled()
  })

  it("XS-17: opens chat when the Companion has not been turned on", async () => {
    mocks.hasResumableSidepanelChat.mockResolvedValueOnce(false)
    mocks.fetchPersonalizationProfile.mockResolvedValueOnce({ enabled: false })

    renderResolver()

    expect(
      await screen.findByTestId("sidepanel-chat-root", undefined, { timeout: 15_000 })
    ).toBeInTheDocument()
  })

  it("XS-17: opens chat when the server cannot say whether the Companion is ready", async () => {
    mocks.hasResumableSidepanelChat.mockResolvedValueOnce(false)
    mocks.getServerCapabilities.mockRejectedValueOnce(new Error("offline"))

    renderResolver()

    expect(
      await screen.findByTestId("sidepanel-chat-root", undefined, { timeout: 15_000 })
    ).toBeInTheDocument()
  })

  it("renders Companion Home from sidepanel / when no resumable chat exists and the Companion is on", async () => {
    mocks.hasResumableSidepanelChat.mockResolvedValueOnce(false)

    render(
      <MemoryRouter>
        <SidepanelHomeResolver />
      </MemoryRouter>
    )

    // First use of the lazy companion chunk (see note above).
    expect(
      await screen.findByTestId("sidepanel-companion-root", undefined, {
        timeout: 15_000
      })
    ).toBeInTheDocument()
  })

  it("renders chat immediately when the route forces chat view", async () => {
    render(
      <MemoryRouter initialEntries={["/?view=chat"]}>
        <SidepanelHomeResolver />
      </MemoryRouter>
    )

    expect(await screen.findByTestId("sidepanel-chat-root")).toBeInTheDocument()
    expect(mocks.hasResumableSidepanelChat).not.toHaveBeenCalled()
  })

  it("renders Companion Home immediately when the route forces companion view", async () => {
    render(
      <MemoryRouter initialEntries={["/?view=companion"]}>
        <SidepanelHomeResolver />
      </MemoryRouter>
    )

    expect(await screen.findByTestId("sidepanel-companion-root")).toBeInTheDocument()
    expect(mocks.hasResumableSidepanelChat).not.toHaveBeenCalled()
  })
})

import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

const { runtimeGetURLMock, tabsCreateMock, runtime, errorFeedback } = vi.hoisted(() => ({
  errorFeedback: vi.fn(),
  runtime: { id: "header-extension" as string | undefined },
  runtimeGetURLMock: vi.fn((path: string) => `chrome-extension://test${path}`),
  tabsCreateMock: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/assets/icon.png", () => ({
  default: "chrome-extension://test/icon.png"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({
    children,
    to,
    ...props
  }: {
    children?: React.ReactNode
    to: string
  }) => (
    <a href={to} {...props}>
      {children}
    </a>
  )
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() { return runtime.id },
      getURL: runtimeGetURLMock
    },
    tabs: {
      create: tabsCreateMock
    }
  }
}))

vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => ({
    temporaryChat: false
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: errorFeedback
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasPersona: false }
  })
}))

vi.mock("../StatusDot", () => ({
  StatusDot: () => <span data-testid="status-dot" />
}))

vi.mock("../TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => <div data-testid="tts-clips-drawer" />
}))

describe("SidepanelHeaderSimple WebUI chat handoff", () => {
  beforeEach(() => {
    runtime.id = "header-extension"
    runtimeGetURLMock.mockClear()
    tabsCreateMock.mockClear()
  })

  it("delegates the visible full-screen chat action to the WebUI chat handoff", async () => {
    const onOpenChatInWebUi = vi.fn().mockResolvedValue(undefined)

    render(
      <SidepanelHeaderSimple onOpenChatInWebUi={onOpenChatInWebUi} />
    )

    expect(screen.getByRole("button", { name: "Open full chat in WebUI" })).toHaveAccessibleDescription("Opens the configured WebUI with a draft and selected chat options. Device-local history may be unavailable there.")
    fireEvent.click(screen.getByTestId("chat-open-full-screen"))

    await waitFor(() =>
      expect(onOpenChatInWebUi).toHaveBeenCalledTimes(1)
    )
    expect(runtimeGetURLMock).not.toHaveBeenCalledWith("/options.html#/")
    expect(tabsCreateMock).not.toHaveBeenCalled()
  })
})

import { HistorySelectionContext, historySelectionExpansionPath, parseHistorySelectionHandoff } from "@/hooks/chat/useHistorySelection"
it("uses the configured WebUI handoff with H1 present in the web compact view", async () => {
  runtime.id = undefined
  const onOpenChatInWebUi = vi.fn().mockResolvedValue(undefined)
  const prepareExpansionPath = vi.fn()
  render(<HistorySelectionContext.Provider value={{ prepareExpansionPath } as unknown as import("@/hooks/chat/useHistorySelection").HistorySelectionController}>
    <SidepanelHeaderSimple onOpenChatInWebUi={onOpenChatInWebUi} />
  </HistorySelectionContext.Provider>)
  fireEvent.click(screen.getByTestId("chat-open-full-screen"))
  await waitFor(() => expect(onOpenChatInWebUi).toHaveBeenCalledTimes(1))
  expect(prepareExpansionPath).not.toHaveBeenCalled()
  expect(tabsCreateMock).not.toHaveBeenCalled()
})

it("opens H1 expansion in the extension full page with only a scoped bookmark address", async () => {
  runtime.id = "header-extension"
  const onOpenChatInWebUi = vi.fn()
  const reference = { profile_id: "profile", client_session_id: "view-a", owner_key: "owner", conversation_id: "chat" }
  render(<HistorySelectionContext.Provider value={{ reference, owner: { kind: "local" }, prepareExpansionPath: async () => historySelectionExpansionPath({ reference, owner: { kind: "local" } as any }) } as any}>
    <SidepanelHeaderSimple onOpenChatInWebUi={onOpenChatInWebUi} />
  </HistorySelectionContext.Provider>)
  fireEvent.click(screen.getByRole("button", { name: "Expand in full page" }))
  await waitFor(() => expect(tabsCreateMock).toHaveBeenCalled())
  const url = tabsCreateMock.mock.calls.at(-1)![0].url
  expect(url.startsWith("chrome-extension://test/options.html#/chat?")).toBe(true)
  expect(parseHistorySelectionHandoff(url)).toEqual({ ...reference, owner_kind: "local" })
  expect(onOpenChatInWebUi).not.toHaveBeenCalled()
})

 it("disables expansion until the conversation has restored", () => {
  runtime.id = undefined
  const onOpenChatInWebUi = vi.fn()
  render(<SidepanelHeaderSimple chatReady={false} onOpenChatInWebUi={onOpenChatInWebUi} />)
  const expand = screen.getByTestId("chat-open-full-screen")
  expect(expand).toBeDisabled()
  fireEvent.click(expand)
  expect(onOpenChatInWebUi).not.toHaveBeenCalled()
})

it.each(["unmount", "replacement", "not ready"])("suppresses retired header WebUI failure after %s", async boundary => {
  runtime.id = undefined
  errorFeedback.mockClear()
  const log = vi.spyOn(console, "error").mockImplementation(() => {})
  let reject!: (error: Error) => void
  const onOpen = vi.fn(() => new Promise<void>((_, fail) => { reject = fail }))
  const view = render(<SidepanelHeaderSimple onOpenChatInWebUi={onOpen} />)
  fireEvent.click(screen.getByTestId("chat-open-full-screen"))
  if (boundary === "unmount") view.unmount()
  else view.rerender(<SidepanelHeaderSimple chatReady={boundary !== "not ready"} onOpenChatInWebUi={boundary === "replacement" ? vi.fn() : onOpen} />)
  await act(async () => { reject(new Error("OLD ERROR")) })
  expect(errorFeedback).not.toHaveBeenCalled()
  log.mockRestore()
})


it("suppresses a superseded header opening failure for the same source", async () => {
  runtime.id = undefined
  errorFeedback.mockClear()
  const log = vi.spyOn(console, "error").mockImplementation(() => {})
  const rejects: Array<(error: Error) => void> = []
  const onOpen = vi.fn(() => new Promise<void>((_, reject) => { rejects.push(reject) }))
  render(<SidepanelHeaderSimple onOpenChatInWebUi={onOpen} />)
  fireEvent.click(screen.getByTestId("chat-open-full-screen"))
  fireEvent.click(screen.getByTestId("chat-open-full-screen"))
  await act(async () => { rejects[0](new Error("SUPERSEDED ERROR")) })
  expect(errorFeedback).not.toHaveBeenCalled()
  await act(async () => { rejects[1](new Error("CURRENT ERROR")) })
  expect(errorFeedback).toHaveBeenCalledOnce()
  log.mockRestore()
})

import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import { act, render, screen } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidepanelApp } from "../sidepanel-app"

const authority = vi.hoisted(() => ({
  key: "verified-sidebar-owner" as string | null
}))
vi.mock("@/services/tldw/quick-ingest-authority", () => ({
  useQuickIngestAuthority: () => authority.key
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [resolvePresetMap(), vi.fn(), { isLoading: false }]
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value: any) => value?.defaultValue || value
  })
}))
vi.mock("~/hooks/useSidepanelInit", () => ({
  useSidepanelInit: () => ({
    direction: "ltr",
    t: (_key: string, value: any) => value?.defaultValue
  })
}))
vi.mock("../AppShell", () => ({
  AppShell: ({ children, extras }: any) => (
    <>
      {children}
      {extras}
    </>
  )
}))
vi.mock("@/routes/sidepanel-route-shell", () => ({
  SidepanelRouteShell: () => null
}))
vi.mock("@/components/Common/PersonaBuddy", () => ({
  BuddyShellHost: () => null,
  BuddyShellRenderContextProvider: ({ children }: any) => children
}))
vi.mock("@/components/Common/PersonaBuddy/IndependentBuddyHost", () => ({
  IndependentBuddyHost: () => null
}))
vi.mock("@/components/Common/QuickChatHelper", () => ({
  QuickChatHelperButton: () => null
}))
vi.mock("@/components/Common/PageHelpModalHost", () => ({
  PageHelpModalHost: () => null
}))
vi.mock("@/components/Common/Workflow/WorkflowIntegrationHost", () => ({
  WorkflowIntegrationHost: () => null
}))
vi.mock("@/utils/antd-notification-compat", () => ({
  patchStaticAntdNotificationCompat: () => {}
}))
// The real shared event host remains mounted; only the expensive wizard leaf is replaced.
vi.mock("@/components/Common/QuickIngestWizardModal", () => ({
  QuickIngestWizardModal: ({ open }: any) =>
    open ? <div role="dialog" aria-label="Shared Quick ingest" /> : null
}))

describe("actual sidebar Quick ingest host", () => {
  beforeEach(async () => {
    authority.key = "verified-sidebar-owner"
    sessionStorage.clear()
    delete (window as any).__tldwPendingQuickIngestOpen
    useQuickIngestSessionStore.getState().setAuthority(authority.key)
    useQuickIngestSessionStore.getState().clearSession()
    await useQuickIngestSessionStore.persist.rehydrate()
  })

  it("opens exactly one shared host for the sidebar HTTPS handoff", async () => {
    render(<SidepanelApp />)
    act(() => {
      requestQuickIngestOpen({
        source: "manual",
        url: "https://example.com/sidebar-source"
      })
    })
    expect(
      await screen.findAllByRole("dialog", { name: "Shared Quick ingest" })
    ).toHaveLength(1)
    expect(
      useQuickIngestSessionStore
        .getState()
        .session?.queueItems.map((item) => item.url)
    ).toEqual(["https://example.com/sidebar-source"])
    expect(useQuickIngestSessionStore.getState().session?.authorityKey).toBe(
      authority.key
    )
  })

  it("holds an open event until verified sidebar authority is available", () => {
    authority.key = null
    useQuickIngestSessionStore.getState().setAuthority(null)
    render(<SidepanelApp />)
    act(() => {
      requestQuickIngestOpen({
        source: "manual",
        url: "https://example.com/sidebar-source"
      })
    })
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(useQuickIngestSessionStore.getState().session).toBeNull()
  })
})

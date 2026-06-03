// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import OptionLayout from "../Layout"

const storeMessageOptionMock = vi.hoisted(() =>
  vi.fn(() => ({ historyId: null, serverChatId: null }))
)

vi.mock("@/hooks/useLayoutEffectsOwner", () => ({
  useLayoutEffectsOwner: () => false
}))

vi.mock("@/hooks/useStorageMigrations", () => ({
  useStorageMigrations: () => undefined
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({})
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: () => ""
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: unknown) => unknown) =>
    selector({
      historyId: null,
      serverChatId: null
    })
}))

vi.mock("@/utils/settings-return", () => ({
  setSettingsReturnTo: () => undefined
}))

vi.mock("@/utils/ocr", () => ({
  processImageForOCR: () => Promise.resolve("")
}))

vi.mock("@/context/demo-mode", () => ({
  DemoModeProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  useDemoMode: () => ({ demoMode: false })
}))

describe("OptionLayout shell overrides", () => {
  afterEach(() => {
    delete (
      globalThis as typeof globalThis & {
        __tldwOptionShell?: unknown
      }
    ).__tldwOptionShell
    storeMessageOptionMock.mockClear()
  })

  it("does not clear another shell override if this render never applied one", () => {
    vi.useFakeTimers()
    const externalShell: {
      mounted: boolean
      ownerId: string
      setOverrides?: (overrides: unknown) => void
    } = {
      mounted: true,
      ownerId: "root-shell"
    }
    ;(
      globalThis as typeof globalThis & {
        __tldwOptionShell?: typeof externalShell & {
          setOverrides?: (overrides: unknown) => void
        }
      }
    ).__tldwOptionShell = externalShell

    const { unmount } = render(
      <MemoryRouter>
        <OptionLayout hideHeader>
          <div>Nested content</div>
        </OptionLayout>
      </MemoryRouter>
    )

    const otherOwnerSetOverrides = vi.fn()
    externalShell.setOverrides = otherOwnerSetOverrides

    unmount()

    expect(otherOwnerSetOverrides).not.toHaveBeenCalled()
    vi.useRealTimers()
  })
})

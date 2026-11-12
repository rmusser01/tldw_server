/**
 * XS-07 (#3105): the header's "Rename conversation" edits the chat's full
 * title. The header shows the tab label, which is truncated to 40 characters;
 * renaming from that text would rename the server chat to the truncated text.
 */
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

vi.mock("@/assets/icon.png", () => ({ default: "chrome-extension://test/icon.png" }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback ?? _key })
}))
vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...props }: { children?: React.ReactNode; to: string }) => (
    <a href={to} {...props}>
      {children}
    </a>
  )
}))
vi.mock("antd", () => ({
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))
vi.mock("wxt/browser", () => ({ browser: { runtime: { getURL: (path: string) => path } } }))
vi.mock("@/hooks/useMessage", () => ({ useMessage: () => ({ temporaryChat: false }) }))
vi.mock("@/hooks/useAntdNotification", () => ({ useAntdNotification: () => ({ error: vi.fn() }) }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasPersona: false } })
}))
vi.mock("../StatusDot", () => ({ StatusDot: () => <span data-testid="status-dot" /> }))
vi.mock("../TtsClipsDrawer", () => ({ TtsClipsDrawer: () => null }))

const fullTitle = "Quarterly planning review for the northern region sales team"
const truncatedLabel = `${fullTitle.slice(0, 40)}...`

/** Start editing the header title and wait until the field holds an editable title. */
const startEditing = async () => {
  fireEvent.click(screen.getByRole("button", { name: "Rename conversation" }))
  const field = await screen.findByRole("textbox", { name: "Rename conversation" })
  await waitFor(() => expect(field).toBeEnabled())
  return field as HTMLInputElement
}

describe("SidepanelHeaderSimple title rename", () => {
  it("edits the chat's full title, not the truncated label it shows", async () => {
    const onRenameTitle = vi.fn()
    render(
      <SidepanelHeaderSimple
        activeTitle={truncatedLabel}
        onRenameTitle={onRenameTitle}
        loadEditableTitle={async () => fullTitle}
      />
    )
    const field = await startEditing()
    expect(field).toHaveValue(fullTitle)
    fireEvent.change(field, { target: { value: `${field.value} v2` } })
    fireEvent.blur(field)

    expect(onRenameTitle).toHaveBeenCalledWith(`${fullTitle} v2`)
  })

  it("does nothing when the full title is saved unchanged", async () => {
    const onRenameTitle = vi.fn()
    render(
      <SidepanelHeaderSimple
        activeTitle={truncatedLabel}
        onRenameTitle={onRenameTitle}
        loadEditableTitle={async () => fullTitle}
      />
    )
    const field = await startEditing()
    expect(field).toHaveValue(fullTitle)
    fireEvent.blur(field)

    expect(onRenameTitle).not.toHaveBeenCalled()
  })

  it("falls back to the label when no full title is known", async () => {
    const onRenameTitle = vi.fn()
    render(
      <SidepanelHeaderSimple
        activeTitle="Short chat"
        onRenameTitle={onRenameTitle}
        loadEditableTitle={async () => null}
      />
    )
    const field = await startEditing()
    expect(field).toHaveValue("Short chat")
    fireEvent.change(field, { target: { value: "Short chat v2" } })
    fireEvent.blur(field)

    expect(onRenameTitle).toHaveBeenCalledWith("Short chat v2")
  })
})

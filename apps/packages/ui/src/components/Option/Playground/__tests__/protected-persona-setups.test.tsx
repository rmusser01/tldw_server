import React from "react"
import { ConfigProvider } from "antd"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { HistorySelectionContext, type HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { SavedRolePlaySetupsPanel } from "../SavedRolePlaySetupsPanel"
import { PlaygroundStartupTemplateModal } from "../PlaygroundStartupTemplateModal"
import { createStartupTemplateBundle } from "../startup-template-bundles"

const translate = (key: string, fallback?: string, values?: { name?: string }) =>
  (fallback ?? key).replace("{{name}}", values?.name ?? "")
const cases = [
  { label: "protected persona", protectedHistory: true, kind: "persona" as const, unavailable: true },
  { label: "protected character", protectedHistory: true, kind: "character" as const, unavailable: false },
  { label: "legacy persona", protectedHistory: false, kind: "persona" as const, unavailable: false }
]
const controller = {} as HistorySelectionController // Only context presence is this unit's capability boundary.
const wrap = (content: React.ReactNode, protectedHistory: boolean) => <ConfigProvider theme={{ token: { motion: false } }}>
  <HistorySelectionContext.Provider value={protectedHistory ? controller : null}>{content}</HistorySelectionContext.Provider>
</ConfigProvider>
const bundle = (kind: "persona" | "character") => createStartupTemplateBundle({
  name: "Saved setup", source: "role-play-setup", selectedModel: null, systemPrompt: "",
  rolePlay: { source: "role-play-setup", identity: { kind, id: "fixture-identity", name: "Fixture identity" }, behavior: null, scene: null, generation: null, context: null }
})

describe.each(cases)("$label setup choices", ({ protectedHistory, kind, unavailable }) => {
  it("keeps a saved setup readable and removable while qualifying Apply before callbacks", async () => {
    const setup = bundle(kind)
    const original = structuredClone(setup)
    const apply = vi.fn()
    const preview = vi.fn()
    const user = userEvent.setup()
    render(wrap(<SavedRolePlaySetupsPanel setups={[setup]} draftName="" nameFallback="New setup"
      onDraftNameChange={vi.fn()} onSaveCurrent={vi.fn()} onPreviewSetup={preview} onApplySetup={apply}
      onRenameSetup={vi.fn()} onDeleteSetup={vi.fn()} t={translate} />, protectedHistory))
    const action = screen.getByRole("button", { name: "Apply Saved setup" })
    if (unavailable) {
      expect(action).toBeDisabled()
      expect(action).toHaveAccessibleDescription(/tracked persona.*unavailable.*overlay/i)
    } else expect(action).toBeEnabled()
    await user.click(action)
    if (unavailable) expect(apply).not.toHaveBeenCalled()
    else expect(apply).toHaveBeenCalledWith(setup)
    await user.click(screen.getByRole("button", { name: "Preview Saved setup" }))
    expect(preview).toHaveBeenCalledWith(setup.id)
    expect(screen.getByRole("button", { name: "Delete Saved setup" })).toBeEnabled()
    expect(setup).toEqual(original)
  })

  it("qualifies startup-preview Apply with visible guidance and preserves Delete", async () => {
    const setup = bundle(kind)
    const apply = vi.fn()
    const user = userEvent.setup()
    render(wrap(<PlaygroundStartupTemplateModal preview={setup} onClose={vi.fn()} onDelete={vi.fn()} onApply={apply}
      promptDescription={null} promptResolution={null} preset={undefined} t={translate} />, protectedHistory))
    const action = await screen.findByRole("button", { name: "Apply template" })
    if (unavailable) {
      expect(action).toBeDisabled()
      expect(action).toHaveAccessibleDescription(/tracked persona.*unavailable.*overlay/i)
    } else expect(action).toBeEnabled()
    await user.click(action)
    if (unavailable) expect(apply).not.toHaveBeenCalled()
    else expect(apply).toHaveBeenCalledOnce()
    expect(screen.getByRole("button", { name: "Delete template" })).toBeEnabled()
  })
})

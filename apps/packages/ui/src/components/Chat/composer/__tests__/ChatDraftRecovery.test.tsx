import { useStoreMessageOption } from "@/store/option"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatDraftRecovery } from "../ChatDraftRecovery"
import type { RecoveredChatDraft } from "../hooks/useComposerText"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("antd", () => ({
  Modal: ({
    open,
    title,
    children,
    onCancel
  }: {
    open: boolean
    title: string
    children: React.ReactNode
    onCancel: () => void
  }) =>
    open ? (
      <section role="dialog" aria-label={title}>
        <button onClick={onCancel}>Close</button>
        {children}
      </section>
    ) : null
}))
const draft: RecoveredChatDraft = {
  key: "private-key",
  surfaceKey: "surface",
  content: "MY UNSENT TEXT",
  updatedAt: 1,
  handle: {
    ownerHash: "a".repeat(64),
    address: {
      kind: "fresh",
      id: "retained",
      temporary: true,
      scope: { type: "global" }
    }
  }
}
const mount = () => {
  const listDrafts = vi
    .fn()
    .mockResolvedValue({ drafts: [draft], unavailable: false })
  const copyDraft = vi.fn().mockResolvedValue(true)
  const onRestore = vi.fn().mockResolvedValue(true)
  const view = render(
    <ChatDraftRecovery
      listDrafts={listDrafts}
      copyDraft={copyDraft}
      onRestore={onRestore}
    />
  )
  return { ...view, listDrafts, copyDraft, onRestore }
}
describe("retained composer recovery choices", () => {
  beforeEach(() => {
    useStoreMessageOption.getState().startNewChatDraft()
  })
  it("shows a bounded excerpt and original mode, then re-reads before an explicit recovery", async () => {
    const view = mount()
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    const choice = await screen.findByRole("button", {
      name: /Temporary draft.*MY UNSENT TEXT/
    })
    fireEvent.click(choice)
    await waitFor(() =>
      expect(view.onRestore).toHaveBeenCalledWith(draft, expect.any(Function))
    )
    expect(view.listDrafts).toHaveBeenCalledTimes(2)
  })
  it("retains the current view when a listed record is no longer available", async () => {
    const view = mount()
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    const choice = await screen.findByRole("button", { name: /MY UNSENT TEXT/ })
    view.listDrafts.mockResolvedValue({ drafts: [], unavailable: false })
    fireEvent.click(choice)
    await screen.findByRole("alert")
    expect(view.onRestore).not.toHaveBeenCalled()
  })
  it("clears private previews and ignores a late list after accepted New", async () => {
    const view = mount()
    let finish!: (value: unknown) => void
    view.listDrafts.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    await act(async () => {
      finish({ drafts: [draft], unavailable: false })
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText("MY UNSENT TEXT")).not.toBeInTheDocument()
    expect(view.onRestore).not.toHaveBeenCalled()
  })
  it("distinguishes unread storage from no retained drafts and offers Retry", async () => {
    const view = mount()
    view.listDrafts.mockResolvedValue({ drafts: [], unavailable: true })
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    await screen.findByText(/Some saved drafts could not be read/)
    expect(
      screen.queryByText("No other retained drafts on this browser.")
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(view.listDrafts).toHaveBeenCalledTimes(2))
  })
  it("keeps the recovery choice open if navigation or copy is declined", async () => {
    const view = mount()
    view.onRestore.mockResolvedValue(false)
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    fireEvent.click(
      await screen.findByRole("button", { name: /MY UNSENT TEXT/ })
    )
    await screen.findByRole("alert")
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })
  it.each(["Close", "unmount", "ownership loss"])(
    "cancels a held copy after %s without accepting New",
    async (boundary) => {
      const view = mount()
      let finish!: (value: boolean) => void
      view.copyDraft.mockImplementation(
        () =>
          new Promise<boolean>((resolve) => {
            finish = resolve
          })
      )
      view.onRestore.mockImplementation(async (_draft, copy) => {
        if (!(await copy("surface"))) return false
        return useStoreMessageOption.getState().startNewChatDraft()
      })
      fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
      fireEvent.click(
        await screen.findByRole("button", { name: /MY UNSENT TEXT/ })
      )
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      const source = useStoreMessageOption.getState().chatDraftAddress
      const revision = useStoreMessageOption.getState().chatDraftRevision
      if (boundary === "Close")
        fireEvent.click(screen.getByRole("button", { name: "Close" }))
      else if (boundary === "unmount") view.unmount()
      else
        view.rerender(
          <ChatDraftRecovery
            listDrafts={view.listDrafts}
            copyDraft={view.copyDraft}
            onRestore={view.onRestore}
            disabled
          />
        )
      await act(async () => {
        finish(true)
      })
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
      expect(useStoreMessageOption.getState().chatDraftRevision).toBe(revision)
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expect(screen.queryByText("MY UNSENT TEXT")).not.toBeInTheDocument()
    }
  )
  it("clears an open prior-owner preview as soon as ownership readiness is lost", async () => {
    const view = mount()
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    await screen.findByText("MY UNSENT TEXT")
    const revision = useStoreMessageOption.getState().chatDraftRevision
    view.rerender(
      <ChatDraftRecovery
        listDrafts={view.listDrafts}
        copyDraft={view.copyDraft}
        onRestore={view.onRestore}
        disabled
      />
    )
    expect(useStoreMessageOption.getState().chatDraftRevision).toBe(revision)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText("MY UNSENT TEXT")).not.toBeInTheDocument()
  })
})

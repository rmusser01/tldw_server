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


it("shows active unknown creation without requiring a nonempty draft and offers Find conversations", async () => {
  const onFind = vi.fn()
  render(<ChatDraftRecovery listDrafts={vi.fn().mockResolvedValue({ drafts: [], unavailable: false })} copyDraft={vi.fn()}
    onRestore={vi.fn()} onFindConversations={onFind} creationRecovery={{ unavailable: false, active: {
      version: 1, operationId: "c".repeat(36), handle: { ...draft.handle, address: { ...draft.handle.address, temporary: false } },
      surface: "playground", createdAt: 1, inputText: "", state: "unconfirmed"
    } }} />)
  expect(screen.getByText("Server save unconfirmed")).toBeInTheDocument()
  expect(screen.getByText(/another chat may create a duplicate/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Find conversations" }))
  expect(onFind).toHaveBeenCalledOnce()
  expect(screen.queryByRole("button", { name: "Review created chat" })).not.toBeInTheDocument()
})

it("keeps acknowledged creation distinct from uncertainty after history review fails", async () => {
  const review = vi.fn().mockResolvedValue(false)
  render(<ChatDraftRecovery listDrafts={vi.fn()} copyDraft={vi.fn()} onRestore={vi.fn()}
    onReviewCreatedChat={review} creationRecovery={{ unavailable: false, active: {
      version: 1, operationId: "c".repeat(36), handle: { ...draft.handle, address: { ...draft.handle.address, temporary: false } },
      surface: "playground", createdAt: 1, inputText: "original unsent text", state: "acknowledged", chatId: "saved-id"
    } }} />)
  expect(review).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Review created chat" }))
  await waitFor(() => expect(review).toHaveBeenCalledOnce())
  expect(await screen.findByText(/Could not verify the created chat/)).toBeInTheDocument()
  expect(screen.getByText("Chat already created")).toBeInTheDocument()
  expect(screen.queryByText("Server save unconfirmed")).not.toBeInTheDocument()
})


it("uses the owned Copy action and ignores its late failure after New", async () => {
  let reject!: (reason: Error) => void
  const copy = vi.fn(() => new Promise<boolean>((_resolve, fail) => { reject = fail }))
  render(<ChatDraftRecovery listDrafts={vi.fn()} copyDraft={vi.fn()} onRestore={vi.fn()}
    onCopyCreationText={copy} creationRecovery={{ unavailable: false, active: {
      version: 1, operationId: "c".repeat(36), handle: { ...draft.handle, address: { ...draft.handle.address, temporary: false } },
      surface: "playground", createdAt: 1, inputText: "owned original attempt", state: "unconfirmed"
    } }} />)
  expect(copy).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Copy attempted text" }))
  expect(copy).toHaveBeenCalledOnce()
  await act(async () => { useStoreMessageOption.getState().startNewChatDraft(); reject(new Error("Clipboard unavailable")) })
  expect(screen.queryByRole("alert")).not.toBeInTheDocument()
})


it("discloses Copy before reviewing a bound chat with different retained input", async () => {
  const review = vi.fn().mockResolvedValue(true)
  render(<ChatDraftRecovery listDrafts={vi.fn()} copyDraft={vi.fn()} onReviewCreatedChat={review}
    creationRecovery={{ unavailable: false, copyInputRequired: true, active: {
      version: 1, operationId: "bound", surface: "playground", createdAt: 1, state: "bound", chatId: "saved",
      inputText: "ORIGINAL INPUT", handle: { ...draft.handle, address: { ...draft.handle.address, kind: "fresh", temporary: false } }
    } }} />)
  fireEvent.click(screen.getByRole("button", { name: "Copy draft and review saved chat" }))
  await waitFor(() => expect(review).toHaveBeenCalledOnce())
})


it("shows acknowledged creation in retained discovery before history qualification", async () => {
  const handle = { ...draft.handle, address: { ...draft.handle.address, temporary: false } }
  const retained = { ...draft, handle, creationAttempt: {
    version: 1 as const, operationId: "created", handle, surface: "playground" as const,
    createdAt: 1, inputText: draft.content, state: "acknowledged" as const, chatId: "saved-id"
  } }
  render(<ChatDraftRecovery listDrafts={vi.fn().mockResolvedValue({ drafts: [retained], unavailable: false })}
    copyDraft={vi.fn()} onRestore={vi.fn()} />)
  fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
  expect(await screen.findByText(/Chat already created/)).toBeInTheDocument()
  expect(screen.queryByText("Server save unconfirmed")).not.toBeInTheDocument()
})

describe("draft account verification recovery", () => {
  const pending = "Checking your account before restoring or saving drafts…"
  const failed =
    "Your account could not be verified. Draft saving, recovery, and imports are paused. Retry or check your connection settings."

  it("describes a pending owner without reading recovery records or offering another Retry", () => {
    const listDrafts = vi.fn()
    const retry = vi.fn()
    render(
      <ChatDraftRecovery
        listDrafts={listDrafts}
        copyDraft={vi.fn()}
        onRestore={vi.fn()}
        disabled
        draftOwnerVerificationState="pending"
        retryDraftOwnerVerification={retry}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(pending)
    expect(
      screen.getByRole("button", { name: "Recover drafts" })
    ).toBeDisabled()
    expect(
      screen.queryByRole("button", { name: "Retry account verification" })
    ).not.toBeInTheDocument()
    expect(listDrafts).not.toHaveBeenCalled()
    expect(retry).not.toHaveBeenCalled()
  })

  it("offers account Retry while recovery is disabled, then removes the notice when ready", async () => {
    const listDrafts = vi
      .fn()
      .mockResolvedValue({ drafts: [], unavailable: false })
    const retry = vi.fn()
    const props = {
      listDrafts,
      copyDraft: vi.fn(),
      onRestore: vi.fn(),
      retryDraftOwnerVerification: retry
    }
    const view = render(
      <ChatDraftRecovery
        {...props}
        disabled
        draftOwnerVerificationState="error"
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(failed)
    const action = screen.getByRole("button", {
      name: "Retry account verification"
    })
    expect(action).toBeEnabled()
    expect(action).toHaveAttribute("type", "button")
    fireEvent.click(action)
    expect(retry).toHaveBeenCalledOnce()
    expect(listDrafts).not.toHaveBeenCalled()
    expect(
      screen.getByRole("button", { name: "Recover drafts" })
    ).toBeDisabled()
    view.rerender(
      <ChatDraftRecovery
        {...props}
        disabled
        draftOwnerVerificationState="pending"
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(pending)
    expect(
      screen.queryByRole("button", { name: "Retry account verification" })
    ).not.toBeInTheDocument()
    view.rerender(
      <ChatDraftRecovery {...props} draftOwnerVerificationState="ready" />
    )
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(screen.queryByText(failed)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    await screen.findByText("No other retained drafts on this browser.")
    expect(listDrafts).toHaveBeenCalledOnce()
  })

  it("keeps a verified owner distinct from draft hydration or a transient same-owner recheck", () => {
    const retry = vi.fn()
    render(
      <ChatDraftRecovery
        listDrafts={vi.fn()}
        copyDraft={vi.fn()}
        onRestore={vi.fn()}
        disabled
        draftOwnerVerificationState="ready"
        retryDraftOwnerVerification={retry}
      />
    )
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Retry account verification" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Recover drafts" })
    ).toBeDisabled()
    expect(retry).not.toHaveBeenCalled()
  })
})


it.each([false, true])(
  "client UUID unknown recovery discloses the exact Check action with copy required %s",
  async (copyInputRequired) => {
    const review = vi.fn().mockResolvedValue(false)
    const id = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
    render(
      <ChatDraftRecovery
        listDrafts={vi.fn()}
        copyDraft={vi.fn()}
        onReviewCreatedChat={review}
        creationRecovery={{
          unavailable: false,
          copyInputRequired,
          active: {
            version: 1,
            operationId: id,
            clientChatId: id,
            surface: "playground",
            createdAt: 1,
            state: "unconfirmed",
            inputText: "ORIGINAL INPUT",
            handle: {
              ...draft.handle,
              address: { ...draft.handle.address, temporary: false }
            }
          }
        }}
      />
    )
    expect(screen.getByText("Server save unconfirmed")).toBeInTheDocument()
    expect(review).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", {
        name: copyInputRequired
          ? "Copy draft and check server"
          : "Check server for this chat"
      })
    )
    await waitFor(() => expect(review).toHaveBeenCalledOnce())
    expect(screen.getByText("ORIGINAL INPUT")).toBeInTheDocument()
    expect(screen.queryByText("Chat already created")).not.toBeInTheDocument()
  }
)

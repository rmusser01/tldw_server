import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
const reviewButtons = vi.hoisted(() => ({ open: [] as Array<() => void> }))
vi.mock("@/components/Common/Button", async (original) => {
  const { Button } =
    await original<typeof import("@/components/Common/Button")>()
  const React = await import("react")
  return {
    Button: React.forwardRef<HTMLButtonElement, any>((props, ref) => {
      if (props.children === "Open saved reply")
        reviewButtons.open.push(props.onClick)
      return React.createElement(Button, { ...props, ref })
    })
  }
})
vi.mock("react-i18next", async () => {
  const { default: locale } = await import("@/assets/locale/en/playground.json")
  return {
    useTranslation: () => ({
      t: (key: string, value: any) => {
        const translated = key
          .split(".")
          .reduce((node: any, part) => node?.[part], locale)
        return (
          translated ??
          (typeof value === "string" ? value : value?.defaultValue || key)
        )
      }
    })
  }
})
import { HistorySelectionReview } from "../HistorySelectionReview"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
const nodes = Array.from({ length: 20300 }, (_, i) => ({
  id: `m${i}`,
  revision: `r${i}`,
  role: i % 2 ? "assistant" : "user",
  preview: `Message ${i}`,
  parent_id: null,
  settled: true
}))
const controller = () =>
  ({
    status: "legacy_review_required",
    error: null,
    capture: { snapshot: { nodes, source_digest: "all" } },
    view: { cursor: { kind: "empty" } },
    confirm: vi.fn(),
    refresh: vi.fn(),
    choose: vi.fn(),
    pending: null
  }) as unknown as HistorySelectionController

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(480)
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(320)
})

describe("history review", () => {
  it("keeps omitted alternatives in the complete source and confirms the complete proposed order", () => {
    const selection = controller()
    render(<HistorySelectionReview selection={selection} />)
    fireEvent.click(
      screen.getByRole("button", { name: "Review conversation history" })
    )
    const checkbox = screen.getByRole("checkbox", {
      name: /Include.*Message 0/
    })
    fireEvent.click(checkbox)
    expect(screen.getByText("Omitted alternative")).toBeTruthy()
    fireEvent.click(
      screen.getByRole("button", { name: "Before first included message" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm selected history" })
    )
    const [ids, cursor] = (selection.confirm as any).mock.calls[0]
    expect(ids).toHaveLength(20299)
    expect(ids[0]).toBe("m1")
    expect(ids.at(-1)).toBe("m20299")
    expect(cursor).toEqual({ kind: "before_message", message_id: "m1" })
    expect(screen.getAllByRole("checkbox").length).toBeLessThan(100)
  })
  it("preserves draft choices after source drift, labels the error, and returns focus on cancel", () => {
    const selection = controller()
    const { rerender } = render(
      <HistorySelectionReview selection={selection} />
    )
    const trigger = screen.getByRole("button", {
      name: "Review conversation history"
    })
    fireEvent.click(trigger)
    fireEvent.click(
      screen.getByRole("checkbox", { name: /Include.*Message 0/ })
    )
    rerender(
      <HistorySelectionReview
        selection={{
          ...selection,
          status: "stale_selection",
          error: "source_changed"
        }}
      />
    )
    expect(screen.getByRole("alert").textContent).toContain(
      "Conversation history changed"
    )
    expect(
      screen.getByRole("checkbox", { name: /Include.*Message 0/ })
    ).not.toBeChecked()
    fireEvent.click(screen.getByRole("button", { name: "Cancel review" }))
    expect(trigger).toHaveFocus()
  })

  it("does not confirm message IDs from a previously opened conversation", () => {
    const first = {
      ...controller(),
      owner: { kind: "local", conversation_id: "first", owner_key: "owner" },
      capture: { snapshot: { nodes: [nodes[0], nodes[1]], source_digest: "first" } }
    } as unknown as HistorySelectionController
    const second = {
      ...controller(),
      owner: { kind: "local", conversation_id: "second", owner_key: "owner" },
      capture: {
        snapshot: {
          nodes: [{ ...nodes[0], id: "other-message" }],
          source_digest: "second"
        }
      }
    } as unknown as HistorySelectionController
    const { rerender } = render(<HistorySelectionReview selection={first} />)
    fireEvent.click(
      screen.getByRole("button", { name: "Review conversation history" })
    )
    rerender(<HistorySelectionReview selection={second} />)
    expect(
      screen.queryByRole("button", { name: "Confirm selected history" })
    ).toBeNull()
    fireEvent.click(
      screen.getByRole("button", { name: "Review conversation history" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm selected history" })
    )
    expect(second.confirm).toHaveBeenCalledWith(
      ["other-message"],
      { kind: "after_message", message_id: "other-message" }
    )
  })
})

it("reorders included history by keyboard without changing virtual row identity", () => {
  const selection = controller()
  render(<HistorySelectionReview selection={selection} />)
  fireEvent.click(
    screen.getByRole("button", { name: "Review conversation history" })
  )
  const first = screen.getAllByRole("listitem")[0]
  expect(first).toHaveAttribute("data-history-message-id", "m0")
  expect(first).toHaveAttribute("aria-setsize", "20300")
  fireEvent.keyDown(first, { key: "ArrowDown", altKey: true })
  expect(screen.getAllByRole("listitem")[0]).toBe(first)
  fireEvent.click(
    screen.getByRole("button", { name: "Confirm selected history" })
  )
  expect((selection.confirm as any).mock.calls[0][0].slice(0, 3)).toEqual([
    "m1",
    "m0",
    "m2"
  ])
})

it("shows recoverable text separately with copy and dismiss but no replay action", () => {
  const selection = {
    ...controller(),
    status: "ready",
    recoveries: [
      {
        scope: { profile_id: "p", client_session_id: "old" },
        turn: {
          operation_id: "op",
          state: "generated_unsaved",
          input_text: "accepted question",
          result_text: "recover this answer"
        }
      }
    ],
    dismissRecovery: vi.fn()
  } as any
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("recover this answer")).toBeTruthy()
  expect(
    screen.getByRole("button", { name: "Copy recovered text" })
  ).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery" }))
  expect(selection.dismissRecovery).toHaveBeenCalledWith(
    selection.recoveries[0]
  )
  expect(
    screen.queryByRole("button", { name: /resend|retry|settle/i })
  ).toBeNull()
})

it("shows confirmed native refusal with exact input, Copy and Dismiss without uncertain outcome or replay", async () => {
  const copy = vi.fn(async () => undefined)
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: copy }
  })
  const selection = {
    ...controller(),
    status: "ready",
    recoveries: [
      {
        scope: { profile_id: "p", client_session_id: "old" },
        turn: {
          operation_id: "refused",
          persistence: "server",
          state: "refused",
          input_text: "  original refused question  ",
          result_text: ""
        }
      }
    ],
    dismissRecovery: vi.fn()
  } as unknown as HistorySelectionController
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("Message not added")).toBeVisible()
  expect(
    screen.getByText(/Your original input is kept here/)
  ).toBeVisible()
  expect(screen.queryByText(/unknown|Turn needs review/i)).toBeNull()
  fireEvent.click(screen.getByText("Inspect original input and result"))
  expect(screen.getByText("original refused question")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" }))
  expect(copy).toHaveBeenCalledWith("  original refused question  ")
  fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery" }))
  expect(selection.dismissRecovery).toHaveBeenCalledWith(
    selection.recoveries[0]
  )
  expect(
    screen.queryByRole("button", { name: /resend|retry|settle/i })
  ).toBeNull()
})

const recoveryController = (
  persistence: "server" | "client" | undefined,
  state: "accepted_unsent" | "generated_unsaved",
  resultText = ""
) => {
  const origin = {
    owner_key: "original-owner",
    conversation_id: "original-chat",
    view_session_id: "old-view",
    selection_revision: 1,
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "empty" }
  }
  return {
    ...controller(),
    status: "ready",
    recoveries: [
      {
        scope: {
          profile_id: "original-profile",
          client_session_id: "original-client"
        },
        turn: {
          operation_id: "native-interrupted",
          persistence,
          state,
          origin_view: origin,
          owner_key: origin.owner_key,
          conversation_id: origin.conversation_id,
          selection_digest: "selected",
          request_context_digest: "prepared",
          input_id: "accepted-input",
          ...(persistence !== "server"
            ? { assistant_id: "client-result" }
            : {}),
          created_at: 1,
          input_text: "original accepted question",
          input_images: [],
          result_text: resultText,
          admission: {
            version: 1,
            owner_key: origin.owner_key,
            conversation_id: origin.conversation_id,
            input_message_id: "accepted-input",
            input_message_revision: "r1",
            selection_digest: "selected"
          }
        }
      }
    ],
    dismissRecovery: vi.fn()
  } as unknown as HistorySelectionController
}

it.each(["accepted_unsent", "generated_unsaved"] as const)(
  "native %s recovery keeps known admission and uncertain response visible",
  (state) => {
    const selection = recoveryController(
      "server",
      state,
      state === "generated_unsaved" ? "unacknowledged generated text" : ""
    )
    render(<HistorySelectionReview selection={selection} />)
    expect(
      screen.getByText("User input accepted; response outcome unknown")
    ).toBeVisible()
    expect(
      screen.queryByText("User input accepted; response not saved")
    ).toBeNull()
    expect(screen.queryByText("User admission outcome unknown")).toBeNull()
    fireEvent.click(screen.getByText("Inspect original input and result"))
    expect(screen.getByText("original accepted question")).toBeVisible()
    if (state === "generated_unsaved")
      expect(screen.getByText("unacknowledged generated text")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: /resend|retry|settle/i })
    ).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery" }))
    expect(selection.dismissRecovery).toHaveBeenCalledWith(
      selection.recoveries[0]
    )
    expect(selection.recoveries[0].scope.profile_id).toBe("original-profile")
  }
)

it.each([undefined, "client"] as const)(
  "client-managed recovery retains accepted-but-unsent wording (%s)",
  (persistence) => {
    render(
      <HistorySelectionReview
        selection={recoveryController(persistence, "accepted_unsent")}
      />
    )
    expect(
      screen.getByText("User input accepted; response not saved")
    ).toBeVisible()
    expect(
      screen.queryByText("User input accepted; response outcome unknown")
    ).toBeNull()
  }
)

it("client-managed generated text retains its existing review label", () => {
  render(
    <HistorySelectionReview
      selection={recoveryController(
        "client",
        "generated_unsaved",
        "client generated text"
      )}
    />
  )
  expect(screen.getByText("Generated response needs review")).toBeVisible()
  expect(
    screen.queryByText("User input accepted; response outcome unknown")
  ).toBeNull()
})
it("retains unresolved fork separately from sends with candidate inspection and deliberate-new-action control", () => {
  const selection = {...controller(), status: "ready", forkOperations: [{operation_id: "op", owner_key: "original", state: "partial", candidate_child_id: "child", active_intent: "intent", result: {state: "partial", code: "response_lost"}}], inspectForkOperation: vi.fn(), allowNewFork: vi.fn()} as any
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("Fork incomplete")).toBeTruthy()
  expect(screen.getByText(/Some content may be missing/)).toBeTruthy()
  fireEvent.click(screen.getByRole("button", {name: /Inspect copy/}))
  expect(selection.inspectForkOperation).toHaveBeenCalledWith(selection.forkOperations[0])
  expect(selection.allowNewFork).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", {name: "Allow a new fork action"}))
  expect(selection.allowNewFork).toHaveBeenCalledWith(selection.forkOperations[0])
  expect(screen.queryByRole("button", {name: /Retry fork/})).toBeNull()
})

it.each(['dispatching', 'unknown', 'partial'] as const)('scopes %s copy guarantees to application attempts and discloses possible server duplicates', state => {
  const selection = { ...controller(), status: 'ready', forkOperations: [{ operation_id: 'op', owner_key: 'original', state }], inspectForkOperation: vi.fn(), allowNewFork: vi.fn() } as unknown as HistorySelectionController
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText(/The app will not start another attempt automatically/)).toBeVisible()
  expect(screen.getByText(/connection failure may have left multiple server copies/)).toBeVisible()
  expect(screen.queryByText(/No copy will be retried automatically/)).toBeNull()
})


describe("available history-path actions (TASK-13550)", () => {
  it("hides both history choices for a ready empty source", () => {
    const selection = {
      ...controller(),
      status: "ready",
      capture: { snapshot: { nodes: [], source_digest: "empty" } }
    } as unknown as HistorySelectionController
    render(<HistorySelectionReview selection={selection} />)
    expect(
      screen.queryByRole("button", { name: "Use no prior messages" })
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Review conversation history" })
    ).toBeNull()
  })

  it("retains the empty-path action for a ready nonempty source", () => {
    const selection = {
      ...controller(),
      status: "ready",
      capture: { snapshot: { nodes: [nodes[0]], source_digest: "one" } }
    } as unknown as HistorySelectionController
    render(<HistorySelectionReview selection={selection} />)
    expect(
      screen.getByRole("button", { name: "Review conversation history" })
    ).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Use no prior messages" })
    )
    expect(selection.choose).toHaveBeenCalledWith({ kind: "empty" })
  })

  it("retains expansion and recovery when there is no history to omit", () => {
    const selection = {
      ...controller(),
      status: "ready",
      capture: { snapshot: { nodes: [], source_digest: "empty" } },
      recoveries: [
        {
          scope: { profile_id: "profile", client_session_id: "session" },
          turn: {
            operation_id: "retained",
            state: "generated_unsaved",
            input_text: "question",
            result_text: "Retained answer"
          }
        }
      ],
      dismissRecovery: vi.fn()
    } as unknown as HistorySelectionController
    const onExpand = vi.fn()
    render(<HistorySelectionReview selection={selection} onExpand={onExpand} />)
    expect(screen.getByText("Retained answer")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Expand in full page" }))
    expect(onExpand).toHaveBeenCalledOnce()
    expect(
      screen.getByRole("button", { name: "Dismiss recovery" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Use no prior messages" })
    ).toBeNull()
  })

  it.each([
    { status: "loading", title: "Loading selected history" },
    { status: "stale_selection", title: "Conversation history changed" },
    {
      status: "unsupported_history_capability",
      title: "Selected history unavailable"
    }
  ])("retains $status feedback for an empty source", ({ status, title }) => {
    const selection = {
      ...controller(),
      status,
      error: "source_changed",
      capture: { snapshot: { nodes: [], source_digest: "empty" } }
    } as unknown as HistorySelectionController
    render(<HistorySelectionReview selection={selection} />)
    expect(screen.getByText(title)).toBeInTheDocument()
  })
})


const completedEditedSelection = () =>
  ({
    ...controller(),
    status: "ready",
    owner: { kind: "native" },
    recoveries: [
      {
        scope: { profile_id: "profile", client_session_id: "origin" },
        turn: {
          operation_id: "edited",
          action: "edit_resend",
          outcome: "complete",
          settled_message_id: "saved-answer",
          persistence: "server",
          state: "generated_unsaved",
          admission: {},
          input_text: "Edited question",
          result_text: "Saved edited reply"
        }
      }
    ],
    openRecoveryResult: vi.fn(async () => true),
    dismissRecovery: vi.fn()
  }) as unknown as HistorySelectionController

it("a saved edited reply has truthful copy and opens only on an explicit action", async () => {
  const selection = completedEditedSelection()
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("Edited reply saved")).toBeTruthy()
  expect(screen.queryByText(/response outcome unknown/i)).toBeNull()
  expect(selection.openRecoveryResult).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
  await waitFor(() =>
    expect(selection.openRecoveryResult).toHaveBeenCalledOnce()
  )
  expect(selection.openRecoveryResult).toHaveBeenCalledWith(
    selection.recoveries[0],
    expect.any(Function)
  )
})

it("explicit recovery leaves an open message editor intact until it is finished", async () => {
  const selection = completedEditedSelection()
  const editor = document.createElement("form")
  editor.setAttribute("data-chat-message-editor", "")
  const text = document.createElement("textarea")
  text.value = "Newer unsaved text"
  editor.append(text)
  document.body.append(editor)
  try {
    render(<HistorySelectionReview selection={selection} />)
    fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
    expect(selection.openRecoveryResult).not.toHaveBeenCalled()
    expect(screen.getByRole("alert").textContent).toContain(
      "Finish or cancel your current message edit"
    )
    expect(text.value).toBe("Newer unsaved text")
    editor.remove()
    fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
    await waitFor(() =>
      expect(selection.openRecoveryResult).toHaveBeenCalledOnce()
    )
  } finally {
    editor.remove()
  }
})

it("a held recovery action cannot be duplicated and keeps a failed reply available", async () => {
  const selection = completedEditedSelection()
  let release!: (opened: boolean) => void
  vi.mocked(selection.openRecoveryResult).mockImplementation(
    () =>
      new Promise((done) => {
        release = done
      })
  )
  render(<HistorySelectionReview selection={selection} />)
  const button = screen.getByRole("button", { name: "Open saved reply" })
  fireEvent.click(button)
  fireEvent.click(button)
  expect(selection.openRecoveryResult).toHaveBeenCalledOnce()
  await act(async () => {
    release(false)
  })
  expect(screen.getByRole("alert").textContent).toContain(
    "Could not open the saved reply"
  )
  expect(selection.dismissRecovery).not.toHaveBeenCalled()
  expect(
    screen.getByRole("button", { name: "Open saved reply" })
  ).not.toBeDisabled()
})

it("a retained Open callback cannot adopt a same-ID draft restore before React commits", async () => {
  const { usePlaygroundSessionStore } = await import(
    "@/store/playground-session"
  )
  const selection = completedEditedSelection()
  render(<HistorySelectionReview selection={selection} />)
  const retained = reviewButtons.open.at(-1)!
  usePlaygroundSessionStore.setState({
    restoreRevision: usePlaygroundSessionStore.getState().restoreRevision + 1
  })
  retained()
  expect(selection.openRecoveryResult).not.toHaveBeenCalled()
})

it("replacement recovery opens without waiting for an old held read and late cleanup cannot unlock it", async () => {
  const { useStoreMessageOption } = await import("@/store/option")
  const first = completedEditedSelection(),
    second = completedEditedSelection()
  second.owner = {
    kind: "native",
    owner_key: "next-owner",
    conversation_id: "next-chat"
  } as never
  let releaseFirst!: (value: boolean) => void,
    releaseSecond!: (value: boolean) => void
  vi.mocked(first.openRecoveryResult).mockImplementation(
    () =>
      new Promise((done) => {
        releaseFirst = done
      })
  )
  vi.mocked(second.openRecoveryResult).mockImplementation(
    () =>
      new Promise((done) => {
        releaseSecond = done
      })
  )
  const mounted = render(<HistorySelectionReview selection={first} />)
  fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
  useStoreMessageOption.getState().startNewChatDraft()
  mounted.rerender(<HistorySelectionReview selection={second} />)
  const next = screen.getByRole("button", { name: "Open saved reply" })
  expect(next).not.toBeDisabled()
  fireEvent.click(next)
  expect(second.openRecoveryResult).toHaveBeenCalledOnce()
  await act(async () => {
    releaseFirst(false)
  })
  expect(next).toBeDisabled()
  await act(async () => {
    releaseSecond(false)
  })
  expect(next).not.toBeDisabled()
})


it.each(["complete", "refused", "unknown"])("native regeneration %s has accurate review copy and requires an explicit result opening", async state => {
  const selection = completedEditedSelection()
  selection.recoveries[0].turn.action = "regenerate"
  if (state !== "complete") Object.assign(selection.recoveries[0].turn, { outcome: undefined, settled_message_id: undefined, state })
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText(state === "complete" ? "New reply saved" : state === "refused" ? "New reply refused" : "New reply unconfirmed")).toBeTruthy()
  expect(screen.queryByText("Edited reply saved")).toBeNull()
  expect(screen.queryByText("Message not added")).toBeNull()
  expect(selection.openRecoveryResult).not.toHaveBeenCalled()
  if (state === "complete") {
    fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
    await waitFor(() => expect(selection.openRecoveryResult).toHaveBeenCalledOnce())
  } else expect(screen.queryByRole("button", { name: "Open saved reply" })).toBeNull()
})


describe("shared native send-conflict choices", () => {
  it("offers loading latest or a deliberate branch without submitting or changing the selected boundary", () => {
    const selection = {
      ...controller(),
      status: "stale_selection",
      error: "history_branch_changed",
      sendConflict: true,
      loadLatest: vi.fn(),
      branchHere: vi.fn(),
      fence: () => () => true
    }
    render(
      <HistorySelectionReview
        selection={selection as unknown as HistorySelectionController}
      />
    )
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Updated in another window or tab"
    )
    fireEvent.click(screen.getByRole("button", { name: "Load latest messages" }))
    fireEvent.click(screen.getByRole("button", { name: "Branch here" }))
    expect(selection.loadLatest).toHaveBeenCalledWith(
      selection.view,
      expect.any(Function),
      expect.any(Function)
    )
    expect(selection.branchHere).toHaveBeenCalledWith(
      selection.view,
      expect.any(Function),
      expect.any(Function)
    )
    expect(selection.choose).not.toHaveBeenCalled()
    expect(selection.refresh).not.toHaveBeenCalled()
  })

  it("refused input never falsely diagnoses unsupported context", () => {
    const selection = {
      ...controller(),
      status: "ready",
      recoveries: [
        {
          scope: { profile_id: "p", client_session_id: "c" },
          turn: {
            operation_id: "refused",
            persistence: "client",
            state: "refused",
            input_text: "exact unsent input",
            result_text: ""
          }
        }
      ]
    }
    render(
      <HistorySelectionReview
        selection={selection as unknown as HistorySelectionController}
      />
    )
    expect(
      screen.getByText(/Your original input is kept here/)
    ).not.toHaveTextContent("unsupported context")
  })
})

it("an armed branch can be cancelled without sending and retained native images are downloadable", () => {
  const selection = controller()
  Object.assign(selection, {
    status: "ready",
    branchPending: true,
    choose: vi.fn(),
    recoveries: [
      {
        scope: {},
        turn: {
          operation_id: "image-refusal",
          state: "refused",
          input_text: "Exact image input",
          input_images: [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6VbMAAAAASUVORK5CYII=",
            "https://untrusted.test/private.png"
          ],
          result_text: ""
        }
      }
    ]
  })
  render(<HistorySelectionReview selection={selection} />)
  fireEvent.click(screen.getByRole("button", { name: "Cancel branch" }))
  expect(selection.choose).toHaveBeenCalledWith(selection.view!.cursor)
  expect(
    screen.getByRole("link", { name: "Download retained image" })
  ).toHaveAttribute("download", "recovered-image-1")
  expect(screen.getAllByRole("link")).toHaveLength(1)
})

it("retired recovery Copy does not write a departed account's input to the clipboard", () => {
  const selection = controller()
  let current = true
  const copy = vi.fn()
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: copy }
  })
  Object.assign(selection, {
    status: "ready",
    fence: () => () => current,
    recoveries: [
      {
        scope: {},
        turn: {
          operation_id: "source-refusal",
          state: "refused",
          input_text: "Private original",
          result_text: ""
        }
      }
    ]
  })
  render(<HistorySelectionReview selection={selection} />)
  current = false
  fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" }))
  expect(copy).not.toHaveBeenCalled()
})


it("a proactive update keeps composer focus and makes no unsaved-message claim", () => {
  const selection = {
    ...controller(),
    status: "stale_selection",
    error: "history_branch_changed",
    sendConflict: true,
    freshnessConflict: true,
    loadLatest: vi.fn(),
    branchHere: vi.fn(),
    fence: () => () => true
  }
  const input = document.createElement("textarea")
  input.value = "Exact draft.  \nSecond line."
  document.body.append(input)
  input.focus()
  try {
    render(
      <HistorySelectionReview
        selection={selection as unknown as HistorySelectionController}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(
      "Updated in another window or tab"
    )
    expect(screen.queryByText(/Your message was not added/)).toBeNull()
    expect(document.activeElement).toBe(input)
    expect(input.value).toBe("Exact draft.  \nSecond line.")
  } finally {
    input.remove()
  }
})

it.each(["Load latest messages", "Branch here"])(
  "clears finished-editor feedback when %s is admitted",
  (action) => {
    const selection = {
      ...controller(),
      status: "stale_selection",
      error: "history_branch_changed",
      sendConflict: true,
      freshnessConflict: true,
      loadLatest: vi.fn(),
      branchHere: vi.fn(),
      fence: () => () => true
    }
    render(
      <HistorySelectionReview
        selection={selection as unknown as HistorySelectionController}
      />
    )
    const editor = document.createElement("form")
    editor.dataset.chatMessageEditor = ""
    document.body.append(editor)
    try {
      fireEvent.click(screen.getByRole("button", { name: action }))
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Finish or cancel your current message edit"
      )
      expect(selection.loadLatest).not.toHaveBeenCalled()
      expect(selection.branchHere).not.toHaveBeenCalled()
      editor.remove()
      fireEvent.click(screen.getByRole("button", { name: action }))
      expect(
        action === "Branch here" ? selection.branchHere : selection.loadLatest
      ).toHaveBeenCalledOnce()
      expect(
        screen.queryByText(/Finish or cancel your current message edit/)
      ).toBeNull()
    } finally {
      editor.remove()
    }
  }
)

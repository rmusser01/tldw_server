import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { parseRoleplayResumeRecovery } from "@/services/tldw/roleplay-resume"
const roleplayRetry = vi.hoisted(() => ({ send: vi.fn() }))
vi.mock("@/hooks/chat/roleplay-resume-send", () => ({
  sendSavedRoleplay: roleplayRetry.send
}))
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
    dismissRecovery: vi.fn(async () => true)
  } as any
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("recover this answer")).toBeTruthy()
  expect(
    screen.getByRole("button", { name: "Copy recovered text" })
  ).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery" }))
  expect(selection.dismissRecovery).toHaveBeenCalledWith(
    selection.recoveries[0], expect.any(Function)
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
    dismissRecovery: vi.fn(async () => true)
  } as unknown as HistorySelectionController
  render(<HistorySelectionReview selection={selection} />)
  fireEvent.click(screen.getByText(/Retained outcomes \(1\)/))
  expect(screen.getByText("Message not added")).toBeVisible()
  expect(
    screen.getByText(/Your original input is kept here/)
  ).toBeVisible()
  expect(screen.queryByText(/unknown|Turn needs review/i)).toBeNull()
  fireEvent.click(screen.getByText("Inspect original input and result"))
  expect(screen.getByText("original refused question")).toBeVisible()
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" })) })
  expect(copy).toHaveBeenCalledWith("  original refused question  ")
  fireEvent.click(screen.getByRole("button", { name: "Dismiss recovery" }))
  expect(selection.dismissRecovery).toHaveBeenCalledWith(
    selection.recoveries[0], expect.any(Function)
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
    dismissRecovery: vi.fn(async () => true)
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
    fireEvent.click(screen.getByText(/Retained outcomes \(1\)/))
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
      selection.recoveries[0], expect.any(Function)
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
    fireEvent.click(screen.getByText(/Retained outcomes \(1\)/))
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
  fireEvent.click(screen.getByText(/Retained outcomes \(1\)/))
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
      dismissRecovery: vi.fn(async () => true)
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
    dismissRecovery: vi.fn(async () => true)
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


describe("bounded retained outcomes", () => {
  const view = {
    owner_key: "owner",
    conversation_id: "chat",
    view_session_id: "visible-view",
    selection_revision: 3,
    interpretation: { kind: "parent_graph_v1" as const },
    cursor: { kind: "after_message" as const, message_id: "reply" }
  }
  const receipt = (id: string, current = false) => ({
    scope: {
      profile_id: "profile",
      client_session_id: current ? "writer" : "older-writer"
    },
    turn: {
      operation_id: id,
      state: "refused" as const,
      input_text: `Exact ${id}.  \nNext line.`,
      result_text: "",
      created_at: Number(id.slice(1)),
      origin_view: { ...view }
    }
  })
  const selectionWith = (recoveries: ReturnType<typeof receipt>[]) =>
    ({
      ...controller(),
      owner: { kind: "native", owner_key: "owner", conversation_id: "chat" },
      view,
      bookmarkScope: { profile_id: "profile", client_session_id: "writer" },
      status: "stale_selection",
      sendConflict: true,
      loadLatest: vi.fn(),
      branchHere: vi.fn(),
      fence: () => () => true,
      recoveries,
      dismissRecovery: vi.fn(async () => true)
    }) as unknown as HistorySelectionController
  it("keeps current choices and only its qualified retained result ahead of five collapsed older records", () => {
    const older = Array.from({ length: 5 }, (_, index) =>
        receipt(`r${index + 1}`)
      ),
      current = receipt("r6", true)
    const original = [
      older[2],
      current,
      ...older.filter((entry) => entry !== older[2])
    ]
    const selection = selectionWith(original)
    render(<HistorySelectionReview selection={selection} />)
    const summary = screen.getByText(/Retained outcomes.*5/)
    const copies = screen.getAllByRole("button", {
      name: "Copy recovered text"
    })
    expect(copies[0]).toBeVisible()
    copies.slice(1).forEach((button) => expect(button).not.toBeVisible())
    expect(
      screen
        .getByRole("button", { name: "Load latest messages" })
        .compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(summary.closest("details")).not.toHaveAttribute("open")
    fireEvent.click(summary)
    expect(
      screen.getAllByRole("button", { name: "Copy recovered text" })
    ).toHaveLength(6)
    expect(selection.recoveries).toBe(original)
    for (const entry of original)
      expect(
        screen.getByText(entry.turn.input_text, {
          normalizer: (value) => value
        })
      ).toBeInTheDocument()
  })
  it("preserves an explicit expansion during same-owner refresh, but closes after leaving and returning", async () => {
    const selection = selectionWith([receipt("r1")])
    const mounted = render(<HistorySelectionReview selection={selection} />)
    fireEvent.click(screen.getByText(/Retained outcomes \(1\)/))
    await waitFor(() =>
      expect(
        screen.getByText(/Retained outcomes \(1\)/).closest("details")
      ).toHaveAttribute("open")
    )
    mounted.rerender(
      <HistorySelectionReview
        selection={{
          ...selection,
          recoveries: [...selection.recoveries, receipt("r2")]
        }}
      />
    )
    expect(
      screen.getByText(/Retained outcomes \(2\)/).closest("details")
    ).toHaveAttribute("open")
    mounted.rerender(
      <HistorySelectionReview
        selection={{
          ...selection,
          owner: {
            kind: "native",
            owner_key: "other-owner",
            conversation_id: "other-chat"
          },
          recoveries: []
        }}
      />
    )
    expect(screen.queryByText(/Retained outcomes/)).toBeNull()
    mounted.rerender(<HistorySelectionReview selection={selection} />)
    expect(
      screen.getByText(/Retained outcomes \(1\)/).closest("details")
    ).not.toHaveAttribute("open")
  })
  it("keeps distinct operations with identical text and sorts their actions by recency without changing the journal", async () => {
    const first = receipt("r1"),
      second = receipt("r2")
    first.turn.input_text = second.turn.input_text =
      "Identical input.  \nExact next line."
    const entries = [first, second],
      selection = selectionWith(entries)
    render(<HistorySelectionReview selection={selection} />)
    const summary = screen.getByText(/Retained outcomes \(2\)/)
    fireEvent.click(summary)
    const cards = [
      ...summary
        .closest("details")!
        .querySelectorAll<HTMLElement>('[role="status"]')
    ]
    expect(cards).toHaveLength(2)
    expect(
      screen.getAllByText(first.turn.input_text, {
        normalizer: (value) => value
      })
    ).toHaveLength(2)
    for (const card of cards) {
      await act(async () => { fireEvent.click(within(card).getByRole("button", { name: "Dismiss recovery" })) })
    }
    expect(vi.mocked(selection.dismissRecovery).mock.calls).toEqual([
      [second, expect.any(Function)],
      [first, expect.any(Function)]
    ])
    expect(selection.recoveries).toBe(entries)
    expect(entries.map((entry) => entry.turn.operation_id)).toEqual([
      "r1",
      "r2"
    ])
  })
  it("retains saved-result actions and an uncertain partial reply inside the disclosure", async () => {
    const saved = completedEditedSelection().recoveries[0]
    const partial = {
      ...receipt("r3"),
      turn: {
        ...receipt("r3").turn,
        state: "unknown" as const,
        input_text: "Uncertain question",
        result_text: "Retained partial reply"
      }
    }
    const selection = {
      ...selectionWith([]),
      status: "ready" as const,
      recoveries: [saved, partial],
      openRecoveryResult: vi.fn(async () => true)
    }
    render(<HistorySelectionReview selection={selection} />)
    const summary = screen.getByText(/Retained outcomes \(2\)/)
    expect(
      screen.getByRole("button", { name: "Open saved reply" })
    ).not.toBeVisible()
    expect(selection.openRecoveryResult).not.toHaveBeenCalled()
    fireEvent.click(summary)
    const partialCard = screen
      .getByText("Retained partial reply")
      .closest('[role="status"]')!
    fireEvent.click(
      within(partialCard as HTMLElement).getByText(
        "Inspect original input and result"
      )
    )
    expect(screen.getByText("Retained partial reply")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Open saved reply" }))
    await waitFor(() =>
      expect(selection.openRecoveryResult).toHaveBeenCalledWith(
        saved,
        expect.any(Function)
      )
    )
    expect(selection.dismissRecovery).not.toHaveBeenCalled()
  })
  it("counts only displayed retained records and never promotes a foreign writer into the current result", () => {
    const entries = Array.from({ length: 5 }, (_, index) =>
      receipt(`r${index + 1}`)
    )
    const selection = selectionWith(entries)
    render(
      <HistorySelectionReview
        selection={selection}
        hideRecovery={(entry) => entry.turn.operation_id === "r2"}
      />
    )
    const summary = screen.getByText(/Retained outcomes.*4/)
    screen
      .getAllByRole("button", { name: "Copy recovered text" })
      .forEach((button) => expect(button).not.toBeVisible())
    fireEvent.click(summary)
    expect(
      screen.getAllByRole("button", { name: "Copy recovered text" })
    ).toHaveLength(4)
    expect(
      screen.queryByText(entries[1].turn.input_text, {
        normalizer: (value) => value
      })
    ).toBeNull()
    const record = screen
      .getByText(entries[0].turn.input_text, { normalizer: (value) => value })
      .closest('[role="status"]')!
    fireEvent.click(record.querySelectorAll("button")[1])
    expect(selection.dismissRecovery).toHaveBeenCalledWith(entries[0], expect.any(Function))
  })
})


describe("acknowledged recovery actions", () => {
  const mountAcknowledgement = async () => {
    const { useStoreMessageOption } = await import("@/store/option")
    const { useComposerEvents } = await import("@/hooks/useComposerEvents")
    useStoreMessageOption.getState().startNewChatDraft()
    const view = {
      owner_key: "owner",
      conversation_id: "chat",
      view_session_id: "writer",
      selection_revision: 1,
      interpretation: { kind: "parent_graph_v1" as const },
      cursor: { kind: "empty" as const }
    }
    const entry = {
      scope: { profile_id: "profile", client_session_id: "writer" },
      turn: {
        operation_id: "ack-op",
        origin_view: view,
        created_at: 1,
        owner_key: "owner",
        conversation_id: "chat",
        state: "generated_unsaved" as const,
        input_text: "  exact original input\n  ",
        result_text: "<think>raw reasoning</think>  exact result\n  "
      }
    }
    let epoch = 1
    let ownerCurrent = true
    const selection = {
      ...controller(),
      status: "ready",
      view,
      recoveries: [entry],
      owner: {
        kind: "local",
        profile_id: "profile",
        owner_key: "owner",
        conversation_id: "chat",
        validate_lease: () => ownerCurrent
      },
      bookmarkScope: entry.scope,
      fence: () => {
        const token = epoch
        return () => epoch === token
      },
      dismissRecovery: vi.fn(async () => true)
    } as unknown as HistorySelectionController
    const focusRequests = vi.fn()
    function Composer() {
      const input = React.useRef<HTMLTextAreaElement>(null)
      const onFocusComposer = React.useCallback(() => {
        focusRequests()
        input.current?.focus()
      }, [])
      useComposerEvents({ onFocusComposer })
      return (
        <>
          <textarea
            ref={input}
            aria-label="ACK composer"
            defaultValue={"Exact draft.  \nKeep this line."}
          />
          <button>Other editor control</button>
          <HistorySelectionReview selection={selection} />
        </>
      )
    }
    const mounted = render(<Composer />)
    return {
      mounted,
      selection,
      entry,
      focusRequests,
      retire: () => {
        epoch += 1
      },
      expireOwner: () => {
        ownerCurrent = false
      },
      newDraft: () => useStoreMessageOption.getState().startNewChatDraft(),
      rerender: () => mounted.rerender(<Composer />)
    }
  }

  it("refuses clipboard access from an expired rendered owner before any React commit", async () => {
    const probe = await mountAcknowledgement()
    const copy = vi.fn(async () => undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: copy }
    })
    probe.expireOwner()
    fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" }))
    expect(copy).not.toHaveBeenCalled()
    expect(screen.queryByText("Recovered text copied.")).toBeNull()
  })

  it("copies exact raw bytes and announces success only after the clipboard acknowledges", async () => {
    const probe = await mountAcknowledgement()
    let finish!: () => void
    const copy = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve
        })
    )
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: copy }
    })
    const button = screen.getByRole("button", { name: "Copy recovered text" })
    fireEvent.click(button)
    expect(copy).toHaveBeenCalledExactlyOnceWith(probe.entry.turn.result_text)
    expect(button).toBeDisabled()
    expect(screen.queryByText("Recovered text copied.")).toBeNull()
    await act(async () => {
      finish()
    })
    expect(screen.getByText("Recovered text copied.")).toHaveAttribute(
      "role",
      "status"
    )
    expect(probe.selection.recoveries).toEqual([probe.entry])
    expect(probe.focusRequests).not.toHaveBeenCalled()
  })

  it("retains the text and offers a usable retry after a clipboard rejection", async () => {
    const probe = await mountAcknowledgement()
    const copy = vi
      .fn()
      .mockRejectedValueOnce(new Error("clipboard denied"))
      .mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: copy }
    })
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Copy recovered text" })
      )
    })
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Could not copy recovered text. Select it in the details or try Copy again."
    )
    expect(
      screen.getByText(probe.entry.turn.result_text, {
        normalizer: (value) => value
      })
    ).toBeInTheDocument()
    expect(screen.queryByText("Recovered text copied.")).toBeNull()
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Copy recovered text" })
      )
    })
    expect(screen.getByText("Recovered text copied.")).toBeInTheDocument()
    expect(copy).toHaveBeenNthCalledWith(2, probe.entry.turn.result_text)
  })

  it("an old clipboard completion cannot publish into or clear a newer attempt", async () => {
    const probe = await mountAcknowledgement()
    let oldFinish!: () => void
    let newFinish!: () => void
    const copy = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            oldFinish = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            newFinish = resolve
          })
      )
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: copy }
    })
    fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" }))
    act(() => {
      probe.newDraft()
      probe.rerender()
    })
    fireEvent.click(screen.getByRole("button", { name: "Copy recovered text" }))
    await act(async () => {
      oldFinish()
    })
    expect(screen.queryByText("Recovered text copied.")).toBeNull()
    expect(
      screen.getByRole("button", { name: "Copy recovered text" })
    ).toBeDisabled()
    await act(async () => {
      newFinish()
    })
    expect(screen.getByText("Recovered text copied.")).toBeInTheDocument()
    expect(copy).toHaveBeenCalledTimes(2)
  })

  it.each(["refused", "failed"])(
    "does not announce or remove a recovery after %s dismissal",
    async (mode) => {
      const probe = await mountAcknowledgement()
      const dismiss = vi.mocked(probe.selection.dismissRecovery)
      if (mode === "failed")
        dismiss.mockRejectedValueOnce(new Error("journal unavailable"))
      else dismiss.mockResolvedValueOnce(false)
      await act(async () => {
        fireEvent.click(
          screen.getByRole("button", { name: "Dismiss recovery" })
        )
      })
      expect(screen.getByRole("alert")).toHaveTextContent(
        "The recovery could not be dismissed. Your text is still available; try again."
      )
      expect(screen.queryByText("Recovery dismissed.")).toBeNull()
      expect(
        screen.getByText(probe.entry.turn.result_text, {
          normalizer: (value) => value
        })
      ).toBeInTheDocument()
      expect(probe.selection.recoveries).toEqual([probe.entry])
      expect(probe.focusRequests).not.toHaveBeenCalled()
    }
  )

  it.each(["Copy recovered text", "Dismiss recovery"])(
    "an old rendered %s cannot adopt a new same-ID H1 epoch",
    async (label) => {
      const probe = await mountAcknowledgement()
      const copy = vi.fn(async () => undefined)
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText: copy }
      })
      probe.retire()
      fireEvent.click(screen.getByRole("button", { name: label }))
      expect(copy).not.toHaveBeenCalled()
      expect(probe.selection.dismissRecovery).not.toHaveBeenCalled()
    }
  )

  it.each([
    "current",
    "focus ABA",
    "pointer",
    "new input",
    "New",
    "H1",
    "editor",
    "dialog",
    "hidden",
    "unmount"
  ])(
    "acknowledged Dismiss returns only its owned removed-control focus (%s)",
    async (boundary) => {
      const probe = await mountAcknowledgement()
      const frames = new Map<number, FrameRequestCallback>()
      let id = 0
      const raf = vi
        .spyOn(globalThis, "requestAnimationFrame")
        .mockImplementation((callback) => {
          frames.set(++id, callback)
          return id
        })
      const cancel = vi
        .spyOn(globalThis, "cancelAnimationFrame")
        .mockImplementation((key) => {
          frames.delete(key)
        })
      const visibility = vi
        .spyOn(document, "visibilityState", "get")
        .mockReturnValue("visible")
      const dismiss = screen.getByRole("button", { name: "Dismiss recovery" })
      const input = screen.getByRole("textbox", { name: "ACK composer" })
      const other = screen.getByRole("button", { name: "Other editor control" })
      let extra: HTMLElement | undefined
      try {
        dismiss.focus()
        await act(async () => {
          fireEvent.click(dismiss)
        })
        expect(probe.selection.dismissRecovery).toHaveBeenCalledWith(
          probe.entry,
          expect.any(Function)
        )
        if (boundary === "focus ABA") {
          other.focus()
          dismiss.focus()
        } else if (boundary === "pointer") fireEvent.pointerDown(other)
        else if (boundary === "new input")
          fireEvent.input(input, { target: { value: "Newer exact input.  " } })
        else if (boundary === "New") probe.newDraft()
        else if (boundary === "H1") probe.retire()
        else if (boundary === "editor") {
          extra = document.createElement("form")
          extra.dataset.chatMessageEditor = ""
          document.body.append(extra)
        } else if (boundary === "dialog") {
          extra = document.createElement("div")
          extra.setAttribute("role", "dialog")
          document.body.append(extra)
        } else if (boundary === "hidden") visibility.mockReturnValue("hidden")
        else if (boundary === "unmount") probe.mounted.unmount()
        if (boundary !== "unmount") {
          probe.selection.recoveries = []
          act(() => probe.rerender())
        }
        const pending = [...frames.values()]
        await act(async () => {
          pending.forEach((callback) => callback(0))
        })
        expect(probe.focusRequests).toHaveBeenCalledTimes(
          boundary === "current" ? 1 : 0
        )
        if (boundary === "current") {
          expect(input).toHaveFocus()
          expect(input).toHaveValue("Exact draft.  \nKeep this line.")
          expect(screen.getByText("Recovery dismissed.")).toHaveAttribute(
            "role",
            "status"
          )
        }
      } finally {
        extra?.remove()
        visibility.mockRestore()
        cancel.mockRestore()
        raf.mockRestore()
        probe.mounted.unmount()
      }
    }
  )
})


const roleplaySelection = (saved: boolean) => {
  const selection = completedEditedSelection()
  const original = {
    version: 1,
    conversation_id: "review-resume",
    input: {
      message_id: "resume-input",
      content: "Exact original input",
      expected: {
        version: 1,
        snapshot_digest: "sha256:" + "c".repeat(64),
        settings_version: 1,
        history_version: 1,
        tail: { message_id: null, message_version: null }
      }
    },
    assistant_message_id: "saved-answer"
  }
  const append = {
    version: 1,
    conversation_id: original.conversation_id,
    message_id: original.input.message_id,
    message_version: 1,
    parent_message_id: null,
    fence: {
      ...original.input.expected,
      history_version: 2,
      tail: { message_id: original.input.message_id, message_version: 1 }
    }
  }
  const completion = {
    chat_id: original.conversation_id,
    character_id: 12,
    provider: "openai",
    model: "original-model",
    saved: true,
    user_message_id: original.input.message_id,
    assistant_message_id: original.assistant_message_id,
    assistant_content: "Original reply",
    tldw_roleplay_resume_v1: {
      version: 1,
      input_message_id: original.input.message_id,
      input_message_version: 1,
      assistant_message_id: original.assistant_message_id,
      expected: append.fence,
      assistant_message_version: 1,
      fence: {
        ...append.fence,
        history_version: 3,
        tail: { message_id: original.assistant_message_id, message_version: 1 }
      },
      unsaved_reason: null
    }
  }
  const protocol = parseRoleplayResumeRecovery({
    intent: original,
    append,
    ...(saved ? { completion } : {})
  })
  selection.recoveries = [
    {
      scope: { profile_id: "profile", client_session_id: "original" },
      turn: {
        operation_id: "resume-review",
        owner_key: "resume-owner",
        conversation_id: original.conversation_id,
        origin_view: {
          owner_key: "resume-owner",
          conversation_id: original.conversation_id,
          view_session_id: "original",
          selection_revision: 1,
          interpretation: { kind: "parent_graph_v1" },
          cursor: { kind: "empty" }
        },
        persistence: "server",
        created_at: 1,
        input_text: original.input.content,
        input_images: [],
        input_id: append.message_id,
        result_text: saved ? completion.assistant_content : "",
        state: saved ? "generated_unsaved" : "accepted_unsent",
        ...(saved
          ? {
              assistant_id: original.assistant_message_id,
              outcome: "complete",
              settled_message_id: original.assistant_message_id
            }
          : {}),
        roleplay_resume: protocol
      }
    }
  ] as typeof selection.recoveries
  selection.fence = () => () => true
  selection.getCurrent = () =>
    selection as unknown as ReturnType<HistorySelectionController["getCurrent"]>
  return selection
}

it("a saved original Roleplay reply can be explicitly opened without an Edit or Regenerate action", () => {
  const selection = roleplaySelection(true)
  render(<HistorySelectionReview selection={selection} />)
  expect(screen.getByText("Reply saved")).toBeTruthy()
  expect(screen.getByRole("button", { name: "Open saved reply" })).toBeTruthy()
  expect(screen.queryByText(/Edited reply saved/)).toBeNull()
})


it("Retry owns shared response activity and keeps dismissal disabled until its original attempt settles", async () => {
  useStoreMessageOption.getState().startNewChatDraft()
  const selection = roleplaySelection(false)
  let finish!: () => void
  roleplayRetry.send.mockImplementationOnce(
    () =>
      new Promise<void>((resolve) => {
        finish = resolve
      })
  )
  render(<HistorySelectionReview selection={selection} />)
  fireEvent.click(
    screen.getByRole("button", { name: "Retry original request" })
  )
  await waitFor(() => expect(roleplayRetry.send).toHaveBeenCalledOnce())
  expect(useStoreMessageOption.getState().isProcessing).toBe(true)
  expect(useStoreMessageOption.getState().streaming).toBe(true)
  expect(
    screen.getByRole("button", { name: "Dismiss recovery" })
  ).toHaveProperty("disabled", true)
  await act(async () => {
    finish()
    await Promise.resolve()
  })
  expect(useStoreMessageOption.getState().isProcessing).toBe(false)
  expect(useStoreMessageOption.getState().streaming).toBe(false)
})

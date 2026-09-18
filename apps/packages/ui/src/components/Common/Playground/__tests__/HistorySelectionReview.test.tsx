import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
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

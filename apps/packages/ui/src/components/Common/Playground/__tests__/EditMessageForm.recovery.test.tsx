// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { EditMessageForm } from "../EditMessageForm"
import { useStoreMessageOption } from "@/store/option"
const history = vi.hoisted(() => ({ epoch: 0, enabled: false }))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () =>
    history.enabled
      ? {
          fence: () => {
            const epoch = history.epoch
            return () => epoch === history.epoch
          }
        }
      : null
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) =>
      (
        ({
          save: "Save",
          saveAndSubmit: "Save & Send",
          cancel: "Cancel",
          saving: "Saving..."
        }) as Record<string, string>
      )[key] ??
      fallback ??
      key
  })
}))
const clipboardDescriptor = Object.getOwnPropertyDescriptor(
  navigator,
  "clipboard"
)
let draftRevision: number
beforeEach(() => {
  draftRevision = useStoreMessageOption.getState().chatDraftRevision
})
afterEach(() => {
  useStoreMessageOption.setState({ chatDraftRevision: draftRevision })
  history.enabled = false
  history.epoch = 0
  if (clipboardDescriptor)
    Object.defineProperty(navigator, "clipboard", clipboardDescriptor)
  else Reflect.deleteProperty(navigator, "clipboard")
})
const draft =
  "  {{char}} greets {{user}}.\n<think>Authored roleplay directions</think>\n\nExact edit  "
const wireError = (status: number, detail: unknown) =>
  Object.assign(new Error("Server response"), { status, details: { detail } })
const mount = (outcome: unknown, rejected = false) => {
  const close = vi.fn(),
    submit = vi.fn(() =>
      rejected ? Promise.reject(outcome) : Promise.resolve(outcome)
    )
  const props = {
    value: "Original",
    isBot: false,
    onClose: close,
    onSumbit: submit
  }
  const view = render(<EditMessageForm {...props} />)
  fireEvent.change(screen.getByRole("textbox"), { target: { value: draft } })
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
  return { ...view, props, close, submit }
}
describe("current message edit recovery", () => {
  it.each([
    {
      name: "conflict",
      outcome: wireError(409, "Version mismatch"),
      rejected: true,
      expected: /saved message has changed/i
    },
    {
      name: "busy",
      outcome: { status: "skipped", reason: "response_in_progress" },
      expected: /reply is still in progress/i
    },
    {
      name: "account",
      outcome: wireError(412, {
        code: "request_config_scope_changed",
        message: "private account detail"
      }),
      rejected: true,
      expected: /server or signed-in account changed/i
    },
    {
      name: "history",
      outcome: new Error("unsupported_history_capability"),
      rejected: true,
      expected: /does not support this edit action/i
    },
    {
      name: "resend",
      outcome: new Error("unsupported_history_edit_and_send:private-id"),
      rejected: true,
      expected: /does not support this edit action/i
    },
    {
      name: "payload",
      outcome: new Error("unsupported_message_payload"),
      rejected: true,
      expected: /plain-text messages/i
    },
    {
      name: "sync",
      outcome: wireError(400, {
        error_code: "sync_v2_message_edit_not_supported",
        message: "private sync detail"
      }),
      rejected: true,
      expected: /sync setup does not support editing/i
    },
    {
      name: "protected Edit source changed",
      outcome: {
        status: "failed",
        errorMessage: "native_history_edit_target_changed"
      },
      expected: /saved message has changed/i
    },
    {
      name: "protected Edit old server",
      outcome: {
        status: "failed",
        errorMessage: "native_history_edit_unsupported"
      },
      expected: /does not support this edit action/i
    },
    {
      name: "changed edit",
      outcome: { status: "skipped", reason: "stale_selection" },
      expected: /conversation or edited text changed/i
    },
    {
      name: "setup",
      outcome: { status: "skipped", reason: "validation_failed" },
      expected: /review the conversation setup/i
    },
    {
      name: "unknown receipt",
      outcome: new Error("message_edit_unconfirmed"),
      rejected: true,
      expected: /confirm whether your edit was saved/i
    },
    {
      name: "unknown transport",
      outcome: new Error(
        "https://private.example/api/v1/messages/u Bearer secret-token-value"
      ),
      rejected: true,
      expected: /confirm whether your edit was saved/i
    }
  ])(
    "$name gives usable guidance and retains exact authored text",
    async ({ outcome, rejected, expected }) => {
      const view = mount(outcome, rejected)
      const alert = await screen.findByRole("alert")
      expect(alert).toHaveTextContent(expected)
      expect(alert).toHaveTextContent("Your edited text is kept")
      expect(alert).not.toHaveTextContent(
        /unsupported_|sync_v2|private|secret-token|api\/v1|Try saving again/i
      )
      expect(screen.getByRole("textbox")).toHaveValue(draft)
      expect(view.close).not.toHaveBeenCalled()
      expect(view.submit).toHaveBeenCalledOnce()
    }
  )
  it("unknown write recovery requires checking saved state before another save", async () => {
    mount(new Error("Lost response"), true)
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /reload this conversation to check.*before saving again/i
    )
  })
  it("a changed source requires reloading the conversation instead of reopening a stale editor", async () => {
    const view = mount({ status: "skipped", reason: "stale_selection" })
    await screen.findByRole("alert")
    view.rerender(
      <EditMessageForm {...view.props} value="Changed saved source" />
    )
    expect(screen.getByRole("alert")).toHaveTextContent(
      /reload this conversation/i
    )
    expect(screen.getByRole("textbox")).toHaveValue(draft)
  })
  it("Copy edit preserves whitespace, placeholders and authored reasoning verbatim", async () => {
    const writeText = vi.fn(async () => {})
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })
    const view = mount(new Error("Lost response"), true)
    await screen.findByRole("alert")
    fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Edit copied")
    )
    expect(writeText).toHaveBeenCalledExactlyOnceWith(draft)
    expect(screen.getByRole("textbox")).toHaveValue(draft)
    expect(view.close).not.toHaveBeenCalled()
    expect(view.submit).toHaveBeenCalledOnce()
  })
  it.each(["rejection", "missing"])(
    "Copy edit reports %s without claiming success or clearing the draft",
    async (kind) => {
      const writeText = vi.fn(async () => {
        throw new Error("Clipboard denied")
      })
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: kind === "missing" ? undefined : { writeText }
      })
      mount(new Error("Lost response"), true)
      await screen.findByRole("alert")
      fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
      await waitFor(() =>
        expect(screen.getByRole("status")).toHaveTextContent(
          /select your edited text.*copy/i
        )
      )
      expect(screen.getByRole("status")).not.toHaveTextContent("Edit copied")
      expect(screen.getByRole("textbox")).toHaveValue(draft)
    }
  )
  it.each([
    "typing",
    "undo",
    "cancel",
    "source",
    "same-id-epoch",
    "same-id-draft",
    "unmount"
  ])(
    "held copy cannot certify a newer or retired editor after %s",
    async (boundary) => {
      let release!: () => void
      const held = new Promise<void>((accept) => {
        release = accept
      })
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText: vi.fn(() => held) }
      })
      history.enabled = true
      const view = mount(new Error("Lost response"), true)
      await screen.findByRole("alert")
      fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
      if (boundary === "same-id-epoch") {
        history.epoch++
        view.rerender(<EditMessageForm {...view.props} />)
      }
      if (boundary === "same-id-draft") {
        const revision = useStoreMessageOption.getState().chatDraftRevision
        useStoreMessageOption.setState({ chatDraftRevision: revision + 1 })
        view.rerender(<EditMessageForm {...view.props} />)
      }
      if (boundary === "undo") {
        fireEvent.change(screen.getByRole("textbox"), {
          target: { value: "Newer edit" }
        })
        fireEvent.change(screen.getByRole("textbox"), {
          target: { value: draft }
        })
      }
      if (boundary === "typing")
        fireEvent.change(screen.getByRole("textbox"), {
          target: { value: "Newer edit" }
        })
      if (boundary === "cancel")
        fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
      if (boundary === "source")
        view.rerender(<EditMessageForm {...view.props} value="Other source" />)
      if (boundary === "unmount") view.unmount()
      if (boundary !== "cancel" && boundary !== "unmount")
        await waitFor(() =>
          expect(
            screen.getByRole("button", { name: "Copy edit" })
          ).toBeEnabled()
        )
      await act(async () => {
        release()
        await held
      })
      expect(screen.queryByText("Edit copied")).toBeNull()
      if (boundary !== "unmount")
        expect(screen.getByRole("textbox")).toHaveValue(
          boundary === "typing" ? "Newer edit" : draft
        )
    }
  )
})

it.each([
  wireError(409, { code: "future_error", message: "message_changed" }),
  wireError(409, {
    code: "request_config_scope_changed",
    error_code: "sync_v2_message_edit_not_supported"
  }),
  new Error("provider wrote unsupported_history_capability near private-id"),
  Object.assign(new Error("hostile"), { code: "__proto__" })
])(
  "unknown or contradictory codes stay uncertain without exposing details",
  async (outcome) => {
    mount(outcome, true)
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /confirm whether your edit was saved/i
    )
  }
)

it("a wrapped declared account error keeps its recovery", async () => {
  mount(
    new Error("Save failed", {
      cause: wireError(412, { code: "request_config_scope_changed" })
    }),
    true
  )
  expect(await screen.findByRole("alert")).toHaveTextContent(
    /server or signed-in account changed/i
  )
})

it("a completed copy stops certifying the draft after edit and Undo", async () => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn(async () => {}) }
  })
  mount(new Error("Lost response"), true)
  await screen.findByRole("alert")
  fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
  await waitFor(() =>
    expect(screen.getByRole("status")).toHaveTextContent("Edit copied")
  )
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Other edit" }
  })
  fireEvent.change(screen.getByRole("textbox"), { target: { value: draft } })
  expect(screen.queryByText("Edit copied")).toBeNull()
})

it("a completed copy stops certifying identical text after a new history epoch", async () => {
  history.enabled = true
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn(async () => {}) }
  })
  const view = mount(new Error("Lost response"), true)
  await screen.findByRole("alert")
  fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
  await waitFor(() =>
    expect(screen.getByRole("status")).toHaveTextContent("Edit copied")
  )
  history.epoch++
  view.rerender(<EditMessageForm {...view.props} />)
  expect(screen.queryByText("Edit copied")).toBeNull()
})

it("a retired copy completion cannot clear a newer copy's pending state", async () => {
  let releaseOld!: () => void, releaseNew!: () => void
  const old = new Promise<void>((accept) => {
      releaseOld = accept
    }),
    next = new Promise<void>((accept) => {
      releaseNew = accept
    })
  const writeText = vi
    .fn()
    .mockImplementationOnce(() => old)
    .mockImplementationOnce(() => next)
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText }
  })
  const view = mount(new Error("Lost response"), true)
  try {
    await screen.findByRole("alert")
    fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Newer edit" }
    })
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Copy edit" })).toBeEnabled()
    )
    fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
    await act(async () => {
      releaseOld()
      await old
    })
    expect(screen.getByRole("button", { name: "Copy edit" })).toBeDisabled()
    expect(screen.getByRole("status")).toHaveTextContent("Copying edit")
    expect(screen.queryByText("Edit copied")).toBeNull()
    await act(async () => {
      releaseNew()
      await next
    })
    expect(screen.getByRole("status")).toHaveTextContent("Edit copied")
    expect(writeText.mock.calls).toEqual([[draft], ["Newer edit"]])
  } finally {
    releaseOld()
    releaseNew()
    view.unmount()
  }
})

it.each(["history", "draft"])(
  "a retired rendered Copy cannot adopt a newer %s source before rerender",
  async (boundary) => {
    history.enabled = true
    const writeText = vi.fn(async () => {})
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })
    const view = mount(new Error("Lost response"), true)
    await screen.findByRole("alert")
    if (boundary === "history") history.epoch++
    else
      useStoreMessageOption.setState({
        chatDraftRevision:
          useStoreMessageOption.getState().chatDraftRevision + 1
      })
    fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
    expect(writeText).not.toHaveBeenCalled()
    view.rerender(<EditMessageForm {...view.props} />)
    fireEvent.click(screen.getByRole("button", { name: "Copy edit" }))
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Edit copied")
    )
    expect(writeText).toHaveBeenCalledExactlyOnceWith(draft)
  }
)

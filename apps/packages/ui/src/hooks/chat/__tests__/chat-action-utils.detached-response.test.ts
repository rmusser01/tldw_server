import { afterEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import type { NativeHistoryEditResend } from "@/types/chat-modes"
import { withDetachedHistoryResponse } from "../chat-action-utils"

// Only the private presentation lifetime is under test; mounted dispatch tests validate the intent.
const edited = { historyEditResend: {} as NativeHistoryEditResend }
const held = () => {
  let release!: () => void
  const promise = new Promise<void>(resolve => { release = resolve })
  return { promise, release }
}
afterEach(() => { useStoreMessageOption.setState({ detachedHistoryResponse: null }) })

describe("detached response presentation lifetime", () => {
  it("installs identity before the dispatch and clears an admitted early return", async () => {
    const submit = withDetachedHistoryResponse(
      async (_params, beginResponse) => {
        beginResponse()
        expect(
          useStoreMessageOption.getState().detachedHistoryResponse
        ).toBeTruthy()
        return "skipped"
      }
    )
    expect(await submit(edited)).toBe("skipped")
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
  })

  it("clears the same attempt after a dispatch exception", async () => {
    const submit = withDetachedHistoryResponse(
      async (_params, beginResponse) => {
        beginResponse()
        throw new Error("preflight refusal")
      }
    )
    await expect(submit(edited)).rejects.toThrow("preflight refusal")
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
  })

  it("a late attempt cannot retire its replacement even when both use the same intent", async () => {
    const first = held(),
      second = held()
    const submit = withDetachedHistoryResponse(
      async (
        {
          gate
        }: { gate: Promise<void>; historyEditResend?: NativeHistoryEditResend },
        beginResponse
      ) => {
        beginResponse()
        await gate
      }
    )
    const a = submit({ ...edited, gate: first.promise })
    const original = useStoreMessageOption.getState().detachedHistoryResponse
    const b = submit({ ...edited, gate: second.promise })
    const replacement = useStoreMessageOption.getState().detachedHistoryResponse
    expect(replacement).not.toBe(original)
    first.release()
    await a
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBe(
      replacement
    )
    second.release()
    await b
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
  })

  it("normal dispatch clears obsolete presentation before starting and cannot clear a later edit", async () => {
    useStoreMessageOption.setState({ detachedHistoryResponse: {} })
    const first = held(),
      second = held()
    const submit = withDetachedHistoryResponse(
      async (
        {
          gate,
          historyEditResend
        }: { gate: Promise<void>; historyEditResend?: NativeHistoryEditResend },
        beginResponse
      ) => {
        beginResponse()
        if (!historyEditResend)
          expect(
            useStoreMessageOption.getState().detachedHistoryResponse
          ).toBeNull()
        await gate
      }
    )
    const normal = submit({ gate: first.promise })
    const editedReply = submit({ ...edited, gate: second.promise })
    const replacement = useStoreMessageOption.getState().detachedHistoryResponse
    first.release()
    await normal
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBe(
      replacement
    )
    second.release()
    await editedReply
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
  })

  it("an account reset cannot be undone by a retiring request", async () => {
    const gate = held()
    const submit = withDetachedHistoryResponse(
      async (_params, beginResponse) => {
        beginResponse()
        await gate.promise
      }
    )
    const pending = submit(edited)
    window.dispatchEvent(
      new CustomEvent("tldw:auth-principal-changed", {
        detail: { kind: "logout" }
      })
    )
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
    gate.release()
    await pending
    expect(useStoreMessageOption.getState().detachedHistoryResponse).toBeNull()
  })
})


it("an early refusal before beginning response presentation leaves the active marker intact", async () => {
  const current = {}
  useStoreMessageOption.setState({ detachedHistoryResponse: current })
  const submit = withDetachedHistoryResponse(async () => "skipped")
  expect(await submit(edited)).toBe("skipped")
  expect(useStoreMessageOption.getState().detachedHistoryResponse).toBe(current)
})

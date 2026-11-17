import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useCharacterInlineEdit } from "../hooks/useCharacterInlineEdit"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { updateCharacter: vi.fn() }
}))

beforeEach(() => vi.clearAllMocks())

const setup = (field: "name" | "description") => {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } }
  })
  const notification = {
    open: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    destroy: vi.fn()
  }
  const record = {
    id: "inline-1",
    name: "Original",
    description: "Original",
    version: 7
  }
  const hook = renderHook(
    () =>
      useCharacterInlineEdit({
        t: (key) => key,
        notification,
        qc: client,
        data: [record]
      }),
    {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      )
    }
  )
  act(() => hook.result.current.startInlineEdit(record, field))
  act(() =>
    hook.result.current.setInlineEdit(
      (previous) => previous && { ...previous, value: "Changed" }
    )
  )
  return { ...hook, notification }
}

it.each(["name", "description"] as const)(
  "submits concurrent inline %s saves only once",
  async (field) => {
    let resolve!: (value: object) => void
    const pending = new Promise<object>((yes) => {
      resolve = yes
    })
    vi.mocked(tldwClient.updateCharacter).mockReturnValue(pending)
    const hook = setup(field)
    await act(async () => {
      hook.result.current.saveInlineEdit()
      hook.result.current.saveInlineEdit()
    })
    await waitFor(() => expect(tldwClient.updateCharacter).toHaveBeenCalled())
    expect(tldwClient.updateCharacter).toHaveBeenCalledTimes(1)
    expect(tldwClient.updateCharacter).toHaveBeenCalledWith(
      "inline-1",
      { [field]: "Changed" },
      7
    )
    await act(async () => resolve({ id: "inline-1", version: 8 }))
    await waitFor(() => expect(hook.result.current.inlineEdit).toBeNull())
  }
)

it.each(["name", "description"] as const)(
  "retains a failed inline %s draft and permits one later retry",
  async (field) => {
    let reject!: (error: Error) => void
    const pending = new Promise<never>((_, no) => {
      reject = no
    })
    vi.mocked(tldwClient.updateCharacter)
      .mockReturnValueOnce(pending)
      .mockResolvedValueOnce({ id: "inline-1", version: 8 })
    const hook = setup(field)
    await act(async () => hook.result.current.saveInlineEdit())
    await waitFor(() =>
      expect(tldwClient.updateCharacter).toHaveBeenCalledTimes(1)
    )
    await act(async () => reject(new Error("Save unavailable")))
    await waitFor(() =>
      expect(hook.notification.error).toHaveBeenCalledTimes(1)
    )
    expect(hook.result.current.inlineEdit?.value).toBe("Changed")
    await act(async () => hook.result.current.saveInlineEdit())
    await waitFor(() => expect(hook.result.current.inlineEdit).toBeNull())
    expect(tldwClient.updateCharacter).toHaveBeenCalledTimes(2)
    expect(
      vi
        .mocked(tldwClient.updateCharacter)
        .mock.calls.map((call) => call.slice(1))
    ).toEqual([
      [{ [field]: "Changed" }, 7],
      [{ [field]: "Changed" }, 7]
    ])
  }
)

import { StrictMode, type PropsWithChildren } from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  improvePrompt: vi.fn()
}))

vi.mock("@/services/prompt-improvement", async (importOriginal) => {
  const original = await importOriginal<
    typeof import("@/services/prompt-improvement")
  >()
  return {
    ...original,
    improvePrompt: (...args: unknown[]) => mocks.improvePrompt(...args)
  }
})

import { usePromptAssist, type PromptTargetAdapter } from "../usePromptAssist"
import type { PromptImproveResponse } from "@/services/prompt-improvement"

const route = {
  selected_model: "openai/gpt-5-mini",
  provider_hint: "openai"
}

const response = (
  operationId: string,
  overrides: Partial<PromptImproveResponse> = {}
): PromptImproveResponse => ({
  schema_version: 1,
  operation_id: operationId,
  status: "improved",
  improved_text: "Improved draft",
  findings: [],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  meta_prompt_version: "prompt-improvement-v1",
  ...overrides
})

const createAdapter = (target: "system" | "user_message" = "system") => {
  let text = target === "system" ? "System draft" : "User draft @alice"
  let revision = "r1"
  let rawState: unknown = { override: undefined, selectedPromptId: "saved-1" }
  const apply = vi.fn((candidate: string) => {
    text = candidate
  })
  const captureUndo = vi.fn(() => rawState)
  const restoreUndo = vi.fn((snapshot: unknown) => {
    rawState = snapshot
  })
  const adapter: PromptTargetAdapter = {
    target,
    read: () => text,
    readRevision: () => revision,
    apply,
    captureUndo,
    restoreUndo
  }
  return {
    adapter,
    apply,
    captureUndo,
    restoreUndo,
    setText: (value: string) => {
      text = value
    },
    setRevision: (value: string) => {
      revision = value
    },
    setRawState: (value: unknown) => {
      rawState = value
    }
  }
}

const createImmutableAdapter = (
  text: string,
  revision: string,
  apply = vi.fn()
): PromptTargetAdapter => ({
  target: "system",
  read: () => text,
  readRevision: () => revision,
  apply,
  captureUndo: vi.fn(() => ({ text, revision })),
  restoreUndo: vi.fn()
})

describe("usePromptAssist", () => {
  let nextId = 0

  beforeEach(() => {
    mocks.improvePrompt.mockReset()
    nextId = 0
    vi.spyOn(globalThis.crypto, "randomUUID").mockImplementation(
      () =>
        `00000000-0000-4000-8000-${String(++nextId).padStart(12, "0")}` as `${string}-${string}-${string}-${string}-${string}`
    )
  })

  it("captures a UUID/text/revision/route snapshot and auto-applies only this target", async () => {
    const target = createAdapter("user_message")
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id, { improved_text: "Improved draft @alice" })
    )
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        readRecognizedTokens: () => [
          { kind: "mention", value: "@alice" },
          { kind: "attachment_reference", value: "private attachment body" }
        ],
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    await act(async () => {
      await result.current.improveNow()
    })

    expect(mocks.improvePrompt).toHaveBeenCalledWith({
      operation_id: "00000000-0000-4000-8000-000000000001",
      target: "user_message",
      text: "User draft @alice",
      model_selection: route,
      protected_tokens: [{ kind: "mention", value: "@alice", occurrences: 1 }]
    })
    expect(target.captureUndo).toHaveBeenCalledTimes(1)
    expect(target.apply).toHaveBeenCalledWith("Improved draft @alice")
    expect(result.current.state).toMatchObject({ status: "applied" })
  })

  it("does not auto-apply when text, revision, or route becomes stale", async () => {
    const target = createAdapter()
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    let activeRoute = route
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => activeRoute,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    let pending: Promise<void>
    act(() => {
      pending = result.current.improveNow()
    })
    target.setText("Edited while analyzing")
    target.setRevision("r2")
    activeRoute = { selected_model: "anthropic/claude", provider_hint: "anthropic" }
    await act(async () => {
      resolveRequest?.(response("unused"))
      await pending
    })

    expect(target.apply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      notice: "draft_changed"
    })
  })

  it("uses the latest immutable adapter without invalidating an unchanged target", async () => {
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    const oldApply = vi.fn()
    const latestApply = vi.fn()
    const wrapper = ({ children }: PropsWithChildren) => (
      <StrictMode>{children}</StrictMode>
    )
    const { result, rerender } = renderHook(
      ({ adapter }: { adapter: PromptTargetAdapter }) =>
        usePromptAssist({
          adapter,
          readActiveRoute: () => route,
          contextKey: "conversation-1",
          surfaceOpen: true
        }),
      {
        initialProps: {
          adapter: createImmutableAdapter("System draft", "r1", oldApply)
        },
        wrapper
      }
    )

    let pending: Promise<void>
    act(() => {
      pending = result.current.improveNow()
    })
    rerender({ adapter: createImmutableAdapter("System draft", "r1", latestApply) })
    await act(async () => {
      resolveRequest?.(response("unused"))
      await pending
    })

    expect(oldApply).not.toHaveBeenCalled()
    expect(latestApply).toHaveBeenCalledWith("Improved draft")
  })

  it("never applies through a stale immutable adapter after a draft rerender", async () => {
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    const oldApply = vi.fn()
    const latestApply = vi.fn()
    const { result, rerender } = renderHook(
      ({ adapter }: { adapter: PromptTargetAdapter }) =>
        usePromptAssist({
          adapter,
          readActiveRoute: () => route,
          contextKey: "conversation-1",
          surfaceOpen: true
        }),
      { initialProps: { adapter: createImmutableAdapter("System draft", "r1", oldApply) } }
    )

    let pending: Promise<void>
    act(() => {
      pending = result.current.improveNow()
    })
    rerender({ adapter: createImmutableAdapter("Newer draft", "r2", latestApply) })
    await act(async () => {
      resolveRequest?.(response("unused"))
      await pending
    })

    expect(oldApply).not.toHaveBeenCalled()
    expect(latestApply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      notice: "draft_changed"
    })
  })

  it("does not auto-apply after an immutable route rerender", async () => {
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    const apply = vi.fn()
    const adapter = createImmutableAdapter("System draft", "r1", apply)
    const { result, rerender } = renderHook(
      ({ activeRoute }: { activeRoute: typeof route }) =>
        usePromptAssist({
          adapter,
          readActiveRoute: () => activeRoute,
          contextKey: "conversation-1",
          surfaceOpen: true
        }),
      { initialProps: { activeRoute: route } }
    )

    let pending: Promise<void>
    act(() => {
      pending = result.current.improveNow()
    })
    rerender({
      activeRoute: {
        selected_model: "anthropic/claude",
        provider_hint: "anthropic"
      }
    })
    await act(async () => {
      resolveRequest?.(response("unused"))
      await pending
    })

    expect(apply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      notice: "route_changed"
    })
  })

  it.each([
    {
      label: "placeholder multiset",
      original: "System {{name}} {{name}}",
      candidate: "System {{name}}",
      recognized: []
    },
    {
      label: "protected token count",
      original: "Ask @alice and @alice",
      candidate: "Ask @alice",
      recognized: [{ kind: "mention", value: "@alice" }]
    },
    {
      label: "Markdown fence structure",
      original: "Use code:\n```ts\nconst x = 1\n```",
      candidate: "Use code:\n```ts\nconst x = 1",
      recognized: []
    },
    {
      label: "XML wrapper structure",
      original: "<Context>Keep this</Context>",
      candidate: "<Context>Keep this",
      recognized: []
    }
  ])("upgrades a false server review flag for corrupted $label", async ({ original, candidate, recognized }) => {
    const apply = vi.fn()
    const adapter = createImmutableAdapter(original, "r1", apply)
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id, {
        improved_text: candidate,
        review_required: false
      })
    )
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter,
        readActiveRoute: () => route,
        readRecognizedTokens: () => recognized,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    await act(async () => result.current.improveNow())

    expect(apply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      notice: "review_required"
    })
  })

  it.each([
    { contextKey: "conversation-2", surfaceOpen: true },
    { contextKey: "conversation-1", surfaceOpen: false }
  ])("discards a deferred response after lifecycle change %#", async (nextProps) => {
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    const apply = vi.fn()
    const adapter = createImmutableAdapter("System draft", "r1", apply)
    const { result, rerender } = renderHook(
      (props: { contextKey: string; surfaceOpen: boolean }) =>
        usePromptAssist({
          adapter,
          readActiveRoute: () => route,
          contextKey: props.contextKey,
          surfaceOpen: props.surfaceOpen
        }),
      { initialProps: { contextKey: "conversation-1", surfaceOpen: true } }
    )

    act(() => void result.current.improveNow())
    rerender(nextProps)
    await act(async () => {
      resolveRequest?.(response("unused"))
      await Promise.resolve()
    })

    expect(apply).not.toHaveBeenCalled()
    expect(result.current.state.status).toBe("idle")
  })

  it("always reviews in review mode and applies only against the captured original", async () => {
    const target = createAdapter()
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    await act(async () => {
      await result.current.reviewChanges()
    })
    expect(result.current.state.status).toBe("reviewing")

    act(() => {
      result.current.editCandidate("Edited candidate")
      target.setText("New live draft")
      result.current.applyCandidate()
    })
    expect(target.apply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      replaceConfirmationRequired: true
    })
  })

  it("requires exact revision freshness before review apply and confirms through the latest adapter", async () => {
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const oldApply = vi.fn()
    const latestApply = vi.fn()
    const { result, rerender } = renderHook(
      ({ adapter }: { adapter: PromptTargetAdapter }) =>
        usePromptAssist({
          adapter,
          readActiveRoute: () => route,
          contextKey: "conversation-1",
          surfaceOpen: true
        }),
      { initialProps: { adapter: createImmutableAdapter("System draft", "r1", oldApply) } }
    )

    await act(async () => result.current.reviewChanges())
    rerender({ adapter: createImmutableAdapter("System draft", "r2", latestApply) })
    act(() => result.current.applyCandidate())

    expect(oldApply).not.toHaveBeenCalled()
    expect(latestApply).not.toHaveBeenCalled()
    expect(result.current.state).toMatchObject({
      status: "reviewing",
      replaceConfirmationRequired: true
    })

    act(() => result.current.confirmReplaceCurrent())
    expect(latestApply).toHaveBeenCalledWith("Improved draft")
  })

  it("captures the exact live undo snapshot immediately before confirmed replacement", async () => {
    const target = createAdapter()
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    await act(async () => {
      await result.current.reviewChanges()
    })
    const exactLiveSnapshot = { override: "new custom override" }
    target.setText("New live draft")
    target.setRawState(exactLiveSnapshot)
    act(() => result.current.applyCandidate())
    act(() => result.current.confirmReplaceCurrent())

    expect(target.captureUndo).toHaveBeenCalledTimes(1)
    expect(target.apply).toHaveBeenCalledWith("Improved draft")
    act(() => result.current.undo())
    expect(target.restoreUndo).toHaveBeenCalledWith(exactLiveSnapshot)
    expect(result.current.state.undo).toBeNull()
  })

  it("restores an exact undefined raw system snapshot through the adapter", async () => {
    const target = createAdapter()
    target.setRawState(undefined)
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    await act(async () => result.current.improveNow())
    act(() => result.current.undo())

    expect(target.restoreUndo).toHaveBeenCalledWith(undefined)
  })

  it("prevents duplicate submissions and Retry takes a fresh UUID/snapshot", async () => {
    const target = createAdapter()
    let rejectRequest: ((error: Error) => void) | undefined
    mocks.improvePrompt
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectRequest = reject
          })
      )
      .mockImplementationOnce(async (request) => response(request.operation_id))
    const { result } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    let first: Promise<void>
    act(() => {
      first = result.current.improveNow()
      void result.current.improveNow()
    })
    expect(mocks.improvePrompt).toHaveBeenCalledTimes(1)
    await act(async () => {
      rejectRequest?.(Object.assign(new Error("offline"), {
        code: "provider_unavailable",
        retryable: true
      }))
      await first
    })
    expect(result.current.state).toMatchObject({
      status: "failed",
      error: {
        code: "provider_unavailable",
        message: "The prompt improvement service is unavailable."
      }
    })
    target.setText("Fresh retry draft")
    target.setRevision("r2")
    await act(async () => result.current.retry())

    expect(mocks.improvePrompt).toHaveBeenCalledTimes(2)
    expect(mocks.improvePrompt.mock.calls[1][0]).toMatchObject({
      operation_id: "00000000-0000-4000-8000-000000000002",
      text: "Fresh retry draft"
    })
  })

  it("clears undo on edits, send/save, context navigation, close, dismiss, and new operations", async () => {
    const target = createAdapter()
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const { result, rerender } = renderHook(
      (props: { contextKey: string; surfaceOpen: boolean }) =>
        usePromptAssist({
          adapter: target.adapter,
          readActiveRoute: () => route,
          contextKey: props.contextKey,
          surfaceOpen: props.surfaceOpen
        }),
      { initialProps: { contextKey: "conversation-1", surfaceOpen: true } }
    )

    await act(async () => result.current.improveNow())
    act(() => result.current.notifyTargetEdited())
    expect(result.current.state.undo).toBeNull()

    await act(async () => result.current.improveNow())
    act(() => result.current.notifySendOrSave())
    expect(result.current.state.undo).toBeNull()

    await act(async () => result.current.improveNow())
    rerender({ contextKey: "conversation-2", surfaceOpen: true })
    expect(result.current.state).toMatchObject({ status: "idle", undo: null })

    await act(async () => result.current.improveNow())
    rerender({ contextKey: "conversation-2", surfaceOpen: false })
    expect(result.current.state).toMatchObject({ status: "idle", undo: null })

    rerender({ contextKey: "conversation-2", surfaceOpen: true })
    await act(async () => result.current.improveNow())
    act(() => result.current.dismiss())
    expect(result.current.state).toMatchObject({ status: "idle", undo: null })
  })

  it("disposes a late response after unmount without mutating the adapter", async () => {
    const target = createAdapter()
    let resolveRequest: ((value: PromptImproveResponse) => void) | undefined
    mocks.improvePrompt.mockImplementation(
      (request) =>
        new Promise((resolve) => {
          resolveRequest = () => resolve(response(request.operation_id))
        })
    )
    const { result, unmount } = renderHook(() =>
      usePromptAssist({
        adapter: target.adapter,
        readActiveRoute: () => route,
        contextKey: "conversation-1",
        surfaceOpen: true
      })
    )

    act(() => void result.current.improveNow())
    unmount()
    await act(async () => {
      resolveRequest?.(response("unused"))
      await Promise.resolve()
    })

    expect(target.apply).not.toHaveBeenCalled()
    expect(target.captureUndo).not.toHaveBeenCalled()
  })

  it("remains mounted across the StrictMode effect probe", async () => {
    const target = createAdapter()
    mocks.improvePrompt.mockImplementation(async (request) =>
      response(request.operation_id)
    )
    const wrapper = ({ children }: PropsWithChildren) => (
      <StrictMode>{children}</StrictMode>
    )
    const { result } = renderHook(
      () =>
        usePromptAssist({
          adapter: target.adapter,
          readActiveRoute: () => route,
          contextKey: "conversation-1",
          surfaceOpen: true
        }),
      { wrapper }
    )

    await act(async () => result.current.improveNow())

    expect(target.apply).toHaveBeenCalledWith("Improved draft")
    expect(result.current.state.status).toBe("applied")
  })
})

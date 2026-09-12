import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import * as serverOnline from "@/hooks/useServerOnline"
import type { PromptImproveModelSelection } from "@/services/prompt-improvement"
import {
  clearRecipePersistenceScoped,
  markRecipePersistenceScoped
} from "@/services/recipe-persistence-uncertainty"
import * as recipeAuthority from "@/services/recipe-persistence-uncertainty"
import {
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PromptAssistComposerAction } from "../PromptAssistComposerAction"
import { useComposerText } from "../hooks/useComposerText"

const ownerA = "recipe-owner:sha256:" + "a".repeat(64)
const ownerB = "recipe-owner:sha256:" + "b".repeat(64)
const revisionA = "recipe-authorization:sha256:" + "a".repeat(64)

const mocks = vi.hoisted(() => ({
  realCapabilities: false,
  runtimeId: undefined as string | undefined,
  sendMessage: vi.fn(),
  updatePrompt: vi.fn(),
  markPromptSyncError: vi.fn(),
  autoSyncPrompt: vi.fn(),
  shouldAutoSyncWorkspacePrompts: vi.fn(),
  fetchPromptCapabilities: vi.fn(),
  getAllPrompts: vi.fn(),
  improvePrompt: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return mocks.runtimeId
      },
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
    }
  }
}))

const drawerLifecycleMocks = vi.hoisted(() => ({
  deferClose: false,
  completeClose: null as null | (() => void)
}))

const draftBucketMocks = vi.hoisted(() => {
  const records = new Map<string, { value: string; updatedAt: number }>()
  return {
    records,
    get: vi.fn(async (key: string) => records.get(key) ?? null),
    set: vi.fn(async (key: string, value: string) => {
      records.set(key, { value, updatedAt: Date.now() })
    }),
    remove: vi.fn(async (key: string) => {
      records.delete(key)
    }),
    cleanup: vi.fn(async () => 0)
  }
})

vi.mock("@/services/settings/local-bucket", () => ({
  createLocalRegistryBucket: () => ({
    get: draftBucketMocks.get,
    set: draftBucketMocks.set,
    remove: draftBucketMocks.remove,
    cleanup: draftBucketMocks.cleanup,
    buildKey: (key: string) => `registry:draft:${key}`
  })
}))

vi.mock("@/services/prompt-sync", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/prompt-sync")>()),
  autoSyncPrompt: (...args: unknown[]) => mocks.autoSyncPrompt(...args),
  shouldAutoSyncWorkspacePrompts: () => mocks.shouldAutoSyncWorkspacePrompts()
}))

vi.mock("@/services/prompts-api", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/services/prompts-api")>()
  return {
    ...original,
    revalidatePromptCapabilities: () =>
      mocks.realCapabilities
        ? original.revalidatePromptCapabilities()
        : mocks.fetchPromptCapabilities(),
    fetchPromptCapabilities: (...args: unknown[]) =>
      mocks.realCapabilities
        ? original.fetchPromptCapabilities()
        : mocks.fetchPromptCapabilities(...args)
  }
})

vi.mock("@/db/dexie/helpers", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/db/dexie/helpers")>()
  return {
    ...original,
    getAllPrompts: (...args: unknown[]) => mocks.getAllPrompts(...args),
    updatePrompt: (...args: unknown[]) => mocks.updatePrompt(...args),
    markPromptSyncError: (...args: unknown[]) =>
      mocks.markPromptSyncError(...args)
  }
})

vi.mock("@/services/prompt-improvement", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/services/prompt-improvement")>()
  return {
    ...original,
    improvePrompt: (...args: unknown[]) => mocks.improvePrompt(...args)
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValue?: string,
      options?: Record<string, string | number>
    ) =>
      (defaultValue ?? _key).replace(/{{(\w+)}}/g, (_, key) =>
        String(options?.[key] ?? "")
      )
  })
}))

vi.mock("antd", async (importOriginal) => {
  const original = await importOriginal<typeof import("antd")>()
  return {
    ...original,
    Drawer: ({
      open,
      title,
      children,
      onClose,
      size,
      width,
      afterOpenChange,
      focusable
    }: {
      open?: boolean
      title?: React.ReactNode
      children?: React.ReactNode
      onClose?: () => void
      size?: number | string
      width?: number | string
      afterOpenChange?: (open: boolean) => void
      focusable?: { focusTriggerAfterClose?: boolean }
    }) => {
      const wasOpenRef = React.useRef(false)
      React.useEffect(() => {
        if (open) {
          wasOpenRef.current = true
          afterOpenChange?.(true)
          return
        }
        if (!wasOpenRef.current) return
        wasOpenRef.current = false
        const completeClose = () => {
          afterOpenChange?.(false)
          if (focusable?.focusTriggerAfterClose !== false) {
            document.body.tabIndex = -1
            document.body.focus()
          }
        }
        if (drawerLifecycleMocks.deferClose) {
          drawerLifecycleMocks.completeClose = completeClose
        } else {
          completeClose()
        }
      }, [afterOpenChange, focusable?.focusTriggerAfterClose, open])

      return open ? (
        <aside
          role="dialog"
          aria-label={String(title)}
          data-drawer-size={String(size)}
          data-drawer-width={String(width)}
          data-focus-trigger-after-close={String(
            focusable?.focusTriggerAfterClose
          )}
          onKeyDown={(event) => {
            if (event.key === "Escape") onClose?.()
          }}>
          <button
            type="button"
            aria-label="Close prompt improvement drawer"
            onClick={onClose}
          />
          <button
            type="button"
            aria-label="Prompt improvement drawer backdrop"
            onClick={onClose}
          />
          {children}
        </aside>
      ) : null
    }
  }
})

const availableCapabilities = {
  availability: "available" as const,
  prompt_improvement_v1: { supported: true, limits: null },
  single_text_recipe_v2: { supported: false }
}

const improvementResponse = (operationId: string) => ({
  schema_version: 1 as const,
  operation_id: operationId,
  status: "improved" as const,
  improved_text: "Improved user draft",
  findings: [],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  meta_prompt_version: "prompt-improvement-v1"
})

type HarnessProps = {
  initialDraft?: string
  modelSelection?: PromptImproveModelSelection | null
  promptAssistContextKey?: string
  promptAssistBackendKey?: string | null
  promptAssistAuthorizationRevision?: string | null
  sending?: boolean
  surfaceOpen?: boolean
  narrow?: boolean
  onSubmit?: () => void
  onSelectModel?: () => void
  draftEnabled?: boolean
  strictMode?: boolean
  collisionGeometry?: boolean
}

function Harness({
  initialDraft = "Original user draft",
  modelSelection = {
    selected_model: "openai/gpt-5-mini",
    provider_hint: "openai"
  },
  promptAssistContextKey = "conversation-1",
  promptAssistBackendKey = "backend-a",
  promptAssistAuthorizationRevision = "authorization-one",
  sending = false,
  surfaceOpen = true,
  narrow = false,
  onSubmit,
  onSelectModel,
  draftEnabled = false,
  collisionGeometry = false
}: HarnessProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const initializedRef = React.useRef(false)
  const [sendPending, setSendPending] = React.useState(false)
  const activeAttemptRef = React.useRef<number | null>(null)
  const queuedAttemptRef = React.useRef<number | null>(null)
  const composer = useComposerText({
    draftKey: "tldw:test:prompt-assist-composer",
    textareaRef,
    draftEnabled
  })

  React.useLayoutEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true
    composer.form.setFieldValue("message", initialDraft)
  }, [composer.form, initialDraft])

  return (
    <form
      data-prompt-assist-collision-surface={collisionGeometry ? "" : undefined}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit?.()
      }}>
      <textarea
        ref={textareaRef}
        aria-label="User draft"
        {...composer.form.getInputProps("message")}
      />
      <output aria-label="Committed user draft">
        {composer.form.values.message}
      </output>
      <button
        type="button"
        onClick={() => {
          setSendPending(true)
          activeAttemptRef.current = composer.beginPromptAssistReset()
        }}>
        Begin send
      </button>
      <button
        type="button"
        onClick={() => {
          activeAttemptRef.current = composer.beginPromptAssistReset()
        }}>
        Reset before pending
      </button>
      <button type="button" onClick={() => setSendPending(true)}>
        Mark send pending
      </button>
      <button
        type="button"
        onClick={() => {
          composer.clearDraft()
          if (activeAttemptRef.current !== null) {
            composer.markPromptAssistAttemptSaved(activeAttemptRef.current)
          }
          setSendPending(false)
        }}>
        Finish successful send
      </button>
      <button type="button" onClick={() => setSendPending(false)}>
        Finish failed send
      </button>
      <button type="button" onClick={() => setSendPending(false)}>
        Reject send
      </button>
      <button
        type="button"
        onClick={() => {
          if (queuedAttemptRef.current === null) {
            queuedAttemptRef.current =
              activeAttemptRef.current ?? composer.beginPromptAssistReset()
          }
          composer.markPromptAssistAttemptSaved(queuedAttemptRef.current)
        }}>
        Finish queued send
      </button>
      {collisionGeometry ? (
        <button type="button" aria-label="Intermediate composer control">
          Intermediate composer control
        </button>
      ) : null}
      <div
        data-testid={
          collisionGeometry ? "sidepanel-send-action-cluster" : undefined
        }>
        <PromptAssistComposerAction
          form={composer.form}
          messageRevision={composer.messageRevision}
          promptAssistMutation={composer.promptAssistMutation}
          promptAssistSavedAttemptId={composer.promptAssistSavedAttemptId}
          modelSelection={modelSelection}
          promptAssistContextKey={promptAssistContextKey}
          promptAssistBackendKey={promptAssistBackendKey}
          promptAssistAuthorizationRevision={promptAssistAuthorizationRevision}
          sending={sending || sendPending}
          surfaceOpen={surfaceOpen}
          narrow={narrow}
          onSelectModel={onSelectModel}
          onReturnFocus={composer.textAreaFocus}
        />
      </div>
    </form>
  )
}

const renderHarness = (props: HarnessProps = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const renderTree = (nextProps: HarnessProps) => {
    const tree = (
      <QueryClientProvider client={queryClient}>
        <Harness {...nextProps} />
      </QueryClientProvider>
    )
    return nextProps.strictMode ? (
      <React.StrictMode>{tree}</React.StrictMode>
    ) : (
      tree
    )
  }
  const view = render(renderTree(props))
  return {
    ...view,
    queryClient,
    rerenderHarness: (nextProps: HarnessProps) =>
      view.rerender(renderTree(nextProps))
  }
}

const openActions = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: "Improve prompt" }))
}

const improveNow = async (user: ReturnType<typeof userEvent.setup>) => {
  await openActions(user)
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeEnabled()
  )
  await user.click(screen.getByRole("button", { name: /Improve now/ }))
  await screen.findByRole("button", { name: "Undo improvement" })
}

describe("PromptAssistComposerAction entry and request contract", () => {
  afterEach(async () => {
    mocks.runtimeId = undefined
    await clearRecipePersistenceScoped("scoped-recipe", ownerA)
    clearRuntimeAuthOverride()
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(false)
    vi.spyOn(
      recipeAuthority,
      "resolveRecipePersistenceOwnerView"
    ).mockResolvedValue({
      ownerId: ownerA,
      authorizationRevision: revisionA
    })
    draftBucketMocks.records.clear()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
    mocks.getAllPrompts.mockResolvedValue([])
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  it("uses the compact upward composer trigger", async () => {
    const user = userEvent.setup()
    renderHarness()

    const trigger = screen.getByRole("button", { name: "Improve prompt" })
    expect(trigger).toHaveAttribute("title", "Improve prompt")
    expect(trigger).toHaveClass("h-11", "w-11")
    expect(trigger).not.toHaveTextContent("Improve my prompt")

    await user.click(trigger)
    const actions = screen.getByRole("group", {
      name: "Prompt improvement actions"
    })
    expect(actions).toHaveClass("fixed")
    expect(actions.parentElement).toBe(document.body)
  })

  it("keeps local recipe work usable when the real extension owner facade rejects", async () => {
    vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
    vi.mocked(recipeAuthority.resolveRecipePersistenceOwnerView).mockRestore()
    mocks.runtimeId = "adapter-extension"
    const pending = createDeferred<void>()
    mocks.sendMessage.mockImplementation(async () => {
      await pending.promise
      throw new Error("extension authority disconnected")
    })
    const user = userEvent.setup()
    const view = renderHarness({ promptAssistBackendKey: null })

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await user.type(
      await screen.findByLabelText("Current value for Task (not saved)"),
      "Keep local work"
    )
    const preview = (
      screen.getByLabelText("Compiled prompt preview") as HTMLTextAreaElement
    ).value
    expect(preview).toContain("Keep local work")
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Apply to user message" })
    ).toBeEnabled()
    await act(async () => {
      pending.resolve()
      await pending.promise
    })
    expect(mocks.sendMessage).toHaveBeenCalledExactlyOnceWith({
      type: "tldw:recipe-owner:resolve"
    })
    expect(mocks.fetchPromptCapabilities).not.toHaveBeenCalled()
    expect(
      view.queryClient
        .getQueryCache()
        .getAll()
        .some((query) => String(query.queryKey[1]).startsWith("recipe-owner:"))
    ).toBe(false)
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" })
    )
    expect(screen.getByLabelText("User draft")).toHaveValue(preview)
  })

  it.each(["same-owner reopen", "owner replacement"])(
    "revalidates through real prompts-api/api-send for %s",
    async (scenario) => {
      vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
      mocks.realCapabilities = true
      mocks.runtimeId = "adapter-extension"
      const old = createDeferred<unknown>()
      const fresh = createDeferred<unknown>()
      const capabilities = (authorized: boolean) => ({
        ok: true,
        status: 200,
        data: {
          prompt_improvement_v1: {
            supported: true,
            limits: {
              max_request_bytes: 100000,
              max_draft_chars: 10000,
              max_candidate_chars: 10000,
              max_raw_output_chars: 10000,
              max_findings: 10,
              max_finding_text_chars: 1000,
              max_provider_chars: 100,
              max_model_chars: 100,
              max_meta_prompt_version_chars: 100,
              max_warning_chars: 1000,
              max_warnings: 10,
              max_protected_tokens: 100,
              max_protected_token_kind_chars: 100,
              max_protected_token_chars: 1000,
              max_protected_token_occurrences: 100,
              max_protected_token_total_chars: 10000
            }
          },
          single_text_recipe_v2: { supported: true },
          prompt_persistence: {
            create_authorized: authorized,
            update_authorized: authorized
          }
        }
      })
      mocks.sendMessage
        .mockReturnValueOnce(old.promise)
        .mockReturnValueOnce(fresh.promise)
      const user = userEvent.setup()
      const view = renderHarness({ promptAssistBackendKey: null })

      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      await user.click(
        screen.getByRole("button", { name: /Build from recipe/ })
      )
      await waitFor(() => expect(mocks.sendMessage).toHaveBeenCalledTimes(1))
      await user.click(await screen.findByRole("button", { name: "Back" }))
      const nextOwner = scenario === "owner replacement" ? ownerB : ownerA
      vi.mocked(
        recipeAuthority.resolveRecipePersistenceOwnerView
      ).mockResolvedValue({
        ownerId: nextOwner,
        authorizationRevision: revisionA
      })
      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      await user.click(
        screen.getByRole("button", { name: /Build from recipe/ })
      )
      try {
        await waitFor(() => expect(mocks.sendMessage).toHaveBeenCalledTimes(2))
        expect(
          mocks.sendMessage.mock.calls.map(([message]) => message)
        ).toEqual([
          {
            type: "tldw:request",
            payload: { path: "/api/v1/prompts/capabilities", method: "GET" }
          },
          {
            type: "tldw:request",
            payload: { path: "/api/v1/prompts/capabilities", method: "GET" }
          }
        ])
        await act(async () => {
          old.resolve(capabilities(true))
          await old.promise
        })
        expect(
          screen.getByRole("button", { name: "Save as new recipe" })
        ).toBeDisabled()
        await act(async () => {
          fresh.resolve(capabilities(false))
          await fresh.promise
        })
        await waitFor(() => expect(view.queryClient.isFetching()).toBe(0))
        expect(
          view.queryClient.getQueryData([
            "promptCapabilities",
            nextOwner,
            revisionA
          ])
        ).toMatchObject({
          prompt_persistence: {
            create_authorized: false,
            update_authorized: false
          }
        })
        expect(
          screen.getByRole("button", { name: "Save as new recipe" })
        ).toBeDisabled()
      } finally {
        await act(async () => {
          old.resolve(capabilities(true))
          fresh.resolve(capabilities(false))
          await Promise.all([old.promise, fresh.promise])
        })
      }
    }
  )

  it.each([
    ["advanced normalized base", true],
    ["runtime-key override", false],
    ["manual/cookie source", false],
    ["same-sub bearer rotation", true],
    ["principal replacement", false],
    ["missing cookie principal", false]
  ])("reopens with fresh authority for %s", async (scenario, sameOwner) => {
    vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
    let config: Parameters<
      typeof recipeAuthority.resolveRecipeOwnerWithConfig
    >[0] extends () => Promise<infer C>
      ? C
      : never = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "manual-adapter-key"
    }
    if (
      scenario === "same-sub bearer rotation" ||
      scenario === "principal replacement"
    ) {
      config = { ...config, authMode: "multi-user", accessToken: "bearer-one" }
    }
    let principal = "authoritative-user"
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response(JSON.stringify({ id: principal }), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        })
    )
    vi.mocked(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).mockImplementation(() =>
      recipeAuthority.resolveRecipeOwnerWithConfig(async () => config)
    )
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits: null },
      single_text_recipe_v2: { supported: true },
      prompt_persistence: { create_authorized: true, update_authorized: true }
    })
    const user = userEvent.setup()
    const view = renderHarness({ promptAssistBackendKey: null })
    await openActions(user)
    expect(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    const firstKey = view.queryClient
      .getQueryCache()
      .getAll()
      .find(
        (query) =>
          query.queryKey[0] === "promptCapabilities" &&
          String(query.queryKey[1]).startsWith("recipe-owner:")
      )!.queryKey
    expect(firstKey).toHaveLength(3)
    expect(firstKey[2]).toMatch(/^recipe-authorization:sha256:[a-f0-9]{64}$/)
    await user.click(screen.getByRole("button", { name: "Back" }))
    if (scenario === "advanced normalized base")
      config = { ...config, serverUrl: "HTTP://LOCALHOST:3000///" }
    if (scenario === "runtime-key override")
      setRuntimeSingleUserApiKeyOverride("runtime-adapter-key")
    if (scenario === "same-sub bearer rotation")
      config = { ...config, accessToken: "bearer-two" }
    if (scenario === "principal replacement") principal = "replacement-user"
    if (
      scenario === "manual/cookie source" ||
      scenario === "missing cookie principal"
    ) {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      config = {
        serverUrl: "http://localhost:3000",
        authMode: "single-user",
        authSource: "cookie-session"
      }
    }
    if (scenario === "missing cookie principal")
      vi.stubGlobal("fetch", async () => new Response("{}", { status: 200 }))
    const pending =
      createDeferred<
        Awaited<
          ReturnType<typeof recipeAuthority.resolveRecipePersistenceOwnerView>
        >
      >()
    const resolved = await recipeAuthority.resolveRecipeOwnerWithConfig(
      async () => config
    )
    vi.mocked(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).mockReturnValueOnce(pending.promise)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    expect(
      await screen.findByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
    if (scenario === "principal replacement") {
      await user.type(
        screen.getByLabelText("Current value for Task (not saved)"),
        "Local during principal change"
      )
      expect(
        (
          screen.getByLabelText(
            "Compiled prompt preview"
          ) as HTMLTextAreaElement
        ).value
      ).toContain("Local during principal change")
      expect(
        screen.getByRole("button", { name: "Apply to user message" })
      ).toBeEnabled()
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeDisabled()
    }
    await act(async () => {
      pending.resolve(resolved)
      await pending.promise
    })
    if (scenario === "missing cookie principal") {
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeDisabled()
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
      await user.type(
        screen.getByLabelText("Current value for Task (not saved)"),
        "Still local"
      )
      expect(
        screen.getByRole("button", { name: "Apply to user message" })
      ).toBeEnabled()
    } else {
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Save as new recipe" })
        ).toBeEnabled()
      )
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
      expect(
        view.queryClient.getQueryData([
          "promptCapabilities",
          resolved!.ownerId,
          resolved!.authorizationRevision
        ])
      ).toBeDefined()
      expect(resolved!.ownerId === firstKey[1]).toBe(sameOwner)
      if (scenario === "same-sub bearer rotation")
        expect(resolved!.authorizationRevision).not.toBe(firstKey[2])
    }
  })

  it("does not revive cached recipe authorization after a failed reopen refetch", async () => {
    vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits: null },
      single_text_recipe_v2: { supported: true },
      prompt_persistence: { create_authorized: true, update_authorized: true }
    })
    const user = userEvent.setup()
    renderHarness({ promptAssistBackendKey: null })
    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: "Back" }))
    mocks.fetchPromptCapabilities.mockRejectedValueOnce(
      new Error("authorization unavailable")
    )
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    )
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
  })

  it("ignores an earlier open's pending authorized response after reopening", async () => {
    vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
    const old = createDeferred<unknown>()
    const fresh = createDeferred<unknown>()
    mocks.fetchPromptCapabilities
      .mockReturnValueOnce(old.promise)
      .mockReturnValueOnce(fresh.promise)
    const user = userEvent.setup()
    renderHarness({ promptAssistBackendKey: null })

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
    )
    await user.click(await screen.findByRole("button", { name: "Back" }))
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    )
    await act(async () => {
      old.resolve({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: {
          create_authorized: true,
          update_authorized: true
        }
      })
      await old.promise
    })
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    await act(async () => {
      fresh.resolve({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: {
          create_authorized: false,
          update_authorized: false
        }
      })
      await fresh.promise
    })
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
  })

  it("keeps the authoritative owner locked on reopen after durable marker storage fails", async () => {
    vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
    const definition = structuredClone(CLEAR_TASK_RECIPE.definition)
    definition.assembly_config.target_role = "user"
    definition.blocks.forEach((block) => {
      block.role = "user"
    })
    mocks.getAllPrompts.mockResolvedValue([
      {
        id: "scoped-recipe",
        name: "Scoped recipe",
        title: "Scoped recipe",
        content: "Template body",
        is_system: false,
        createdAt: 1,
        promptFormat: "structured",
        promptSchemaVersion: 2,
        syncStatus: "local",
        structuredPromptDefinition: definition
      }
    ])
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits: null },
      single_text_recipe_v2: { supported: true },
      prompt_persistence: { create_authorized: true, update_authorized: true }
    })
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(true)
    mocks.updatePrompt.mockResolvedValue("scoped-recipe")
    mocks.autoSyncPrompt.mockResolvedValue({
      success: false,
      failureKind: "invalid_server_payload",
      localId: "scoped-recipe",
      recipeOwnership: {
        localId: "scoped-recipe",
        dispatch: { state: "dispatched", actualOwnerId: ownerA }
      }
    })
    mocks.markPromptSyncError.mockRejectedValue(
      new Error("durable storage unavailable")
    )
    const user = userEvent.setup()
    renderHarness({ promptAssistBackendKey: null })

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await screen.findByRole("option", { name: "Scoped recipe" })
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recipe source" }),
      "saved:scoped-recipe"
    )
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Update recipe" })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: "Update recipe" }))
    await waitFor(() =>
      expect(mocks.markPromptSyncError).toHaveBeenCalledWith("scoped-recipe")
    )
    expect(
      await recipeAuthority.readRecipePersistenceUncertainty(
        "scoped-recipe",
        ownerA
      )
    ).toBe("scoped")
    await user.click(screen.getByRole("button", { name: "Back" }))
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await screen.findByRole("option", { name: "Scoped recipe" })
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Recipe source" }),
      "saved:scoped-recipe"
    )
    expect(screen.getByRole("button", { name: "Update recipe" })).toBeDisabled()
    expect(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).toHaveBeenCalledTimes(2)
  })

  it.each([
    [ownerA, "first-credential", false],
    [ownerA, "refreshed-credential", false],
    [ownerB, "first-credential", true]
  ])(
    "scopes composer recipe ownership to %s, independently of %s",
    async (backend, revision, enabled) => {
      vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
      await markRecipePersistenceScoped("scoped-recipe", ownerA)
      vi.mocked(
        recipeAuthority.resolveRecipePersistenceOwnerView
      ).mockResolvedValue({
        ownerId: backend,
        authorizationRevision: revision
      })
      const definition = structuredClone(CLEAR_TASK_RECIPE.definition)
      definition.assembly_config.target_role = "user"
      definition.blocks.forEach((block) => {
        block.role = "user"
      })
      mocks.getAllPrompts.mockResolvedValue([
        {
          id: "scoped-recipe",
          name: "Scoped recipe",
          title: "Scoped recipe",
          content: "Task",
          is_system: false,
          createdAt: 1,
          promptFormat: "structured",
          promptSchemaVersion: 2,
          syncStatus: "local",
          structuredPromptDefinition: definition
        }
      ])
      mocks.fetchPromptCapabilities.mockResolvedValue({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: {
          create_authorized: true,
          update_authorized: true
        }
      })
      const user = userEvent.setup()
      const view = renderHarness({
        promptAssistBackendKey: "unrelated-legacy-scope"
      })
      await openActions(user)
      await user.click(
        screen.getByRole("button", { name: /Build from recipe/ })
      )
      await screen.findByRole("option", { name: "Scoped recipe" })
      await user.selectOptions(
        screen.getByRole("combobox", { name: "Recipe source" }),
        "saved:scoped-recipe"
      )
      await waitFor(() => expect(view.queryClient.isFetching()).toBe(0))
      expect(
        view.queryClient.getQueryData(["promptCapabilities", backend, revision])
      ).toBeDefined()
      const update = screen.getByRole("button", { name: "Update recipe" })
      if (enabled) expect(update).toBeEnabled()
      else expect(update).toBeDisabled()
    }
  )

  it("builds locally without a model and restores an exact whitespace Unicode draft", async () => {
    const user = userEvent.setup()
    const original = "  Draft 🧪\n\n"
    renderHarness({ initialDraft: original, modelSelection: null })

    await openActions(user)
    const build = screen.getByRole("button", { name: /Build from recipe/ })
    expect(build).toBeEnabled()
    await user.click(build)
    expect(
      await screen.findByRole("dialog", { name: "Build from recipe" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Draft 🧪"
    )

    await user.type(
      await screen.findByLabelText("Current value for Task (not saved)"),
      "Explain Unicode safely."
    )
    const preview = (
      screen.getByLabelText("Compiled prompt preview") as HTMLTextAreaElement
    ).value
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" })
    )

    expect(screen.getByLabelText("User draft")).toHaveValue(preview)
    expect(mocks.improvePrompt).not.toHaveBeenCalled()
    expect(screen.getByText("Recipe applied.").parentElement).toHaveClass(
      "fixed"
    )
    expect(
      screen.getByText("Recipe applied.").parentElement?.parentElement
    ).toBe(document.body)
    await user.click(screen.getByRole("button", { name: "Undo recipe" }))
    expect(
      (screen.getByLabelText("User draft") as HTMLTextAreaElement).value
    ).toBe(original)
  })

  it("keeps persistence unknown without an authoritative backend identity", async () => {
    const user = userEvent.setup()
    vi.mocked(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).mockResolvedValue(null)
    renderHarness({ promptAssistBackendKey: null })

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))

    expect(
      await screen.findByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).not.toHaveBeenCalled()
  })

  it("gates cached recipe authorization while open-time revalidation is pending and revoked", async () => {
    const revoked = createDeferred<{
      availability: "available"
      prompt_improvement_v1: { supported: true; limits: null }
      single_text_recipe_v2: { supported: true }
      prompt_persistence: {
        create_authorized: false
        update_authorized: false
      }
    }>()
    mocks.fetchPromptCapabilities
      .mockResolvedValueOnce({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: {
          create_authorized: true,
          update_authorized: true
        }
      })
      .mockReturnValueOnce(revoked.promise)
    const user = userEvent.setup()
    renderHarness()
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
    )

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    )
    expect(
      await screen.findByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()

    await act(async () => {
      revoked.resolve({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: {
          create_authorized: false,
          update_authorized: false
        }
      })
      await revoked.promise
    })
    expect(
      screen.getByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
  })

  it("isolates capability cache when authorization changes within one backend scope", async () => {
    const nextCapabilities = createDeferred<typeof availableCapabilities>()
    mocks.fetchPromptCapabilities
      .mockResolvedValueOnce(availableCapabilities)
      .mockReturnValueOnce(nextCapabilities.promise)
    const user = userEvent.setup()
    const view = renderHarness({
      promptAssistBackendKey: "stable-backend",
      promptAssistAuthorizationRevision: "authorization-one"
    })
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
    )
    await openActions(user)
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeEnabled()

    view.rerenderHarness({
      promptAssistBackendKey: "stable-backend",
      promptAssistAuthorizationRevision: "authorization-two"
    })

    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    )
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
  })

  it("discards unapplied runtime values when the recipe builder closes and reopens", async () => {
    const user = userEvent.setup()
    renderHarness({ initialDraft: "" })

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    const runtime = await screen.findByLabelText(
      "Current value for Task (not saved)"
    )
    await user.type(runtime, "Temporary runtime")
    expect(
      (screen.getByLabelText("User draft") as HTMLTextAreaElement).value
    ).toBe("")
    await user.click(
      screen.getByRole("button", { name: "Close prompt improvement drawer" })
    )

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    expect(
      screen.getByLabelText("Current value for Task (not saved)")
    ).toHaveValue("")
    expect(
      (screen.getByLabelText("User draft") as HTMLTextAreaElement).value
    ).toBe("")
  })

  it("closes recipe mode on Escape from a recipe input and restores draft, runtime, and trigger focus", async () => {
    const user = userEvent.setup()
    const original = "  Exact draft 🧪\n\n"
    renderHarness({ initialDraft: original })

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    const runtime = await screen.findByLabelText(
      "Current value for Task (not saved)"
    )
    await user.type(runtime, "Temporary runtime")
    expect(runtime).toHaveFocus()
    await user.keyboard("{Escape}")

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Build from recipe" })
      ).not.toBeInTheDocument()
    )
    expect(screen.getByLabelText("User draft")).toHaveValue(original)
    expect(screen.getByRole("button", { name: "Improve prompt" })).toHaveFocus()

    await openActions(user)
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    expect(
      await screen.findByLabelText("Current value for Task (not saved)")
    ).toHaveValue("")
    expect(screen.getByLabelText("User draft")).toHaveValue(original)
  })

  it("disables both actions for a whitespace-only user draft", async () => {
    const user = userEvent.setup()
    renderHarness({ initialDraft: "   " })

    await openActions(user)

    expect(
      await screen.findByText("Write a draft to enable prompt improvement.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: /Review changes/ })
    ).toBeDisabled()
  })

  it("disables both actions and offers recovery when the model is missing", async () => {
    const onSelectModel = vi.fn()
    const user = userEvent.setup()
    renderHarness({ modelSelection: null, onSelectModel })

    await openActions(user)

    expect(
      await screen.findByText("Select a chat model to improve this draft.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: /Review changes/ })
    ).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Select model" }))
    expect(onSelectModel).toHaveBeenCalledTimes(1)
  })

  it("fails closed when the backend capability is unsupported", async () => {
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "unavailable",
      prompt_improvement_v1: { supported: false, limits: null },
      single_text_recipe_v2: { supported: false }
    })
    const user = userEvent.setup()
    renderHarness()

    await openActions(user)

    expect(
      await screen.findByText(
        "Prompt improvement requires a newer server version."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
  })

  it("disables the entry point while a message is sending", () => {
    renderHarness({ sending: true })

    expect(
      screen.getByRole("button", { name: "Improve prompt" })
    ).toBeDisabled()
  })

  it("sends only the independent user draft and active route in Improve now mode", async () => {
    const user = userEvent.setup()
    renderHarness({ initialDraft: "Independent user-only draft" })

    await openActions(user)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Improve now/ })).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Improve now/ }))

    expect(
      await screen.findByText("Improved user draft", {
        selector: "output"
      })
    ).toBeInTheDocument()
    expect(mocks.improvePrompt).toHaveBeenCalledTimes(1)
    expect(mocks.improvePrompt.mock.calls[0]?.[0]).toMatchObject({
      target: "user_message",
      text: "Independent user-only draft",
      model_selection: {
        selected_model: "openai/gpt-5-mini",
        provider_hint: "openai"
      },
      protected_tokens: []
    })
    expect(mocks.improvePrompt.mock.calls[0]?.[0]).not.toHaveProperty(
      "messages"
    )
    expect(mocks.improvePrompt.mock.calls[0]?.[0]).not.toHaveProperty(
      "attachments"
    )
  })

  it("keeps Review changes in the shared panel without mutating the user draft", async () => {
    const user = userEvent.setup()
    renderHarness()

    await openActions(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    expect(
      await screen.findByRole("dialog", { name: "Prompt improvement" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("textbox", { name: "Improved prompt candidate" })
    ).toHaveValue("Improved user draft")
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Original user draft"
    )
    expect(mocks.improvePrompt.mock.calls[0]?.[0]).toMatchObject({
      target: "user_message",
      text: "Original user draft"
    })
  })
})

type Deferred<T> = {
  promise: Promise<T>
  resolve: (value: T) => void
}

const createDeferred = <T,>(): Deferred<T> => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

const resolveDeferred = async <T,>(deferred: Deferred<T>, value: T) => {
  await act(async () => {
    deferred.resolve(value)
    await deferred.promise
    await Promise.resolve()
  })
}

const requestedOperationId = () =>
  mocks.improvePrompt.mock.calls.at(-1)?.[0].operation_id as string

beforeEach(() => {
  mocks.realCapabilities = false
  mocks.runtimeId = undefined
  mocks.sendMessage.mockReset()
  drawerLifecycleMocks.deferClose = false
  drawerLifecycleMocks.completeClose = null
})

describe("PromptAssistComposerAction deferred lifecycle ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    draftBucketMocks.records.clear()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
  })

  it("never overwrites typing committed while Improve now is in flight", async () => {
    const deferred = createDeferred<ReturnType<typeof improvementResponse>>()
    mocks.improvePrompt.mockReturnValue(deferred.promise)
    const user = userEvent.setup()
    renderHarness()

    await openActions(user)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Improve now/ })).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Improve now/ }))
    await user.clear(screen.getByRole("textbox", { name: "User draft" }))
    await user.type(
      screen.getByRole("textbox", { name: "User draft" }),
      "Edited while analyzing"
    )

    await resolveDeferred(deferred, improvementResponse(requestedOperationId()))

    expect(
      await screen.findByText("Edited while analyzing", { selector: "output" })
    ).toBeInTheDocument()
    expect(
      await screen.findByText(
        "The draft changed while this result was open. Applying normally will not overwrite it."
      )
    ).toBeInTheDocument()
  })

  it.each(["improve", "review"] as const)(
    "discards a deferred %s response after route, backend, or context ownership changes",
    async (mode) => {
      const deferred = createDeferred<ReturnType<typeof improvementResponse>>()
      mocks.improvePrompt.mockReturnValue(deferred.promise)
      const user = userEvent.setup()
      const { rerenderHarness } = renderHarness()

      await openActions(user)
      const actionName = mode === "improve" ? /Improve now/ : /Review changes/
      await waitFor(() =>
        expect(screen.getByRole("button", { name: actionName })).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: actionName }))

      rerenderHarness({
        modelSelection: {
          selected_model: "anthropic/claude-sonnet-4",
          provider_hint: "anthropic"
        },
        promptAssistBackendKey: "backend-b",
        promptAssistContextKey: "conversation-2"
      })
      await resolveDeferred(
        deferred,
        improvementResponse(requestedOperationId())
      )

      await waitFor(() => {
        expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
          "Original user draft"
        )
        expect(
          screen.queryByRole("dialog", { name: "Prompt improvement" })
        ).not.toBeInTheDocument()
      })
    }
  )

  it.each(["Cancel", "owner close"] as const)(
    "%s ignores a late provider completion",
    async (closeMethod) => {
      const deferred = createDeferred<ReturnType<typeof improvementResponse>>()
      mocks.improvePrompt.mockReturnValue(deferred.promise)
      const user = userEvent.setup()
      renderHarness()

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Improve now/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Improve now/ }))
      await screen.findByRole("dialog", { name: "Prompt improvement" })
      await user.click(
        closeMethod === "Cancel"
          ? screen.getByRole("button", { name: "Cancel" })
          : screen.getByRole("button", {
              name: "Close prompt improvement drawer"
            })
      )

      await resolveDeferred(
        deferred,
        improvementResponse(requestedOperationId())
      )

      await waitFor(() => {
        expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
          "Original user draft"
        )
        expect(
          screen.queryByRole("dialog", { name: "Prompt improvement" })
        ).not.toBeInTheDocument()
      })
    }
  )
})

describe("PromptAssistComposerAction review application", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  it("edits the review candidate and applies only that candidate to the owner", async () => {
    const user = userEvent.setup()
    renderHarness()

    await openActions(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    const candidate = await screen.findByRole("textbox", {
      name: "Improved prompt candidate"
    })
    await user.clear(candidate)
    await user.type(candidate, "Edited review candidate")
    await user.click(screen.getByRole("button", { name: "Apply to draft" }))

    expect(
      await screen.findByText("Edited review candidate", {
        selector: "output"
      })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("dialog", { name: "Prompt improvement" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
    expect(screen.getByText("Improvement applied.").parentElement).toHaveClass(
      "fixed"
    )
    expect(
      screen.getByText("Improvement applied.").parentElement?.parentElement
    ).toBe(document.body)
  })

  it("requires confirmation before replacing a draft edited after review began", async () => {
    const user = userEvent.setup()
    renderHarness()

    await openActions(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    const candidate = await screen.findByRole("textbox", {
      name: "Improved prompt candidate"
    })
    await user.clear(candidate)
    await user.type(candidate, "Confirmed replacement candidate")

    const liveDraft = screen.getByRole("textbox", { name: "User draft" })
    await user.clear(liveDraft)
    await user.type(liveDraft, "Newer live user draft")
    await user.click(screen.getByRole("button", { name: "Apply to draft" }))

    expect(
      screen.getByRole("button", { name: "Replace current draft" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Newer live user draft"
    )
    await user.click(
      screen.getByRole("button", { name: "Replace current draft" })
    )
    await user.click(screen.getByRole("button", { name: "Confirm replace" }))

    expect(
      await screen.findByText("Confirmed replacement candidate", {
        selector: "output"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
  })
})

describe("PromptAssistComposerAction feedback collision ownership", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("hides feedback when an intermediate composer control consumes the only gap, then restores the same Undo", async () => {
    const makeRect = (x: number, y: number, width: number, height: number) =>
      ({
        x,
        y,
        width,
        height,
        top: y,
        right: x + width,
        bottom: y + height,
        left: x,
        toJSON: () => ({ x, y, width, height })
      }) as DOMRect
    let viewportHeight = 240
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(360)
    vi.spyOn(window, "innerHeight", "get").mockImplementation(
      () => viewportHeight
    )
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      function () {
        if (this.classList.contains("fixed")) {
          return makeRect(8, 0, 344, 82)
        }
        if (this.getAttribute("aria-label") === "User draft") {
          return makeRect(8, 0, 344, 60)
        }
        if (
          this.getAttribute("aria-label") === "Intermediate composer control"
        ) {
          return makeRect(117, 89, 44, 44)
        }
        if (this.dataset.testid === "sidepanel-send-action-cluster") {
          return makeRect(250, 160, 102, 44)
        }
        if (this.getAttribute("aria-label") === "Improve prompt") {
          return makeRect(250, 160, 44, 44)
        }
        return makeRect(0, 0, 0, 0)
      }
    )

    const user = userEvent.setup()
    renderHarness({
      initialDraft: "Exact draft before collision",
      collisionGeometry: true
    })
    await openActions(user)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Improve now/ })).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Improve now/ }))
    await screen.findByText("Improved user draft", { selector: "output" })

    const feedback = screen.getByText("Improvement applied.")
      .parentElement as HTMLElement
    await waitFor(() => expect(feedback.style.visibility).toBe("hidden"))
    expect(screen.getByText("Undo improvement")).toBeInTheDocument()

    viewportHeight = 500
    fireEvent(window, new Event("resize"))
    await waitFor(() => expect(feedback.style.visibility).toBe(""))
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Exact draft before collision"
    )
  })
})

describe("PromptAssistComposerAction exact Undo lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  it("restores the exact prior draft once, then clears Undo", async () => {
    const user = userEvent.setup()
    renderHarness({ initialDraft: "Exact raw draft before improvement" })

    await improveNow(user)
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Improved user draft"
    )
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))

    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Exact raw draft before improvement"
    )
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
  })

  it("clears persistent Undo after a manual owner edit", async () => {
    const user = userEvent.setup()
    renderHarness()

    await improveNow(user)
    const liveDraft = screen.getByRole("textbox", { name: "User draft" })
    await user.type(liveDraft, " manually extended")

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Undo improvement" })
      ).not.toBeInTheDocument()
    )
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Improved user draft manually extended"
    )
  })

  it("clears Undo when the user manually empties the draft during an existing stream", async () => {
    const user = userEvent.setup()
    const { rerenderHarness } = renderHarness()

    await improveNow(user)
    rerenderHarness({ sending: true })
    await user.clear(screen.getByRole("textbox", { name: "User draft" }))

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Undo improvement" })
      ).not.toBeInTheDocument()
    )
  })

  it("replaces prior Undo when a new operation is started and completed", async () => {
    let resultNumber = 0
    mocks.improvePrompt.mockImplementation(async (request) => ({
      ...improvementResponse(request.operation_id),
      improved_text: `Improved result ${++resultNumber}`
    }))
    const user = userEvent.setup()
    renderHarness()

    await improveNow(user)
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Improved result 1"
    )
    await improveNow(user)
    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Improved result 2"
    )
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))

    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
      "Improved result 1"
    )
  })

  it("clears Undo when a successful send resets the existing owner", async () => {
    const user = userEvent.setup()
    renderHarness()

    await improveNow(user)
    await user.click(screen.getByRole("button", { name: "Begin send" }))
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Finish successful send" })
    )

    expect(screen.getByLabelText("Committed user draft")).toHaveTextContent("")
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Undo improvement" })
      ).not.toBeInTheDocument()
    )
  })

  it.each([
    ["structured failure", false],
    ["structured failure", true],
    ["rejected send", false],
    ["rejected send", true]
  ] as const)(
    "retains Undo after a batched optimistic reset and %s (StrictMode=%s)",
    async (failure, strictMode) => {
      const user = userEvent.setup()
      renderHarness({ strictMode })

      await improveNow(user)
      await user.click(screen.getByRole("button", { name: "Begin send" }))
      await user.click(
        screen.getByRole("button", {
          name:
            failure === "rejected send" ? "Reject send" : "Finish failed send"
        })
      )

      expect(
        screen.getByRole("button", { name: "Undo improvement" })
      ).toBeInTheDocument()
      await user.click(screen.getByRole("button", { name: "Undo improvement" }))
      expect(screen.getByLabelText("Committed user draft")).toHaveTextContent(
        "Original user draft"
      )
    }
  )

  it.each(["structured failure", "rejected send"] as const)(
    "retains Undo when pending state follows the reset before a %s",
    async (failure) => {
      const user = userEvent.setup()
      renderHarness()

      await improveNow(user)
      await user.click(
        screen.getByRole("button", { name: "Reset before pending" })
      )
      await user.click(
        screen.getByRole("button", { name: "Mark send pending" })
      )
      await user.click(
        screen.getByRole("button", {
          name:
            failure === "rejected send" ? "Reject send" : "Finish failed send"
        })
      )

      expect(
        screen.getByRole("button", { name: "Undo improvement" })
      ).toBeInTheDocument()
    }
  )

  it("clears Undo when queue success lands while another send is active", async () => {
    const user = userEvent.setup()
    renderHarness()

    await improveNow(user)
    await user.click(screen.getByRole("button", { name: "Begin send" }))
    await user.click(screen.getByRole("button", { name: "Finish queued send" }))

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Undo improvement" })
      ).not.toBeInTheDocument()
    )
  })

  it.each(["Playground", "Sidepanel"])(
    "keeps newer %s Undo when an older queued item later succeeds",
    async () => {
      const user = userEvent.setup()
      renderHarness()

      await improveNow(user)
      await user.click(
        screen.getByRole("button", { name: "Finish queued send" })
      )

      const liveDraft = screen.getByRole("textbox", { name: "User draft" })
      await user.clear(liveDraft)
      await user.type(liveDraft, "Newer independent draft")
      await improveNow(user)
      await user.click(
        screen.getByRole("button", { name: "Finish queued send" })
      )

      expect(
        screen.getByRole("button", { name: "Undo improvement" })
      ).toBeInTheDocument()
    }
  )

  it.each(["surface", "context"] as const)(
    "clears Undo after a %s ownership change",
    async (change) => {
      const user = userEvent.setup()
      const { rerenderHarness } = renderHarness()

      await improveNow(user)
      rerenderHarness(
        change === "surface"
          ? { surfaceOpen: false }
          : { promptAssistContextKey: "conversation-2" }
      )
      if (change === "surface") {
        rerenderHarness({ surfaceOpen: true })
      }

      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: "Undo improvement" })
        ).not.toBeInTheDocument()
      )
    }
  )

  it("keeps applied actions persistent and inspection read-only without a second Apply", async () => {
    const user = userEvent.setup()
    renderHarness()

    await improveNow(user)
    expect(
      screen.queryByRole("dialog", { name: "Prompt improvement" })
    ).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "View changes" }))

    const drawer = screen.getByRole("dialog", { name: "Prompt improvement" })
    expect(drawer).toHaveAttribute("data-drawer-size", "480")
    expect(drawer).toHaveAttribute("data-drawer-width", "undefined")
    expect(screen.getByText("Applied changes")).toBeInTheDocument()
    expect(
      screen.getByRole("textbox", { name: "Improved prompt candidate" })
    ).toHaveAttribute("readonly")
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Apply to draft" })
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Close" }))
    expect(
      screen.getByRole("button", { name: "View changes" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
  })

  it("persists the existing owner after Apply and exact Undo", async () => {
    const user = userEvent.setup()
    renderHarness({ draftEnabled: true })

    await waitFor(() =>
      expect(draftBucketMocks.set).toHaveBeenLastCalledWith(
        "tldw:test:prompt-assist-composer",
        "Original user draft"
      )
    )
    await improveNow(user)
    await waitFor(() =>
      expect(draftBucketMocks.set).toHaveBeenLastCalledWith(
        "tldw:test:prompt-assist-composer",
        "Improved user draft"
      )
    )
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))

    await waitFor(() =>
      expect(draftBucketMocks.set).toHaveBeenLastCalledWith(
        "tldw:test:prompt-assist-composer",
        "Original user draft"
      )
    )
  })
})

describe("PromptAssistComposerAction owner surface", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    draftBucketMocks.records.clear()
    mocks.fetchPromptCapabilities.mockResolvedValue(availableCapabilities)
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  it("does not submit the composer when Enter or Escape is handled in review", async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    renderHarness({ onSubmit })

    await openActions(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    const candidate = await screen.findByRole("textbox", {
      name: "Improved prompt candidate"
    })
    candidate.focus()
    await user.keyboard("{Enter}")
    expect(onSubmit).not.toHaveBeenCalled()

    await user.keyboard("{Escape}")
    expect(
      screen.queryByRole("dialog", { name: "Prompt improvement" })
    ).not.toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it.each([
    [false, "480"],
    [true, "100vw"]
  ] as const)(
    "owns one responsive Drawer when narrow=%s",
    async (narrow, expectedWidth) => {
      const user = userEvent.setup()
      renderHarness({ narrow })

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Review changes/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Review changes/ }))

      const drawers = await screen.findAllByRole("dialog", {
        name: "Prompt improvement"
      })
      expect(drawers).toHaveLength(1)
      expect(drawers[0]).toHaveAttribute("data-drawer-size", expectedWidth)
      expect(drawers[0]).toHaveAttribute("data-drawer-width", "undefined")
      expect(drawers[0]).toHaveAttribute(
        "data-focus-trigger-after-close",
        "false"
      )
    }
  )

  it.each(["X", "backdrop", "Escape", "Cancel"] as const)(
    "returns desktop focus after the Drawer fully closes via %s",
    async (closeMethod) => {
      const user = userEvent.setup()
      renderHarness()
      const textarea = screen.getByRole("textbox", { name: "User draft" })

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Review changes/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Review changes/ }))
      const drawer = await screen.findByRole("dialog", {
        name: "Prompt improvement"
      })

      if (closeMethod === "Escape") {
        fireEvent.keyDown(drawer, { key: "Escape" })
      } else {
        await user.click(
          screen.getByRole("button", {
            name:
              closeMethod === "X"
                ? "Close prompt improvement drawer"
                : closeMethod === "backdrop"
                  ? "Prompt improvement drawer backdrop"
                  : "Cancel"
          })
        )
      }

      await waitFor(() => expect(textarea).toHaveFocus())
    }
  )

  it.each(["Apply", "Confirm apply"] as const)(
    "returns focus after the Drawer fully closes through %s",
    async (applyMethod) => {
      const user = userEvent.setup()
      renderHarness()
      const textarea = screen.getByRole("textbox", { name: "User draft" })

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Review changes/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Review changes/ }))
      await screen.findByRole("dialog", { name: "Prompt improvement" })

      if (applyMethod === "Confirm apply") {
        await user.type(textarea, " newer")
      }
      await user.click(screen.getByRole("button", { name: "Apply to draft" }))
      if (applyMethod === "Confirm apply") {
        await user.click(
          screen.getByRole("button", { name: "Replace current draft" })
        )
        await user.click(
          screen.getByRole("button", { name: "Confirm replace" })
        )
      }

      await waitFor(() => expect(textarea).toHaveFocus())
    }
  )

  it.each(["surface", "context"] as const)(
    "does not let an interrupted close steal focus after a %s ownership change",
    async (ownershipChange) => {
      drawerLifecycleMocks.deferClose = true
      const user = userEvent.setup()
      const { rerenderHarness } = renderHarness()

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Review changes/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Review changes/ }))
      await screen.findByRole("dialog", { name: "Prompt improvement" })
      await user.click(
        screen.getByRole("button", {
          name: "Close prompt improvement drawer"
        })
      )

      if (ownershipChange === "surface") {
        rerenderHarness({ surfaceOpen: false })
        rerenderHarness({ surfaceOpen: true })
      } else {
        rerenderHarness({ promptAssistContextKey: "conversation-2" })
      }

      await openActions(user)
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Review changes/ })
        ).toBeEnabled()
      )
      await user.click(screen.getByRole("button", { name: /Review changes/ }))
      const candidate = await screen.findByRole("textbox", {
        name: "Improved prompt candidate"
      })
      candidate.focus()
      expect(candidate).toHaveFocus()

      act(() => {
        const completeClose = drawerLifecycleMocks.completeClose
        drawerLifecycleMocks.completeClose = null
        completeClose?.()
      })

      expect(candidate).toHaveFocus()
    }
  )

  it("closes applied inspection without clearing Undo and restores desktop focus", async () => {
    const user = userEvent.setup()
    renderHarness()
    const textarea = screen.getByRole("textbox", { name: "User draft" })

    await improveNow(user)
    await user.click(screen.getByRole("button", { name: "View changes" }))
    await user.click(
      screen.getByRole("button", {
        name: "Close prompt improvement drawer"
      })
    )

    expect(textarea).toHaveFocus()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
  })

  it("uses the mobile-aware focus callback when the narrow Drawer owner closes", async () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue("iPhone")
    const blur = vi.spyOn(HTMLTextAreaElement.prototype, "blur")
    const user = userEvent.setup()
    renderHarness({ narrow: true })

    await openActions(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeEnabled()
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    await user.click(
      screen.getByRole("button", {
        name: "Close prompt improvement drawer"
      })
    )

    expect(blur).toHaveBeenCalled()
    expect(
      screen.getByRole("textbox", { name: "User draft" })
    ).not.toHaveFocus()
  })

  it("returns desktop focus to the existing textarea after auto-Apply and Undo", async () => {
    const user = userEvent.setup()
    renderHarness()
    const textarea = screen.getByRole("textbox", { name: "User draft" })

    await improveNow(user)
    expect(textarea).toHaveFocus()

    await user.click(screen.getByRole("button", { name: "Undo improvement" }))
    expect(textarea).toHaveFocus()
  })

  it("does not force focus onto the textarea after mobile Apply or Undo", async () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue("iPhone")
    const user = userEvent.setup()
    renderHarness({ narrow: true })
    const textarea = screen.getByRole("textbox", { name: "User draft" })

    await improveNow(user)
    expect(textarea).not.toHaveFocus()

    await user.click(screen.getByRole("button", { name: "Undo improvement" }))
    expect(textarea).not.toHaveFocus()
  })
})

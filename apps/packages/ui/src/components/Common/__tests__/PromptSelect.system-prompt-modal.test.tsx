import * as serverOnline from "@/hooks/useServerOnline"
import {
  clearRecipePersistenceScoped,
  markRecipePersistenceScoped
} from "@/services/recipe-persistence-uncertainty"
import * as recipeAuthority from "@/services/recipe-persistence-uncertainty"
import {
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"
import { OPEN_PROMPT_SELECT_EVENT } from "@/utils/prompt-select-events"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CLEAR_TASK_RECIPE } from "../PromptAssist/recipes/built-in-recipes"
import { PromptSelect } from "../PromptSelect"

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
  getAllPrompts: vi.fn(async () => []),
  getPromptById: vi.fn(async () => undefined),
  improvePrompt: vi.fn(),
  fetchPromptCapabilities: vi.fn()
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

const registryLabels = vi.hoisted(() => ({
  loading: "Loading via registry"
}))

const commonLoadingResource = vi.hoisted(() => ({
  title: "Loading title from common",
  description: "Loading description from common",
  content: "Loading content from common"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => {
      if (key === "common:loading") return commonLoadingResource
      if (key === "common:loading.title") return commonLoadingResource.title
      return fallback || key
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/db/dexie/helpers", () => ({
  updatePrompt: (...args: unknown[]) => mocks.updatePrompt(...args),
  markPromptSyncError: (...args: unknown[]) =>
    mocks.markPromptSyncError(...args),
  getAllPrompts: mocks.getAllPrompts,
  getPromptById: mocks.getPromptById
}))

vi.mock("@/services/prompt-improvement", async (importActual) => {
  const actual =
    await importActual<typeof import("@/services/prompt-improvement")>()
  return {
    ...actual,
    improvePrompt: (...args: unknown[]) => mocks.improvePrompt(...args)
  }
})

vi.mock("@/services/prompt-sync", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/prompt-sync")>()),
  autoSyncPrompt: (...args: unknown[]) => mocks.autoSyncPrompt(...args),
  shouldAutoSyncWorkspacePrompts: () => mocks.shouldAutoSyncWorkspacePrompts()
}))

vi.mock("@/services/prompts-api", async (importActual) => {
  const actual = await importActual<typeof import("@/services/prompts-api")>()
  return {
    ...actual,
    revalidatePromptCapabilities: () =>
      mocks.realCapabilities
        ? actual.revalidatePromptCapabilities()
        : mocks.fetchPromptCapabilities(),
    fetchPromptCapabilities: () =>
      mocks.realCapabilities
        ? actual.fetchPromptCapabilities()
        : mocks.fetchPromptCapabilities()
  }
})

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "loading" ? registryLabels.loading : state.label
        }
      }
    )
  }
})

vi.mock("antd", async () => {
  const React = await import("react")

  const Input = React.forwardRef<HTMLInputElement, any>((props, ref) => (
    <input
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
      onKeyDownCapture={props.onKeyDownCapture}
      onKeyDown={props.onKeyDown}
    />
  ))

  const TextArea = React.forwardRef<HTMLTextAreaElement, any>((props, ref) => (
    <textarea
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder ?? "System prompt"}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
    />
  ))

  ;(Input as any).TextArea = TextArea

  const renderMenuItems = (items: any[] = []) =>
    items.map((item) => {
      if (!item) return null
      if (item.type === "group") {
        return (
          <div key={item.label}>
            <div>{item.label}</div>
            {renderMenuItems(item.children)}
          </div>
        )
      }
      if (item.key === "empty") {
        return <div key="empty">{item.label}</div>
      }
      return (
        <button
          key={item.key}
          type="button"
          role="menuitem"
          onClick={() => item.onClick?.()}>
          {item.label}
        </button>
      )
    })

  const Dropdown = ({
    open,
    onOpenChange,
    menu,
    popupRender,
    children
  }: any) => {
    const menuNode = <div role="menu">{renderMenuItems(menu?.items)}</div>

    return (
      <div>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? (popupRender ? popupRender(menuNode) : menuNode) : null}
      </div>
    )
  }

  const Modal = ({ open, title, children, footer }: any) =>
    open ? (
      <div
        role="dialog"
        aria-label={typeof title === "string" ? title : undefined}>
        <div>{title}</div>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    ) : null

  return {
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Dropdown,
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description ?? "Empty"}</div>
    ),
    Input,
    Modal
  }
})

const buildPrompt = (overrides: Record<string, unknown> = {}) => ({
  id: "prompt-1",
  title: "Prompt One",
  content: "Template body",
  is_system: true,
  createdAt: Date.now(),
  ...overrides
})

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const improvementResponse = (
  operationId: string,
  improvedText = "Improved system draft"
) => ({
  schema_version: 1 as const,
  operation_id: operationId,
  status: "improved" as const,
  improved_text: improvedText,
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

const renderPromptSelect = (
  overrides: Partial<React.ComponentProps<typeof PromptSelect>> = {}
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  const props: React.ComponentProps<typeof PromptSelect> = {
    selectedSystemPrompt: "prompt-1",
    systemPrompt: "",
    setSystemPrompt: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn(),
    selectedModel: "gpt-5-mini",
    currentProvider: "openai",
    promptAssistContextKey: "conversation-1",
    promptAssistBackendKey: "backend-a",
    onSelectModel: vi.fn(),
    ...overrides
  }

  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <PromptSelect {...props} />
      </QueryClientProvider>
    ),
    props,
    queryClient
  }
}

const openEditor = async (
  user: ReturnType<typeof userEvent.setup>,
  expectedValue = "Template body"
) => {
  await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
  await user.click(
    await screen.findByRole("menuitem", { name: /edit system prompt/i })
  )
  await screen.findByDisplayValue(expectedValue)
}

const applyImprovementNow = async (
  user: ReturnType<typeof userEvent.setup>
) => {
  await user.click(screen.getByRole("button", { name: "Improve prompt" }))
  await user.click(screen.getByRole("button", { name: /Improve now/ }))
  await screen.findByRole("button", { name: "Undo improvement" })
}

describe("PromptSelect system prompt modal", () => {
  afterEach(async () => {
    mocks.runtimeId = undefined
    await clearRecipePersistenceScoped("scoped-recipe", ownerA)
    clearRuntimeAuthOverride()
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.realCapabilities = false
    mocks.runtimeId = undefined
    mocks.sendMessage.mockReset()
    mocks.shouldAutoSyncWorkspacePrompts.mockResolvedValue(false)
    vi.spyOn(
      recipeAuthority,
      "resolveRecipePersistenceOwnerView"
    ).mockResolvedValue({
      ownerId: ownerA,
      authorizationRevision: revisionA
    })
    mocks.getAllPrompts.mockResolvedValue([buildPrompt()])
    mocks.getPromptById.mockResolvedValue(buildPrompt())
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "available",
      prompt_improvement_v1: {
        supported: true,
        limits: null
      },
      single_text_recipe_v2: { supported: false }
    })
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
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
    const view = renderPromptSelect({ promptAssistBackendKey: null })
    await openEditor(user)
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
      screen.getByRole("button", { name: "Apply to system prompt" })
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
      screen.getByRole("button", { name: "Apply to system prompt" })
    )
    expect(view.props.setSystemPrompt).toHaveBeenCalledWith(preview)
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
      const view = renderPromptSelect({ promptAssistBackendKey: null })
      await openEditor(user)
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
    const view = renderPromptSelect({ promptAssistBackendKey: null })
    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
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
        screen.getByRole("button", { name: "Apply to system prompt" })
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
        screen.getByRole("button", { name: "Apply to system prompt" })
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
    renderPromptSelect({ promptAssistBackendKey: null })
    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
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
    renderPromptSelect({ promptAssistBackendKey: null })
    await openEditor(user)
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
        prompt_persistence: { create_authorized: true, update_authorized: true }
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

    mocks.getAllPrompts.mockResolvedValue([
      {
        id: "scoped-recipe",
        name: "Scoped recipe",
        title: "Scoped recipe",
        content: "Template body",
        is_system: true,
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
    renderPromptSelect({ promptAssistBackendKey: null })
    await openEditor(user)
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
    "scopes system recipe ownership to %s, independently of %s",
    async (backend, revision, enabled) => {
      vi.spyOn(serverOnline, "useServerOnline").mockReturnValue(true)
      await markRecipePersistenceScoped("scoped-recipe", ownerA)
      vi.mocked(
        recipeAuthority.resolveRecipePersistenceOwnerView
      ).mockResolvedValue({
        ownerId: backend,
        authorizationRevision: revision
      })
      mocks.getAllPrompts.mockResolvedValue([
        {
          ...buildPrompt(),
          id: "scoped-recipe",
          name: "Scoped recipe",
          title: "Scoped recipe",
          promptFormat: "structured",
          promptSchemaVersion: 2,
          syncStatus: "local",
          structuredPromptDefinition: structuredClone(
            CLEAR_TASK_RECIPE.definition
          )
        }
      ])
      mocks.fetchPromptCapabilities.mockResolvedValue({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: true },
        prompt_persistence: { create_authorized: true, update_authorized: true }
      })
      const user = userEvent.setup()
      const view = renderPromptSelect({
        promptAssistBackendKey: "unrelated-legacy-scope"
      })
      await openEditor(user)
      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
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

  it.each([undefined, "", "  Override 🧪\n\n"])(
    "applies a local recipe and restores exact override %s and template identity",
    async (original) => {
      const user = userEvent.setup()
      const { props } = renderPromptSelect({ systemPrompt: original })
      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" })
      )
      await user.click(
        await screen.findByRole("menuitem", { name: /edit system prompt/i })
      )
      expect(await screen.findByLabelText("Enter system prompt")).toHaveValue(
        original || "Template body"
      )

      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      const build = screen.getByRole("button", { name: /Build from recipe/ })
      expect(build).toBeEnabled()
      await user.click(build)
      expect(
        await screen.findByRole("region", { name: "Recipe builder" })
      ).toBeInTheDocument()
      expect(props.setSystemPrompt).not.toHaveBeenCalled()
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()

      await user.type(
        screen.getByLabelText("Current value for Task (not saved)"),
        "Preserve system identity."
      )
      const preview = (
        screen.getByLabelText("Compiled prompt preview") as HTMLTextAreaElement
      ).value
      await user.click(
        screen.getByRole("button", { name: "Apply to system prompt" })
      )
      expect(props.setSystemPrompt).toHaveBeenLastCalledWith(preview)
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()

      await user.click(screen.getByRole("button", { name: "Undo recipe" }))
      expect(props.setSystemPrompt).toHaveBeenLastCalledWith(original)
      expect(props.setSelectedSystemPrompt).toHaveBeenLastCalledWith("prompt-1")
      expect(screen.getByLabelText("Enter system prompt")).toHaveValue(
        original || "Template body"
      )
    }
  )

  describe.each([undefined, "", "  Override 🧪\n\n"])(
    "recipe followed by improvement with original override %s",
    (original) => {
      it.each([
        ["Improve now", "success"],
        ["Improve now", "cancel"],
        ["Improve now", "failure"],
        ["Review changes", "success"],
        ["Review changes", "cancel"],
        ["Review changes", "failure"]
      ])("consumes recipe Undo at %s start through %s", async (mode, outcome) => {
        const user = userEvent.setup()
        const pending = createDeferred<void>()
        mocks.improvePrompt.mockImplementation(async (request) => {
          await pending.promise
          if (outcome === "failure") throw new Error("provider unavailable")
          return improvementResponse(
            request.operation_id,
            request.text + "\nRefined."
          )
        })
        const view = renderPromptSelect({ systemPrompt: original })
        await user.click(
          await screen.findByRole("button", { name: "selectAPrompt" })
        )
        await user.click(
          await screen.findByRole("menuitem", { name: /edit system prompt/i })
        )
        expect(await screen.findByLabelText("Enter system prompt")).toHaveValue(
          original || "Template body"
        )
        await user.click(screen.getByRole("button", { name: "Improve prompt" }))
        await user.click(
          screen.getByRole("button", { name: /Build from recipe/ })
        )
        await user.type(
          await screen.findByLabelText("Current value for Task (not saved)"),
          "Preserve system identity."
        )
        const compiled = (
          screen.getByLabelText("Compiled prompt preview") as HTMLTextAreaElement
        ).value
        await user.click(
          screen.getByRole("button", { name: "Apply to system prompt" })
        )
        expect(view.props.setSystemPrompt).toHaveBeenLastCalledWith(compiled)
        expect(
          screen.getByRole("button", { name: "Undo recipe" })
        ).toBeInTheDocument()
        // Reflect the conversation owner's accepted override, as the real shell does.
        view.rerender(
          <QueryClientProvider client={view.queryClient}>
            <PromptSelect {...view.props} systemPrompt={compiled} />
          </QueryClientProvider>
        )
        await user.click(screen.getByRole("button", { name: "Improve prompt" }))
        await user.click(screen.getByRole("button", { name: new RegExp(mode) }))
        expect(mocks.improvePrompt).toHaveBeenCalledExactlyOnceWith(
          expect.objectContaining({ target: "system", text: compiled })
        )
        expect(
          screen.queryByRole("button", { name: "Undo recipe" })
        ).not.toBeInTheDocument()

        if (outcome === "cancel") {
          // Cancel while transport is still pending: invalidation belongs to start,
          // not success. A late successful response must not revive either Undo.
          await user.click(screen.getByRole("button", { name: "Cancel" }))
          expect(
            screen.queryByRole("button", { name: "Undo recipe" })
          ).not.toBeInTheDocument()
        }
        await act(async () => {
          pending.resolve()
          await pending.promise
        })
        if (outcome === "failure") {
          await screen.findByRole("alert")
          await user.click(screen.getByRole("button", { name: "Cancel" }))
        } else if (outcome === "success") {
          if (mode === "Review changes") {
            expect(
              await screen.findByRole("textbox", {
                name: "Improved prompt candidate"
              })
            ).toHaveValue(compiled + "\nRefined.")
            expect(view.props.setSystemPrompt).toHaveBeenCalledTimes(1)
            await user.click(
              screen.getByRole("button", { name: "Apply to draft" })
            )
          }
          await screen.findByRole("button", { name: "Undo improvement" })
          expect(
            screen.queryByRole("button", { name: "Undo recipe" })
          ).not.toBeInTheDocument()
          expect(screen.getByLabelText("Enter system prompt")).toHaveValue(
            compiled + "\nRefined."
          )
          await user.click(
            screen.getByRole("button", { name: "Undo improvement" })
          )
        }
        expect(screen.getByLabelText("Enter system prompt")).toHaveValue(compiled)
        expect(view.props.setSystemPrompt).toHaveBeenLastCalledWith(compiled)
        expect(view.props.setSystemPrompt).toHaveBeenCalledTimes(
          outcome === "success" ? 3 : 1
        )
        expect(
          screen.queryByRole("button", { name: "Undo recipe" })
        ).not.toBeInTheDocument()
        expect(
          screen.queryByRole("button", { name: "Undo improvement" })
        ).not.toBeInTheDocument()
        expect(view.props.setSelectedSystemPrompt).not.toHaveBeenCalled()
        expect(view.props.setSelectedQuickPrompt).not.toHaveBeenCalled()
        expect(mocks.updatePrompt).not.toHaveBeenCalled()
        await user.click(screen.getByRole("button", { name: "Reset" }))
        expect(
          await screen.findByDisplayValue("Template body")
        ).toBeInTheDocument()
        expect(view.props.setSelectedSystemPrompt).not.toHaveBeenCalled()
      })
    }
  )

  it("gates cached recipe authorization while open-time revalidation is pending and revoked", async () => {
    const user = userEvent.setup()
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
    renderPromptSelect()
    await waitFor(() =>
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(1)
    )

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
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

  it("returns from recipe mode without changing an empty system draft", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: undefined,
      selectedModel: null
    })
    await openEditor(user, "")

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    await user.type(
      await screen.findByLabelText("Current value for Task (not saved)"),
      "Unapplied runtime"
    )
    await user.click(screen.getByRole("button", { name: "Back" }))

    expect(screen.getByDisplayValue("")).toBeInTheDocument()
    expect(props.setSystemPrompt).not.toHaveBeenCalled()
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
  })

  it("opens an editor modal with the effective selected template content", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
  })

  it("keeps the prompt trigger visible while the prompt library is loading", async () => {
    mocks.getAllPrompts.mockReturnValue(new Promise(() => {}))

    renderPromptSelect({
      selectedSystemPrompt: undefined
    })

    expect(
      await screen.findByRole("button", { name: /loading prompts/i })
    ).toBeInTheDocument()
    expect(screen.getByText("Loading prompts")).toBeInTheDocument()
    expect(
      screen.queryByRole("status", { name: /loading prompts/i })
    ).not.toBeInTheDocument()
  })

  it("shows prompt library errors with a retry action", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockRejectedValueOnce(new Error("dexie unavailable"))
    renderPromptSelect({
      selectedSystemPrompt: undefined
    })

    await user.click(
      await screen.findByRole("button", { name: /prompt library unavailable/i })
    )

    expect(
      await screen.findByRole("menuitem", {
        name: /prompt library unavailable/i
      })
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("menuitem", { name: /retry prompt library/i })
    )

    expect(mocks.getAllPrompts).toHaveBeenCalledTimes(2)
  })

  it("saves edited prompt content through setSystemPrompt", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    const textarea = await screen.findByDisplayValue("Template body")
    await user.clear(textarea)
    await user.type(textarea, "Conversation override")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith(
        "Conversation override"
      )
    })
  })

  it("clears redundant overrides when the saved text matches the selected template", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    const textarea = await screen.findByDisplayValue("Conversation override")
    await user.clear(textarea)
    await user.type(textarea, "Template body")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    })
  })

  it("shows override-active copy when the live system prompt differs from the template", async () => {
    const user = userEvent.setup()
    renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    expect(await screen.findByText(/override active/i)).toBeInTheDocument()
  })

  it("resets to an empty prompt when the selected template cannot be resolved", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      selectedSystemPrompt: "missing-prompt"
    })

    mocks.getAllPrompts.mockResolvedValue([])
    mocks.getPromptById.mockRejectedValue(new Error("missing"))

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: /reset/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    })
  })

  it("renders the scalar common loading title while resolving editor content", async () => {
    const user = userEvent.setup()
    mocks.getPromptById.mockReturnValue(new Promise(() => {}))
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    expect(
      await screen.findByText("Loading title from common")
    ).toBeInTheDocument()
  })

  it("does not commit an editor-open lookup after the modal closes", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof buildPrompt>>()
    const nextOpen = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById.mockReturnValueOnce(pending.promise)
    mocks.getPromptById.mockReturnValueOnce(nextOpen.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => {
      pending.resolve(buildPrompt({ content: "Late open content" }))
      await pending.promise
    })

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    })
    await user.click(screen.getByRole("button", { name: "selectAPrompt" }))
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    expect(
      screen.queryByDisplayValue("Late open content")
    ).not.toBeInTheDocument()
  })

  it("lets a newer editor-open lookup win over an older completion", async () => {
    const user = userEvent.setup()
    const older = createDeferred<ReturnType<typeof buildPrompt>>()
    const newer = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "selectAPrompt" }))
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    await act(async () => {
      newer.resolve(buildPrompt({ content: "Newest open content" }))
      await newer.promise
    })
    expect(
      await screen.findByDisplayValue("Newest open content")
    ).toBeInTheDocument()
    await act(async () => {
      older.resolve(buildPrompt({ content: "Older open content" }))
      await older.promise
    })

    await waitFor(() => {
      expect(
        screen.getByDisplayValue("Newest open content")
      ).toBeInTheDocument()
      expect(
        screen.queryByDisplayValue("Older open content")
      ).not.toBeInTheDocument()
    })
  })

  it("does not let a late editor-open lookup overwrite typing", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById.mockReturnValueOnce(pending.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    const editor = screen.getByRole("textbox", { name: "Enter system prompt" })
    await user.type(editor, "Typed while opening")
    await act(async () => {
      pending.resolve(buildPrompt({ content: "Late open content" }))
      await pending.promise
    })

    await waitFor(() => {
      expect(editor).toHaveValue("Typed while opening")
    })
  })

  it("does not commit a Reset lookup after close or unmount", async () => {
    const user = userEvent.setup()
    const afterClose = createDeferred<ReturnType<typeof buildPrompt>>()
    const afterUnmount = createDeferred<ReturnType<typeof buildPrompt>>()
    const first = renderPromptSelect({ systemPrompt: "Conversation override" })
    await openEditor(user, "Conversation override")
    mocks.getPromptById.mockReturnValueOnce(afterClose.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => {
      afterClose.resolve(buildPrompt({ content: "Late reset after close" }))
      await afterClose.promise
    })
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    )
    expect(first.props.setSystemPrompt).not.toHaveBeenCalled()

    first.unmount()
    const second = renderPromptSelect({ systemPrompt: "Conversation override" })
    await openEditor(user, "Conversation override")
    mocks.getPromptById.mockReturnValueOnce(afterUnmount.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    second.unmount()
    await act(async () => {
      afterUnmount.resolve(buildPrompt({ content: "Late reset after unmount" }))
      await afterUnmount.promise
    })

    await waitFor(() => {
      expect(second.props.setSystemPrompt).not.toHaveBeenCalled()
    })
  })

  it("lets a newer Reset win and rejects a Reset completion after typing", async () => {
    const user = userEvent.setup()
    const older = createDeferred<ReturnType<typeof buildPrompt>>()
    const newer = createDeferred<ReturnType<typeof buildPrompt>>()
    const afterTyping = createDeferred<ReturnType<typeof buildPrompt>>()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })
    await openEditor(user, "Conversation override")
    mocks.getPromptById
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await act(async () => {
      newer.resolve(buildPrompt({ content: "Newest reset content" }))
      await newer.promise
    })
    expect(
      await screen.findByDisplayValue("Newest reset content")
    ).toBeInTheDocument()
    await act(async () => {
      older.resolve(buildPrompt({ content: "Older reset content" }))
      await older.promise
    })
    await waitFor(() => {
      expect(
        screen.queryByDisplayValue("Older reset content")
      ).not.toBeInTheDocument()
    })

    mocks.getPromptById.mockReturnValueOnce(afterTyping.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    const editor = screen.getByRole("textbox", { name: "Enter system prompt" })
    await user.clear(editor)
    await user.type(editor, "Typed after reset")
    await act(async () => {
      afterTyping.resolve(buildPrompt({ content: "Late reset content" }))
      await afterTyping.promise
    })

    await waitFor(() => expect(editor).toHaveValue("Typed after reset"))
    expect(props.setSystemPrompt).toHaveBeenLastCalledWith(
      "Newest reset content"
    )
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["context", { promptAssistContextKey: "conversation-2" }]
  ])(
    "rejects a Reset lookup after a %s lifecycle change",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const pending = createDeferred<ReturnType<typeof buildPrompt>>()
      const rendered = renderPromptSelect({
        systemPrompt: "Conversation override"
      })
      await openEditor(user, "Conversation override")
      mocks.getPromptById.mockReturnValueOnce(pending.promise)
      await user.click(screen.getByRole("button", { name: "Reset" }))

      rendered.rerender(
        <QueryClientProvider client={rendered.queryClient}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )
      await act(async () => {
        pending.resolve(buildPrompt({ content: "Late lifecycle reset" }))
        await pending.promise
      })

      await waitFor(() => {
        expect(rendered.props.setSystemPrompt).not.toHaveBeenCalled()
      })
    }
  )

  it("closes the prompt dropdown when Escape is pressed from search", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    expect(await screen.findByRole("menu")).toBeInTheDocument()

    const search = await screen.findByRole("textbox", {
      name: "Search prompts..."
    })
    search.focus()
    await user.keyboard("{Escape}")

    await waitFor(() => {
      expect(screen.queryByRole("menu")).not.toBeInTheDocument()
    })
  })

  it("returns focus to the launching rail trigger after prompt selection", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined
    })
    render(
      <button type="button" data-testid="cockpit-prompt-select-trigger">
        Select prompt from rail
      </button>
    )
    const trigger = screen.getByTestId("cockpit-prompt-select-trigger")
    trigger.focus()

    window.dispatchEvent(
      new CustomEvent(OPEN_PROMPT_SELECT_EVENT, {
        detail: {
          returnFocusSelector: "[data-testid='cockpit-prompt-select-trigger']",
          source: "playground-cockpit"
        }
      })
    )

    await user.click(
      await screen.findByRole("menuitem", { name: /Prompt One/i })
    )

    await waitFor(() => {
      expect(props.setSelectedSystemPrompt).toHaveBeenCalledWith("prompt-1")
      expect(trigger).toHaveFocus()
    })
  })

  it("keeps current system prompt recovery actions visible when there are no saved prompts", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )

    expect(await screen.findByText(/no saved prompts/i)).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /edit current system prompt/i })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /clear current system prompt/i })
    ).toBeInTheDocument()
  })

  it("keeps current system prompt recovery actions visible when saved prompts exist", async () => {
    const user = userEvent.setup()
    renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )

    expect(await screen.findByText(/Prompt One/i)).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /edit current system prompt/i })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /clear current system prompt/i })
    ).toBeInTheDocument()
  })

  it("edits and saves a current custom prompt when the prompt library is empty", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", {
        name: /edit current system prompt/i
      })
    )

    const textarea = await screen.findByDisplayValue("Stay in character.")
    await user.clear(textarea)
    await user.type(textarea, "Speak as the station chief.")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith(
        "Speak as the station chief."
      )
    })
  })

  it("clears a current custom prompt when the prompt library is empty", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", {
        name: /clear current system prompt/i
      })
    )

    expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "selectAPrompt" })
      ).toHaveFocus()
    })
  })

  it("reviews the effective system draft in the existing modal and applies only a scoped override", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(
      await screen.findByRole("button", { name: "Improve prompt" })
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    expect(
      await screen.findByRole("textbox", {
        name: "Improved prompt candidate"
      })
    ).toHaveValue("Improved system draft")
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
    expect(mocks.improvePrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        target: "system",
        text: "Template body",
        model_selection: {
          selected_model: "gpt-5-mini",
          provider_hint: "openai"
        }
      })
    )

    await user.click(screen.getByRole("button", { name: "Apply to draft" }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith(
        "Improved system draft"
      )
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
      expect(screen.getByDisplayValue("Improved system draft")).toHaveFocus()
    })
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reset" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
  })

  it("normalizes an applied candidate matching the selected template to no override", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id, "Template body")
    )
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(
      await screen.findByRole("button", { name: "Improve prompt" })
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    await user.click(
      await screen.findByRole("button", { name: "Apply to draft" })
    )

    expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
  })

  it.each([undefined, "", "Custom override"])(
    "Undo restores the exact raw override state %s",
    async (rawOverride) => {
      const user = userEvent.setup()
      const { props } = renderPromptSelect({ systemPrompt: rawOverride })

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" })
      )
      await user.click(
        await screen.findByRole("menuitem", { name: /edit system prompt/i })
      )
      await user.click(
        await screen.findByRole("button", { name: "Improve prompt" })
      )
      await user.click(screen.getByRole("button", { name: /Improve now/ }))
      await user.click(
        await screen.findByRole("button", { name: "Undo improvement" })
      )

      expect(props.setSystemPrompt).toHaveBeenLastCalledWith(rawOverride)
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
    }
  )

  it("Reset restores the selected template without changing its identity", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Reset" }))

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
    expect(props.setSystemPrompt).toHaveBeenCalledWith("Template body")
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
  })

  it("assist Cancel restores the draft captured on entry without stacking a modal", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof improvementResponse>>()
    mocks.improvePrompt.mockReturnValue(pending.promise)
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    const editor = await screen.findByDisplayValue("Template body")
    await user.clear(editor)
    await user.type(editor, "Unsaved editor draft")
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    await user.click(await screen.findByRole("button", { name: "Cancel" }))

    expect(
      await screen.findByDisplayValue("Unsaved editor draft")
    ).toHaveFocus()
    expect(props.setSystemPrompt).not.toHaveBeenCalled()
  })

  it("returns focus after confirming replacement of a stale reviewed draft", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof improvementResponse>>()
    mocks.improvePrompt.mockReturnValue(pending.promise)
    const rendered = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    pending.resolve(
      improvementResponse(
        mocks.improvePrompt.mock.calls[0][0].operation_id,
        "Replacement candidate"
      )
    )

    expect(
      await screen.findByRole("textbox", {
        name: "Improved prompt candidate"
      })
    ).toHaveValue("Replacement candidate")
    await user.type(
      screen.getByDisplayValue("Template body"),
      " changed while reviewing"
    )

    await user.click(screen.getByRole("button", { name: "Apply to draft" }))
    await user.click(
      await screen.findByRole("button", { name: "Replace current draft" })
    )
    await user.click(screen.getByRole("button", { name: "Confirm replace" }))

    expect(rendered.props.setSystemPrompt).toHaveBeenCalledWith(
      "Replacement candidate"
    )
    await waitFor(() => {
      expect(screen.getByDisplayValue("Replacement candidate")).toHaveFocus()
    })
  })

  it("keeps a polite no-change result visible in the editor", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockImplementation(async (request) => ({
      ...improvementResponse(request.operation_id, "Template body"),
      status: "no_change" as const
    }))
    renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Improve now/ }))

    expect(
      await screen.findByText("No useful improvement found.")
    ).toBeInTheDocument()
    expect(screen.getByDisplayValue("Template body")).toBeInTheDocument()
  })

  it("typing after Apply invalidates Undo while keeping the editor usable", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    const editor = screen.getByDisplayValue("Improved system draft")
    await user.type(editor, " with a local edit")

    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument()
  })

  it("Reset after Apply clears Undo and restores the selected template", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    await user.click(screen.getByRole("button", { name: "Reset" }))

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
    expect(props.setSystemPrompt).toHaveBeenLastCalledWith("Template body")
  })

  it("Save after Apply closes the modal and consumes Undo", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    await openEditor(user)
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["conversation", { promptAssistContextKey: "conversation-2" }]
  ])(
    "%s changes after Apply invalidate Undo",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const rendered = renderPromptSelect()

      await openEditor(user)
      await applyImprovementNow(user)

      rendered.rerender(
        <QueryClientProvider client={new QueryClient()}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )

      await waitFor(() => {
        expect(
          screen.queryByRole("button", { name: "Undo improvement" })
        ).not.toBeInTheDocument()
      })
    }
  )

  it("fails closed while prompt-improvement capability is unknown", async () => {
    const user = userEvent.setup()
    mocks.fetchPromptCapabilities.mockReturnValue(new Promise(() => {}))
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.improvePrompt).not.toHaveBeenCalled()
  })

  it("closes the editor and hands missing idle model recovery to its owner", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({ selectedModel: null })

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: "Select model" }))

    expect(props.onSelectModel).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(mocks.improvePrompt).not.toHaveBeenCalled()
  })

  it("closes the editor and hands result-failure model recovery to its owner", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockRejectedValueOnce({
      code: "missing_model",
      retryable: false
    })
    const { props } = renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Improve now/ }))
    await user.click(
      await screen.findByRole("button", { name: "Select model" })
    )

    expect(props.onSelectModel).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("fails closed without an authoritative backend identity", async () => {
    const user = userEvent.setup()
    vi.mocked(
      recipeAuthority.resolveRecipePersistenceOwnerView
    ).mockResolvedValue(null)
    renderPromptSelect({ promptAssistBackendKey: null })

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
    expect(
      await screen.findByRole("button", { name: "Save as new recipe" })
    ).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).not.toHaveBeenCalled()
  })

  it("does not reuse supported capabilities for the same user after credential claims change", async () => {
    const user = userEvent.setup()
    const backendB = createDeferred<{
      availability: "available"
      prompt_improvement_v1: { supported: false; limits: null }
      single_text_recipe_v2: { supported: false }
    }>()
    mocks.fetchPromptCapabilities
      .mockResolvedValueOnce({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: false }
      })
      .mockReturnValueOnce(backendB.promise)
    const firstProps = {
      promptAssistBackendKey: "stable-backend",
      promptAssistAuthorizationRevision: "authorization-one"
    }
    const rendered = renderPromptSelect(firstProps)

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(
      await screen.findByRole("button", { name: /Improve now/ })
    ).toBeEnabled()

    rendered.rerender(
      <QueryClientProvider client={rendered.queryClient}>
        <PromptSelect
          {...rendered.props}
          promptAssistBackendKey="stable-backend"
          promptAssistAuthorizationRevision="authorization-two"
        />
      </QueryClientProvider>
    )

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    expect(
      rendered.queryClient.getQueryCache().find({
        queryKey: ["promptCapabilities", "stable-backend", "authorization-two"]
      })?.options.retry
    ).toBe(false)
  })

  it("ignores an old backend capability response while the new backend is unresolved", async () => {
    const user = userEvent.setup()
    const backendA =
      createDeferred<ReturnType<typeof mocks.fetchPromptCapabilities>>()
    const backendB =
      createDeferred<ReturnType<typeof mocks.fetchPromptCapabilities>>()
    mocks.fetchPromptCapabilities
      .mockReturnValueOnce(backendA.promise)
      .mockReturnValueOnce(backendB.promise)
    const rendered = renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()

    rendered.rerender(
      <QueryClientProvider client={rendered.queryClient}>
        <PromptSelect {...rendered.props} promptAssistBackendKey="backend-b" />
      </QueryClientProvider>
    )
    backendA.resolve({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits: null },
      single_text_recipe_v2: { supported: false }
    })

    await waitFor(() => {
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["context", { promptAssistContextKey: "conversation-2" }],
    ["backend", { promptAssistBackendKey: "backend-b" }]
  ])(
    "%s changes invalidate an in-flight result without overwriting or stealing focus",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const pending = createDeferred<ReturnType<typeof improvementResponse>>()
      mocks.improvePrompt.mockReturnValue(pending.promise)
      const rendered = renderPromptSelect()
      render(<button type="button">Focus sentinel</button>)

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" })
      )
      await user.click(
        await screen.findByRole("menuitem", { name: /edit system prompt/i })
      )
      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      await user.click(screen.getByRole("button", { name: /Improve now/ }))

      rendered.rerender(
        <QueryClientProvider client={new QueryClient()}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )
      const focusSentinel = screen.getByRole("button", {
        name: "Focus sentinel"
      })
      focusSentinel.focus()
      pending.resolve(
        improvementResponse(
          mocks.improvePrompt.mock.calls[0][0].operation_id,
          "Late candidate"
        )
      )

      await waitFor(() => expect(mocks.improvePrompt).toHaveBeenCalledTimes(1))
      expect(rendered.props.setSystemPrompt).not.toHaveBeenCalled()
      expect(focusSentinel).toHaveFocus()
    }
  )
})

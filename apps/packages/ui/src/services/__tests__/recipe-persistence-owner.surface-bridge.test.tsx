import { PromptAssistComposerAction } from "@/components/Chat/composer/PromptAssistComposerAction"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import { PromptSelect } from "@/components/Common/PromptSelect"
import { pullFromStudio } from "@/services/prompt-sync"
import { RecipePersistenceRegistry } from "@/services/recipe-persistence-registry"
import {
  readRecipePersistenceUncertainty,
  resolveRecipePersistenceOwnerView
} from "@/services/recipe-persistence-uncertainty"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type BackgroundListener = (
  message: unknown,
  sender: { id: string },
  reply: (value: unknown) => void
) => unknown

type StoredPrompt = Record<string, unknown> & {
  id?: string
  syncStatus?: string
}

const boundary = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    runtimeId: undefined as string | undefined,
    sendMessage: vi.fn(),
    listeners: new Set<BackgroundListener>(),
    values: new Map<string, unknown>(),
    rows: new Map<string, StoredPrompt>(),
    nextId: "surface-recipe-id",
    mutationCount: 0,
    observations: [] as Array<{
      id: string
      ownerId: string
      state: string
      body: Record<string, unknown>
    }>,
    createOutcome: "success" as "success" | "ambiguous"
  }
})

vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn() })
  return {
    browser: {
      runtime: {
        get id() {
          return boundary.runtimeId
        },
        getURL: (path: string) => `chrome-extension://test-extension${path}`,
        sendMessage: (...args: unknown[]) => boundary.sendMessage(...args),
        onConnect: event(),
        onStartup: event(),
        onMessage: {
          addListener: (listener: BackgroundListener) =>
            boundary.listeners.add(listener)
        }
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        session: { get: async () => ({}), set: async () => {} },
        onChanged: event()
      },
      alarms: {
        clear: async () => true,
        create: async () => {},
        onAlarm: event()
      },
      tabs: {
        query: async () => [],
        create: vi.fn(),
        sendMessage: async () => {}
      },
      action: { onClicked: event() },
      contextMenus: { create: vi.fn(), removeAll: vi.fn(), onClicked: event() },
      i18n: { getMessage: (key: string) => key }
    }
  }
})

vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: async (key: string) => boundary.values.get(key),
    set: async (key: string, value: unknown) => {
      boundary.values.set(key, value)
    },
    remove: async (key: string) => {
      boundary.values.delete(key)
    }
  })
}))

vi.mock("@/services/settings/local-bucket", () => ({
  createLocalRegistryBucket: () => ({
    get: async () => null,
    set: async () => {},
    remove: async () => {},
    cleanup: async () => 0,
    buildKey: (key: string) => `test:${key}`
  })
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({
    defaultProjectId: 42,
    autoSyncWorkspacePrompts: true
  }),
  setPromptStudioDefaults: async (value: unknown) => value
}))

vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => boundary.rows.get(id),
      update: async (
        id: string,
        fields:
          | Record<string, unknown>
          | ((row: StoredPrompt) => void | boolean)
      ) => {
        const current = boundary.rows.get(id)
        if (!current) return 0
        const row = structuredClone(current)
        if (typeof fields === "function") {
          if (fields(row) === false) return 1
        } else {
          Object.assign(row, fields)
        }
        boundary.rows.set(id, row)
        return 1
      },
      add: async (row: { id: string }) => {
        boundary.rows.set(row.id, structuredClone(row))
      },
      where: (field: string) => ({
        equals: (value: unknown) => ({
          first: async () =>
            [...boundary.rows.values()].find((row) => row[field] === value)
        })
      })
    }
  }
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => boundary.nextId,
  getAllPrompts: async () => structuredClone([...boundary.rows.values()]),
  getPromptById: async (id: string) => structuredClone(boundary.rows.get(id)),
  savePrompt: async (fields: Record<string, unknown>) => {
    const now = Date.now()
    const row = {
      ...structuredClone(fields),
      id: boundary.nextId,
      createdAt: now,
      updatedAt: now,
      syncStatus: "local",
      sourceSystem: "workspace"
    }
    boundary.rows.set(row.id, row)
    return structuredClone(row)
  },
  updatePrompt: async (row: StoredPrompt & { id: string }) => {
    boundary.rows.set(row.id, structuredClone(row))
    return row.id
  },
  markPromptSyncError: async (id: string) => {
    const row = boundary.rows.get(id)
    if (row) row.syncStatus = "error"
    return id
  },
  permanentlyDeletePrompt: async (id: string) => {
    boundary.rows.delete(id)
    return id
  },
  restorePromptSnapshot: async (row: StoredPrompt & { id: string }) => {
    boundary.rows.set(row.id, structuredClone(row))
    return row.id
  }
}))

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/utils/is-private-mode", () => ({ isFireFoxPrivateMode: false }))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string | { defaultValue?: string; [key: string]: unknown },
      options?: Record<string, unknown>
    ) => {
      const template =
        typeof fallback === "string"
          ? fallback
          : typeof fallback?.defaultValue === "string"
            ? fallback.defaultValue
            : key
      const variables = typeof fallback === "object" ? fallback : options
      return template.replace(/{{(\w+)}}/g, (_, name) =>
        String(variables?.[name] ?? "")
      )
    }
  })
}))

vi.mock("antd", async () => {
  const React = await import("react")
  type InputProps = React.InputHTMLAttributes<HTMLInputElement> & {
    "aria-label"?: string
  }
  type TextAreaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
    "aria-label"?: string
  }
  type MenuItem = {
    key?: React.Key
    type?: string
    label?: React.ReactNode
    children?: MenuItem[]
    onClick?: () => void
  }
  const InputBase = React.forwardRef<HTMLInputElement, InputProps>(
    (props, ref) => (
      <input
        ref={ref}
        aria-label={props["aria-label"] ?? props.placeholder}
        value={props.value}
        defaultValue={props.defaultValue}
        onChange={props.onChange}
        onKeyDown={props.onKeyDown}
        onKeyDownCapture={props.onKeyDownCapture}
      />
    )
  )
  const TextArea = React.forwardRef<HTMLTextAreaElement, TextAreaProps>(
    (props, ref) => (
      <textarea
        ref={ref}
        aria-label={props["aria-label"] ?? props.placeholder ?? "System prompt"}
        value={props.value}
        defaultValue={props.defaultValue}
        onChange={props.onChange}
      />
    )
  )
  const Input = Object.assign(InputBase, { TextArea })
  const renderMenuItems = (items: MenuItem[] = []): React.ReactNode =>
    items.map((item, index) => {
      if (!item) return null
      if (item.type === "group")
        return (
          <div key={`group-${index}`}>{renderMenuItems(item.children)}</div>
        )
      if (item.key === "empty") return <div key="empty">{item.label}</div>
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
  }: {
    open?: boolean
    onOpenChange?: (open: boolean) => void
    menu?: { items?: MenuItem[] }
    popupRender?: (node: React.ReactNode) => React.ReactNode
    children?: React.ReactNode
  }) => {
    const menuNode = <div role="menu">{renderMenuItems(menu?.items)}</div>
    return (
      <div>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? (popupRender ? popupRender(menuNode) : menuNode) : null}
      </div>
    )
  }
  const Modal = ({
    open,
    title,
    children,
    footer
  }: {
    open?: boolean
    title?: React.ReactNode
    children?: React.ReactNode
    footer?: React.ReactNode
  }) =>
    open ? (
      <div
        role="dialog"
        aria-label={typeof title === "string" ? title : undefined}>
        {title}
        {children}
        {footer}
      </div>
    ) : null
  const Drawer = ({
    open,
    title,
    children,
    onClose
  }: {
    open?: boolean
    title?: React.ReactNode
    children?: React.ReactNode
    onClose?: () => void
  }) =>
    open ? (
      <aside role="dialog" aria-label={String(title)}>
        <button type="button" aria-label="Close drawer" onClick={onClose} />
        {children}
      </aside>
    ) : null
  return {
    Drawer,
    Dropdown,
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description ?? "Empty"}</div>
    ),
    Input,
    Modal,
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

const config = (
  apiKey: string,
  serverUrl = "https://recipes.example.test"
) => ({
  serverUrl,
  authMode: "single-user",
  authSource: "manual",
  credentialSource: "manual",
  apiKey,
  apiKeyPersistence: "device",
  apiKeyServerOrigin: new URL(serverUrl).origin
})

const limits = Object.fromEntries(
  [
    "max_request_bytes",
    "max_draft_chars",
    "max_candidate_chars",
    "max_raw_output_chars",
    "max_findings",
    "max_finding_text_chars",
    "max_provider_chars",
    "max_model_chars",
    "max_meta_prompt_version_chars",
    "max_warning_chars",
    "max_warnings",
    "max_protected_tokens",
    "max_protected_token_kind_chars",
    "max_protected_token_chars",
    "max_protected_token_occurrences",
    "max_protected_token_total_chars"
  ].map((key) => [key, 1000])
)

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  })

const serverPrompt = (id = 101) => ({
  id,
  project_id: 42,
  name: "Surface recipe",
  version_number: 1,
  updated_at: "2026-09-11T00:00:00Z",
  prompt_format: "structured",
  prompt_schema_version: 2,
  prompt_definition: structuredClone(CLEAR_TASK_RECIPE.definition),
  system_prompt: "",
  user_prompt: ""
})

const successfulCapabilities = () =>
  json({
    prompt_improvement_v1: { supported: true, limits },
    single_text_recipe_v2: { supported: true },
    prompt_persistence: {
      create_authorized: true,
      update_authorized: true
    }
  })

const installNetwork = () => {
  vi.stubGlobal(
    "fetch",
    async (input: string | URL | Request, init?: RequestInit) => {
      const url = new URL(
        typeof input === "string" || input instanceof URL ? input : input.url
      )
      const method = String(
        init?.method ?? (input instanceof Request ? input.method : "GET")
      ).toUpperCase()
      if (url.pathname.endsWith("/api/v1/prompts/capabilities"))
        return successfulCapabilities()
      if (
        url.pathname.endsWith("/api/v1/prompt-studio/prompts/create") &&
        method === "POST"
      ) {
        boundary.mutationCount += 1
        const reserve = vi
          .mocked(RecipePersistenceRegistry.prototype.reserve)
          .mock.calls.at(-1)
        const id = String(reserve?.[0] ?? "")
        const ownerId = String(reserve?.[1] ?? "")
        boundary.observations.push({
          id,
          ownerId,
          state: await readRecipePersistenceUncertainty(id, ownerId),
          body: JSON.parse(String(init?.body ?? "{}"))
        })
        if (boundary.createOutcome === "ambiguous")
          throw new Error("connection lost")
        return json({ success: true, data: serverPrompt() })
      }
      if (url.pathname.includes("/api/v1/prompt-studio/prompts/get/101")) {
        return json({ success: true, data: serverPrompt() })
      }
      throw new Error(`Unexpected network request: ${method} ${url.pathname}`)
    }
  )
}

const sendToBackground = (message: unknown): Promise<unknown> =>
  new Promise((resolve) => {
    const listener = [...boundary.listeners][0]
    if (!listener) throw new Error("Missing real background listener")
    listener(message, { id: "test-extension" }, resolve)
  })

const startBackground = async () => {
  boundary.runtimeId = "test-extension"
  boundary.listeners.clear()
  const browserWindow = globalThis.window
  vi.stubGlobal("window", undefined)
  const background = (await import("@/entries/background")).default
  background.main()
  vi.stubGlobal("window", browserWindow)
  boundary.sendMessage.mockImplementation(sendToBackground)
}

const queryClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false } } })

function ComposerHarness({ initialDraft = "Original user draft" }) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const initialized = React.useRef(false)
  const composer = useComposerText({
    draftKey: "recipe-surface-bridge",
    textareaRef,
    draftEnabled: false
  })
  React.useLayoutEffect(() => {
    if (initialized.current) return
    initialized.current = true
    composer.form.setFieldValue("message", initialDraft)
  }, [composer.form, initialDraft])
  return (
    <div>
      <textarea
        ref={textareaRef}
        aria-label="User draft"
        {...composer.form.getInputProps("message")}
      />
      <output aria-label="Committed user draft">
        {composer.form.values.message}
      </output>
      <PromptAssistComposerAction
        form={composer.form}
        messageRevision={composer.messageRevision}
        promptAssistMutation={composer.promptAssistMutation}
        promptAssistSavedAttemptId={composer.promptAssistSavedAttemptId}
        modelSelection={{
          selected_model: "gpt-5-mini",
          provider_hint: "openai"
        }}
        promptAssistContextKey="surface-bridge"
        promptAssistBackendKey={null}
        surfaceOpen
      />
    </div>
  )
}

const renderComposer = () => {
  const client = queryClient()
  return {
    ...render(
      <QueryClientProvider client={client}>
        <ComposerHarness />
      </QueryClientProvider>
    ),
    client
  }
}

const renderSystemModal = () => {
  const client = queryClient()
  const setSystemPrompt = vi.fn()
  return {
    ...render(
      <QueryClientProvider client={client}>
        <PromptSelect
          selectedSystemPrompt={undefined}
          systemPrompt="Original system draft"
          setSystemPrompt={setSystemPrompt}
          setSelectedSystemPrompt={vi.fn()}
          setSelectedQuickPrompt={vi.fn()}
          selectedModel="gpt-5-mini"
          currentProvider="openai"
          promptAssistContextKey="system-surface-bridge"
          promptAssistBackendKey={null}
        />
      </QueryClientProvider>
    ),
    client,
    setSystemPrompt
  }
}

const openComposerRecipe = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(
    await screen.findByRole("button", { name: "Improve prompt" })
  )
  await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
  await screen.findByRole("region", { name: "Structured recipe builder" })
}

const openSystemRecipe = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
  await user.click(
    await screen.findByRole("menuitem", { name: /edit.*system prompt/i })
  )
  await screen.findByDisplayValue("Original system draft")
  await user.click(screen.getByRole("button", { name: "Improve prompt" }))
  await user.click(screen.getByRole("button", { name: /Build from recipe/ }))
  await screen.findByRole("region", { name: "Structured recipe builder" })
}

const fillTask = async (
  user: ReturnType<typeof userEvent.setup>,
  value: string
) => {
  await user.clear(
    screen.getByRole("textbox", { name: "Current value for Task (not saved)" })
  )
  await user.type(
    screen.getByRole("textbox", { name: "Current value for Task (not saved)" }),
    value
  )
  return (
    screen.getByRole("textbox", {
      name: "Compiled prompt preview"
    }) as HTMLTextAreaElement
  ).value
}

const findOwnerKey = (client: QueryClient) =>
  client
    .getQueryCache()
    .getAll()
    .find(
      (query) =>
        query.queryKey[0] === "promptCapabilities" &&
        String(query.queryKey[1]).startsWith("recipe-owner:")
    )?.queryKey

const chooseSavedRecipe = async (user: ReturnType<typeof userEvent.setup>) => {
  await screen.findByRole("option", { name: /Surface recipe|Untitled recipe/ })
  await user.selectOptions(
    screen.getByRole("combobox", { name: "Recipe source" }),
    `saved:${boundary.nextId}`
  )
}

beforeEach(() => {
  vi.spyOn(RecipePersistenceRegistry.prototype, "reserve")
  boundary.runtimeId = undefined
  boundary.sendMessage.mockReset()
  boundary.listeners.clear()
  boundary.values.clear()
  boundary.rows.clear()
  boundary.nextId = "surface-recipe-id"
  boundary.mutationCount = 0
  boundary.observations = []
  boundary.createOutcome = "success"
  boundary.values.set("tldwConfig", config("owner-a-key"))
  installNetwork()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("actual recipe surface adapter bridges", () => {
  it("carries the system modal owner and exact local ID through immutable direct dispatch, reconciliation, and local Apply", async () => {
    const expectedOwner = await resolveRecipePersistenceOwnerView()
    const user = userEvent.setup()
    const view = renderSystemModal()
    await openSystemRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    const preview = await fillTask(user, "system bridge task")
    await user.click(screen.getByRole("button", { name: "Save as new recipe" }))
    await waitFor(() =>
      expect(boundary.rows.get(boundary.nextId)).toMatchObject({
        serverId: 101,
        syncStatus: "synced"
      })
    )

    expect(findOwnerKey(view.client)).toEqual([
      "promptCapabilities",
      expectedOwner!.ownerId,
      expectedOwner!.authorizationRevision
    ])
    expect(RecipePersistenceRegistry.prototype.reserve).toHaveBeenCalledWith(
      boundary.nextId,
      expectedOwner!.ownerId,
      expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
      )
    )
    expect(boundary.observations).toEqual([
      expect.objectContaining({
        id: boundary.nextId,
        ownerId: expectedOwner!.ownerId,
        state: "unknown_owner",
        body: expect.objectContaining({
          project_id: 42,
          prompt_schema_version: 2
        })
      })
    ])
    expect(boundary.mutationCount).toBe(1)
    expect(
      await readRecipePersistenceUncertainty(
        boundary.nextId,
        expectedOwner!.ownerId
      )
    ).toBe("clear")

    await user.click(
      screen.getByRole("button", { name: "Apply to system prompt" })
    )
    expect(view.setSystemPrompt).toHaveBeenCalledWith(preview)
    expect(boundary.mutationCount).toBe(1)
  })

  it("carries the composer owner through direct ambiguity, cross-owner reopen, and exact-owner reconciliation while Apply stays local", async () => {
    boundary.nextId = "composer-ambiguous-id"
    boundary.createOutcome = "ambiguous"
    const ownerA = await resolveRecipePersistenceOwnerView()
    const user = userEvent.setup()
    const first = renderComposer()
    await openComposerRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    await fillTask(user, "ambiguous composer task")
    await user.click(screen.getByRole("button", { name: "Save as new recipe" }))
    await screen.findByText(/server outcome could not be verified/i)
    expect(boundary.mutationCount).toBe(1)
    expect(RecipePersistenceRegistry.prototype.reserve).toHaveBeenCalledWith(
      boundary.nextId,
      ownerA!.ownerId,
      expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
      )
    )
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("scoped")
    first.unmount()

    boundary.values.set("tldwConfig", config("owner-b-key"))
    const ownerB = await resolveRecipePersistenceOwnerView()
    expect(ownerB!.ownerId).not.toBe(ownerA!.ownerId)
    const reopened = renderComposer()
    await openComposerRecipe(user)
    await chooseSavedRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Update recipe" })
      ).toBeDisabled()
    )
    const preview = await fillTask(user, "still local under owner B")
    expect(
      screen.getByRole("button", { name: "Apply to user message" })
    ).toBeEnabled()
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" })
    )
    expect(screen.getByText("Recipe applied.")).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "User draft" })).toHaveValue(
      preview
    )
    expect(boundary.mutationCount).toBe(1)
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("scoped")

    const ownerBPull = await pullFromStudio(101, boundary.nextId)
    expect(ownerBPull).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true,
      recipeOwnership: {
        dispatch: { state: "dispatched", actualOwnerId: ownerB!.ownerId }
      }
    })
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("scoped")
    boundary.values.set("tldwConfig", config("owner-a-key"))
    expect(await pullFromStudio(101, boundary.nextId)).toMatchObject({
      success: true,
      recipeOwnership: {
        dispatch: { state: "dispatched", actualOwnerId: ownerA!.ownerId }
      }
    })
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("clear")
    expect(boundary.mutationCount).toBe(1)
    reopened.unmount()
  })

  it("routes the shared extension composer adapter through the actual background listener exactly once", async () => {
    boundary.nextId = "extension-sidepanel-id"
    await startBackground()
    const owner = await resolveRecipePersistenceOwnerView()
    const user = userEvent.setup()
    const view = renderComposer()
    await openComposerRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    const preview = await fillTask(user, "extension sidepanel task")
    await user.click(screen.getByRole("button", { name: "Save as new recipe" }))
    await waitFor(() =>
      expect(boundary.rows.get(boundary.nextId)).toMatchObject({
        serverId: 101,
        syncStatus: "synced"
      })
    )

    expect(findOwnerKey(view.client)).toEqual([
      "promptCapabilities",
      owner!.ownerId,
      owner!.authorizationRevision
    ])
    expect(RecipePersistenceRegistry.prototype.reserve).toHaveBeenCalledWith(
      boundary.nextId,
      owner!.ownerId,
      expect.stringMatching(/^[0-9a-f-]{36}$/)
    )
    expect(boundary.observations).toEqual([
      expect.objectContaining({
        id: boundary.nextId,
        ownerId: owner!.ownerId,
        state: "unknown_owner"
      })
    ])
    expect(boundary.mutationCount).toBe(1)
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, owner!.ownerId)
    ).toBe("clear")
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" })
    )
    expect(screen.getByRole("textbox", { name: "User draft" })).toHaveValue(
      preview
    )
    expect(boundary.mutationCount).toBe(1)
  })

  it("keeps an extension response-loss quarantine across owner reopen and exposes only local Apply and Forget", async () => {
    boundary.nextId = "extension-unknown-id"
    await startBackground()
    const ownerA = await resolveRecipePersistenceOwnerView()
    const deliver = boundary.sendMessage.getMockImplementation()!
    boundary.sendMessage.mockImplementation(async (message: unknown) => {
      const result = await deliver(message)
      const request = message as {
        type?: unknown
        payload?: { method?: unknown }
      }
      if (
        request?.type === "tldw:request" &&
        String(request?.payload?.method).toUpperCase() === "POST"
      ) {
        throw new Error("response port closed")
      }
      return result
    })
    const user = userEvent.setup()
    const first = renderComposer()
    await openComposerRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Save as new recipe" })
      ).toBeEnabled()
    )
    await fillTask(user, "unknown extension task")
    await user.click(screen.getByRole("button", { name: "Save as new recipe" }))
    await screen.findByText(/server outcome could not be verified/i)
    expect(boundary.mutationCount).toBe(1)
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("unknown_owner")
    first.unmount()

    boundary.values.set("tldwConfig", config("owner-b-key"))
    const ownerB = await resolveRecipePersistenceOwnerView()
    const reopened = renderComposer()
    await openComposerRecipe(user)
    await chooseSavedRecipe(user)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Update recipe" })
      ).toBeDisabled()
    )
    const preview = await fillTask(user, "unknown but locally editable")
    expect(
      screen.getByRole("button", { name: "Apply to user message" })
    ).toBeEnabled()
    await user.click(
      screen.getByRole("button", { name: "Apply to user message" })
    )
    expect(screen.getByRole("textbox", { name: "User draft" })).toHaveValue(
      preview
    )
    expect(boundary.mutationCount).toBe(1)

    reopened.unmount()
    const recovery = renderComposer()
    await openComposerRecipe(user)
    await chooseSavedRecipe(user)
    await user.click(
      await screen.findByRole("button", {
        name: "Forget unresolved operation"
      })
    )
    await user.click(screen.getByRole("button", { name: "Confirm forget" }))
    await screen.findByText(/Unresolved operation forgotten/i)
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerB!.ownerId)
    ).toBe("clear")
    expect(
      await readRecipePersistenceUncertainty(boundary.nextId, ownerA!.ownerId)
    ).toBe("scoped")
    expect(boundary.mutationCount).toBe(1)
    recovery.unmount()
  })
})

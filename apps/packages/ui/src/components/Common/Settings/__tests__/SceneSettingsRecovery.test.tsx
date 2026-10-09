import React from "react"
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConfigProvider, Form, Input } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useActorEditorPrefs, useActorStore } from "@/store/actor"
import {
  HistorySelectionContext,
  useHistorySelection,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"
import { createDefaultActorSettings } from "@/types/actor"
import { ActorPopout } from "../ActorPopout"
import { ActorEditor } from "../ActorEditor"
import { CurrentChatModelSettings } from "../CurrentChatModelSettings"

const io = vi.hoisted(() => ({
  persistent: true,
  values: new Map<string, unknown>(),
  get: vi.fn(),
  set: vi.fn(),
  modelSettings: vi.fn(),
  realActorEditor: false,
  characterId: "character-one",
  retries: [] as Array<() => unknown>,
  accountCurrent: () => true,
  finishes: [] as Array<(values: Record<string, unknown>) => unknown>
}))
vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    get hasPersistentBackend() {
      return io.persistent
    },
    get: io.get,
    set: io.set,
    getAll: async () => ({}),
    watch: () => () => undefined,
    unwatch: () => undefined,
    remove: async (key: string) => {
      io.values.delete(key)
    }
  })
}))
vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  const CapturedForm = (props: React.ComponentProps<typeof actual.Form>) => {
    if (props.onFinish) io.finishes.push(props.onFinish)
    return React.createElement(actual.Form, props)
  }
  const CapturedButton = (
    props: React.ComponentProps<typeof actual.Button>
  ) => {
    if (props.children === "Retry" && props.onClick)
      io.retries.push(() => props.onClick!({} as React.MouseEvent<HTMLElement>))
    return React.createElement(actual.Button, props)
  }
  return {
    ...actual,
    Button: CapturedButton,
    Form: Object.assign(CapturedForm, actual.Form)
  }
})
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useStoreMessageOption()
}))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [
    { id: io.characterId, name: "Mira" },
    vi.fn(),
    { isLoading: false, isCurrent: io.accountCurrent }
  ]
}))
vi.mock("@/hooks/useChatDraftOwner", () => ({
  useChatDraftOwner: () => ({
    ownerKey: "default-action-owner",
    isCurrent: io.accountCurrent,
    verificationState: "ready",
    retryVerification: vi.fn()
  })
}))
vi.mock("@/hooks/useSelectedAssistant", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/hooks/useSelectedAssistant")>()),
  useSelectedAssistant: () => [null, vi.fn(), { isCurrent: io.accountCurrent }]
}))
vi.mock("@/hooks/useChatModelCatalog", () => ({
  EMPTY_CHAT_MODELS: [],
  useChatModelCatalog: () => ({ data: [], isLoading: false })
}))
vi.mock("@/services/model-settings", () => ({
  getAllModelSettings: io.modelSettings
}))
vi.mock("@/services/ocr", () => ({ getOCRLanguage: async () => null }))
vi.mock("@/components/Common/system-prompt-utils", () => ({
  resolveSelectedSystemPromptContent: async () => ""
}))
vi.mock("../ActorEditor", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ActorEditor")>()
  return {
    ActorEditor: (props: React.ComponentProps<typeof ActorEditor>) =>
      io.realActorEditor ? (
        <actual.ActorEditor {...props} />
      ) : (
        <Form.Item label="Scene notes" name="actorNotes">
          <Input />
        </Form.Item>
      )
  }
})
vi.mock("@/hooks/useActorWorldBooks", () => ({
  useActorWorldBooks: () => ({
    worldBooks: [],
    worldBooksLoading: false,
    entriesByWorldBook: {},
    entriesLoading: false,
    loadEntriesForWorldBook: vi.fn()
  })
}))
vi.mock("../tabs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../tabs")>()
  return {
    ModelBasicsTab: () => (
      <>
        <Form.Item label="Temperature" name="temperature">
          <Input />
        </Form.Item>
        <Form.Item label="Scene notes" name="actorNotes">
          <Input />
        </Form.Item>
      </>
    ),
    ConversationTab: () => null,
    AdvancedParamsTab: () => null,
    ActorTab: (props: React.ComponentProps<typeof actual.ActorTab>) =>
      io.realActorEditor ? <actual.ActorTab {...props} /> : null
  }
})
vi.mock("../LlamaCppAdvancedControls", () => ({
  LlamaCppAdvancedControls: () => null
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))

const scene = (notes: string) => ({ ...createDefaultActorSettings(), notes })
const target = (serverChatId: string) => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({
    historyId: null,
    serverChatId,
    selectedModel: "openai:gpt-4o-mini",
    serverChatAssistantKind: null
  })
}
const mountEditor = (
  kind: "actor" | "model",
  open = true,
  selection: HistorySelectionController | null = null,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
) => {
  const close = vi.fn()
  const content = (visible: boolean) => (
    <HistorySelectionContext.Provider value={selection}>
      <QueryClientProvider client={client}>
        <ConfigProvider theme={{ token: { motion: false } }}>
          {kind === "actor" ? (
            <ActorPopout open={visible} setOpen={close} />
          ) : (
            <CurrentChatModelSettings
              open={visible}
              setOpen={close}
              useDrawer
            />
          )}
        </ConfigProvider>
      </QueryClientProvider>
    </HistorySelectionContext.Provider>
  )
  const view = render(content(open))
  return {
    ...view,
    close,
    rerenderOpen: (visible: boolean) => view.rerender(content(visible))
  }
}

beforeEach(() => {
  io.persistent = true
  target("scene-one")
  useActorStore.setState({ settings: null, preview: "", tokenCount: 0 })
  io.values.clear()
  io.finishes = []
  io.retries = []
  io.characterId = "character-one"
  io.realActorEditor = false
  io.accountCurrent = () => true
  io.modelSettings.mockReset().mockResolvedValue({ temperature: 0.7 })
  io.get
    .mockReset()
    .mockImplementation(async (key: string) => io.values.get(key))
  io.set.mockReset().mockImplementation(async (key: string, value: unknown) => {
    io.values.set(key, value)
  })
  vi.spyOn(console, "error").mockImplementation(() => undefined)
})

describe.each(["actor", "model"] as const)(
  "%s scene editor recovery",
  (kind) => {
    it("keeps the unread scene unchanged and retries the actual reader", async () => {
      io.values.set("actorSettings:server:scene-one", scene("Keep this scene."))
      io.get.mockRejectedValueOnce(new Error("read unavailable"))
      const view = mountEditor(kind)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Scene settings could not be loaded"
      )
      const save = screen.queryByRole("button", { name: /^save$/i })
      if (save) expect(save).toBeDisabled()
      expect(io.set).not.toHaveBeenCalled()
      fireEvent.click(screen.getByRole("button", { name: /retry/i }))
      expect(await screen.findByLabelText("Scene notes")).toHaveValue(
        "Keep this scene."
      )
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Edited scene." }
      })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() =>
        expect(io.set).toHaveBeenCalledWith(
          "actorSettings:server:scene-one",
          expect.objectContaining({ notes: "Edited scene." })
        )
      )
      await waitFor(() => expect(view.close).toHaveBeenCalledWith(false))
    })

    it("does not use another scene's global UI cache instead of the destination read", async () => {
      useActorStore.setState({
        settings: scene("Another conversation's scene.")
      })
      io.values.set(
        "actorSettings:server:scene-one",
        scene("This conversation's scene.")
      )
      mountEditor(kind)
      expect(await screen.findByLabelText("Scene notes")).toHaveValue(
        "This conversation's scene."
      )
      expect(io.get).toHaveBeenCalledWith("actorSettings:server:scene-one")
    })

    it("retires a held load when the destination changes", async () => {
      let release!: (value: unknown) => void
      io.get.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            release = resolve
          })
      )
      io.values.set("actorSettings:server:scene-two", scene("Second scene."))
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(io.get).toHaveBeenCalledWith("actorSettings:server:scene-one")
      )
      act(() => target("scene-two"))
      view.rerenderOpen(true)
      await waitFor(() =>
        expect(io.get).toHaveBeenCalledWith("actorSettings:server:scene-two")
      )
      await act(async () => release(scene("Late first scene.")))
      expect(await screen.findByLabelText("Scene notes")).toHaveValue(
        "Second scene."
      )
      expect(useActorStore.getState().settings?.notes).toBe("Second scene.")
    })

    it("rejects an older rendered Save after another source has finished loading", async () => {
      io.values.set("actorSettings:server:scene-one", scene("First scene."))
      io.values.set("actorSettings:server:scene-two", scene("Second scene."))
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue("First scene.")
      )
      const oldSave = io.finishes.at(-1)!
      act(() => target("scene-two"))
      view.rerenderOpen(true)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Second scene."
        )
      )
      await act(async () => {
        await oldSave({ actorNotes: "Retired edits." })
      })
      expect(io.set).not.toHaveBeenCalled()
      expect(view.close).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings?.notes).toBe("Second scene.")
    })

    it("retires and reloads a held scene at the actual same-ID H1 boundary", async () => {
      const selection = renderHook(() => useHistorySelection())
      let release!: (value: unknown) => void
      io.get.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            release = resolve
          })
      )
      io.values.set(
        "actorSettings:server:scene-one",
        scene("Current verified scene.")
      )
      const view = mountEditor(kind, true, selection.result.current)
      await waitFor(() =>
        expect(io.get).toHaveBeenCalledWith("actorSettings:server:scene-one")
      )
      act(() => {
        selection.result.current.beginLoad()
      })
      view.rerenderOpen(true)
      await waitFor(() => expect(io.get).toHaveBeenCalledTimes(2))
      await act(async () => release(scene("Retired H1 scene.")))
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Current verified scene."
        )
      )
      expect(useActorStore.getState().settings?.notes).toBe(
        "Current verified scene."
      )
    })

    it("does not reuse a held load after close and reopen with identical IDs", async () => {
      let release!: (value: unknown) => void
      io.get.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            release = resolve
          })
      )
      io.values.set("actorSettings:server:scene-one", scene("Reopened scene."))
      const view = mountEditor(kind)
      await waitFor(() => expect(io.get).toHaveBeenCalledTimes(1))
      view.rerenderOpen(false)
      view.rerenderOpen(true)
      await waitFor(() => expect(io.get).toHaveBeenCalledTimes(2))
      await act(async () => release(scene("Retired closed scene.")))
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Reopened scene."
        )
      )
    })

    it("ignores an older failed scene's Retry after the next chat has unsaved edits", async () => {
      io.get.mockRejectedValueOnce(new Error("read unavailable"))
      io.values.set(
        "actorSettings:server:scene-two",
        scene("Stored second scene.")
      )
      const view = mountEditor(kind)
      await screen.findByRole("alert")
      const oldRetry = io.retries.at(-1)!
      act(() => target("scene-two"))
      view.rerenderOpen(true)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Stored second scene."
        )
      )
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Unsaved second scene." }
      })
      const reads = io.get.mock.calls.length
      await act(async () => {
        await oldRetry()
      })
      expect(io.get).toHaveBeenCalledTimes(reads)
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "Unsaved second scene."
      )
    })

    it("retires a failed read's Retry after a successful same-source retry", async () => {
      io.get.mockRejectedValueOnce(new Error("read unavailable"))
      io.values.set("actorSettings:server:scene-one", scene("Stored scene."))
      mountEditor(kind)
      await screen.findByRole("alert")
      const oldRetry = io.retries.at(-1)!
      fireEvent.click(screen.getByRole("button", { name: /retry/i }))
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Stored scene."
        )
      )
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Unsaved current edit." }
      })
      const reads = io.get.mock.calls.length
      await act(async () => {
        await oldRetry()
      })
      expect(io.get).toHaveBeenCalledTimes(reads)
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "Unsaved current edit."
      )
    })

    it("rejects a retained Save immediately when the verified account retires", async () => {
      let verified = true
      io.accountCurrent = () => verified
      io.values.set("actorSettings:server:scene-one", scene("Current scene."))
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Current scene."
        )
      )
      const save = io.finishes.at(-1)!
      verified = false
      await act(async () => {
        await save({ actorNotes: "Retired account edits." })
      })
      expect(io.set).not.toHaveBeenCalled()
      expect(view.close).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings?.notes).toBe("Current scene.")
    })

    it.each(["account", "character", "restore"] as const)(
      "retires a held read across %s changes and loads the current scene",
      async (boundary) => {
        let verified = true
        io.accountCurrent = () => verified
        let release!: (value: unknown) => void
        io.get.mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              release = resolve
            })
        )
        const view = mountEditor(kind)
        await waitFor(() => expect(io.get).toHaveBeenCalledTimes(1))
        if (boundary === "account") {
          verified = false
          io.accountCurrent = () => true
        } else if (boundary === "character") io.characterId = "character-two"
        else
          act(() =>
            usePlaygroundSessionStore.setState((state) => ({
              restoreRevision: state.restoreRevision + 1
            }))
          )
        io.values.set(
          "actorSettings:server:scene-one",
          scene("Current replacement scene.")
        )
        view.rerenderOpen(true)
        await waitFor(() => expect(io.get).toHaveBeenCalledTimes(2))
        await act(async () => release(scene("Retired read.")))
        await waitFor(() =>
          expect(screen.getByLabelText("Scene notes")).toHaveValue(
            "Current replacement scene."
          )
        )
        expect(useActorStore.getState().settings?.notes).toBe(
          "Current replacement scene."
        )
      }
    )

    it("does not reuse an older rendered Save after New returns to identical IDs", async () => {
      io.values.set("actorSettings:server:scene-one", scene("Initial scene."))
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Initial scene."
        )
      )
      const oldSave = io.finishes.at(-1)!
      act(() => target("scene-one"))
      view.rerenderOpen(true)
      await waitFor(() => expect(io.get).toHaveBeenCalledTimes(2))
      await act(async () => {
        await oldSave({ actorNotes: "Old edits." })
      })
      expect(io.set).not.toHaveBeenCalled()
      expect(view.close).not.toHaveBeenCalled()
    })

    it("retains edits after a rejected storage save and supports current Save retry", async () => {
      io.values.set("actorSettings:server:scene-one", scene("Initial scene."))
      io.set.mockRejectedValueOnce(new Error("write unavailable"))
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Initial scene."
        )
      )
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Kept edited scene." }
      })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Scene settings could not be saved"
      )
      expect(view.close).not.toHaveBeenCalled()
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "Kept edited scene."
      )
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(view.close).toHaveBeenCalledWith(false))
      expect(io.values.get("actorSettings:server:scene-one")).toMatchObject({
        notes: "Kept edited scene."
      })
    })

    it("does not publish or close another scene when a held Save settles late", async () => {
      io.values.set("actorSettings:server:scene-one", scene("Initial scene."))
      io.values.set("actorSettings:server:scene-two", scene("Second scene."))
      let release!: () => void
      io.set.mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            release = resolve
          })
      )
      const view = mountEditor(kind)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Initial scene."
        )
      )
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "First saved edit." }
      })
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(io.set).toHaveBeenCalledTimes(1))
      act(() => target("scene-two"))
      view.rerenderOpen(true)
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "Second scene."
        )
      )
      await act(async () => release())
      expect(view.close).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings?.notes).toBe("Second scene.")
    })

    it("clears the previous Save failure when the next account is unavailable", async () => {
      io.values.set("actorSettings:server:scene-one", scene("Existing scene."))
      io.set.mockRejectedValueOnce(new Error("write unavailable"))
      const view = mountEditor(kind)
      await screen.findByLabelText("Scene notes")
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Scene settings could not be saved"
      )
      io.accountCurrent = () => false
      act(() => target("scene-two"))
      view.rerenderOpen(true)
      await waitFor(() =>
        expect(
          screen.queryByText(/Scene settings could not be saved/)
        ).not.toBeInTheDocument()
      )
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Scene settings could not be loaded"
      )
      expect(io.set).toHaveBeenCalledTimes(1)
    })

    it("reports an unavailable initiating account without leaving scene hydration busy", async () => {
      io.accountCurrent = () => false
      mountEditor(kind)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Scene settings could not be loaded"
      )
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
      expect(
        screen.queryByRole("button", { name: /retry/i })
      ).not.toBeInTheDocument()
      expect(
        screen.getByText(/check your account verification and selected chat/i)
      ).toBeInTheDocument()
    })

    it("disables the actual native scene presets during a held Save", async () => {
      io.realActorEditor = true
      io.values.set("actorSettings:server:scene-one", scene("Current scene."))
      let release!: () => void
      io.set.mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            release = resolve
          })
      )
      const view = mountEditor(kind)
      if (kind === "model")
        fireEvent.click(
          await screen.findByRole("tab", { name: /scene director/i })
        )
      const preset = await screen.findByRole("button", {
        name: /slice of life/i
      })
      expect(preset).toBeEnabled()
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
      await waitFor(() => expect(io.set).toHaveBeenCalledTimes(1))
      expect(preset).toBeDisabled()
      await act(async () => release())
      await waitFor(() => expect(view.close).toHaveBeenCalledWith(false))
    })

    it("stops after a synchronous publication retires the initiating account", async () => {
      let current = true
      io.accountCurrent = () => current
      io.values.set("actorSettings:server:scene-one", scene("Original scene."))
      const view = mountEditor(kind)
      const notes = await screen.findByLabelText("Scene notes")
      fireEvent.change(notes, { target: { value: "Admitted scene." } })
      const preview = useActorStore.getState().preview
      const unsubscribe = useActorStore.subscribe(state => {
        if (state.settings?.notes === "Admitted scene.") current = false
      })
      try {
        fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
        await waitFor(() => expect(io.set).toHaveBeenCalledTimes(1))
        await act(async () => { await Promise.resolve() })
        expect({ closed: view.close.mock.calls.length, preview: useActorStore.getState().preview })
          .toEqual({ closed: 0, preview })
      } finally {
        unsubscribe()
      }
    })

    it("keeps known absence editable with existing defaults", async () => {
      mountEditor(kind)
      expect(await screen.findByLabelText("Scene notes")).toHaveValue("")
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    })
  }
)

it("uses cached model defaults while freshly reading the scene if the model-settings reader rejects", async () => {
  io.modelSettings.mockRejectedValue(new Error("model defaults unavailable"))
  io.values.set(
    "actorSettings:server:scene-one",
    scene("Fresh readable scene.")
  )
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  client.setQueryData(["fetchModelConfig2", true, "character-one"], {
    temperature: 0.7
  })
  mountEditor("model", true, null, client)
  expect(await screen.findByLabelText("Scene notes")).toHaveValue(
    "Fresh readable scene."
  )
  expect(io.modelSettings).not.toHaveBeenCalled()
  expect(io.get).toHaveBeenCalledWith("actorSettings:server:scene-one")
  expect(screen.queryByRole("alert")).not.toBeInTheDocument()
})


it("stops model-default writes if scope publication retires the source", async () => {
  let current = true
  io.accountCurrent = () => current
  useStoreChatModelSettings.getState().reset()
  useStoreChatModelSettings.getState().setActiveSettingsScope("foreign:model")
  const view = mountEditor("model")
  const temperature = await screen.findByLabelText("Temperature")
  fireEvent.change(temperature, { target: { value: "0.22" } })
  const unsubscribe = useStoreChatModelSettings.subscribe(state => {
    if (state.activeSettingsScope !== "foreign:model") current = false
  })
  try {
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
    await waitFor(() => expect(io.set).toHaveBeenCalledTimes(1))
    await act(async () => { await Promise.resolve() })
    expect({ closed: view.close.mock.calls.length, temperature: useStoreChatModelSettings.getState().temperature })
      .toEqual({ closed: 0, temperature: undefined })
  } finally {
    unsubscribe()
  }
})

it("does not apply a held Scene Director reset to a newly selected scene", async () => {
  io.realActorEditor = true
  useActorEditorPrefs.getState().setEditorMode("advanced")
  io.values.set("actorSettings:server:scene-one", scene("Original scene."))
  io.values.set("actorSettings:server:scene-two", scene("New scene."))
  mountEditor("model")
  fireEvent.click(await screen.findByRole("tab", { name: /scene director/i }))
  fireEvent.click(await screen.findByRole("button", { name: "Reset for this chat" }))
  const confirm = await screen.findByRole("button", { name: "Confirm", exact: true })
  act(() => target("scene-two"))
  await waitFor(() => expect(useActorStore.getState().settings?.notes).toBe("New scene."))
  fireEvent.click(confirm)
  expect(screen.getAllByLabelText("Scene notes").map(input => (input as HTMLInputElement).value))
    .toEqual(["New scene."])
}, 15_000)


it("keeps Scene Director presets staged when Cancel is chosen", async () => {
  io.realActorEditor = true
  useActorEditorPrefs.getState().setEditorMode("advanced")
  io.values.set("actorSettings:server:scene-one", scene("Original scene."))
  const view = mountEditor("actor")
  const preset = await screen.findByRole("button", { name: /slice of life/i })
  const before = useActorStore.getState().settings
  fireEvent.click(preset)
  fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
  expect({ closed: view.close.mock.calls, scene: useActorStore.getState().settings, writes: io.set.mock.calls })
    .toEqual({ closed: [[false]], scene: before, writes: [] })
}, 15_000)

it("keeps a newer Scene Director edit when a held character profile returns", async () => {
  io.realActorEditor = true
  useActorEditorPrefs.getState().setEditorMode("advanced")
  io.values.set("actorSettings:server:scene-one", scene("Original scene."))
  let release!: (value: ReturnType<typeof scene>) => void
  const pending = new Promise<ReturnType<typeof scene>>(resolve => { release = resolve })
  io.get.mockImplementation((key: string) => key === "actorProfile:character-one" ? pending : Promise.resolve(io.values.get(key)))
  mountEditor("actor")
  fireEvent.click(await screen.findByRole("button", { name: /apply.*default/i }))
  await waitFor(() => expect(io.get).toHaveBeenCalledWith("actorProfile:character-one"))
  fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
  const notes = await screen.findByLabelText("Scene notes")
  fireEvent.change(notes, { target: { value: "Newer edit." } })
  await act(async () => { release(scene("Old profile.")); await pending })
  expect(notes).toHaveValue("Newer edit.")
}, 15_000)


it("keeps the standalone scene draft when dirty-close is cancelled", async () => {
  io.values.set("actorSettings:server:scene-one", scene("Original scene."))
  const view = mountEditor("actor")
  const notes = await screen.findByLabelText("Scene notes")
  fireEvent.change(notes, { target: { value: "Staged scene." } })
  fireEvent.click(screen.getByRole("button", { name: "Close", exact: true }))
  fireEvent.click(await screen.findByRole("button", { name: "Keep editing", exact: true }))
  expect({ closed: view.close.mock.calls.length, notes: (notes as HTMLInputElement).value, writes: io.set.mock.calls })
    .toEqual({ closed: 0, notes: "Staged scene.", writes: [] })
})


it.each(["actor", "model"] as const)("keeps a held %s scene Save unchanged when an older profile read returns", async kind => {
  io.realActorEditor = true
  useActorEditorPrefs.getState().setEditorMode("advanced")
  io.values.set("actorSettings:server:scene-one", scene("Original scene."))
  let releaseProfile!: (value: ReturnType<typeof scene>) => void
  const profile = new Promise<ReturnType<typeof scene>>(resolve => { releaseProfile = resolve })
  io.get.mockImplementation((key: string) => key === "actorProfile:character-one" ? profile : Promise.resolve(io.values.get(key)))
  let releaseSave!: () => void
  const saved = new Promise<void>(resolve => { releaseSave = resolve })
  io.set.mockImplementationOnce(() => saved)
  const view = mountEditor(kind)
  if (kind === "model") fireEvent.click(await screen.findByRole("tab", { name: /scene director/i }))
  fireEvent.click(await screen.findByRole("button", { name: /apply.*default/i }))
  await waitFor(() => expect(io.get).toHaveBeenCalledWith("actorProfile:character-one"))
  fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
  const notes = await screen.findByLabelText("Scene notes")
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
  await waitFor(() => expect(io.set).toHaveBeenCalledTimes(1))
  await act(async () => { releaseProfile(scene("Returned profile.")); await profile })
  expect(notes).toHaveValue("Original scene.")
  await act(async () => { releaseSave(); await saved })
  await waitFor(() => expect(view.close).toHaveBeenCalledWith(false))
  expect(useActorStore.getState().settings?.notes).toBe("Original scene.")
}, 15_000)


const defaultActionScene = (notes: string) => {
  const settings = scene(notes)
  return { ...settings, aspects: settings.aspects.slice(0, 1) }
}
const mountDefaultEditor = async (
  kind: "actor" | "model",
  initial = defaultActionScene("Current staged scene.")
) => {
  io.realActorEditor = true
  useActorEditorPrefs.getState().setEditorMode("advanced")
  io.values.set("actorSettings:server:scene-one", initial)
  const view = mountEditor(kind)
  await waitFor(() =>
    expect(useActorStore.getState().settings?.notes).toBe(initial.notes)
  )
  if (kind === "model")
    fireEvent.click(await screen.findByRole("tab", { name: /scene director/i }))
  const aspects = await screen.findByRole("button", { name: /^aspects/i })
  if (aspects.getAttribute("aria-expanded") !== "true") fireEvent.click(aspects)
  const apply = await screen.findByRole("button", { name: /apply.*default/i })
  const save = screen.getByRole("button", { name: /save as.*default/i })
  return { ...view, save, apply }
}

describe.each(["actor", "model"] as const)(
  "%s explicit character defaults",
  (kind) => {
    it("keeps the staged scene and reports a failed default save until explicit retry ACK", async () => {
      const view = await mountDefaultEditor(kind)
      const original = scene("Previous default.")
      io.values.set("actorProfile:character-one", original)
      io.set.mockRejectedValueOnce(new Error("profile write unavailable"))
      fireEvent.click(view.save)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Character default could not be saved"
      )
      expect(io.values.get("actorProfile:character-one")).toEqual(original)
      expect(view.close).not.toHaveBeenCalled()
      fireEvent.click(view.save)
      expect(
        await screen.findByText("Character default saved on this device.")
      ).toHaveAttribute("role", "status")
      expect(io.values.get("actorProfile:character-one")).toMatchObject({
        notes: "Current staged scene."
      })
      expect(io.values.get("actorSettings:server:scene-one")).toEqual(
        defaultActionScene("Current staged scene.")
      )
    }, 15_000)

    it("refuses a default Save after persistent storage becomes unavailable", async () => {
      const view = await mountDefaultEditor(kind)
      io.persistent = false
      fireEvent.click(view.save)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Character default could not be saved"
      )
      expect(io.set).not.toHaveBeenCalled()
    }, 15_000)

    it("admits one pending default action and waits for actual write acknowledgement", async () => {
      const view = await mountDefaultEditor(kind)
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      io.set.mockImplementationOnce(async (key: string, value: unknown) => {
        await held
        io.values.set(key, value)
      })
      act(() => {
        view.save.click()
        view.save.click()
        view.apply.click()
      })
      await waitFor(() => expect(io.set).toHaveBeenCalledOnce())
      expect(
        await screen.findByText("Saving character default...")
      ).toHaveAttribute("role", "status")
      expect(view.save).toBeDisabled()
      expect(view.apply).toBeDisabled()
      expect(
        screen.queryByText("Character default saved on this device.")
      ).not.toBeInTheDocument()
      await act(async () => {
        release()
        await held
      })
      expect(
        await screen.findByText("Character default saved on this device.")
      ).toBeInTheDocument()
      expect(io.set).toHaveBeenCalledOnce()
      expect(io.get).not.toHaveBeenCalledWith("actorProfile:character-one")
    }, 15_000)

    it.each(["failed", "absent"])(
      "distinguishes a %s default read from a staged application",
      async (outcome) => {
        const view = await mountDefaultEditor(kind)
        if (outcome === "failed")
          io.get.mockImplementation(async (key: string) => {
            if (key === "actorProfile:character-one")
              throw new Error("profile read unavailable")
            return io.values.get(key)
          })
        fireEvent.click(view.apply)
        if (outcome === "failed")
          expect(await screen.findByRole("alert")).toHaveTextContent(
            "Character default could not be loaded"
          )
        else
          expect(
            await screen.findByText(
              "No saved default exists for this character on this device."
            )
          ).toHaveAttribute("role", "status")
        expect(io.set).not.toHaveBeenCalled()
        expect(useActorStore.getState().settings?.notes).toBe(
          "Current staged scene."
        )
      },
      15_000
    )

    it.each([true, false])(
      "stages enabled=%s with the default and preserves it through scene Save",
      async (enabled) => {
        const view = await mountDefaultEditor(kind, {
          ...defaultActionScene("Current staged scene."),
          isEnabled: !enabled
        })
        const profile = { ...scene("Profile scene."), isEnabled: enabled }
        io.values.set("actorProfile:character-one", profile)
        fireEvent.click(view.apply)
        fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
        await waitFor(() =>
          expect(screen.getByLabelText("Scene notes")).toHaveValue(
            profile.notes
          )
        )
        expect(io.set).not.toHaveBeenCalled()
        fireEvent.click(screen.getByRole("button", { name: /^save$/i }))
        await waitFor(() => expect(view.close).toHaveBeenCalledWith(false))
        expect(io.values.get("actorSettings:server:scene-one")).toEqual(profile)
      },
      15_000
    )

    it("suppresses late default Save feedback after actual New returns to the same IDs", async () => {
      const view = await mountDefaultEditor(kind)
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      io.set.mockReturnValueOnce(held)
      fireEvent.click(view.save)
      await waitFor(() => expect(io.set).toHaveBeenCalledOnce())
      act(() => target("scene-one"))
      await act(async () => {
        release()
        await held
      })
      expect(
        screen.queryByText("Character default saved on this device.")
      ).not.toBeInTheDocument()
      expect(view.close).not.toHaveBeenCalled()
    }, 15_000)

    it.each(["same-value", "A/B/A"])(
      "retires a held default Apply on actual %s assistant operation revisions",
      async (change) => {
        const actual = await vi.importActual<
          typeof import("@/hooks/useSelectedAssistant")
        >("@/hooks/useSelectedAssistant")
        const selection = renderHook(() => actual.useSelectedAssistant())
        await waitFor(() =>
          expect(selection.result.current[2].isLoading).toBe(false)
        )
        const a = {
          kind: "character" as const,
          id: "character-one",
          name: "Mira"
        }
        await act(async () => {
          expect(await selection.result.current[1](a)).toBe(true)
        })
        const view = await mountDefaultEditor(kind)
        let release!: (value: ReturnType<typeof scene>) => void
        const held = new Promise<ReturnType<typeof scene>>((resolve) => {
          release = resolve
        })
        io.get.mockImplementation((key: string) =>
          key === "actorProfile:character-one"
            ? held
            : Promise.resolve(io.values.get(key))
        )
        fireEvent.click(view.apply)
        await waitFor(() =>
          expect(io.get).toHaveBeenCalledWith("actorProfile:character-one")
        )
        await act(async () => {
          if (change === "A/B/A")
            expect(
              await selection.result.current[1]({ ...a, id: "character-two" })
            ).toBe(true)
          expect(await selection.result.current[1](a)).toBe(true)
          release(scene("Retired profile."))
          await held
        })
        fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
        expect(await screen.findByLabelText("Scene notes")).toHaveValue(
          "Current staged scene."
        )
        expect(
          screen.queryByText(
            "Character default applied to this draft. Save applies it to this chat."
          )
        ).not.toBeInTheDocument()
      },
      15_000
    )

    it("reports a present malformed falsy default as a load error and retains it", async () => {
      const view = await mountDefaultEditor(kind)
      io.values.set("actorProfile:character-one", false)
      fireEvent.click(view.apply)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Character default could not be loaded"
      )
      expect(io.values.get("actorProfile:character-one")).toBe(false)
      expect(io.set).not.toHaveBeenCalled()
      expect(useActorStore.getState().settings?.notes).toBe(
        "Current staged scene."
      )
    }, 15_000)

    it("retires the saved-default status after a later edit even when its value is restored", async () => {
      const view = await mountDefaultEditor(kind)
      fireEvent.click(view.save)
      expect(
        await screen.findByText("Character default saved on this device.")
      ).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
      const notes = await screen.findByLabelText("Scene notes")
      fireEvent.change(notes, { target: { value: "A later staged edit." } })
      expect(
        screen.queryByText("Character default saved on this device.")
      ).not.toBeInTheDocument()
      fireEvent.change(notes, { target: { value: "Current staged scene." } })
      expect(
        screen.queryByText("Character default saved on this device.")
      ).not.toBeInTheDocument()
      expect(io.set).toHaveBeenCalledOnce()
    }, 15_000)

    it("shows its own staged-default status and retires it after a later Reset", async () => {
      const view = await mountDefaultEditor(kind)
      io.values.set(
        "actorProfile:character-one",
        scene("Applied profile scene.")
      )
      fireEvent.click(view.apply)
      expect(
        await screen.findByText(
          "Character default applied to this draft. Save applies it to this chat."
        )
      ).toBeInTheDocument()
      fireEvent.click(
        screen.getByRole("button", { name: "Reset for this chat" })
      )
      fireEvent.click(
        await screen.findByRole("button", { name: "Confirm", exact: true })
      )
      expect(
        screen.queryByText(
          "Character default applied to this draft. Save applies it to this chat."
        )
      ).not.toBeInTheDocument()
      expect(io.set).not.toHaveBeenCalled()
    }, 15_000)

    it.each(["save", "apply"] as const)(
      "retires held %s feedback on a later edit while keeping physical admission until settlement",
      async (action) => {
        const view = await mountDefaultEditor(kind)
        let release!: (value: ReturnType<typeof scene>) => void
        const held = new Promise<ReturnType<typeof scene>>((resolve) => {
          release = resolve
        })
        if (action === "save") io.set.mockReturnValueOnce(held)
        else
          io.get.mockImplementation((key: string) =>
            key === "actorProfile:character-one"
              ? held
              : Promise.resolve(io.values.get(key))
          )
        fireEvent.click(action === "save" ? view.save : view.apply)
        const pendingMessage =
          action === "save"
            ? "Saving character default..."
            : "Loading character default..."
        expect(await screen.findByText(pendingMessage)).toBeInTheDocument()
        fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
        const notes = await screen.findByLabelText("Scene notes")
        fireEvent.change(notes, {
          target: { value: "Later edit during default IO." }
        })
        expect(screen.queryByText(pendingMessage)).not.toBeInTheDocument()
        fireEvent.click(screen.getByRole("button", { name: /^aspects/i }))
        const pendingSave = screen.getByRole("button", {
          name: /save as.*default/i
        })
        const pendingApply = screen.getByRole("button", {
          name: /apply.*default/i
        })
        expect(pendingSave).toBeDisabled()
        expect(pendingApply).toBeDisabled()
        expect(
          screen.getByText(
            "Waiting for the previous character default action..."
          )
        ).toHaveAttribute("role", "status")
        await act(async () => {
          release(scene("Retired profile."))
          await held
        })
        await waitFor(() => expect(pendingSave).not.toBeDisabled())
        expect(pendingApply).not.toBeDisabled()
        fireEvent.click(screen.getByRole("button", { name: /^scene notes/i }))
        expect(await screen.findByLabelText("Scene notes")).toHaveValue(
          "Later edit during default IO."
        )
        expect(
          screen.queryByText("Character default saved on this device.")
        ).not.toBeInTheDocument()
        expect(
          screen.queryByText(
            "Character default applied to this draft. Save applies it to this chat."
          )
        ).not.toBeInTheDocument()
      },
      15_000
    )

    it("refuses retained Save entry after a structural edit before render and accepts a freshly rendered retry", async () => {
      const view = await mountDefaultEditor(kind)
      fireEvent.change(
        screen.getByPlaceholderText("Aspect name (e.g., Mood, Role, Location)"),
        { target: { value: "New continuity" } }
      )
      const add = screen.getByRole("button", { name: "Add aspect" })
      act(() => {
        add.click()
        view.save.click()
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(io.set).not.toHaveBeenCalled()
      fireEvent.click(screen.getByRole("button", { name: /save as.*default/i }))
      expect(
        await screen.findByText("Character default saved on this device.")
      ).toBeInTheDocument()
      expect(io.values.get("actorProfile:character-one")).toMatchObject({
        aspects: expect.arrayContaining([
          expect.objectContaining({ name: "New continuity" })
        ])
      })
    }, 15_000)

    it("saves an immediate free-aspect form edit without waiting for preview recomputation", async () => {
      const view = await mountDefaultEditor(kind)
      const field = screen.getAllByPlaceholderText(
        "Describe briefly (optional)"
      )[0]
      act(() => {
        fireEvent.change(field, { target: { value: "Immediate role edit." } })
        view.save.click()
      })
      expect(
        await screen.findByText("Character default saved on this device.")
      ).toBeInTheDocument()
      expect(io.values.get("actorProfile:character-one")).toMatchObject({
        aspects: expect.arrayContaining([
          expect.objectContaining({
            id: "user_role",
            value: "Immediate role edit."
          })
        ])
      })
    }, 15_000)
  }
)

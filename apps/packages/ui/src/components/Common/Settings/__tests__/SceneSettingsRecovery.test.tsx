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
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useActorStore } from "@/store/actor"
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
    hasPersistentBackend: true,
    get: io.get,
    set: io.set,
    getAll: async () => ({}),
    watch: () => () => undefined
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
    { isLoading: false }
  ]
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
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

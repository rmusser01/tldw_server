// @vitest-environment jsdom
import React from "react"
import { act, renderHook as renderHookBase, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { executeSlashCommand, useSlashCommands } from "../useSlashCommands"
import { useConnectionStore } from "@/store/connection"
import { OPEN_PROMPT_SELECT_EVENT } from "@/utils/prompt-select-events"

const commandIo = vi.hoisted(() => ({
  listChatCommands: vi.fn<() => Promise<unknown>>(async () => ({ commands: [] })),
  startSavedChat: vi.fn(() => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { listChatCommands: () => commandIo.listChatCommands() }
}))

const connectedState = useConnectionStore.getState().state
let queryClient: QueryClient
const renderHook = <Result, Props>(hook: (props: Props) => Result, options: { initialProps?: Props } = {}) =>
  renderHookBase(hook, {
    ...options,
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
  })

beforeEach(() => {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  commandIo.listChatCommands.mockReset()
  commandIo.listChatCommands.mockResolvedValue({ commands: [] })
  commandIo.startSavedChat.mockClear()
  useConnectionStore.setState({ state: { ...connectedState, isConnected: true } })
})
afterEach(() => {
  queryClient.clear()
  useConnectionStore.setState({ state: connectedState })
})

const baseParams = {
  chatMode: "normal",
  setChatMode: vi.fn(),
  webSearch: false,
  setWebSearch: vi.fn(),
  handleImageUpload: vi.fn(),
  imageBackendDefaultTrimmed: "",
  imageBackendLabel: "none",
  currentMessage: "/",
  onNewChat: () => {
    commandIo.startSavedChat()
  }
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string,
      values?: Record<string, unknown>
    ) => {
      const base = fallback || key
      if (!values) return base
      return Object.entries(values).reduce((acc, [name, value]) => {
        return acc.replaceAll(`{{${name}}}`, String(value))
      }, base)
    }
  })
}))

describe("playground useSlashCommands", () => {
  it("shows slash suggestions with concise descriptions and examples", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "",
        imageBackendLabel: "none",
        setOpenModelSettings: vi.fn(),
        currentMessage: "/gen"
      })
    )

    expect(result.current.showSlashMenu).toBe(true)
    const generateImage = result.current.filteredSlashCommands.find(
      (command) => command.command === "generate-image"
    )
    expect(generateImage).toBeDefined()
    expect(generateImage?.description).toContain("/generate-image:<provider>")
  })

  it("resolves a slash command without running it until it is executed (CC-N1)", () => {
    const setChatMode = vi.fn()
    const setWebSearch = vi.fn()
    const setOpenModelSettings = vi.fn()
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode,
        webSearch: false,
        setWebSearch,
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "flux",
        imageBackendLabel: "Flux",
        setOpenModelSettings,
        currentMessage: "/web"
      })
    )

    const intent = result.current.resolveSubmissionIntent("/web latest news")
    result.current.resolveSubmissionIntent("/model")

    expect(intent).toMatchObject({
      handled: true,
      isImageCommand: false,
      message: "latest news"
    })
    expect(setWebSearch).not.toHaveBeenCalled()
    expect(setOpenModelSettings).not.toHaveBeenCalled()

    act(() => {
      executeSlashCommand(intent)
    })
    expect(setWebSearch).toHaveBeenCalledTimes(1)
    expect(setWebSearch).toHaveBeenCalledWith(true)
  })

  it("leaves plain messages and unknown commands unhandled", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "",
        imageBackendLabel: "none",
        setOpenModelSettings: vi.fn(),
        currentMessage: ""
      })
    )

    const plain = result.current.resolveSubmissionIntent("hello")
    const unknown = result.current.resolveSubmissionIntent("/unknown thing")
    expect(plain).toMatchObject({ handled: false, message: "hello" })
    expect(plain.slashCommand).toBeUndefined()
    expect(unknown).toMatchObject({ handled: false, message: "/unknown thing" })
    expect(() => executeSlashCommand(unknown)).not.toThrow()
  })

  it("parses image commands", () => {
    const { result } = renderHook(() =>
      useSlashCommands({
        chatMode: "normal",
        setChatMode: vi.fn(),
        webSearch: false,
        setWebSearch: vi.fn(),
        handleImageUpload: vi.fn(),
        imageBackendDefaultTrimmed: "flux",
        imageBackendLabel: "Flux",
        setOpenModelSettings: vi.fn(),
        currentMessage: ""
      })
    )

    const imageIntent = result.current.resolveSubmissionIntent(
      "/generate-image:flux city skyline"
    )
    expect(imageIntent).toMatchObject({
      handled: true,
      isImageCommand: true,
      imageBackendOverride: "flux",
      message: "city skyline"
    })
  })
})

// CC-11 (slice a): /new and /prompt, plus the server's chat commands.
describe("playground useSlashCommands: /new, /prompt and server commands", () => {
  it("starts a new saved chat with /new, the header's New chat path", () => {
    const { result } = renderHook(() => useSlashCommands(baseParams))
    const command = result.current.filteredSlashCommands.find((item) => item.command === "new")
    expect(command).toBeDefined()
    const setField = vi.fn()
    act(() => {
      result.current.handleSlashCommandSelect(command!, setField, { current: null })
    })
    expect(commandIo.startSavedChat).toHaveBeenCalledTimes(1)
    expect(setField).toHaveBeenCalledWith("message", "")
  })

  it("runs /new on its own: the text after it is not sent into the new chat", () => {
    const { result } = renderHook(() => useSlashCommands(baseParams))
    const intent = result.current.resolveSubmissionIntent("/new tell me a joke")
    expect(intent).toMatchObject({ handled: true })
    expect(intent.slashCommand).toMatchObject({ command: "new", standalone: true })
    expect(commandIo.startSavedChat).not.toHaveBeenCalled()
    executeSlashCommand(intent)
    expect(commandIo.startSavedChat).toHaveBeenCalledTimes(1)
  })

  it("opens the prompt picker with /prompt", () => {
    const opened: Event[] = []
    const listener = (event: Event) => opened.push(event)
    window.addEventListener(OPEN_PROMPT_SELECT_EVENT, listener)
    try {
      const { result } = renderHook(() => useSlashCommands({ ...baseParams, currentMessage: "/pro" }))
      const command = result.current.filteredSlashCommands.find((item) => item.command === "prompt")
      expect(command).toBeDefined()
      const intent = result.current.resolveSubmissionIntent("/prompt")
      expect(opened).toHaveLength(0)
      executeSlashCommand(intent)
      expect(opened).toHaveLength(1)
      expect((opened[0] as CustomEvent).detail).toMatchObject({ source: "slash-command" })
    } finally {
      window.removeEventListener(OPEN_PROMPT_SELECT_EVENT, listener)
    }
  })

  it("opens the note picker with /note by leaving \"/note \" for it to search (decision [4])", () => {
    const { result } = renderHook(() => useSlashCommands({ ...baseParams, currentMessage: "/no" }))
    const command = result.current.filteredSlashCommands.find((item) => item.command === "note")
    expect(command).toMatchObject({ label: "Attach a note", insertText: "/note " })
    const setField = vi.fn()
    result.current.handleSlashCommandSelect(command!, setField, { current: null })
    expect(setField).toHaveBeenCalledWith("message", "/note ")
  })

  it("clears the chat's saved draft before a standalone command runs, then places the text after it", () => {
    const calls: string[] = []
    const { result } = renderHook(() =>
      useSlashCommands({
        ...baseParams,
        onNewChat: () => calls.push("new chat"),
        clearSourceDraft: () => calls.push("clear source draft")
      })
    )
    const intent = result.current.resolveSubmissionIntent("/new and then this")
    result.current.runStandaloneCommand(intent.slashCommand!, intent.message, (value) =>
      calls.push(`message:${value}`)
    )
    expect(calls).toEqual(["clear source draft", "message:", "new chat", "message:and then this"])
  })

  it("passes the text after /prompt to the picker as its search", () => {
    const opened: CustomEvent[] = []
    const listener = (event: Event) => opened.push(event as CustomEvent)
    window.addEventListener(OPEN_PROMPT_SELECT_EVENT, listener)
    try {
      const clearSourceDraft = vi.fn()
      const { result } = renderHook(() => useSlashCommands({ ...baseParams, clearSourceDraft }))
      const intent = result.current.resolveSubmissionIntent("/prompt weekly review")
      expect(intent.slashCommand).toMatchObject({ command: "prompt", standalone: true, consumesArgument: true })
      const messages: string[] = []
      result.current.runStandaloneCommand(intent.slashCommand!, intent.message, (value) => messages.push(value))
      expect(clearSourceDraft).toHaveBeenCalledTimes(1)
      expect(opened[0]?.detail).toMatchObject({ source: "slash-command", search: "weekly review" })
      expect(messages).toEqual(["", ""])
    } finally {
      window.removeEventListener(OPEN_PROMPT_SELECT_EVENT, listener)
    }
  })

  it("lists the server's chat commands and sends them as typed", async () => {
    commandIo.listChatCommands.mockResolvedValue({
      commands: [
        { name: "time", description: "Show the current time" },
        { name: "weather", description: "Weather for a place", usage: "/weather <place>" }
      ]
    })
    const { result } = renderHook(() => useSlashCommands(baseParams))
    // Server commands follow the local ones, so the highlighted row stays put.
    await waitFor(() =>
      expect(result.current.filteredSlashCommands.map((item) => item.command)).toEqual([
        "search", "web", "vision", "generate-image", "model", "new", "prompt", "note", "time", "weather"
      ])
    )
    const time = result.current.filteredSlashCommands.find((item) => item.command === "time")!
    expect(time).toMatchObject({ source: "server", description: "Show the current time" })
    expect(result.current.filteredSlashCommands.find((item) => item.command === "weather")).toMatchObject({
      description: "Weather for a place · /weather <place>"
    })
    expect(time.action).toBeUndefined()
    // No local action: the draft goes to the server, which runs the command.
    expect(result.current.resolveSubmissionIntent("/time UTC")).toMatchObject({
      handled: false,
      message: "/time UTC"
    })
    // Picking it from the menu leaves "/time " in the composer for its arguments.
    const setField = vi.fn()
    act(() => {
      result.current.handleSlashCommandSelect(time, setField, { current: null })
    })
    expect(setField).toHaveBeenCalledWith("message", "/time ")
  })

  it.each([
    ["an empty list", async () => ({ commands: [] })],
    ["a failed read", async () => { throw new Error("403 Forbidden") }]
  ])("keeps the local commands quietly after %s", async (_label, read) => {
    commandIo.listChatCommands.mockImplementation(read)
    const { result } = renderHook(() => useSlashCommands(baseParams))
    await waitFor(() => expect(commandIo.listChatCommands).toHaveBeenCalled())
    await act(async () => {})
    expect(result.current.filteredSlashCommands.map((item) => item.command)).toEqual([
      "search", "web", "vision", "generate-image", "model", "new", "prompt", "note"
    ])
  })

  // Review: /chat makes no commands request on load; the first "/" asks once.
  it("asks the server for its commands only once the slash menu first opens", async () => {
    const { rerender } = renderHook(({ message }) => useSlashCommands({ ...baseParams, currentMessage: message }), {
      initialProps: { message: "" }
    })
    await act(async () => {})
    expect(commandIo.listChatCommands).not.toHaveBeenCalled()
    rerender({ message: "/" })
    await waitFor(() => expect(commandIo.listChatCommands).toHaveBeenCalledTimes(1))
    rerender({ message: "" })
    rerender({ message: "/t" })
    await act(async () => {})
    expect(commandIo.listChatCommands).toHaveBeenCalledTimes(1)
  })

  it("keeps a local command's behaviour when the server lists the same name", async () => {
    commandIo.listChatCommands.mockResolvedValue({ commands: [{ name: "new", description: "Server new" }] })
    const { result } = renderHook(() => useSlashCommands(baseParams))
    await waitFor(() =>
      expect(result.current.filteredSlashCommands.find((item) => item.command === "new")).toMatchObject({ source: "server" })
    )
    const intent = result.current.resolveSubmissionIntent("/new and more")
    expect(intent.slashCommand).toMatchObject({ command: "new", standalone: true, description: "Server new" })
  })

  it("does not ask the server while disconnected", async () => {
    useConnectionStore.setState({ state: { ...connectedState, isConnected: false } })
    renderHook(() => useSlashCommands(baseParams))
    await act(async () => {})
    expect(commandIo.listChatCommands).not.toHaveBeenCalled()
  })
})

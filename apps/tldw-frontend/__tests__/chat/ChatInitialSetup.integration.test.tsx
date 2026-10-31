// WebUI cookie/session authority and storage readback require the WebUI runtime aliases.
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { createMemoryRouter, RouterProvider } from "react-router-dom"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { useHistorySelection, HistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { TldwSettings } from "@/components/Option/Settings/tldw"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import * as servicePrompts from "@/services/service-prompts"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { conversationDraftKey } from "@/utils/chat-draft-address"
import { SETTINGS_NAVIGATION_REQUEST_EVENT, type SettingsNavigationRequestDetail } from "@/utils/settings-return"
import { activateCookieSessionConfig, clearRuntimeAuthOverride } from "@/services/tldw/runtime-auth-override"

const mocks = vi.hoisted(() => ({ bgRequest: vi.fn(), probe: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest, bgStream: vi.fn(), bgUpload: vi.fn() }))
vi.mock("@/components/Option/Settings/server-health-probe", () => ({ probeServerHealth: mocks.probe }))
vi.mock("@/components/Common/ServerOverviewHint", () => ({ ServerOverviewHint: () => null }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback: unknown) => typeof fallback === "string" ? fallback : key }) }))

const originalLocks = Object.getOwnPropertyDescriptor(navigator, "locks")
const restoreMetadata = vi.fn()
let editor: ReturnType<typeof useComposerText>
function Composer() {
  const address = useStoreMessageOption(state => state.chatDraftAddress)
  React.useLayoutEffect(() => {
    if (!useStoreMessageOption.getState().chatDraftAddress) useStoreMessageOption.getState().startNewChatDraft()
  }, [address])
  const composer = useComposerText({
    draftKey: address ? conversationDraftKey("tldw:playgroundChatDraft", address) : "tldw:playgroundChatDraft",
    draftAddress: address || undefined,
    recoverySurface: "tldw:playgroundChatDraft",
    draftEnabled: address !== null,
    tabScopedDraft: true,
    textareaRef: React.createRef<HTMLTextAreaElement>(),
    getDraftMetadata: () => ({ wasExpanded: true }),
    restoreWithMetadata: restoreMetadata
  })
  React.useLayoutEffect(() => { editor = composer }, [composer])
  const [begin, setBegin] = React.useState<SettingsNavigationRequestDetail["beginInitialSetup"]>()
  const [saving, setSaving] = React.useState(false)
  const [open, setOpen] = React.useState(false)
  return <>
    <textarea aria-label="Chat input" {...composer.form.getInputProps("message")} />
    <output data-testid="saving">{String(saving)}</output>
    <button onClick={() => {
      const detail: SettingsNavigationRequestDetail = { destination: "/settings/tldw" }
      window.dispatchEvent(new CustomEvent(SETTINGS_NAVIGATION_REQUEST_EVENT, { cancelable: true, detail }))
      setBegin(() => detail.beginInitialSetup)
      setOpen(true)
    }}>Set up server</button>
    {open && <TldwSettings beginInitialSetup={begin} assertInitialSetupAuthority={() => {}} onSavingChange={setSaving} />}
  </>
}
function Harness() {
  const selection = useHistorySelection()
  return <HistorySelectionContext.Provider value={selection}><Composer /></HistorySelectionContext.Provider>
}
const mount = () => render(<ConfigProvider><App><RouterProvider router={createMemoryRouter([{ path: "/", element: <Harness /> }])} /></App></ConfigProvider>)

beforeEach(() => {
  restoreMetadata.mockClear()
  localStorage.clear()
  sessionStorage.clear()
  activateCookieSessionConfig()
  clearRuntimeAuthOverride()
  ;(tldwClient as unknown as { config: unknown }).config = null
  useStoreMessageOption.setState({ chatDraftAddress: null, chatDraftTransition: null, chatDraftRevision: 0, messages: [], history: [], historyId: null, serverChatId: null, queuedMessages: [], isLoading: false, isProcessing: false, streaming: false, temporaryChat: false })
  usePlaygroundSessionStore.getState().clearSession()
  usePlaygroundSessionStore.setState({ sourceSelectionRevision: 0 })
  vi.stubEnv("NEXT_PUBLIC_API_URL", "https://separate-api.example.test")
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
  for (const key of ["NEXT_PUBLIC_X_API_KEY", "NEXT_PUBLIC_API_BEARER", "VITE_TLDW_API_KEY", "VITE_TLDW_DEFAULT_API_KEY"]) vi.stubEnv(key, "")
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: async (_name: string, work: () => unknown) => await work() } })
  vi.spyOn(tldwAuth, "testApiKey").mockResolvedValue(true)
  mocks.bgRequest.mockReset().mockResolvedValue({ paths: {} })
  mocks.probe.mockReset().mockResolvedValue({ coreStatus: "healthy", ragStatus: "ready", detail: "Ready" })
  vi.spyOn(console, "warn").mockImplementation(() => {})
  vi.spyOn(console, "error").mockImplementation(() => {})
})
afterEach(() => {
  cleanup(); vi.restoreAllMocks(); vi.unstubAllEnvs(); vi.unstubAllGlobals()
  if (originalLocks) Object.defineProperty(navigator, "locks", originalLocks)
  else Reflect.deleteProperty(navigator, "locks")
})

it.each([
  { message: "FIRST IDEA\nSECOND LINE", image: "", imageFilename: "" },
  { message: "", image: "data:image/png;base64,synthetic-image", imageFilename: "" }
])("retains exact fresh input after real settings/client/Core/owner Save: %j", async values => {
  const set = vi.spyOn(Storage.prototype, "set")
  mount()
  await act(async () => { await vi.dynamicImportSettled() })
  act(() => editor.form.setValues(values))
  const source = useStoreMessageOption.getState().chatDraftAddress
  fireEvent.click(screen.getByRole("button", { name: "Set up server" }))
  fireEvent.change(await serverUrlInput(), { target: { value: "https://initial-api.example.test" } })
  fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "synthetic-initial-key" } })
  expect(set.mock.calls.some(([key]) => key.startsWith("registry:draft:") && key.includes("unresolved"))).toBe(false)
  save()
  await waitFor(() => expect(screen.getByTestId("saving")).toHaveTextContent("false"))
  await waitFor(() => expect(editor.form.values).toEqual(values))
  expect(restoreMetadata.mock.calls.at(-1)).toEqual([values.message, { wasExpanded: true }])
  expect(useStoreMessageOption.getState().chatDraftAddress).not.toBe(source)
  expect(useStoreMessageOption.getState().historyId).toBeNull()
  expect(useStoreMessageOption.getState().serverChatId).toBeNull()
  expect(set.mock.calls.some(([key]) => key.startsWith("registry:draft:") && key.includes("unresolved"))).toBe(false)
  expect(mocks.bgRequest.mock.calls.some(([request]) => request?.path?.includes("/chat"))).toBe(false)
})

async function openSetup(message = "FIRST IDEA\nSECOND LINE") {
  const view = mount()
  await act(async () => { await vi.dynamicImportSettled() })
  act(() => editor.form.setFieldValue("message", message))
  fireEvent.click(screen.getByRole("button", { name: "Set up server" }))
  fireEvent.change(await serverUrlInput(), { target: { value: "https://initial-api.example.test" } })
  fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "synthetic-initial-key" } })
  return view
}
const serverUrlInput = async () => {
  const input = await screen.findByLabelText("Server URL", { exact: true })
  return within(input.closest(".ant-form-item")!).getByRole("textbox", { name: "Server URL", exact: true })
}
const save = () => {
  const label = screen.getByText("common:save", { exact: true })
  const button = label.closest("button")!
  fireEvent.click(within(button.parentElement!).getByRole("button", { name: "common:save", exact: true }))
}
const finishedSave = () => waitFor(() => expect(screen.getByTestId("saving")).toHaveTextContent("false"))
const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>(done => { resolve = done })
  return { promise, resolve }
}

it("saves normally without Web Locks and reports that the draft was not retained", async () => {
  Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
  await openSetup()
  save()
  await finishedSave()
  expect((await tldwClient.getConfig())?.apiKey).toBe("synthetic-initial-key")
  expect(await screen.findByText("The draft could not be retained through this connection change.")).toBeInTheDocument()
  expect(editor.form.values.message).toBe("")
})

it("reports draft loss when device persistence falls back to a session", async () => {
  const original = Storage.prototype.set
  vi.spyOn(Storage.prototype, "set").mockImplementation(async function (key, value) {
    if (key === "tldwConfig" && (value as { apiKeyPersistence?: string })?.apiKeyPersistence === "device") throw new Error("Synthetic device failure")
    return original.call(this, key, value)
  })
  await openSetup()
  save()
  await finishedSave()
  expect((await tldwClient.getConfig())?.apiKeyPersistence).toBe("session")
  expect(await screen.findByText("The draft could not be retained through this connection change.")).toBeInTheDocument()
  expect(editor.form.values.message).toBe("")
})

it.each(["owner before cancellation", "owner after cancellation"])("retires written initial setup safely: %s", async sequence => {
  const written = deferred()
  const release = deferred()
  const original = Storage.prototype.set
  vi.spyOn(Storage.prototype, "set").mockImplementation(async function (key, value) {
    await original.call(this, key, value)
    if (key === "tldwConfig" && (value as { apiKeyPersistence?: string })?.apiKeyPersistence === "device") {
      written.resolve()
      await release.promise
    }
  })
  await openSetup()
  const resolveOwner = vi.spyOn(servicePrompts, "resolveServicePromptScope")
  save()
  await act(async () => { await written.promise })
  if (sequence === "owner before cancellation") {
    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await resolveOwner.mock.results.at(-1)?.value
    })
    await waitFor(() => expect(editor.draftReady).toBe(true))
    act(() => {
      useStoreMessageOption.getState().startNewChatDraft()
      editor.form.setFieldValue("message", "NEW OWNED IDEA")
    })
  } else act(() => editor.form.setFieldValue("message", "CANCELLED UNOWNED IDEA"))
  await act(async () => { release.resolve(); await release.promise })
  await finishedSave()
  if (sequence === "owner before cancellation") expect(editor.form.values.message).toBe("NEW OWNED IDEA")
  else {
    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await resolveOwner.mock.results.at(-1)?.value
    })
    expect(editor.form.values.message).toBe("")
  }
})


it("retains the draft after an acknowledged session-only first Save", async () => {
  await openSetup()
  fireEvent.click(screen.getByRole("checkbox", { name: "Remember on this device" }))
  save()
  await finishedSave()
  expect(editor.form.values.message).toBe("FIRST IDEA\nSECOND LINE")
  expect(restoreMetadata.mock.calls.at(-1)).toEqual(["FIRST IDEA\nSECOND LINE", { wasExpanded: true }])
  expect((await tldwClient.getConfig())?.apiKeyPersistence).toBe("session")
  expect(await screen.findByText("Your draft is retained in this chat. Send it when you are ready.")).toBeInTheDocument()
})

it.each(["edit and erase", "Settings edit and erase", "New ABA", "foreign storage ABA", "principal change", "unmount"])("does not write initial configuration after held validation and %s", async boundary => {
  const entered = deferred()
  const release = deferred()
  vi.mocked(tldwAuth.testApiKey).mockImplementationOnce(async () => { entered.resolve(); await release.promise; return true })
  const set = vi.spyOn(Storage.prototype, "set")
  const view = await openSetup()
  const source = useStoreMessageOption.getState().chatDraftAddress!
  save()
  await act(async () => { await entered.promise })
  act(() => {
    if (boundary === "edit and erase") {
      editor.form.setFieldValue("message", "CHANGED")
      editor.form.setFieldValue("message", "FIRST IDEA\nSECOND LINE")
    } else if (boundary === "Settings edit and erase") {
      fireEvent.change(screen.getByRole("textbox", { name: "Server URL" }), { target: { value: "https://changed.example.test" } })
      fireEvent.change(screen.getByRole("textbox", { name: "Server URL" }), { target: { value: "https://initial-api.example.test" } })
    } else if (boundary === "New ABA") {
      useStoreMessageOption.getState().startNewChatDraft()
      useStoreMessageOption.getState().setChatDraftAddress(source)
    } else if (boundary === "foreign storage ABA") {
      const foreign = JSON.stringify({ serverUrl: "https://foreign.example.test", authMode: "single-user", apiKey: "synthetic-foreign" })
      window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: null, newValue: foreign }))
      window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: foreign, newValue: null }))
    } else if (boundary === "principal change") window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    else view.unmount()
  })
  await act(async () => { release.resolve(); await release.promise })
  if (boundary !== "unmount") await finishedSave()
  expect(set.mock.calls.some(([key]) => key === "tldwConfig")).toBe(false)
  expect(set.mock.calls.some(([key]) => key.startsWith("registry:draft:") && key.includes("unresolved"))).toBe(false)
})

it("cancels a waiting handoff on a newer input while owner verification is held", async () => {
  const ownerRelease = deferred()
  const resolved = servicePrompts.resolveServicePromptScope
  vi.spyOn(servicePrompts, "resolveServicePromptScope").mockImplementation(async options => {
    const scope = await resolved(options)
    await ownerRelease.promise
    return scope
  })
  await openSetup()
  save()
  await waitFor(() => expect(localStorage.getItem("tldwConfig")).toContain("synthetic-initial-key"))
  await waitFor(() => expect(editor.form.values.message).toBe(""))
  act(() => editor.form.setFieldValue("message", "NEWER IDEA"))
  await finishedSave()
  expect(editor.form.values.message).toBe("NEWER IDEA")
  await act(async () => { ownerRelease.resolve(); await ownerRelease.promise })
  await waitFor(() => expect(editor.draftReady).toBe(true))
  expect(editor.form.values.message).toBe("NEWER IDEA")
})

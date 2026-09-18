import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwRequest } from "@/services/tldw/request-core"
import { TldwSettings } from "../tldw"

// Enable Plasmo's real localStorage fallback in jsdom, where extension storage
// is absent. Config loading, merging, serialization and saving remain real.
vi.mock("@/utils/safe-storage", async importOriginal => {
  const actual = await importOriginal<typeof import("@/utils/safe-storage")>()
  return { ...actual, createSafeStorage: (options = {}) => actual.createSafeStorage({ ...options, allCopied: true }) }
})
vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("@/store/connection", () => ({ useConnectionStore: { getState: () => ({ checkOnce: vi.fn() }) } }))
vi.mock("@/components/Common/ServerOverviewHint", () => ({ ServerOverviewHint: () => null }))
vi.mock("../server-health-probe", () => ({ probeServerHealth: vi.fn().mockResolvedValue({ ok: true, status: 200 }) }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback: unknown) => typeof fallback === "string" ? fallback : key
}) }))
vi.mock("react-router-dom", () => ({
  Link: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useNavigate: () => vi.fn()
}))

type TimeoutConfig = TldwConfig & {
  chatRequestTimeoutMs?: number
  chatStartupTimeoutMs?: number
  ragRequestTimeoutMs?: number
}
const target: TimeoutConfig = {
  serverUrl: "https://timeouts.example.test", authMode: "multi-user", authSource: "manual",
  accessToken: "synthetic-access", refreshToken: "synthetic-refresh"
}
let storage: Storage

beforeEach(async () => {
  localStorage.clear()
  sessionStorage.clear()
  storage = new Storage({ area: "local", allCopied: true })
  await storage.set("tldwConfig", target)
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => new Response(JSON.stringify({ paths: {} }), { status: 200 })))
  vi.spyOn(tldwClient, "ragHealth").mockResolvedValue({} as never)
  await tldwClient.initialize()
})
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

const mount = () => render(<ConfigProvider theme={{ token: { motion: false } }}><App><TldwSettings /></App></ConfigProvider>)
const load = async (config = target) => {
  await storage.set("tldwConfig", config)
  await tldwClient.initialize()
  mount()
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl))
}
const save = async () => {
  const update = vi.spyOn(tldwClient, "updateConfig")
  const previousCalls = update.mock.calls.length
  fireEvent.click(screen.getByRole("button", { name: "common:save", exact: true }))
  await waitFor(() => expect(update.mock.calls.length).toBeGreaterThan(previousCalls))
  await update.mock.results[previousCalls].value
  await waitFor(async () => expect((await storage.get<TimeoutConfig>("tldwConfig"))?.ragRequestTimeoutMs).toBeDefined())
  await waitFor(() => expect(screen.getByRole("button", { name: "common:save", exact: true })).not.toBeDisabled())
  return (await storage.get<TimeoutConfig>("tldwConfig"))!
}
const generation = (config: TimeoutConfig) => [config.chatRequestTimeoutMs, config.chatStartupTimeoutMs, config.ragRequestTimeoutMs]
const advanced = () => fireEvent.click(screen.getByText("settings:tldw.advancedTimeouts"))

it("ordinary Settings save persists generation defaults without opening Advanced, and reload preserves them", async () => {
  await load()
  const saved = await save()
  expect(generation(saved)).toEqual([120000, 120000, 120000])
  expect(saved).toMatchObject({ requestTimeoutMs: 10000, streamIdleTimeoutMs: 15000, mediaRequestTimeoutMs: 60000, uploadRequestTimeoutMs: 60000 })
  cleanup()
  await load(saved)
  expect(generation(await save())).toEqual([120000, 120000, 120000])
})

it("a first server connection with no stored config uses the same generation defaults", async () => {
  await storage.remove("tldwConfig")
  await tldwClient.initialize()
  mount()
  const server = await screen.findByRole("textbox", { name: "Server URL" })
  fireEvent.change(server, { target: { value: target.serverUrl } })
  fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "synthetic-first-key" } })
  expect(generation(await save())).toEqual([120000, 120000, 120000])
})

it("Extended and explicit Reset persist their generation budgets across reload", async () => {
  await load()
  advanced()
  fireEvent.click(screen.getByText("settings:tldw.timeoutPresetExtended"))
  const extended = await save()
  expect(generation(extended)).toEqual([240000, 240000, 240000])
  cleanup()
  await load(extended)
  advanced()
  await waitFor(() =>
    expect(screen.getByRole("radio", { name: "settings:tldw.timeoutPresetExtended" })).toBeChecked()
  )
  fireEvent.click(screen.getByRole("button", { name: "settings:tldw.reset" }))
  expect(generation(await save())).toEqual([120000, 120000, 120000])
})

it.each([10000, 237000])("preserves explicit stored %i ms generation overrides on ordinary save and reload", async timeout => {
  await load({ ...target, chatRequestTimeoutMs: timeout, chatStartupTimeoutMs: timeout, ragRequestTimeoutMs: timeout })
  const saved = await save()
  expect(generation(saved)).toEqual([timeout, timeout, timeout])
  cleanup()
  await load(saved)
  advanced()
  expect(screen.getByText("settings:tldw.timeoutPresetCustom")).toBeInTheDocument()
  expect(generation(await save())).toEqual([timeout, timeout, timeout])
})

it("invalid zero generation values fall back to generation presets instead of the generic timeout", async () => {
  await load({ ...target, chatRequestTimeoutMs: 0, chatStartupTimeoutMs: 0, ragRequestTimeoutMs: 0 })
  expect(generation(await save())).toEqual([120000, 120000, 120000])
})

it("clicking Balanced from a preserved Custom configuration applies the selected preset", async () => {
  await load({ ...target, chatRequestTimeoutMs: 10000, chatStartupTimeoutMs: 10000, ragRequestTimeoutMs: 10000 })
  advanced()
  expect(screen.getByText("settings:tldw.timeoutPresetCustom")).toBeInTheDocument()
  expect(screen.getByRole("radio", { name: "settings:tldw.timeoutPresetBalanced" })).not.toBeChecked()
  expect(screen.getByRole("radio", { name: "settings:tldw.timeoutPresetExtended" })).not.toBeChecked()
  fireEvent.click(screen.getByRole("radio", { name: "settings:tldw.timeoutPresetBalanced" }))
  const saved = await save()
  expect(generation(saved)).toEqual([120000, 120000, 120000])
  cleanup()
  await load(saved)
  advanced()
  expect(screen.getByRole("radio", { name: "settings:tldw.timeoutPresetBalanced" })).toBeChecked()
})

it("keeps auth, login, and timeout segmented radios independently selected", async () => {
  await load({ ...target, accessToken: undefined, refreshToken: undefined })
  advanced()

  const multiUser = screen.getByRole("radio", { name: "Multi User (Login)" })
  const magicLink = screen.getByRole("radio", { name: "Magic link" })
  const balanced = screen.getByRole("radio", { name: "settings:tldw.timeoutPresetBalanced" })

  fireEvent.click(multiUser)
  fireEvent.click(magicLink)
  fireEvent.click(balanced)
  expect(multiUser).toBeChecked()
  expect(magicLink).toBeChecked()
  expect(balanced).toBeChecked()
  expect(new Set([
    multiUser.getAttribute("name"),
    magicLink.getAttribute("name"),
    balanced.getAttribute("name")
  ]).size).toBe(3)
})

const runDelayedGeneration = async (config: TldwConfig) => {
  cleanup()
  vi.useFakeTimers()
  let signal: AbortSignal | undefined
  const fetchFn = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((resolve, reject) => {
    signal = init?.signal || undefined
    const timer = setTimeout(() => resolve(new Response(JSON.stringify({ answer: "Grounded answer" }), {
      status: 200, headers: { "content-type": "application/json" }
    })), 19689)
    signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")) }, { once: true })
  }))
  const result = tldwRequest({ path: "/api/v1/rag/search", method: "POST", body: { query: "Synthetic source question" } }, {
    getConfig: async () => config, fetchFn
  })
  await vi.advanceTimersByTimeAsync(11000)
  const abortedAtElevenSeconds = signal?.aborted
  await vi.advanceTimersByTimeAsync(10000)
  return { response: await result, abortedAtElevenSeconds, calls: fetchFn.mock.calls.length }
}

it("the actual request layer accepts a generation longer than 10 seconds using ordinary Settings save output", async () => {
  await load()
  const outcome = await runDelayedGeneration(await save())
  expect(outcome.abortedAtElevenSeconds).toBe(false)
  expect(outcome.response).toMatchObject({ ok: true, data: { answer: "Grounded answer" } })
  expect(outcome.calls).toBe(1)
})

it("the same actual request layer still honors a deliberate 10-second RAG limit", async () => {
  await load({ ...target, ragRequestTimeoutMs: 10000 })
  const outcome = await runDelayedGeneration(await save())
  expect(outcome.abortedAtElevenSeconds).toBe(true)
  expect(outcome.response).toMatchObject({ ok: false, code: "REQUEST_TIMEOUT" })
  expect(outcome.calls).toBe(1)
})

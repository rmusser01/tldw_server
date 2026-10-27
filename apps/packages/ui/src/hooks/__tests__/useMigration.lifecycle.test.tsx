import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, expect, it, vi, type MockInstance } from "vitest"
import { useMigration } from "../useMigration"

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  set: vi.fn(),
  hasLegacyData: vi.fn(),
  migrate: vi.fn(),
  open: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: mocks.get, set: mocks.set }) }))
vi.mock("~/db/dexie/migration", () => ({ hasLegacyData: mocks.hasLegacyData, runAllMigrations: mocks.migrate }))
vi.mock("../useAntdMessage", () => ({ useAntdMessage: () => ({ open: mocks.open }) }))

const clients = new Set<QueryClient>()
let reloadTimerSpy: MockInstance<typeof window.setTimeout> | undefined

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal("chrome", { storage: { local: {} } })
  mocks.set.mockResolvedValue(undefined)
  mocks.hasLegacyData.mockResolvedValue(false)
  mocks.migrate.mockResolvedValue(undefined)
})
afterEach(() => {
  cleanup()
  for (const client of clients) client.clear()
  clients.clear()
  for (const call of reloadTimerSpy?.mock.results ?? []) {
    if (call.type === "return") window.clearTimeout(call.value)
  }
  reloadTimerSpy = undefined
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it("admits the composer once after the initial migration check, including its idle render", async () => {
  let release!: (value: boolean) => void
  mocks.get.mockImplementation(() => new Promise<boolean>(resolve => { release = resolve }))
  const mounted = vi.fn()
  const unmounted = vi.fn()
  const loading: boolean[] = []
  function Composer() {
    React.useLayoutEffect(() => { mounted(); return () => unmounted() }, [])
    return <textarea aria-label="Imported source" defaultValue="ORBIT-742 complete source" />
  }
  function Shell() {
    const migration = useMigration()
    loading.push(migration.isLoading)
    return migration.isLoading ? <p>Checking history</p> : <Composer />
  }
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  clients.add(client)
  render(<QueryClientProvider client={client}><Shell /></QueryClientProvider>)
  await waitFor(() => expect(mocks.get).toHaveBeenCalled())
  expect(loading[0]).toBe(true)
  expect(mounted).not.toHaveBeenCalled()
  await act(async () => release(true))
  await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue("ORBIT-742 complete source"))
  expect(mounted).toHaveBeenCalledTimes(1)
  expect(unmounted).not.toHaveBeenCalled()
  expect(mocks.migrate).not.toHaveBeenCalled()
})

it("continues to block while a real legacy migration runs and records success", async () => {
  let release!: () => void
  mocks.get.mockResolvedValue(false)
  mocks.hasLegacyData.mockResolvedValue(true)
  mocks.migrate.mockImplementation(() => new Promise<void>(resolve => { release = resolve }))
  function Shell() {
    const migration = useMigration()
    return <output>{migration.isLoading ? "checking" : migration.isSuccess ? "migrated" : "failed"}</output>
  }
  const client = new QueryClient()
  clients.add(client)
  render(<QueryClientProvider client={client}><Shell /></QueryClientProvider>)
  await waitFor(() => expect(mocks.migrate).toHaveBeenCalledOnce())
  expect(screen.getByRole("status")).toHaveTextContent("checking")
  expect(mocks.set).not.toHaveBeenCalled()
  // Suppress only the existing delayed reload after a successful extension migration.
  reloadTimerSpy = vi.spyOn(window, "setTimeout")
  await act(async () => release())
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("migrated"))
  expect(mocks.set).toHaveBeenCalledWith("isMigrated", true)
})

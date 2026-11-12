// @vitest-environment jsdom
import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import * as service from "@/services/moderation"
import { useBlocklist, type BlocklistState } from "../hooks/useBlocklist"
import { useModerationSettings, type ModerationSettingsState } from "../hooks/useModerationSettings"
import { PolicyStatusBadges } from "../components/PolicyStatusBadges"

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/services/moderation", () => ({
  getModerationSettings: vi.fn(), getEffectivePolicy: vi.fn(),
  getBlocklist: vi.fn(), getManagedBlocklist: vi.fn(), lintBlocklist: vi.fn(),
  appendManagedBlocklist: vi.fn(), deleteManagedBlocklistItem: vi.fn(),
  updateBlocklist: vi.fn()
}))

let lines: string[]
let client: QueryClient
let blocklist: BlocklistState
let settings: ModerationSettingsState
const mutations = ["append", "appendLine", "delete", "replace", "undo"] as const
type Mutation = typeof mutations[number]

function Harness() {
  const currentBlocklist = useBlocklist()
  const currentSettings = useModerationSettings()
  React.useEffect(() => {
    blocklist = currentBlocklist
    settings = currentSettings
  })
  const userSettings = useModerationSettings("alice")
  return <>
    <section aria-label="Server policy">
      <PolicyStatusBadges ruleCount={currentSettings.policyQuery.data?.blocklist_count} />
    </section>
    <section aria-label="Alice policy">
      <PolicyStatusBadges ruleCount={userSettings.policyQuery.data?.blocklist_count} />
    </section>
  </>
}

async function setup() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
  client.setQueryData(["moderation-policy", "inactive-user"], { blocklist_count: 25 })
  render(<QueryClientProvider client={client}><Harness /></QueryClientProvider>)
  await screen.findByText("5 rules")
  await screen.findByText("15 rules")
  await act(async () => { await blocklist.loadManaged(); await blocklist.loadRaw() })
  act(() => settings.setDraft((draft) => ({ ...draft, piiEnabled: true })))
}

async function prepare(kind: Mutation) {
  if (kind === "append") act(() => blocklist.setManagedLine("ORBIT-UAT-FLAG-742"))
  if (kind === "replace" || kind === "undo") {
    await act(async () => { await blocklist.previewRawReplace("replacement") })
  }
  if (kind === "undo") {
    await act(async () => { await blocklist.confirmRawReplace() })
    client.setQueryData(["moderation-policy", "inactive-user"], { blocklist_count: 21 })
  }
}

async function mutate(kind: Mutation) {
  if (kind === "append") await blocklist.appendManaged()
  if (kind === "appendLine") await blocklist.appendLine("ORBIT-UAT-FLAG-742")
  if (kind === "delete") await blocklist.deleteManaged(0)
  if (kind === "replace") await blocklist.confirmRawReplace()
  if (kind === "undo") await blocklist.undoRawReplace()
}

beforeEach(() => {
  vi.resetAllMocks()
  lines = ["one", "two", "three", "four", "five"]
  vi.mocked(service.getModerationSettings).mockResolvedValue({ pii_enabled: false, categories_enabled: [], effective: {} })
  vi.mocked(service.getEffectivePolicy).mockImplementation(async (userId) => ({
    blocklist_count: lines.length + (userId ? 10 : 0)
  }))
  vi.mocked(service.getBlocklist).mockImplementation(async () => [...lines])
  vi.mocked(service.getManagedBlocklist).mockImplementation(async () => ({
    data: { version: "version", items: lines.map((line, id) => ({ id, line, ok: true, pattern_type: "literal", action: "block" })) }, etag: "version"
  }))
  vi.mocked(service.lintBlocklist).mockImplementation(async (input) => ({ items: [], valid_count: input.lines?.length ?? 1, invalid_count: 0 }))
  vi.mocked(service.appendManagedBlocklist).mockImplementation(async (_version, line) => {
    lines.push(line); return { version: "next", index: lines.length - 1, count: lines.length }
  })
  vi.mocked(service.deleteManagedBlocklistItem).mockImplementation(async (_version, id) => {
    lines.splice(id, 1); return { version: "next", count: lines.length }
  })
  vi.mocked(service.updateBlocklist).mockImplementation(async (next) => {
    lines = [...next]; return { status: "ok", count: lines.length }
  })
})
afterEach(() => { cleanup(); client?.clear() })

describe("blocklist mutations refresh effective policy", () => {
  it.each(mutations)("%s refreshes actual policy badges and preserves unsaved settings", async (kind) => {
    await setup()
    await prepare(kind)
    await act(async () => { await mutate(kind) })
    const expected = { append: 6, appendLine: 6, delete: 4, replace: 1, undo: 5 }[kind]
    await screen.findByText(`${expected} rules`)
    await screen.findByText(`${expected + 10} rules`)
    expect(client.getQueryState(["moderation-policy", "inactive-user"])?.isInvalidated).toBe(true)
    expect(settings.draft.piiEnabled).toBe(true)
    expect(settings.isDirty).toBe(true)
    expect(service.getModerationSettings).toHaveBeenCalledTimes(1)
  })

  it.each(mutations)("failed %s preserves cached policy and draft", async (kind) => {
    await setup()
    await prepare(kind)
    const before = client.getQueryData(["moderation-policy", "server"])
    const reads = vi.mocked(service.getEffectivePolicy).mock.calls.length
    const error = new Error("write rejected")
    vi.mocked(service.appendManagedBlocklist).mockRejectedValue(error)
    vi.mocked(service.deleteManagedBlocklistItem).mockRejectedValue(error)
    vi.mocked(service.updateBlocklist).mockRejectedValue(error)
    await act(async () => { await expect(mutate(kind)).rejects.toThrow("write rejected") })
    expect(client.getQueryData(["moderation-policy", "server"])).toEqual(before)
    expect(client.getQueryState(["moderation-policy", "inactive-user"])?.isInvalidated).toBe(false)
    expect(service.getEffectivePolicy).toHaveBeenCalledTimes(reads)
    expect(settings.draft.piiEnabled).toBe(true)
  })

  it("Validate and replacement preview do not refresh policy", async () => {
    await setup()
    const reads = vi.mocked(service.getEffectivePolicy).mock.calls.length
    await act(async () => { await blocklist.lintLine("ORBIT-UAT-FLAG-742"); await blocklist.previewRawReplace("preview") })
    expect(screen.getByText("5 rules")).toBeInTheDocument()
    expect(service.getEffectivePolicy).toHaveBeenCalledTimes(reads)
    expect(client.getQueryState(["moderation-policy", "inactive-user"])?.isInvalidated).toBe(false)
  })

  it("accepted append refreshes policy even when subsequent managed reload fails", async () => {
    await setup()
    vi.mocked(service.getManagedBlocklist).mockRejectedValueOnce(new Error("reload failed"))
    await act(async () => { await expect(blocklist.appendLine("new rule")).rejects.toThrow("reload failed") })
    await waitFor(() => expect(screen.getByText("6 rules")).toBeInTheDocument())
  })
})

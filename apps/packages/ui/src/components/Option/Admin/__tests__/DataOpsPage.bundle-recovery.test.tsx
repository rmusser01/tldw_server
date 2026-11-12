import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"

const mocks = vi.hoisted(() => ({ request: vi.fn(), download: vi.fn() }))
const authority = vi.hoisted(() => ({ config: { serverUrl: "https://c09.example", authMode: "single-user" as const, apiKey: "c09-fixture-only" }, authorityLoading: false }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: () => authority }))
vi.mock("@/services/background-proxy", async (original) => ({
  ...await original<typeof import("@/services/background-proxy")>(), bgRequest: mocks.request
}))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: mocks.download }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key }) }))

const artifact = "c09-owned-backup.zip"
const validations = [{ dataset: "chacha", compatible: true, message: "ok" }, { dataset: "media", compatible: true, message: "ok" }]
const preview = { status: "compatible", validations, warnings: [], datasets_restored: [] }
const uploads = (dry: boolean) => mocks.request.mock.calls.map(([request]) => request).filter((request) => request.path.includes("/bundles/import?") && request.path.includes(`dry_run=${dry}`))
const file = (name = artifact) => {
  const result = new File([new Uint8Array([80, 75, 3, 4])], name, { type: "application/zip" })
  Object.defineProperty(result, "arrayBuffer", { value: async () => new Uint8Array([80, 75, 3, 4]).buffer })
  return result
}
const mount = async () => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>)
  fireEvent.click(screen.getByRole("tab", { name: "Bundles" }))
  const panel = screen.getByRole("tabpanel", { name: "Bundles" })
  await within(panel).findByText(artifact, { selector: "code" })
  return panel
}
const choose = (panel: HTMLElement, value = file()) => {
  fireEvent.change(within(panel).getByLabelText("Backup ZIP file"), { target: { files: [value] } })
  fireEvent.change(within(panel).getByRole("spinbutton", { name: "Restore target user ID" }), { target: { value: "7" } })
}
const validate = async (panel: HTMLElement) => {
  fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
  await waitFor(() => expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeEnabled())
}

beforeEach(() => {
  vi.clearAllMocks()
  authority.config = { serverUrl: "https://c09.example", authMode: "single-user", apiKey: "c09-fixture-only" }
  vi.spyOn(message, "success").mockImplementation(() => undefined as never)
  vi.spyOn(message, "error").mockImplementation(() => undefined as never)
  mocks.request.mockImplementation(async ({ path, method }) => {
    if (method === "GET" && path === "/api/v1/admin/backups/bundles") return { items: [{ bundle_id: artifact, datasets: ["chacha", "media"], user_id: 7 }] }
    if (method === "GET" && path.endsWith("/download")) return new Uint8Array([80, 75, 3, 4]).buffer
    if (method === "POST" && path.includes("/bundles/import?")) return preview
    if (method === "GET") return { items: [] }
    throw new Error("Unexpected request")
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("Admin bundle recovery contract", () => {
  it("downloads the exact listed artifact through authenticated binary transport", async () => {
    const panel = await mount()
    fireEvent.click(within(panel).getByRole("button", { name: `Download ${artifact}` }))
    await waitFor(() => expect(mocks.download).toHaveBeenCalledWith(expect.any(Blob), artifact))
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ path: `/api/v1/admin/backups/bundles/${artifact}/download`, method: "GET", responseType: "arrayBuffer", configSnapshot: authority.config, abortSignal: expect.any(AbortSignal) }))
    const blob = mocks.download.mock.calls[0][0] as Blob
    const bytes = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(blob)
    })
    expect(Array.from(new Uint8Array(bytes))).toEqual([80, 75, 3, 4])
    expect(uploads(true)).toHaveLength(0)
    expect(uploads(false)).toHaveLength(0)
  })

  it("previews exact scope then requires confirmation and admits only one pending restore", async () => {
    const panel = await mount()
    choose(panel)
    await validate(panel)
    expect(uploads(true)).toHaveLength(1)
    expect(uploads(true)[0]).toMatchObject({ path: "/api/v1/admin/backups/bundles/import?user_id=7&dry_run=true", method: "POST", body: expect.any(FormData), preferDirect: true, configSnapshot: authority.config, abortSignal: expect.any(AbortSignal) })
    expect((uploads(true)[0].body as FormData).getAll("file")).toHaveLength(1)
    expect((uploads(true)[0].body as FormData).get("file")).toMatchObject({ name: artifact })
    expect(uploads(false)).toHaveLength(0)
    fireEvent.click(within(panel).getByRole("button", { name: "Restore validated backup" }))
    const dialog = await screen.findByRole("dialog", { name: "Restore backup to this server?" })
    await waitFor(() => expect(within(dialog).getByText(artifact, { exact: true })).toBeVisible())
    expect(within(dialog).getByText("User 7", { exact: true })).toBeVisible()
    expect(within(dialog).getByText("chacha, media", { exact: true })).toBeVisible()
    let finish!: (value: unknown) => void
    mocks.request.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
    const confirm = within(dialog).getByRole("button", { name: "Restore backup" })
    try {
      fireEvent.click(confirm); fireEvent.click(confirm)
      await waitFor(() => expect(uploads(false)).toHaveLength(1))
      expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled()
      expect(within(panel).getByLabelText("Backup ZIP file")).toBeDisabled()
      finish({ status: "imported", datasets_restored: ["chacha", "media"], warnings: [], safety_snapshots: {} })
      await waitFor(() => expect(message.success).toHaveBeenCalledWith("Backup restored successfully"))
      expect(uploads(false)).toHaveLength(1)
    } finally {
      finish?.({ status: "imported", datasets_restored: ["chacha", "media"], warnings: [], safety_snapshots: {} })
      await Promise.resolve()
    }
  }, 30000)

  it("retires validation when either target or file changes", async () => {
    const panel = await mount(); choose(panel); await validate(panel)
    fireEvent.change(within(panel).getByRole("spinbutton", { name: "Restore target user ID" }), { target: { value: "8" } })
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    await validate(panel)
    fireEvent.change(within(panel).getByLabelText("Backup ZIP file"), { target: { files: [file("different.zip")] } })
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

  it("retains input on failed validation and never turns incompatibility into permission to restore", async () => {
    const panel = await mount(); choose(panel)
    mocks.request.mockRejectedValueOnce(new Error("service unavailable"))
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(message.error).toHaveBeenCalled())
    expect(within(panel).getByRole("spinbutton", { name: "Restore target user ID" })).toHaveValue("7")
    await waitFor(() => expect(within(panel).getByRole("button", { name: "Validate backup" })).toBeEnabled())
    mocks.request.mockResolvedValueOnce({ ...preview, status: "incompatible", validations: [{ dataset: "media", compatible: false, message: "Newer schema" }] })
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(uploads(true)).toHaveLength(2))
    await waitFor(() => expect(within(panel).getByText("media", { selector: "strong" }).parentElement).toHaveTextContent(/^media: Newer schema$/))
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(false)).toHaveLength(0)
  }, 30000)


  it.each(["invalid_bundle", "invalid_bundle_manifest"])("explains %s without exposing an internal rejection code", async (code) => {
    const panel = await mount(); choose(panel)
    mocks.request.mockRejectedValueOnce(new Error(code))
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("This backup is damaged or uses an unsupported format. Choose a complete backup ZIP from a compatible server."))
    expect((within(panel).getByLabelText("Backup ZIP file") as HTMLInputElement).files?.[0].name).toBe(artifact)
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(true)).toHaveLength(1)
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

  it.each(["invalid_bundle", "invalid_bundle_manifest"])("explains transport-wrapped %s without exposing an internal rejection code", async (code) => {
    const panel = await mount(); choose(panel)
    mocks.request.mockRejectedValueOnce(Object.assign(new Error(`${code} (POST /api/v1/admin/backups/bundles/import?user_id=7&dry_run=true)`), { status: 400 }))
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("This backup is damaged or uses an unsupported format. Choose a complete backup ZIP from a compatible server."))
    expect((within(panel).getByLabelText("Backup ZIP file") as HTMLInputElement).files?.[0].name).toBe(artifact)
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(true)).toHaveLength(1)
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

  it("keeps authorization denial when explaining a backup rejection", async () => {
    const panel = await mount(); choose(panel)
    mocks.request.mockRejectedValueOnce(Object.assign(new Error("invalid_bundle"), { status: 403 }))
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(screen.getByText("You don't have permission to access data operations.")).toBeVisible())
    expect(uploads(true)).toHaveLength(1)
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

  it("rejects malformed validation messages without exposing Restore", async () => {
    const panel = await mount(); choose(panel)
    mocks.request.mockResolvedValueOnce({ ...preview, validations: [{ dataset: "media", compatible: true, message: { invalid: true } }] })
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("The server did not return a valid backup validation result."))
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

  it.each(["partial", "lost response", "malformed response"])("keeps %s unconfirmed without automatic replay", async (kind) => {
    const panel = await mount(); choose(panel); await validate(panel)
    fireEvent.click(within(panel).getByRole("button", { name: "Restore validated backup" }))
    const dialog = await screen.findByRole("dialog", { name: "Restore backup to this server?" })
    await waitFor(() => expect(dialog).toBeVisible())
    if (kind === "lost response") mocks.request.mockRejectedValueOnce(new Error("response lost"))
    else mocks.request.mockResolvedValueOnce(kind === "partial"
      ? { status: "imported", datasets_restored: ["chacha"], warnings: ["Media restore failed"], rollback_failures: ["Media rollback failed"] }
      : { status: "imported", datasets_restored: null, warnings: [] })
    const confirm = within(dialog).getByRole("button", { name: "Restore backup" })
    fireEvent.click(confirm)
    await waitFor(() => expect(message.error).toHaveBeenCalled())
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeEnabled())
    expect(message.success).not.toHaveBeenCalled()
    expect(within(panel).getByText("Restore was not confirmed. Inspect the target data before retrying.")).toBeVisible()
    expect(within(panel).getByRole("spinbutton", { name: "Restore target user ID" })).toHaveValue("7")
    expect((within(panel).getByLabelText("Backup ZIP file") as HTMLInputElement).files?.[0].name).toBe(artifact)
    expect(confirm).toBeDisabled()
    fireEvent.click(confirm)
    expect(uploads(false)).toHaveLength(1)
    if (kind === "partial") {
      expect(within(panel).getByText("Media restore failed")).toBeVisible()
      expect(within(panel).getByText("Media rollback failed")).toBeVisible()
    }
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
  }, 30000)

  it.each(["server", "credential"])("retires a pending restore when the %s authority changes", async (changed) => {
    const page = () => <ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>
    const view = render(page())
    fireEvent.click(screen.getByRole("tab", { name: "Bundles" }))
    const panel = screen.getByRole("tabpanel", { name: "Bundles" })
    await within(panel).findByText(artifact, { selector: "code" })
    choose(panel); await validate(panel)
    fireEvent.click(within(panel).getByRole("button", { name: "Restore validated backup" }))
    const dialog = await screen.findByRole("dialog", { name: "Restore backup to this server?" })
    await waitFor(() => expect(dialog).toBeVisible())
    let finish!: (value: unknown) => void
    mocks.request.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
    try {
      fireEvent.click(within(dialog).getByRole("button", { name: "Restore backup" }))
      await waitFor(() => expect(uploads(false)).toHaveLength(1))
      const admitted = uploads(false)[0]
      authority.config = changed === "server" ? { ...authority.config, serverUrl: "https://other-c09.example" } : { ...authority.config, apiKey: "c09-new-principal-only" }
      view.rerender(page())
      await waitFor(() => expect(admitted.abortSignal.aborted).toBe(true))
      finish({ status: "imported", datasets_restored: ["chacha", "media"], warnings: [] })
      await within(screen.getByRole("tabpanel", { name: "Bundles" })).findByText(artifact, { selector: "code" })
      expect(message.success).not.toHaveBeenCalled()
      expect(message.error).not.toHaveBeenCalled()
      expect(screen.queryByRole("dialog", { name: "Restore backup to this server?" })).toBeNull()
      expect(screen.getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
      expect(uploads(false)).toHaveLength(1)
      expect(admitted.configSnapshot.serverUrl).toBe("https://c09.example")
      expect(admitted.configSnapshot.apiKey).toBe("c09-fixture-only")
    } finally {
      finish?.({ status: "imported", datasets_restored: ["chacha", "media"], warnings: [] })
      await Promise.resolve()
    }
  }, 30000)
  it("hands PostgreSQL database recovery to the offline helper without an online restore", async () => {
    const panel = await mount()
    fireEvent.change(within(panel).getByLabelText("Backup ZIP file"), { target: { files: [file("database.zip")] } })
    mocks.request.mockResolvedValueOnce({ ...preview, validations: [{ dataset: "postgresql", compatible: true, message: "Offline whole-database recovery required" }] })
    fireEvent.click(within(panel).getByRole("button", { name: "Validate backup" }))
    await waitFor(() => expect(within(panel).getByText(/Stop all API, Jobs and Scheduler clients/)).toBeVisible())
    expect(within(panel).getByText(/pg_backup_restore.py restore-bundle/)).toBeVisible()
    expect(within(panel).getByText(/https:\/\/c09.example/)).toBeVisible()
    expect(within(panel).getByRole("button", { name: "Restore validated backup" })).toBeDisabled()
    expect(uploads(true)).toHaveLength(1)
    expect(uploads(true)[0].path).not.toContain("user_id")
    expect(uploads(false)).toHaveLength(0)
  }, 30000)

})

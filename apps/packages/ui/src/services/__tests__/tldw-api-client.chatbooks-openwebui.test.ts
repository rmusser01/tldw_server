import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient Chatbooks OpenWebUI import contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("sends source_format for OpenWebUI preview uploads", async () => {
    mocks.bgUpload.mockResolvedValue({ openwebui_preview: { chat_count: 0 } })

    const client = new TldwApiClient()
    const file = new File(["[]"], "openwebui.json", { type: "application/json" })
    await client.previewChatbook(file, { source_format: "openwebui_json" })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/preview",
        method: "POST",
        fields: { source_format: "openwebui_json" },
        file: expect.objectContaining({
          name: "openwebui.json",
          type: "application/json"
        })
      })
    )
  })

  it("sends source_format for OpenWebUI database preview uploads", async () => {
    mocks.bgUpload.mockResolvedValue({ openwebui_db_preview: { user_count: 0, users: [] } })

    const client = new TldwApiClient()
    const file = new File(["SQLite format 3"], "webui.db", { type: "application/vnd.sqlite3" })
    await client.previewChatbook(file, { source_format: "openwebui_db" })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/preview",
        method: "POST",
        fields: { source_format: "openwebui_db" },
        file: expect.objectContaining({
          name: "webui.db",
          type: "application/vnd.sqlite3"
        })
      })
    )
  })

  it("serializes import options as multipart fields", async () => {
    mocks.bgUpload.mockResolvedValue({ success: true })

    const client = new TldwApiClient()
    const file = new File(["[]"], "openwebui.json", { type: "application/json" })
    await client.importChatbook(file, {
      source_format: "openwebui_json",
      conflict_resolution: "rename",
      prefix_imported: true,
      import_media: false,
      import_embeddings: false,
      async_mode: true,
      content_selections: { conversation: ["conv-1"] }
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/import",
        method: "POST",
        fields: expect.objectContaining({
          source_format: "openwebui_json",
          conflict_resolution: "rename",
          prefix_imported: "true",
          import_media: "false",
          import_embeddings: "false",
          async_mode: "true",
          content_selections: JSON.stringify({ conversation: ["conv-1"] })
        })
      })
    )
  })

  it("serializes OpenWebUI database selected user import fields", async () => {
    mocks.bgUpload.mockResolvedValue({ success: true })

    const client = new TldwApiClient()
    const file = new File(["SQLite format 3"], "webui.db", { type: "application/vnd.sqlite3" })
    await client.importChatbook(file, {
      source_format: "openwebui_db",
      selected_openwebui_user_id: "user-a",
      conflict_resolution: "skip",
      prefix_imported: false,
      import_media: false,
      import_embeddings: false,
      async_mode: true
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/import",
        method: "POST",
        fields: expect.objectContaining({
          source_format: "openwebui_db",
          selected_openwebui_user_id: "user-a",
          conflict_resolution: "skip",
          prefix_imported: "false",
          import_media: "false",
          import_embeddings: "false",
          async_mode: "true"
        })
      })
    )
  })

  it("previews OpenWebUI attachment hydration with a JSON request", async () => {
    mocks.bgRequest.mockResolvedValue({ summary: { referenced_files: 1 } })

    const client = new TldwApiClient()
    const payload = {
      openwebui_data_root: "/srv/openwebui",
      scope: {
        conversation_ids: ["conv-a"],
        source_user_id: "ow-user"
      },
      process_supported_files: false
    }

    await client.previewOpenWebUIHydration(payload)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/openwebui/hydration/preview",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })
    )
  })

  it("creates an OpenWebUI attachment hydration job with a JSON request", async () => {
    mocks.bgRequest.mockResolvedValue({ job_id: "42", status: "queued" })

    const client = new TldwApiClient()
    const payload = {
      openwebui_data_root: "/srv/openwebui",
      scope: {
        conversation_ids: ["conv-a"],
        source_user_id: "ow-user"
      },
      process_supported_files: true
    }

    await client.createOpenWebUIHydrationJob(payload)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/openwebui/hydration/jobs",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })
    )
  })

  it("gets an OpenWebUI attachment hydration job by id", async () => {
    mocks.bgRequest.mockResolvedValue({ job_id: "42", status: "completed" })

    const client = new TldwApiClient()
    await client.getOpenWebUIHydrationJob("job/42")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/chatbooks/openwebui/hydration/jobs/job%2F42",
        method: "GET"
      })
    )
  })
})

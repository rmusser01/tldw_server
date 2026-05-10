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
})

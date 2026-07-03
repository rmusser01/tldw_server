import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("../background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args)
}))

import { visualIdentityMethods } from "../tldw/domains/visual-identities"

describe("visual identity API domain contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("fetches visual identity capabilities", async () => {
    mocks.bgRequest.mockResolvedValue({
      upload_max_bytes: 1024,
      archive_max_bytes: 2048,
      max_dimension: 2048,
      max_frame_count: 120,
      supported_mime_types: ["image/png"],
      avif_enabled: false
    })

    await visualIdentityMethods.getVisualIdentityCapabilities.call({})

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/visual-identities/capabilities",
      method: "GET"
    })
  })

  it("uploads an expression asset using the backend file field", async () => {
    mocks.bgUpload.mockResolvedValue({ id: 8 })
    const file = {
      name: "happy.webp",
      type: "image/webp",
      data: Uint8Array.from([1, 2, 3])
    }

    await visualIdentityMethods.uploadVisualIdentityPackAsset.call({}, 5, {
      expression_key: "happy",
      draft_id: 7,
      file,
      timeoutMs: 45_000
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/visual-identities/packs/5/assets",
        method: "POST",
        fields: {
          expression_key: "happy",
          draft_id: 7
        },
        file,
        fileFieldName: "file",
        timeoutMs: 45_000
      })
    )
  })

  it("omits optional upload fields when they are not set", async () => {
    mocks.bgUpload.mockResolvedValue({ id: 8 })
    const file = {
      name: "neutral.png",
      type: "image/png",
      data: Uint8Array.from([1, 2, 3])
    }

    await visualIdentityMethods.uploadVisualIdentityPackAsset.call({}, 5, {
      expression_key: "neutral",
      file
    })

    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).toEqual({
      expression_key: "neutral"
    })
    expect(Object.keys(fields)).toEqual(["expression_key"])
  })

  it("imports a generated file asset with source context", async () => {
    mocks.bgRequest.mockResolvedValue({ id: 12 })

    await visualIdentityMethods.createVisualIdentityAssetFromGeneratedFile.call({}, 5, {
      generated_file_id: 42,
      expression_key: "happy",
      draft_id: 7,
      source_feature: "vn_assets",
      source_context: { vn_item_id: 29, vn_slot_label: "Happy" },
      idempotency_key: "vn-assets:42:happy"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/visual-identities/packs/5/assets/from-generated-file",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        generated_file_id: 42,
        expression_key: "happy",
        draft_id: 7,
        source_feature: "vn_assets",
        source_context: { vn_item_id: 29, vn_slot_label: "Happy" },
        idempotency_key: "vn-assets:42:happy"
      }
    })
  })

  it("starts ZIP import using the archive field", async () => {
    mocks.bgUpload.mockResolvedValue({ draft_id: 4, status: "queued" })
    const archive = {
      name: "expressions.zip",
      type: "application/zip",
      data: Uint8Array.from([1])
    }

    await visualIdentityMethods.startVisualIdentityZipImport.call({}, {
      archive,
      title: "Imported Expressions",
      pack_id: 12,
      idempotency_key: "idem-1"
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/visual-identities/imports/zip",
        method: "POST",
        fields: {
          title: "Imported Expressions",
          pack_id: 12,
          idempotency_key: "idem-1"
        },
        file: archive,
        fileFieldName: "archive"
      })
    )
  })

  it("omits optional ZIP import fields when backend defaults should apply", async () => {
    mocks.bgUpload.mockResolvedValue({ draft_id: 4, status: "queued" })
    const archive = {
      name: "expressions.zip",
      type: "application/zip",
      data: Uint8Array.from([1])
    }

    await visualIdentityMethods.startVisualIdentityZipImport.call({}, {
      archive,
      idempotency_key: "idem-1"
    })

    const fields = mocks.bgUpload.mock.calls[0][0].fields
    expect(fields).toEqual({
      idempotency_key: "idem-1"
    })
    expect(Object.keys(fields)).toEqual(["idempotency_key"])
  })

  it("resolves actor bindings with expression query parameters", async () => {
    mocks.bgRequest.mockResolvedValue({ asset_id: 9 })

    await visualIdentityMethods.resolveVisualIdentityBinding.call({}, {
      actor_kind: "character",
      actor_id: 123,
      expression_key: "happy",
      manual_override_expression_key: "excited",
      mood_expression_key: "joy"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path:
        "/api/v1/visual-identities/bindings/resolve?actor_kind=character&actor_id=123&expression_key=happy&manual_override_expression_key=excited&mood_expression_key=joy",
      method: "GET"
    })
  })

  it("builds immutable asset content paths", () => {
    expect(visualIdentityMethods.getVisualIdentityAssetContentPath.call({}, 5, 8)).toBe(
      "/api/v1/visual-identities/packs/5/assets/8/content"
    )
  })
})

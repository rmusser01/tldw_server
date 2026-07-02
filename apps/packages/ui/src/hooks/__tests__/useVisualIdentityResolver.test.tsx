import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  useVisualIdentityExpressionAvailability,
  useVisualIdentityResolver
} from "../useVisualIdentityResolver"

describe("useVisualIdentityResolver", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("resolves the active actor expression through the client", async () => {
    const resolveVisualIdentityBinding = vi.fn(async () => ({
      actor_kind: "character" as const,
      actor_id: 7,
      pack_id: 1,
      pack_version_id: 2,
      expression_key: "happy",
      requested_expression_key: "happy",
      asset_id: 9,
      storage_relpath: "visual_identities/asset.webp",
      fallback_reason: "requested",
      is_animated: true,
      content_type: "image/webp",
      asset_url: "/api/v1/visual-identities/packs/1/assets/9/content"
    }))

    const { result } = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: 7,
        expressionKey: "happy",
        client: { resolveVisualIdentityBinding }
      })
    )

    await waitFor(() => {
      expect(result.current.resolution?.asset_url).toBe(
        "/api/v1/visual-identities/packs/1/assets/9/content"
      )
    })
    expect(resolveVisualIdentityBinding).toHaveBeenCalledWith({
      actor_kind: "character",
      actor_id: 7,
      expression_key: "happy",
      manual_override_expression_key: null,
      mood_expression_key: null
    })
  })

  it("does not resolve without an actor id", () => {
    const resolveVisualIdentityBinding = vi.fn()

    const { result } = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: null,
        expressionKey: "happy",
        client: { resolveVisualIdentityBinding }
      })
    )

    expect(result.current.resolution).toBeNull()
    expect(resolveVisualIdentityBinding).not.toHaveBeenCalled()
  })

  it("marks an expression unavailable when resolution falls back to another asset", async () => {
    const resolveVisualIdentityBinding = vi.fn(
      async (request: { expression_key?: string }) => ({
        actor_kind: "character" as const,
        actor_id: 99,
        pack_id: 1,
        pack_version_id: 2,
        expression_key:
          request.expression_key === "happy" ? "happy" : "neutral",
        requested_expression_key: request.expression_key,
        asset_id: request.expression_key === "happy" ? 9 : 1,
        storage_relpath: "visual_identities/asset.webp",
        fallback_reason:
          request.expression_key === "happy" ? "requested" : "default",
        is_animated: false,
        content_type: "image/webp",
        asset_url: "/api/v1/visual-identities/packs/1/assets/9/content"
      })
    )

    const { result } = renderHook(() =>
      useVisualIdentityExpressionAvailability({
        actorKind: "character",
        actorId: 99,
        expressions: ["happy", "sad"],
        client: { resolveVisualIdentityBinding }
      })
    )

    await waitFor(() => {
      expect(result.current.availability).toEqual({
        happy: true,
        sad: false
      })
    })
  })
})

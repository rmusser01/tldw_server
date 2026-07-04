import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  clearVisualIdentityResolverCaches,
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
      asset_url: "/api/v1/visual-identities/packs/1/assets/9/content",
      preview_url: null
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
      mood_expression_key: null,
      override_pack_id: null,
      override_pack_version_id: null,
      role_id: null,
      role_label: null,
      allow_override_fallback: null
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

  it("keeps override pack resolutions out of the default binding cache", async () => {
    const resolveVisualIdentityBinding = vi.fn(
      async (request: { override_pack_id?: number | null }) => ({
        actor_kind: "character" as const,
        actor_id: 731,
        pack_id: request.override_pack_id ?? 1,
        pack_version_id: request.override_pack_id ? 22 : 2,
        expression_key: "happy",
        requested_expression_key: "happy",
        asset_id: request.override_pack_id ? 990 : 9,
        storage_relpath: "visual_identities/asset.webp",
        fallback_reason: null,
        is_animated: false,
        content_type: "image/webp",
        asset_url: `/api/v1/visual-identities/packs/${request.override_pack_id ?? 1}/assets/${request.override_pack_id ? 990 : 9}/content`,
        preview_url: null
      })
    )

    const { result, rerender } = renderHook(
      (props: { overridePackId?: number | null }) =>
        useVisualIdentityResolver({
          actorKind: "character",
          actorId: 731,
          expressionKey: "happy",
          overridePackId: props.overridePackId,
          client: { resolveVisualIdentityBinding }
        }),
      { initialProps: { overridePackId: null } }
    )

    await waitFor(() => {
      expect(result.current.resolution?.asset_id).toBe(9)
    })

    rerender({ overridePackId: 77 })

    await waitFor(() => {
      expect(result.current.resolution?.asset_id).toBe(990)
    })
    expect(resolveVisualIdentityBinding.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(resolveVisualIdentityBinding).toHaveBeenCalledWith({
      actor_kind: "character",
      actor_id: 731,
      expression_key: "happy",
      manual_override_expression_key: null,
      mood_expression_key: null,
      override_pack_id: 77,
      override_pack_version_id: null,
      role_id: null,
      role_label: null,
      allow_override_fallback: null
    })
  })

  it("clears cached resolver results when Visual Identity bindings change", async () => {
    const firstClient = {
      resolveVisualIdentityBinding: vi.fn(async () => ({
        actor_kind: "character" as const,
        actor_id: 318,
        pack_id: 1,
        pack_version_id: 2,
        expression_key: "happy",
        requested_expression_key: "happy",
        asset_id: 9,
        storage_relpath: "visual_identities/old.webp",
        fallback_reason: null,
        is_animated: false,
        content_type: "image/webp",
        asset_url: "/old.webp",
        preview_url: null
      }))
    }
    const secondClient = {
      resolveVisualIdentityBinding: vi.fn(async () => ({
        actor_kind: "character" as const,
        actor_id: 318,
        pack_id: 3,
        pack_version_id: 4,
        expression_key: "happy",
        requested_expression_key: "happy",
        asset_id: 10,
        storage_relpath: "visual_identities/new.webp",
        fallback_reason: null,
        is_animated: false,
        content_type: "image/webp",
        asset_url: "/new.webp",
        preview_url: null
      }))
    }

    const first = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: 318,
        expressionKey: "happy",
        client: firstClient
      })
    )
    await waitFor(() => {
      expect(first.result.current.resolution?.asset_id).toBe(9)
    })
    first.unmount()

    clearVisualIdentityResolverCaches()

    const second = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: 318,
        expressionKey: "happy",
        client: secondClient
      })
    )
    await waitFor(() => {
      expect(second.result.current.resolution?.asset_id).toBe(10)
    })
    expect(secondClient.resolveVisualIdentityBinding).toHaveBeenCalledTimes(1)
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
        asset_url: "/api/v1/visual-identities/packs/1/assets/9/content",
        preview_url: null
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

  it("deduplicates concurrent resolver requests for the same actor expression", async () => {
    const resolveVisualIdentityBinding = vi.fn(
      async () =>
        new Promise<any>((resolve) =>
          setTimeout(
            () =>
              resolve({
                actor_kind: "character" as const,
                actor_id: 42,
                pack_id: 1,
                pack_version_id: 2,
                expression_key: "happy",
                requested_expression_key: "happy",
                asset_id: 9,
                storage_relpath: "visual_identities/asset.webp",
                fallback_reason: null,
                is_animated: false,
                content_type: "image/webp",
                asset_url: "/asset.webp",
                preview_url: null
              }),
            0
          )
        )
    )

    const first = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: 42,
        expressionKey: "happy",
        client: { resolveVisualIdentityBinding }
      })
    )
    const second = renderHook(() =>
      useVisualIdentityResolver({
        actorKind: "character",
        actorId: 42,
        expressionKey: "happy",
        client: { resolveVisualIdentityBinding }
      })
    )

    await waitFor(() => {
      expect(first.result.current.resolution?.asset_id).toBe(9)
      expect(second.result.current.resolution?.asset_id).toBe(9)
    })
    expect(resolveVisualIdentityBinding).toHaveBeenCalledTimes(1)
  })
})

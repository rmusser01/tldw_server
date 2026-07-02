import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  VisualIdentityActorKind,
  VisualIdentityResolveResponse
} from "@/types/visual-identities"

export type VisualIdentityResolverClient = {
  resolveVisualIdentityBinding: (request: {
    actor_kind: VisualIdentityActorKind
    actor_id: number | string
    expression_key?: string
    manual_override_expression_key?: string | null
    mood_expression_key?: string | null
  }) => Promise<VisualIdentityResolveResponse>
}

export type UseVisualIdentityResolverOptions = {
  actorKind?: VisualIdentityActorKind | null
  actorId?: number | string | null
  expressionKey?: string | null
  manualOverrideExpressionKey?: string | null
  moodExpressionKey?: string | null
  enabled?: boolean
  client?: VisualIdentityResolverClient
}

export type UseVisualIdentityResolverResult = {
  resolution: VisualIdentityResolveResponse | null
  isLoading: boolean
  error: unknown
  refresh: () => void
}

export type UseVisualIdentityExpressionAvailabilityOptions = {
  actorKind?: VisualIdentityActorKind | null
  actorId?: number | string | null
  expressions?: readonly string[]
  enabled?: boolean
  client?: VisualIdentityResolverClient
}

export type UseVisualIdentityExpressionAvailabilityResult = {
  availability: Record<string, boolean>
  isLoading: boolean
  error: unknown
  refresh: () => void
}

const resolutionCache = new Map<string, VisualIdentityResolveResponse | null>()
const availabilityCache = new Map<string, Record<string, boolean>>()

const buildResolverCacheKey = ({
  actorKind,
  actorId,
  expressionKey,
  manualOverrideExpressionKey,
  moodExpressionKey
}: Required<
  Pick<
    UseVisualIdentityResolverOptions,
    | "actorKind"
    | "actorId"
    | "expressionKey"
    | "manualOverrideExpressionKey"
    | "moodExpressionKey"
  >
>) =>
  JSON.stringify([
    actorKind,
    String(actorId),
    expressionKey || "neutral",
    manualOverrideExpressionKey || "",
    moodExpressionKey || ""
  ])

export const useVisualIdentityResolver = ({
  actorKind,
  actorId,
  expressionKey = "neutral",
  manualOverrideExpressionKey = null,
  moodExpressionKey = null,
  enabled = true,
  client = tldwClient
}: UseVisualIdentityResolverOptions): UseVisualIdentityResolverResult => {
  const normalizedActorId =
    actorId === null || actorId === undefined || String(actorId).trim() === ""
      ? null
      : actorId
  const canResolve = Boolean(enabled && actorKind && normalizedActorId != null)
  const [revision, setRevision] = React.useState(0)
  const [resolution, setResolution] =
    React.useState<VisualIdentityResolveResponse | null>(null)
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<unknown>(null)

  const cacheKey = React.useMemo(() => {
    if (!canResolve || !actorKind || normalizedActorId == null) return null
    return buildResolverCacheKey({
      actorKind,
      actorId: normalizedActorId,
      expressionKey,
      manualOverrideExpressionKey,
      moodExpressionKey
    })
  }, [
    actorKind,
    canResolve,
    expressionKey,
    manualOverrideExpressionKey,
    moodExpressionKey,
    normalizedActorId
  ])

  React.useEffect(() => {
    if (!canResolve || !actorKind || normalizedActorId == null || !cacheKey) {
      setResolution(null)
      setIsLoading(false)
      setError(null)
      return
    }

    if (resolutionCache.has(cacheKey) && revision === 0) {
      setResolution(resolutionCache.get(cacheKey) ?? null)
      setIsLoading(false)
      setError(null)
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)

    client
      .resolveVisualIdentityBinding({
        actor_kind: actorKind,
        actor_id: normalizedActorId,
        expression_key: expressionKey || "neutral",
        manual_override_expression_key: manualOverrideExpressionKey || null,
        mood_expression_key: moodExpressionKey || null
      })
      .then((nextResolution) => {
        if (cancelled) return
        resolutionCache.set(cacheKey, nextResolution)
        setResolution(nextResolution)
      })
      .catch((nextError) => {
        if (cancelled) return
        setError(nextError)
        setResolution(null)
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [
    actorKind,
    cacheKey,
    canResolve,
    client,
    expressionKey,
    manualOverrideExpressionKey,
    moodExpressionKey,
    normalizedActorId,
    revision
  ])

  const refresh = React.useCallback(() => {
    if (cacheKey) {
      resolutionCache.delete(cacheKey)
    }
    setRevision((value) => value + 1)
  }, [cacheKey])

  return { resolution, isLoading, error, refresh }
}

const buildAvailabilityCacheKey = ({
  actorKind,
  actorId,
  expressionKeySignature
}: {
  actorKind: VisualIdentityActorKind
  actorId: number | string
  expressionKeySignature: string
}) => JSON.stringify([actorKind, String(actorId), expressionKeySignature])

const buildExpressionKeySignature = (expressions: readonly string[]): string =>
  Array.from(
    new Set(
      expressions
        .map((expression) => expression.trim())
        .filter(Boolean)
    )
  )
    .sort()
    .join("\u0001")

export const useVisualIdentityExpressionAvailability = ({
  actorKind,
  actorId,
  expressions = [],
  enabled = true,
  client = tldwClient
}: UseVisualIdentityExpressionAvailabilityOptions): UseVisualIdentityExpressionAvailabilityResult => {
  const normalizedActorId =
    actorId === null || actorId === undefined || String(actorId).trim() === ""
      ? null
      : actorId
  const expressionKeySignature = buildExpressionKeySignature(expressions)
  const expressionKeys = React.useMemo(
    () =>
      expressionKeySignature
        ? expressionKeySignature.split("\u0001").filter(Boolean)
        : [],
    [expressionKeySignature]
  )
  const canResolve = Boolean(
    enabled && actorKind && normalizedActorId != null && expressionKeys.length > 0
  )
  const [revision, setRevision] = React.useState(0)
  const [availability, setAvailability] = React.useState<Record<string, boolean>>({})
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<unknown>(null)

  const cacheKey = React.useMemo(() => {
    if (!canResolve || !actorKind || normalizedActorId == null) return null
    return buildAvailabilityCacheKey({
      actorKind,
      actorId: normalizedActorId,
      expressionKeySignature
    })
  }, [actorKind, canResolve, expressionKeySignature, normalizedActorId])

  React.useEffect(() => {
    if (!canResolve || !actorKind || normalizedActorId == null || !cacheKey) {
      setAvailability({})
      setIsLoading(false)
      setError(null)
      return
    }

    if (availabilityCache.has(cacheKey) && revision === 0) {
      setAvailability(availabilityCache.get(cacheKey) ?? {})
      setIsLoading(false)
      setError(null)
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)

    Promise.all(
      expressionKeys.map(async (expressionKey) => {
        try {
          const resolved = await client.resolveVisualIdentityBinding({
            actor_kind: actorKind,
            actor_id: normalizedActorId,
            expression_key: expressionKey
          })
          return [
            expressionKey,
            Boolean(resolved.asset_id && resolved.expression_key === expressionKey),
            null
          ] as const
        } catch (nextError) {
          return [expressionKey, false, nextError] as const
        }
      })
    )
      .then((entries) => {
        if (cancelled) return
        const nextAvailability: Record<string, boolean> = {}
        let nextError: unknown = null
        for (const [expressionKey, hasAsset, entryError] of entries) {
          nextAvailability[expressionKey] = hasAsset
          nextError = nextError ?? entryError
        }
        availabilityCache.set(cacheKey, nextAvailability)
        setAvailability(nextAvailability)
        setError(nextError)
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [
    actorKind,
    cacheKey,
    canResolve,
    client,
    expressionKeySignature,
    expressionKeys,
    normalizedActorId,
    revision
  ])

  const refresh = React.useCallback(() => {
    if (cacheKey) {
      availabilityCache.delete(cacheKey)
    }
    setRevision((value) => value + 1)
  }, [cacheKey])

  return { availability, isLoading, error, refresh }
}

export default useVisualIdentityResolver

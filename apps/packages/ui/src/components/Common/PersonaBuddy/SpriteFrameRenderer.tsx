import React from "react"

import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualFrame,
  PersonaVisualManifest,
  PersonaVisualStateId
} from "@/types/persona-visuals"
import {
  acquirePersonaVisualAsset,
  type PersonaVisualAssetHandle
} from "@/services/persona-visual-assets"

import { normalizeFrames } from "./personaVisualAssets"
import type {
  PersonaVisualRenderError,
  PersonaVisualRenderErrorHandler
} from "./personaVisualTypes"

export type SpriteFrameRendererProps = {
  manifest: PersonaVisualManifest
  assets: Record<string, PersonaVisualAsset>
  state?: PersonaVisualStateId
  requestedState?: PersonaVisualStateId
  generation?: number
  reducedMotion?: boolean
  fallbackLabel: string
  className?: string
  onRenderError?: PersonaVisualRenderErrorHandler
  onReady?: () => void
  onFailure?: (error: PersonaVisualRenderError) => void
  onComplete?: () => void
}

type ResolvedAnimation = {
  animation: PersonaVisualAnimation
  animationId: string
}

export const normalizePersonaVisualFrames = normalizeFrames

export const resolveAnimationForState = (
  manifest: PersonaVisualManifest,
  state: PersonaVisualStateId
): ResolvedAnimation | null => {
  const stateOrder = [
    state,
    ...(manifest.fallbacks?.[state] || []),
    ...(state === "idle" ? [] : ["idle" as PersonaVisualStateId])
  ]
  for (const candidate of stateOrder) {
    const animationId = manifest.states?.[candidate]?.animation_id
    if (!animationId) continue
    const animation = manifest.animations?.[animationId]
    if (animation) return { animation, animationId }
  }
  return null
}

const resolveInitialFrameIndex = (
  animation: PersonaVisualAnimation,
  frames: PersonaVisualFrame[]
): number => {
  if (
    typeof animation.preview_frame === "number" &&
    animation.preview_frame >= 0 &&
    animation.preview_frame < frames.length
  ) {
    return animation.preview_frame
  }
  if (animation.preview_asset_id) {
    const previewIndex = frames.findIndex(
      (frame) => frame.asset_id === animation.preview_asset_id
    )
    if (previewIndex >= 0) return previewIndex
  }
  return 0
}

const resolveFrameDuration = (
  frame: PersonaVisualFrame,
  animation: PersonaVisualAnimation
): number => {
  if (typeof frame.duration_ms === "number" && frame.duration_ms > 0) {
    return frame.duration_ms
  }
  if (typeof animation.frame_rate === "number" && animation.frame_rate > 0) {
    return Math.max(16, Math.round(1000 / animation.frame_rate))
  }
  return 100
}

const renderFrame = ({
  frame,
  asset,
  assetUrl,
  visualState,
  fallbackLabel,
  className
}: {
  frame: PersonaVisualFrame
  asset: PersonaVisualAsset
  assetUrl: string
  visualState: PersonaVisualStateId
  fallbackLabel: string
  className?: string
}) => {
  const sharedProps = {
    "data-testid": "persona-visual-frame",
    "data-visual-state": visualState,
    className
  }
  if (frame.region) {
    const region = frame.region
    const offsetX = region.x === 0 ? 0 : -region.x
    const offsetY = region.y === 0 ? 0 : -region.y
    return (
      <div
        {...sharedProps}
        role="img"
        aria-label={fallbackLabel}
        style={{
          width: `${region.width}px`,
          height: `${region.height}px`,
          backgroundImage: `url(${assetUrl})`,
          backgroundPosition: `${offsetX}px ${offsetY}px`,
          backgroundSize:
            asset.width && asset.height
              ? `${asset.width}px ${asset.height}px`
              : undefined,
          backgroundRepeat: "no-repeat"
        }}
      />
    )
  }
  return (
    <img
      {...sharedProps}
      alt={fallbackLabel}
      src={assetUrl}
      width={asset.width ?? undefined}
      height={asset.height ?? undefined}
      draggable={false}
    />
  )
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)

const hasUnsupportedRegion = (
  frame: PersonaVisualFrame,
  asset: PersonaVisualAsset
): boolean => {
  if (!frame.region) return false
  const { x, y, width, height } = frame.region
  if (
    !isFiniteNumber(x) ||
    !isFiniteNumber(y) ||
    !isFiniteNumber(width) ||
    !isFiniteNumber(height)
  ) {
    return true
  }
  if (x < 0 || y < 0 || width <= 0 || height <= 0) {
    return true
  }
  if (isFiniteNumber(asset.width) && x + width > asset.width) {
    return true
  }
  if (isFiniteNumber(asset.height) && y + height > asset.height) {
    return true
  }
  return false
}

export const SpriteFrameRenderer: React.FC<SpriteFrameRendererProps> = ({
  manifest,
  assets,
  state,
  requestedState,
  generation = 0,
  reducedMotion = false,
  fallbackLabel,
  className,
  onRenderError,
  onReady,
  onFailure,
  onComplete
}) => {
  const visualState = requestedState ?? state ?? "idle"
  const resolved = React.useMemo(
    () => resolveAnimationForState(manifest, visualState),
    [manifest, visualState]
  )
  const frames = React.useMemo(
    () => normalizePersonaVisualFrames(resolved?.animation),
    [resolved]
  )
  const initialFrameIndex = React.useMemo(
    () =>
      resolved ? resolveInitialFrameIndex(resolved.animation, frames) : 0,
    [frames, resolved]
  )
  const [frameIndex, setFrameIndex] = React.useState(initialFrameIndex)
  const handleRef = React.useRef<PersonaVisualAssetHandle | null>(null)
  const requestRef = React.useRef(0)
  const reportedFailureRef = React.useRef<string | null>(null)
  const generationRef = React.useRef(generation)
  const [presented, setPresented] = React.useState<{
    frame: PersonaVisualFrame
    asset: PersonaVisualAsset
    url: string
    generation: number
  } | null>(null)
  const [loadError, setLoadError] = React.useState<PersonaVisualRenderError | null>(null)

  generationRef.current = generation

  React.useEffect(() => {
    setFrameIndex(initialFrameIndex)
  }, [initialFrameIndex, resolved?.animationId, visualState])

  const frame = frames[frameIndex] ?? frames[0]
  const asset = frame ? assets[frame.asset_id] : null
  const structuralError: PersonaVisualRenderError | null = !resolved || !frame
    ? "missing_animation"
    : !asset
      ? "missing_asset"
      : hasUnsupportedRegion(frame, asset)
        ? "unsupported_region"
        : reducedMotion && asset.mime_type !== "image/png"
          ? "static_asset_unsupported"
          : null

  React.useEffect(() => {
    if (structuralError || !frame || !asset) {
      if (reducedMotion) {
        requestRef.current += 1
        handleRef.current?.release()
        handleRef.current = null
        setPresented(null)
      }
      setLoadError(structuralError)
      const failureKey = structuralError ? `${generation}:${structuralError}` : null
      if (structuralError && reportedFailureRef.current !== failureKey) {
        reportedFailureRef.current = failureKey
        onFailure?.(structuralError)
      }
      return undefined
    }
    const request = ++requestRef.current
    const controller = new AbortController()
    let acquired: PersonaVisualAssetHandle | null = null
    void acquirePersonaVisualAsset(asset, { signal: controller.signal })
      .then((handle) => {
        acquired = handle
        if (
          controller.signal.aborted
          || request !== requestRef.current
          || generation !== generationRef.current
        ) {
          handle.release()
          return
        }
        const previous = handleRef.current
        handleRef.current = handle
        acquired = null
        setPresented({ frame, asset, url: handle.url, generation })
        setLoadError(null)
        reportedFailureRef.current = null
        previous?.release()
        onReady?.()
      })
      .catch(() => {
        if (controller.signal.aborted || request !== requestRef.current) return
        setLoadError("asset_load_failed")
        const failureKey = `${generation}:asset_load_failed`
        if (reportedFailureRef.current !== failureKey) {
          reportedFailureRef.current = failureKey
          onFailure?.("asset_load_failed")
        }
      })
    return () => {
      controller.abort()
      acquired?.release()
    }
  }, [asset, frame, generation, onFailure, onReady, reducedMotion, structuralError])

  React.useEffect(() => () => {
    requestRef.current += 1
    handleRef.current?.release()
    handleRef.current = null
  }, [])

  React.useEffect(() => {
    if (
      reducedMotion
      || !resolved
      || !frame
      || presented?.generation !== generation
      || presented.frame !== frame
      || (frames.length <= 1 && !onComplete)
    ) return undefined
    const currentFrame = frames[frameIndex] ?? frames[0]
    const timer = window.setTimeout(() => {
      if (generationRef.current !== generation) return
      if (frames.length <= 1) {
        onComplete?.()
        return
      }
      setFrameIndex((current) => {
        const next = (current + 1) % frames.length
        if (next === initialFrameIndex) onComplete?.()
        return next
      })
    }, resolveFrameDuration(currentFrame, resolved.animation))
    return () => window.clearTimeout(timer)
  }, [frame, frameIndex, frames, generation, initialFrameIndex, onComplete, presented, reducedMotion, resolved])

  const error = structuralError ?? loadError

  React.useEffect(() => {
    onRenderError?.(error)
  }, [error, onRenderError])

  if (!presented) {
    return <span>{fallbackLabel}</span>
  }

  return renderFrame({
    frame: presented.frame,
    asset: presented.asset,
    assetUrl: presented.url,
    visualState,
    fallbackLabel,
    className
  })
}

export default SpriteFrameRenderer

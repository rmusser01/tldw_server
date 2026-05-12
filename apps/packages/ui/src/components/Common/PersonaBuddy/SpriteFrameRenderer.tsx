import React from "react"

import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualFrame,
  PersonaVisualManifest,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { normalizeFrames } from "./personaVisualAssets"
import type {
  PersonaVisualRenderError,
  PersonaVisualRenderErrorHandler
} from "./personaVisualTypes"

export type SpriteFrameRendererProps = {
  manifest: PersonaVisualManifest
  assets: Record<string, PersonaVisualAsset>
  state: PersonaVisualStateId
  fallbackLabel: string
  className?: string
  onRenderError?: PersonaVisualRenderErrorHandler
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
  visualState,
  fallbackLabel,
  className
}: {
  frame: PersonaVisualFrame
  asset: PersonaVisualAsset
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
    return (
      <div
        {...sharedProps}
        role="img"
        aria-label={fallbackLabel}
        style={{
          width: `${region.width}px`,
          height: `${region.height}px`,
          backgroundImage: `url(${asset.url})`,
          backgroundPosition: `-${region.x}px -${region.y}px`,
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
      src={asset.url}
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
  fallbackLabel,
  className,
  onRenderError
}) => {
  const resolved = React.useMemo(
    () => resolveAnimationForState(manifest, state),
    [manifest, state]
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

  React.useEffect(() => {
    setFrameIndex(initialFrameIndex)
  }, [initialFrameIndex, resolved?.animationId, state])

  React.useEffect(() => {
    if (!resolved || frames.length <= 1) return undefined
    const currentFrame = frames[frameIndex] ?? frames[0]
    const timer = window.setTimeout(() => {
      setFrameIndex((current) => (current + 1) % frames.length)
    }, resolveFrameDuration(currentFrame, resolved.animation))
    return () => window.clearTimeout(timer)
  }, [frameIndex, frames, resolved])

  const frame = frames[frameIndex] ?? frames[0]
  const asset = frame ? assets[frame.asset_id] : null
  const error: PersonaVisualRenderError | null = !resolved || !frame
    ? "missing_animation"
    : !asset
      ? "missing_asset"
      : hasUnsupportedRegion(frame, asset)
        ? "unsupported_region"
        : null

  React.useEffect(() => {
    onRenderError?.(error)
  }, [error, onRenderError])

  if (error || !frame || !asset) {
    return <span>{fallbackLabel}</span>
  }

  return renderFrame({
    frame,
    asset,
    visualState: state,
    fallbackLabel,
    className
  })
}

export default SpriteFrameRenderer

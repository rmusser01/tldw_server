import React from "react"

import type {
  PersonaVisualPack,
  PersonaVisualRendererType,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { getAssetsById, normalizeFrames } from "./personaVisualDiagnostics"
import {
  SpriteFrameRenderer,
  type PersonaVisualRenderError
} from "./SpriteFrameRenderer"

export type PersonaVisualRendererComponentProps = {
  pack: PersonaVisualPack
  state: PersonaVisualStateId
  fallbackLabel: string
  className?: string
  onRenderError?: (error: PersonaVisualRenderError | null) => void
}

export type PersonaVisualRendererRegistration = {
  rendererType: PersonaVisualRendererType
  canRender: (pack: PersonaVisualPack | null | undefined) => boolean
  Component: React.ComponentType<PersonaVisualRendererComponentProps>
}

const SpriteFrameRendererHost: React.FC<PersonaVisualRendererComponentProps> = ({
  pack,
  state,
  fallbackLabel,
  className,
  onRenderError
}) => (
  <SpriteFrameRenderer
    manifest={pack.manifest}
    assets={getAssetsById(pack)}
    state={state}
    fallbackLabel={fallbackLabel}
    className={className}
    onRenderError={onRenderError}
  />
)

const SPRITE_FRAME_REGISTRATION: PersonaVisualRendererRegistration = {
  rendererType: "sprite_frames",
  canRender: (pack) => {
    if (pack?.renderer_type !== "sprite_frames" || !pack.manifest) {
      return false
    }
    const assetsById = getAssetsById(pack)
    return Object.values(pack.manifest.animations || {}).some((animation) =>
      normalizeFrames(animation).some((frame) => Boolean(assetsById[frame.asset_id]))
    )
  },
  Component: SpriteFrameRendererHost
}

const RENDERERS: Partial<
  Record<PersonaVisualRendererType, PersonaVisualRendererRegistration>
> = {
  sprite_frames: SPRITE_FRAME_REGISTRATION
}

export const getPersonaVisualRenderer = (
  rendererType: PersonaVisualRendererType | string | null | undefined
): PersonaVisualRendererRegistration | null => {
  if (!rendererType) return null
  return RENDERERS[String(rendererType) as PersonaVisualRendererType] ?? null
}

export const canRenderPersonaVisualPack = (
  pack: PersonaVisualPack | null | undefined
): boolean => Boolean(pack && getPersonaVisualRenderer(pack.renderer_type)?.canRender(pack))

export const PersonaVisualRendererHost: React.FC<
  PersonaVisualRendererComponentProps
> = (props) => {
  const renderer = getPersonaVisualRenderer(props.pack.renderer_type)
  if (!renderer || !renderer.canRender(props.pack)) {
    return <span>{props.fallbackLabel}</span>
  }
  const Component = renderer.Component
  return <Component {...props} />
}

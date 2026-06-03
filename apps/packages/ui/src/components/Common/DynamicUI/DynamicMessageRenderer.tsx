import React from "react"
import type { DynamicUIEnvelope, DynamicUISurface } from "@/types/dynamic-ui"
import { DynamicUIErrorBoundary } from "./DynamicUIErrorBoundary"
import { DynamicUISourceFallback } from "./DynamicUISourceFallback"
import {
  isDynamicUIEnabledForSurface,
  loadDynamicUIRenderer,
  type DynamicUIRendererComponent
} from "./registry"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const DynamicMessageRenderer = ({
  envelope,
  sourceMessageId,
  surface,
  onAction
}: {
  envelope: DynamicUIEnvelope
  sourceMessageId: string
  sourceText: string
  surface: DynamicUISurface
  onAction?: (payload: unknown) => void
}) => {
  const [Renderer, setRenderer] =
    React.useState<DynamicUIRendererComponent | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const handleAction = React.useCallback(
    (payload: unknown) => {
      if (!onAction) return
      const actionPayload = isRecord(payload) ? payload : { values: payload }
      onAction({
        ...actionPayload,
        renderer: envelope.renderer,
        sourceMessageId
      })
    },
    [envelope.renderer, onAction, sourceMessageId]
  )

  React.useEffect(() => {
    let active = true
    setError(null)
    setRenderer(null)

    if (!isDynamicUIEnabledForSurface(surface)) {
      return () => {
        active = false
      }
    }

    loadDynamicUIRenderer(envelope.renderer)
      .then((module) => {
        if (active) setRenderer(() => module.default)
      })
      .catch((err) => {
        if (!active) return
        setError(
          err instanceof Error ? err.message : "Failed to load renderer."
        )
      })

    return () => {
      active = false
    }
  }, [envelope.renderer, surface])

  const fallbackSource = envelope.source

  if (!isDynamicUIEnabledForSurface(surface)) {
    return <DynamicUISourceFallback source={fallbackSource} surface={surface} />
  }
  if (error) {
    return (
      <DynamicUISourceFallback
        source={fallbackSource}
        error={error}
        surface={surface}
      />
    )
  }
  if (!Renderer) {
    return <DynamicUISourceFallback source={fallbackSource} surface={surface} />
  }

  return (
    <DynamicUIErrorBoundary source={fallbackSource} surface={surface}>
      <Renderer
        envelope={envelope}
        sourceMessageId={sourceMessageId}
        source={envelope.source}
        onAction={handleAction}
      />
    </DynamicUIErrorBoundary>
  )
}

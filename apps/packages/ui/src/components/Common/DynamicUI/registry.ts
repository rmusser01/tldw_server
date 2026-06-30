import type { ComponentType } from "react"
import type { DynamicUIEnvelope, DynamicUISurface } from "@/types/dynamic-ui"

export type DynamicUIRendererProps = {
  envelope: DynamicUIEnvelope
  sourceMessageId: string
  source: string
  onAction?: (payload: unknown) => void
}

export type DynamicUIRendererComponent = ComponentType<DynamicUIRendererProps>

export const isDynamicUIEnabledForSurface = (
  surface: DynamicUISurface
): boolean => surface === "web-chat"

export const loadDynamicUIRenderer = async (
  renderer: DynamicUIEnvelope["renderer"]
): Promise<{ default: DynamicUIRendererComponent }> => {
  if (renderer === "openui") {
    return import("./renderers/OpenUIRenderer")
  }
  throw new Error(`Unsupported dynamic UI renderer: ${renderer}`)
}

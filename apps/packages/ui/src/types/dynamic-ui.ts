export type DynamicUIRendererId = "openui"
export type DynamicUISurface = "web-chat" | "extension-sidepanel" | "workspace" | "artifact"
export type DynamicUIActionType = "submit"

export type DynamicUIEnvelope = {
  renderer: DynamicUIRendererId
  version: "v1"
  source: string
  state?: Record<string, unknown>
  capabilities?: string[]
}

export type DynamicUIRequest = {
  renderer: DynamicUIRendererId
}

export type DynamicUIActionPayload = {
  renderer: DynamicUIRendererId
  sourceMessageId: string
  actionId: string
  actionType: DynamicUIActionType
  values: Record<string, unknown>
}

export type DynamicUIActionUserMetadata = DynamicUIActionPayload & {
  submittedAt: string
}

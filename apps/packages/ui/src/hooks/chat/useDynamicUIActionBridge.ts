import React from "react"
import type { Message } from "@/store/option"
import {
  formatDynamicUIActionUserMessage,
  normalizeDynamicUIActionPayload,
  shouldBlockDynamicUIActionValues
} from "@/utils/dynamic-ui"

export const useDynamicUIActionBridge = ({
  messages,
  onSubmit,
  confirmSensitiveValues
}: {
  messages: Message[]
  onSubmit: (payload: any) => Promise<unknown>
  confirmSensitiveValues: (payload: unknown) => Promise<boolean>
}) =>
  React.useCallback(
    async (rawPayload: unknown) => {
      const currentMessageIds = new Set(
        messages
          .map((message) => message.id)
          .filter((id): id is string => typeof id === "string" && id.length > 0)
      )
      const normalized = normalizeDynamicUIActionPayload(rawPayload, {
        currentMessageIds
      })
      if (!normalized) return

      if (shouldBlockDynamicUIActionValues(normalized.values)) {
        const confirmed = await confirmSensitiveValues(normalized)
        if (!confirmed) return
      }

      const metadata = {
        ...normalized,
        submittedAt: new Date().toISOString()
      }
      await onSubmit({
        message: formatDynamicUIActionUserMessage(metadata),
        image: "",
        userMetadataExtra: {
          dynamic_ui_action: metadata
        }
      })
    },
    [confirmSensitiveValues, messages, onSubmit]
  )

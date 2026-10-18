import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption, type ToolChoice } from "@/store/option"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { resolveChatToolRequest } from "@/utils/chat-tools"

/** Capture before preparation waits; explicit selections retain their own intent. */
export const captureChatDispatchSource = (params: {
  selectedModel: string
  toolChoice?: ToolChoice
  selectionSource?: {
    model?: "global" | "explicit"
    toolChoice?: "global" | "explicit"
  }
}) => {
  const dispatchSettings = () => {
    const settings = useStoreChatModelSettings.getState()
    const selection = useStoreMessageOption.getState()
    const tools = useMcpToolsStore.getState()
    const effectiveToolChoice =
      params.selectionSource?.toolChoice === "global"
        ? selection.toolChoice
        : params.toolChoice ?? selection.toolChoice
    const toolRequest = resolveChatToolRequest({
      tools: tools.chatTools ?? tools.tools,
      toolChoice: effectiveToolChoice,
      mcpHealthState: tools.healthState,
      hasMcp: tools.healthState !== "unavailable"
    })
    return JSON.stringify({
      settings: settings.getEffectiveSettings(),
      settingsScope: settings.activeSettingsScope,
      model:
        params.selectionSource?.model === "global"
          ? selection.selectedModel
          : params.selectedModel,
      toolChoice: effectiveToolChoice,
      tools: toolRequest.tools,
      requestToolChoice: toolRequest.toolChoice
    })
  }
  const captured = dispatchSettings()
  return () => dispatchSettings() === captured
}

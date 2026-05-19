import i18n from "i18next"
import { formatErrorMessage } from "@/utils/format-error-message"
import { parseBillingLimitError } from "@/utils/billing-error"

export const TLDW_ERROR_BUBBLE_PREFIX = "__tldw_error__:"

export type ChatErrorPayload = {
  summary: string
  hint: string
  detail: string
  upgradeUrl?: string
  category?: string
}

export const encodeChatErrorPayload = (payload: ChatErrorPayload): string =>
  `${TLDW_ERROR_BUBBLE_PREFIX}${JSON.stringify(payload)}`

export const decodeChatErrorPayload = (
  message: string
): ChatErrorPayload | null => {
  if (!message || !message.startsWith(TLDW_ERROR_BUBBLE_PREFIX)) {
    return null
  }
  const raw = message.slice(TLDW_ERROR_BUBBLE_PREFIX.length)
  try {
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed.summary !== "string" || typeof parsed.hint !== "string") {
      return null
    }
    return {
      summary: parsed.summary,
      hint: parsed.hint,
      detail: typeof parsed.detail === "string" ? parsed.detail : "",
      upgradeUrl: typeof parsed.upgradeUrl === "string" ? parsed.upgradeUrl : undefined,
      category: typeof parsed.category === "string" ? parsed.category : undefined,
    }
  } catch {
    return null
  }
}

export const buildFriendlyErrorMessage = (rawError: unknown): string => {
  const detail = formatErrorMessage(rawError, "Request failed")
  const lower = detail.toLowerCase()

  let summary: string
  let hint: string

  if (lower.includes("invalid x-api-key")) {
    summary = i18n.t(
      "common:error.friendlyApiKeySummary",
      "We couldn’t reach your tldw server."
    )
    hint = i18n.t(
      "common:error.friendlyApiKeyHint",
      "Your API key may be invalid. Open Settings → tldw server to check your URL and API key, then try again."
    )
  } else if (
    lower.includes("not a valid model id") ||
    lower.includes("invalid model id") ||
    lower.includes("model_not_found") ||
    lower.includes("no such model")
  ) {
    summary = i18n.t(
      "common:error.friendlyModelUnavailableSummary",
      "The selected model is not available."
    )
    hint = i18n.t(
      "common:error.friendlyModelUnavailableHint",
      "Choose a different model or refresh the model list, then try again."
    )
  } else if (
    lower.includes("/api/v1/files/create") &&
    (lower.includes("body.file_type") ||
      lower.includes("file_type") ||
      lower.includes("unsupported_file_type") ||
      lower.includes("unsupported_export_format"))
  ) {
    summary = i18n.t(
      "common:error.imageArtifactsUnsupportedSummary",
      "Your tldw server doesn't support image generation yet."
    )
    hint = i18n.t(
      "common:error.imageArtifactsUnsupportedHint",
      "Update to a tldw_server2 build with file artifacts enabled, then try again."
    )
  } else if (lower.includes("image_backend_unavailable")) {
    summary = i18n.t(
      "common:error.imageBackendUnavailableSummary",
      "Image generation isn't available on your server."
    )
    hint = i18n.t(
      "common:error.imageBackendUnavailableHint",
      "Enable an image backend (e.g., Flux-Klein or ZTurbo) in your tldw server config, then try again."
    )
  } else if (
    lower.includes("no_provider_configured") ||
    lower.includes("no llm providers are configured") ||
    lower.includes("no providers configured") ||
    lower.includes("provider_not_configured") ||
    (lower.includes("provider") &&
      (lower.includes("not configured") || lower.includes("no api key")))
  ) {
    summary = i18n.t(
      "common:error.friendlyNoProviderSummary",
      "No LLM provider is configured on your server."
    )
    hint = i18n.t(
      "common:error.friendlyNoProviderHint",
      "Add an API key for OpenAI, Anthropic, or another provider in your server's .env file, then restart the server and try again."
    )
  } else if (lower.includes("stream timeout: no updates received")) {
    summary = i18n.t(
      "common:error.friendlyTimeoutSummary",
      "Your chat timed out."
    )
    hint = i18n.t(
      "common:error.friendlyTimeoutHint",
      "The server stopped streaming responses. Try again, or open Health & diagnostics to check server status."
    )
  } else if (
    lower.includes("chunkererror") ||
    lower.includes("chunker error") ||
    lower.includes("chunking failed") ||
    lower.includes("unable to chunk")
  ) {
    summary = i18n.t(
      "common:error.friendlyChunkerSummary",
      "This file couldn't be processed."
    )
    hint = i18n.t(
      "common:error.friendlyChunkerHint",
      "The server had trouble splitting the file into chunks. Try a different format or a smaller file."
    )
  } else if (
    lower.includes("timeouterror") ||
    lower.includes("timed out") ||
    lower.includes("request timeout") ||
    lower.includes("gateway timeout") ||
    lower.includes("504")
  ) {
    summary = i18n.t(
      "common:error.friendlyProcessingTimeoutSummary",
      "Processing took too long."
    )
    hint = i18n.t(
      "common:error.friendlyProcessingTimeoutHint",
      "Try a smaller file, or increase the timeout in Settings."
    )
  } else if (
    lower.includes("connectionerror") ||
    lower.includes("connection refused") ||
    lower.includes("econnrefused") ||
    lower.includes("network error") ||
    lower.includes("failed to fetch") ||
    lower.includes("err_connection")
  ) {
    summary = i18n.t(
      "common:error.friendlyConnectionSummary",
      "Lost connection to the server."
    )
    hint = i18n.t(
      "common:error.friendlyConnectionHint",
      "Check that the server is running, then try again. Open Health & diagnostics for more details."
    )
  } else if (
    lower.includes("limit_exceeded") ||
    lower.includes("feature_not_available") ||
    lower.includes("quota exceeded") ||
    lower.includes("upgrade your plan")
  ) {
    const billingInfo = parseBillingLimitError(rawError)
    summary = i18n.t(
      "common:error.billingLimitSummary",
      "You've reached a plan limit."
    )
    hint = billingInfo?.category
      ? i18n.t(
          "common:error.billingLimitHint",
          "Your {{category}} usage has reached the limit for your current plan. Upgrade to continue.",
          { category: billingInfo.category.replace(/_/g, " ") }
        )
      : i18n.t(
          "common:error.billingFeatureHint",
          "This feature is not available on your current plan. Upgrade to unlock it."
        )
    return encodeChatErrorPayload({
      summary,
      hint,
      detail,
      upgradeUrl: billingInfo?.upgradeUrl || "/billing/plans",
      category: billingInfo?.category,
    })
  } else {
    summary = i18n.t(
      "common:error.friendlyGenericSummary",
      "Something went wrong while talking to your tldw server."
    )
    hint = i18n.t(
      "common:error.friendlyGenericHint",
      "Try again in a moment, or open Health & diagnostics to inspect server health."
    )
  }

  return encodeChatErrorPayload({
    summary,
    hint,
    detail
  })
}

export const buildAssistantErrorContent = (
  botMessage: string | undefined,
  rawError: unknown
): string => {
  if (botMessage && String(botMessage).trim().length > 0) {
    return String(botMessage)
  }
  return buildFriendlyErrorMessage(rawError)
}

import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope";
import { deriveScopedUserId } from "@/utils/media-navigation-scope";

/** Explicit authenticated support; unknown and missing fields fail closed. */
export type NotesLanguageCapability = {
  available: boolean;
  reason:
    | "unsupported_server"
    | "authentication_required"
    | "transport_unavailable"
    | "scope_changed"
    | null;
};

export function decodeNotesLanguageCapability(
  value: unknown,
): NotesLanguageCapability {
  return value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as Record<string, unknown>).content_language === true
    ? { available: true, reason: null }
    : { available: false, reason: "unsupported_server" };
}

/** Reuse the existing request guard's target, credential lineage and principal fields. */
export function captureNotesLanguageRequestScope(
  config: TldwConfig | null,
): ServicePromptRequestScope | undefined {
  if (!config) return undefined;
  const userScope = deriveScopedUserId(config);
  const userId =
    config.authMode === "multi-user" &&
    userScope !== "user:anonymous" &&
    userScope.startsWith("user:")
      ? userScope.slice(5)
      : null;
  return {
    config: {
      serverUrl: config.serverUrl,
      authMode: config.authMode,
      authSource: config.authSource,
      orgId: config.orgId,
      expectedRefreshToken: config.refreshToken,
      expectedSingleUserApiKeyScope:
        deriveSingleUserApiKeyCredentialScope(config.authMode, config.apiKey) ??
        undefined,
    },
    userId,
  };
}

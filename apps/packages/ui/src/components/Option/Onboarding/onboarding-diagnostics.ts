import type { TFunction } from "i18next"
import type { ConnectionErrorKind } from "./validation"

export type OnboardingDiagnosticSeverity = "blocking" | "recoverable" | "warning"

export type OnboardingDiagnosticActionId =
  | "edit_api_key"
  | "edit_server_url"
  | "retry"
  | "open_setup"
  | "open_audio_setup"
  | "open_rag_settings"
  | "review_storage_paths"
  | "continue_without_optional"

export type OnboardingDiagnosticAction = {
  id: OnboardingDiagnosticActionId
  label: string
}

export type OnboardingDiagnostic = {
  title: string
  cause: string
  whyItMatters: string
  severity: OnboardingDiagnosticSeverity
  blockingFirstChat?: boolean
  primaryAction: OnboardingDiagnosticAction
  secondaryActions: OnboardingDiagnosticAction[]
}

export type OnboardingReadinessIssueKind =
  | "config_write_failed"
  | "restart_needed"
  | "network_unavailable"
  | "downloads_disabled"
  | "package_installs_disabled"
  | "rag_storage_unavailable"
  | "audio_readiness_failed"

export type OnboardingReadinessDiagnosticOptions = {
  selectedOptionalLane?: boolean
}

const action = (
  id: OnboardingDiagnosticActionId,
  label: string
): OnboardingDiagnosticAction => ({ id, label })

export function buildSetupDiagnostic(
  errorKind: ConnectionErrorKind,
  t: TFunction
): OnboardingDiagnostic | null {
  switch (errorKind) {
    case "auth_invalid":
      return {
        title: t(
          "settings:onboarding.diagnostics.auth.title",
          "API key was not accepted"
        ),
        cause: t(
          "settings:onboarding.diagnostics.auth.cause",
          "The backend rejected the API key during authentication."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.auth.why",
          "A valid key is required before the first chat can run."
        ),
        severity: "blocking",
        primaryAction: action(
          "edit_api_key",
          t("settings:onboarding.diagnostics.actions.editApiKey", "Edit API key")
        ),
        secondaryActions: [
          action("retry", t("common:retry", "Retry")),
          action(
            "edit_server_url",
            t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
          ),
        ],
      }
    case "refused":
      return {
        title: t(
          "settings:onboarding.diagnostics.refused.title",
          "Server is not accepting connections"
        ),
        cause: t(
          "settings:onboarding.diagnostics.refused.cause",
          "The URL is reachable as a local address, but no tldw server responded there."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.refused.why",
          "The setup shell needs a reachable backend before it can validate chat."
        ),
        severity: "blocking",
        primaryAction: action(
          "edit_server_url",
          t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
        ),
        secondaryActions: [action("retry", t("common:retry", "Retry"))],
      }
    case "dns_failed":
      return {
        title: t(
          "settings:onboarding.diagnostics.dns.title",
          "Server address could not be found"
        ),
        cause: t(
          "settings:onboarding.diagnostics.dns.cause",
          "The hostname in the server URL did not resolve."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.dns.why",
          "The browser cannot reach the backend until the URL points to a valid host."
        ),
        severity: "blocking",
        primaryAction: action(
          "edit_server_url",
          t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
        ),
        secondaryActions: [action("retry", t("common:retry", "Retry"))],
      }
    case "timeout":
      return {
        title: t(
          "settings:onboarding.diagnostics.timeout.title",
          "Server response timed out"
        ),
        cause: t(
          "settings:onboarding.diagnostics.timeout.cause",
          "The backend did not answer the setup check quickly enough."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.timeout.why",
          "First chat will also fail until the backend responds consistently."
        ),
        severity: "blocking",
        primaryAction: action("retry", t("common:retry", "Retry")),
        secondaryActions: [
          action(
            "edit_server_url",
            t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
          ),
        ],
      }
    case "cors_blocked":
      return {
        title: t(
          "settings:onboarding.diagnostics.cors.title",
          "Browser access is blocked"
        ),
        cause: t(
          "settings:onboarding.diagnostics.cors.cause",
          "The backend did not allow this WebUI origin for browser requests."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.cors.why",
          "The WebUI cannot complete setup until the backend allows this origin."
        ),
        severity: "blocking",
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [
          action("retry", t("common:retry", "Retry")),
          action(
            "edit_server_url",
            t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
          ),
        ],
      }
    case "ssl_error":
      return {
        title: t(
          "settings:onboarding.diagnostics.ssl.title",
          "Certificate check failed"
        ),
        cause: t(
          "settings:onboarding.diagnostics.ssl.cause",
          "The browser rejected the server certificate."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.ssl.why",
          "Setup cannot continue until the URL and certificate match."
        ),
        severity: "blocking",
        primaryAction: action(
          "edit_server_url",
          t("settings:onboarding.diagnostics.actions.editServerUrl", "Edit server URL")
        ),
        secondaryActions: [action("retry", t("common:retry", "Retry"))],
      }
    case "server_error":
      return {
        title: t(
          "settings:onboarding.diagnostics.server.title",
          "Server reported a setup error"
        ),
        cause: t(
          "settings:onboarding.diagnostics.server.cause",
          "The backend returned an internal error while checking setup."
        ),
        whyItMatters: t(
          "settings:onboarding.diagnostics.server.why",
          "First chat may keep failing until the backend recovers or setup is repaired."
        ),
        severity: "recoverable",
        primaryAction: action("retry", t("common:retry", "Retry")),
        secondaryActions: [
          action(
            "open_setup",
            t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
          ),
        ],
      }
    default:
      return null
  }
}

export function buildReadinessDiagnostic(
  issueKind: OnboardingReadinessIssueKind,
  t: TFunction,
  options: OnboardingReadinessDiagnosticOptions = {}
): OnboardingDiagnostic {
  const selectedOptionalLane = options.selectedOptionalLane === true

  switch (issueKind) {
    case "config_write_failed":
      return {
        title: t(
          "settings:onboarding.readiness.configWrite.title",
          "Setup changes could not be saved"
        ),
        cause: t(
          "settings:onboarding.readiness.configWrite.cause",
          "The backend reported that it could not write the requested setup configuration."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.configWrite.why",
          "The first chat should not run until the server can persist the provider and setup values."
        ),
        severity: "blocking",
        blockingFirstChat: true,
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [action("retry", t("common:retry", "Retry"))],
      }
    case "restart_needed":
      return {
        title: t(
          "settings:onboarding.readiness.restart.title",
          "Server restart is needed"
        ),
        cause: t(
          "settings:onboarding.readiness.restart.cause",
          "The backend accepted a setup change that requires a restart before it is active."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.restart.why",
          "Retry first chat after restarting the backend so the new provider settings are loaded."
        ),
        severity: "recoverable",
        blockingFirstChat: true,
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [action("retry", t("common:retry", "Retry"))],
      }
    case "network_unavailable":
      return {
        title: t(
          "settings:onboarding.readiness.network.title",
          "Network access is unavailable"
        ),
        cause: t(
          "settings:onboarding.readiness.network.cause",
          "The backend cannot reach required external services right now."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.network.why",
          "Hosted providers and package downloads need network access before they can complete."
        ),
        severity: "recoverable",
        blockingFirstChat: true,
        primaryAction: action("retry", t("common:retry", "Retry")),
        secondaryActions: [
          action(
            "open_setup",
            t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
          ),
        ],
      }
    case "downloads_disabled":
      return {
        title: t(
          "settings:onboarding.readiness.downloads.title",
          "Model downloads are disabled"
        ),
        cause: t(
          "settings:onboarding.readiness.downloads.cause",
          "This server is configured not to download model assets during setup."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.downloads.why",
          "Local model and audio setup may need operator action before those features are ready."
        ),
        severity: "recoverable",
        blockingFirstChat: false,
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [
          action(
            "continue_without_optional",
            t(
              "settings:onboarding.readiness.actions.continueWithoutOptional",
              "Continue without optional setup"
            )
          ),
        ],
      }
    case "package_installs_disabled":
      return {
        title: t(
          "settings:onboarding.readiness.installs.title",
          "Package installs are disabled"
        ),
        cause: t(
          "settings:onboarding.readiness.installs.cause",
          "This server will not install missing packages automatically during setup."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.installs.why",
          "Some local or audio features may need manual operator setup before they can run."
        ),
        severity: "recoverable",
        blockingFirstChat: false,
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [
          action(
            "continue_without_optional",
            t(
              "settings:onboarding.readiness.actions.continueWithoutOptional",
              "Continue without optional setup"
            )
          ),
        ],
      }
    case "rag_storage_unavailable": {
      if (selectedOptionalLane) {
        return {
          title: t(
            "settings:onboarding.readiness.ragStorage.title",
            "RAG storage is not ready"
          ),
          cause: t(
            "settings:onboarding.readiness.ragStorage.cause",
            "The selected knowledge or storage path is not ready for ingestion."
          ),
          whyItMatters: t(
            "settings:onboarding.readiness.ragStorage.why",
            "Fix storage before continuing if you chose ingestion or RAG as part of first use."
          ),
          severity: "recoverable",
          blockingFirstChat: true,
          primaryAction: action(
            "open_rag_settings",
            t("settings:onboarding.readiness.actions.openRagSettings", "Open RAG settings")
          ),
          secondaryActions: [
            action(
              "review_storage_paths",
              t(
                "settings:onboarding.readiness.actions.reviewStoragePaths",
                "Review storage paths"
              )
            ),
          ],
        }
      }
      return {
        title: t(
          "settings:onboarding.readiness.ragStorage.title",
          "RAG storage is not ready"
        ),
        cause: t(
          "settings:onboarding.readiness.ragStorage.optionalCause",
          "Knowledge storage has a setup issue, but it is optional before first chat."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.ragStorage.optionalWhy",
          "You can finish first chat now and return to ingestion or RAG setup afterward."
        ),
        severity: "warning",
        blockingFirstChat: false,
        primaryAction: action(
          "continue_without_optional",
          t(
            "settings:onboarding.readiness.actions.continueWithoutOptional",
            "Continue without optional setup"
          )
        ),
        secondaryActions: [
          action(
            "open_rag_settings",
            t("settings:onboarding.readiness.actions.openRagSettings", "Open RAG settings")
          ),
        ],
      }
    }
    case "audio_readiness_failed": {
      if (selectedOptionalLane) {
        return {
          title: t(
            "settings:onboarding.readiness.audio.title",
            "Audio setup is not ready"
          ),
          cause: t(
            "settings:onboarding.readiness.audio.cause",
            "The selected STT or TTS configuration did not pass backend readiness checks."
          ),
          whyItMatters: t(
            "settings:onboarding.readiness.audio.why",
            "Fix audio readiness before continuing if audio was selected for first use."
          ),
          severity: "recoverable",
          blockingFirstChat: true,
          primaryAction: action(
            "open_audio_setup",
            t("settings:onboarding.readiness.actions.openAudioSetup", "Open audio setup")
          ),
          secondaryActions: [
            action("retry", t("common:retry", "Retry")),
            action(
              "continue_without_optional",
              t(
                "settings:onboarding.readiness.actions.continueWithoutOptional",
                "Continue without optional setup"
              )
            ),
          ],
        }
      }
      return {
        title: t(
          "settings:onboarding.readiness.audio.title",
          "Audio setup is not ready"
        ),
        cause: t(
          "settings:onboarding.readiness.audio.optionalCause",
          "Audio has a setup issue, but STT and TTS are optional before first chat."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.audio.optionalWhy",
          "You can finish first chat now and return to audio setup afterward."
        ),
        severity: "warning",
        blockingFirstChat: false,
        primaryAction: action(
          "continue_without_optional",
          t(
            "settings:onboarding.readiness.actions.continueWithoutOptional",
            "Continue without optional setup"
          )
        ),
        secondaryActions: [
          action(
            "open_audio_setup",
            t("settings:onboarding.readiness.actions.openAudioSetup", "Open audio setup")
          ),
        ],
      }
    }
    default:
      return {
        title: t(
          "settings:onboarding.readiness.generic.title",
          "System readiness issue"
        ),
        cause: t(
          "settings:onboarding.readiness.generic.cause",
          "An unexpected readiness issue was detected."
        ),
        whyItMatters: t(
          "settings:onboarding.readiness.generic.why",
          "This may affect some features during first use."
        ),
        severity: "warning",
        blockingFirstChat: false,
        primaryAction: action(
          "open_setup",
          t("settings:onboarding.diagnostics.actions.openSetup", "Open setup recovery")
        ),
        secondaryActions: [],
      }
  }
}

import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import { PageAssistLoader } from "@/components/Common/PageAssistLoader";
import { useSetupReadinessSummary } from "@/hooks/useSetupReadinessSummary";
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding";
import { useConnectionActions } from "@/hooks/useConnectionState";
import {
  normalizeSelectedModel,
  useSelectedModel,
} from "@/hooks/chat/useSelectedModel";
import { useStoreMessageOption } from "@/store/option";
import { createSafeStorage } from "@/utils/safe-storage";
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient";
import { normalizeProviderAvailabilityKey } from "@/services/tldw/model-provider-availability";
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider";
import { servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error";
import { derivePromptAssistAuthorizationRevision } from "@/services/chat-surface-scope";
import type {
  FirstRunMetadata,
  FirstRunState,
  FirstRunStepUpdateRequest,
  SetupProviderSaveResponse,
  FirstChatVerifyResponse,
  SetupCompleteResponse,
} from "@/types/setup-onboarding";
import { SetupPathStep } from "./steps/SetupPathStep";
import { PrivacySecurityStep } from "./steps/PrivacySecurityStep";
import { MultiUserExitPanel } from "./steps/MultiUserExitPanel";
import {
  ProviderSetupStep,
  type ProviderSelection,
  type ProviderSavedPayloadFingerprintState,
  type ProviderValidationViewState,
} from "./steps/ProviderSetupStep";
import { IngestDefaultsStep } from "./steps/IngestDefaultsStep";
import { AudioSetupStep } from "./steps/AudioSetupStep";
import { OptionalAdvancedStep } from "./steps/OptionalAdvancedStep";
import { McpToolsStep } from "./steps/McpToolsStep";
import { FirstChatStep } from "./steps/FirstChatStep";
import { SetupReadinessPanel } from "./SetupReadinessPanel";

type WizardStep =
  | "setup_path"
  | "privacy_security"
  | "provider_setup"
  | "ingest_defaults"
  | "audio_defaults"
  | "optional_advanced"
  | "mcp_tools"
  | "first_chat"
  | "multi_user_exit";

type SoloSetupPath = "docker" | "local";

type UnifiedSetupWizardProps = {
  initialState?: FirstRunState | null;
  initialMetadata?: FirstRunMetadata | null;
  onStateChange?: (state: FirstRunState) => void;
  onComplete?: () => void;
};

const setupPathToBackend = (path: SoloSetupPath) =>
  path === "docker" ? "docker_single_user" : "local_single_user";

const modelStorage = createSafeStorage();
const configStorage = createSafeStorage({ area: "local" });
const setupChatProviderAliases: Record<string, string> = {
  koboldcpp: "kobold",
  custom_openai_api2: "custom-openai-api-2",
};
type SetupModelHandoff = {
  generation: number;
  config: TldwConfig;
  selectionRevision: number;
  eligible: boolean;
  verified?: FirstChatVerifyResponse;
  completed?: SetupCompleteResponse;
};

const stepFromState = (state: FirstRunState | null): WizardStep => {
  const completed = new Set(state?.completed_steps ?? []);
  if (!completed.has("setup_path")) return "setup_path";
  if (!completed.has("privacy_security")) return "privacy_security";
  if (!completed.has("providers")) return "provider_setup";
  if (!completed.has("ingest_defaults")) return "ingest_defaults";
  if (!completed.has("audio_defaults")) return "audio_defaults";
  if (!completed.has("optional_advanced")) return "optional_advanced";
  const mcpToolsData = state?.step_data?.mcp_tools;
  const mcpToolsDone =
    mcpToolsData?.acknowledged === true ||
    mcpToolsData?.validation_state === "skipped";
  if (!mcpToolsDone) return "mcp_tools";
  return "first_chat";
};

const providerSelectionFromState = (
  state: FirstRunState | null,
): ProviderSelection | null => {
  const data = state?.step_data?.providers;
  const provider = data?.default_provider;
  const model = data?.default_model;
  const credentialConfigured =
    data?.default_provider_credential_configured === true;
  if (typeof provider === "string" && typeof model === "string") {
    return { provider, model, credential_configured: credentialConfigured };
  }
  const firstChatProvider = state?.first_chat?.provider;
  const firstChatModel = state?.first_chat?.model;
  if (firstChatProvider && firstChatModel) {
    return {
      provider: firstChatProvider,
      model: firstChatModel,
      credential_configured: credentialConfigured,
    };
  }
  return null;
};

export function UnifiedSetupWizard({
  initialState = null,
  initialMetadata = null,
  onStateChange,
  onComplete,
}: UnifiedSetupWizardProps = {}) {
  const navigate = useNavigate();
  const { t } = useTranslation("settings");
  const { setConfigPartial } = useConnectionActions();
  const { setSelectedModel } = useSelectedModel();
  const handoffRef = React.useRef<SetupModelHandoff | null>(null);
  const handoffGeneration = React.useRef(0);
  const selectionRevision = React.useRef(0);
  React.useEffect(() => {
    const invalidate = () => {
      handoffGeneration.current += 1;
    };
    const watchedConfig = Object.fromEntries(
      ["tldwConfig", "tldwCookieSessionConfig", "tldwManualSessionApiKey"].map(
        (key) => [key, invalidate],
      ),
    );
    configStorage.watch(watchedConfig);
    const unsubscribe = useStoreMessageOption.subscribe((next, previous) => {
      if (next.selectedModel !== previous.selectedModel)
        selectionRevision.current += 1;
    });
    const storageChanged = (event: StorageEvent) => {
      const key = event.key?.replace(/^plasmo-(?:local|sync):/, "");
      if (
        key == null ||
        [
          "tldwConfig",
          "tldwCookieSessionConfig",
          "tldwManualSessionApiKey",
        ].includes(key)
      )
        invalidate();
    };
    window.addEventListener("tldw:config-updated", invalidate);
    window.addEventListener("tldw:auth-principal-changed", invalidate);
    window.addEventListener("storage", storageChanged);
    return () => {
      invalidate();
      configStorage.unwatch(watchedConfig);
      unsubscribe();
      window.removeEventListener("tldw:config-updated", invalidate);
      window.removeEventListener("tldw:auth-principal-changed", invalidate);
      window.removeEventListener("storage", storageChanged);
    };
  }, []);
  const {
    state,
    metadata,
    providerCatalog,
    mcpToolsCatalog,
    audioRecommendations,
    loading,
    error,
    refresh,
    loadProviderCatalog,
    loadMcpToolsCatalog,
    loadAudioRecommendations,
    saveStep,
    skip,
    saveProvider,
    validateProvider,
    saveIngestDefaults,
    saveAudioDefaults,
    saveOptionalAdvanced,
    applyMcpTools,
    validateMcpTools,
    verifyFirstChat,
    complete,
  } = useSetupOnboarding({
    initialState,
    initialMetadata,
    autoLoad: !initialState || !initialMetadata,
  });
  const {
    status: setupReadinessStatus,
    loading: setupReadinessLoading,
    error: setupReadinessError,
    refresh: refreshSetupReadinessStatus,
  } = useSetupReadinessSummary({
    enabled: Boolean(metadata) && metadata.auth_mode !== "multi_user",
  });
  const [step, setStep] = React.useState<WizardStep>(() =>
    stepFromState(initialState),
  );
  const [providerSelection, setProviderSelection] =
    React.useState<ProviderSelection | null>(() =>
      providerSelectionFromState(initialState),
    );
  const [providerSavedProviders, setProviderSavedProviders] = React.useState<
    Record<string, SetupProviderSaveResponse>
  >({});
  const [
    providerSavedPayloadFingerprints,
    setProviderSavedPayloadFingerprints,
  ] = React.useState<ProviderSavedPayloadFingerprintState>({});
  const [providerSavedDefaultProvider, setProviderSavedDefaultProvider] =
    React.useState<string | null>(
      () => providerSelectionFromState(initialState)?.provider ?? null,
    );
  const [providerValidationState, setProviderValidationState] = React.useState<
    Record<string, ProviderValidationViewState>
  >({});
  const [providerEditRevisions, setProviderEditRevisions] = React.useState<
    Record<string, number>
  >({});
  const [savingStep, setSavingStep] = React.useState(false);
  const [skipPending, setSkipPending] = React.useState(false);
  const skipPendingRef = React.useRef(false);
  const [mcpToolsSkipPending, setMcpToolsSkipPending] = React.useState(false);
  const mcpToolsSkipPendingRef = React.useRef(false);
  const [stepError, setStepError] = React.useState<string | null>(null);
  const [loginPending, setLoginPending] = React.useState(false);
  const isMultiUserServer = metadata?.auth_mode === "multi_user";
  // Public progress can omit a saved local path. Keep anonymous resume actionable.
  const activeStep = isMultiUserServer
    ? "multi_user_exit"
    : step === "first_chat" && !providerSelection ? "provider_setup" : step;

  const handleSignIn = async () => {
    setLoginPending(true);
    setStepError(null);
    let stage = "configuration";
    try {
      await setConfigPartial({ authMode: "multi-user" });
      stage = "navigation";
      navigate("/settings/tldw");
    } catch (error) {
      // Config/navigation errors can contain credentials. Report only the
      // operation and built-in type; omit payload, stack, and custom names.
      console.error("Setup sign-in failed", {
        stage,
        errorType:
          error instanceof TypeError ? "TypeError" :
          error instanceof Error ? "Error" : "NonError",
      });
      setStepError(t("onboarding.loginSettingsError", {
        defaultValue: "Login settings could not be opened. Try again.",
      }));
    } finally {
      setLoginPending(false);
    }
  };

  React.useEffect(() => {
    if (!state) return;
    setProviderSelection(
      (current) => current ?? providerSelectionFromState(state),
    );
  }, [state]);

  React.useEffect(() => {
    if (activeStep !== "provider_setup" || providerCatalog.length > 0) return;
    void loadProviderCatalog().catch((err) => {
      console.error("Provider catalog could not be loaded", err);
      setStepError("Provider catalog could not be loaded. Try again.");
    });
  }, [loadProviderCatalog, providerCatalog.length, activeStep]);

  React.useEffect(() => {
    if (step !== "audio_defaults" || audioRecommendations.length > 0) return;
    void loadAudioRecommendations().catch((err) => {
      console.error("Audio recommendations could not be loaded", err);
      setStepError(
        "Audio recommendations could not be loaded. You can continue with defaults.",
      );
    });
  }, [audioRecommendations.length, loadAudioRecommendations, step]);

  const persistStep = React.useCallback(
    async (payload: FirstRunStepUpdateRequest) => {
      setSavingStep(true);
      setStepError(null);
      try {
        const nextState = await saveStep(payload);
        onStateChange?.(nextState);
        return nextState;
      } catch (err) {
        console.error("Setup progress could not be saved", err);
        setStepError("Setup progress could not be saved. Try again.");
        return null;
      } finally {
        setSavingStep(false);
      }
    },
    [onStateChange, saveStep],
  );

  const refreshParentState = React.useCallback(
    async (beforePublish?: () => Promise<void>) => {
      const nextState = await refresh().catch(() => null);
      if (nextState) {
        await beforePublish?.();
        onStateChange?.(nextState);
      }
      return nextState;
    },
    [onStateChange, refresh],
  );

  const refreshSetupReadiness = React.useCallback(() => {
    void refreshSetupReadinessStatus().catch((err) => {
      console.warn("Setup readiness summary could not be refreshed", err);
    });
  }, [refreshSetupReadinessStatus]);

  const handlePathSelect = React.useCallback(
    (path: "docker" | "local" | "multi_user") => {
      if (path === "multi_user") {
        setStepError(null);
        setStep("multi_user_exit");
        return;
      }

      void (async () => {
        const nextState = await persistStep({
          step: "setup_path",
          data: {
            acknowledged: true,
            selected_path: setupPathToBackend(path),
            setup_path_key: setupPathToBackend(path),
            install_method: path,
            deployment_mode: "single_user",
          },
        });
        if (!nextState) return;
        setStep("privacy_security");
      })();
    },
    [persistStep],
  );

  const handlePrivacyContinue = React.useCallback(() => {
    void (async () => {
      const nextState = await persistStep({
        step: "privacy_security",
        data: {
          acknowledged: true,
          local_only: metadata?.connection?.browser_access === "local",
          allow_remote_setup_access: Boolean(metadata?.remote_setup_enabled),
        },
      });
      if (!nextState) return;
      setStep("provider_setup");
    })();
  }, [metadata, persistStep]);

  const handleProviderContinue = React.useCallback(
    (selection: ProviderSelection) => {
      void (async () => {
        const nextState = await persistStep({
          step: "providers",
          data: {
            acknowledged: true,
            default_provider: selection.provider,
            default_model: selection.model,
            default_provider_credential_configured: Boolean(
              selection.credential_configured,
            ),
          },
        });
        if (!nextState) return;
        setProviderSelection(selection);
        setStep("ingest_defaults");
      })();
    },
    [persistStep],
  );

  const saveIngestAndPublish = React.useCallback(
    async (...args: Parameters<typeof saveIngestDefaults>) => {
      const response = await saveIngestDefaults(...args);
      refreshSetupReadiness();
      await refreshParentState();
      return response;
    },
    [refreshParentState, refreshSetupReadiness, saveIngestDefaults],
  );

  const saveAudioAndPublish = React.useCallback(
    async (...args: Parameters<typeof saveAudioDefaults>) => {
      const response = await saveAudioDefaults(...args);
      refreshSetupReadiness();
      await refreshParentState();
      return response;
    },
    [refreshParentState, refreshSetupReadiness, saveAudioDefaults],
  );

  const saveAdvancedAndPublish = React.useCallback(
    async (...args: Parameters<typeof saveOptionalAdvanced>) => {
      const response = await saveOptionalAdvanced(...args);
      refreshSetupReadiness();
      await refreshParentState();
      return response;
    },
    [refreshParentState, refreshSetupReadiness, saveOptionalAdvanced],
  );

  const applyMcpToolsAndPublish = React.useCallback(
    async (...args: Parameters<typeof applyMcpTools>) => {
      const response = await applyMcpTools(...args);
      refreshSetupReadiness();
      await refreshParentState();
      return response;
    },
    [applyMcpTools, refreshParentState, refreshSetupReadiness],
  );

  const validateMcpToolsAndPublish = React.useCallback(
    async (...args: Parameters<typeof validateMcpTools>) => {
      const response = await validateMcpTools(...args);
      refreshSetupReadiness();
      await refreshParentState();
      return response;
    },
    [refreshParentState, refreshSetupReadiness, validateMcpTools],
  );

  const assertHandoffCurrent = React.useCallback(
    async (handoff: SetupModelHandoff) => {
      const config = await tldwClient.getConfig();
      if (
        handoff.generation !== handoffGeneration.current ||
        !config ||
        !servicePromptTargetsMatch(config, handoff.config) ||
        derivePromptAssistAuthorizationRevision(config) !==
          derivePromptAssistAuthorizationRevision(handoff.config)
      ) {
        throw new Error(
          "The connection changed. Return to setup for the current server before finishing.",
        );
      }
    },
    [],
  );

  const verifyFirstChatForHandoff = React.useCallback(
    async (...args: Parameters<typeof verifyFirstChat>) => {
      const generation = ++handoffGeneration.current;
      const revision = selectionRevision.current;
      handoffRef.current = null;
      const config = await tldwClient.getConfig();
      const stored = await modelStorage.get<string | null>("selectedModel");
      if (
        !config?.serverUrl ||
        config.authMode !== "single-user" ||
        generation !== handoffGeneration.current
      ) {
        throw new Error(
          "The setup connection is unavailable. Reconnect before verifying the model.",
        );
      }
      const handoff: SetupModelHandoff = {
        generation,
        config: { ...config },
        selectionRevision: revision,
        eligible:
          revision === selectionRevision.current &&
          !normalizeSelectedModel(
            useStoreMessageOption.getState().selectedModel,
          ) &&
          !normalizeSelectedModel(stored),
      };
      await assertHandoffCurrent(handoff);
      const response = await verifyFirstChat(...args);
      await assertHandoffCurrent(handoff);
      if (response.status === "ready") {
        if (
          normalizeProviderAvailabilityKey(response.provider) !==
            normalizeProviderAvailabilityKey(args[0].provider) ||
          response.model.trim() !== args[0].model.trim()
        ) {
          throw new Error(
            "The verified model did not match the requested setup selection. Verify it again.",
          );
        }
        handoff.verified = response;
        handoffRef.current = handoff;
      }
      return response;
    },
    [assertHandoffCurrent, verifyFirstChat],
  );

  const completeAndPublish = React.useCallback(
    async (...args: Parameters<typeof complete>) => {
      const handoff = handoffRef.current;
      if (!handoff?.verified)
        throw new Error("Verify the current setup model before finishing.");
      await assertHandoffCurrent(handoff);
      const response = handoff.completed ?? (await complete(...args));
      if (!response.success)
        throw new Error(
          response.message || "Setup completion could not be saved.",
        );
      handoff.completed = response;
      await assertHandoffCurrent(handoff);
      if (
        handoff.eligible &&
        handoff.selectionRevision === selectionRevision.current
      ) {
        const verified = handoff.verified;
        // First-run verification precedes browser authentication. Its matching
        // ready response is authoritative even while the protected catalog is
        // unavailable; do not make finishing setup depend on that catalog.
        let qualified = parseProviderQualifiedModelSelection(
          `${verified.provider}:${verified.model.trim()}`,
        );
        if (!qualified.provider) {
          const provider =
            normalizeProviderAvailabilityKey(verified.provider) || "";
          qualified = parseProviderQualifiedModelSelection(
            `${setupChatProviderAliases[provider] || provider}:${verified.model.trim()}`,
          );
        }
        if (!qualified.provider || qualified.modelId !== verified.model.trim())
          throw new Error(
            "The verified model provider could not be selected. Check the provider settings.",
          );
        const write = setSelectedModel(
          `tldw:${qualified.provider}:${qualified.modelId}`,
        );
        // Our own publication may be retried after a rejected device write.
        // Any later user operation still advances beyond this revision.
        handoff.selectionRevision = selectionRevision.current;
        await write;
      }
      await assertHandoffCurrent(handoff);
      await refreshParentState(() => assertHandoffCurrent(handoff));
      return response;
    },
    [assertHandoffCurrent, complete, refreshParentState, setSelectedModel],
  );

  const saveProviderAndRefreshReadiness = React.useCallback(
    async (...args: Parameters<typeof saveProvider>) => {
      try {
        return await saveProvider(...args);
      } finally {
        refreshSetupReadiness();
      }
    },
    [refreshSetupReadiness, saveProvider],
  );

  const validateProviderAndRefreshReadiness = React.useCallback(
    async (...args: Parameters<typeof validateProvider>) => {
      try {
        return await validateProvider(...args);
      } finally {
        refreshSetupReadiness();
      }
    },
    [refreshSetupReadiness, validateProvider],
  );

  const handleSkip = React.useCallback(() => {
    if (skipPendingRef.current) return;
    skipPendingRef.current = true;
    setSkipPending(true);
    setStepError(null);
    void skip({ reason: "user_skip" })
      .then((nextState) => {
        onStateChange?.(nextState);
      })
      .catch((err) => {
        console.error("Setup skip could not be saved", err);
        setStepError("Setup skip could not be saved. Try again.");
      })
      .finally(() => {
        skipPendingRef.current = false;
        setSkipPending(false);
        void refreshSetupReadiness();
      });
  }, [onStateChange, refreshSetupReadiness, skip]);

  if (loading && !state && !metadata) {
    return (
      <PageAssistLoader
        label="Loading setup..."
        description="Reading setup readiness from the server"
      />
    );
  }

  return (
    <div
      data-testid="unified-setup-shell"
      tabIndex={-1}
      className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4 py-8"
    >
      <header className="mb-6">
        <p className="text-xs font-medium uppercase tracking-normal text-text-muted">
          {isMultiUserServer ? "Multi-user connection" : "Solo onboarding"}
        </p>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold text-text">
              First-time setup
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-text-muted">
              {isMultiUserServer
                ? "Sign in to your server with an account created by its administrator."
                : "Configure the minimum needed to reach a successful first chat."}
            </p>
          </div>
          {!isMultiUserServer ? (
            <button
              type="button"
              onClick={handleSkip}
              disabled={skipPending}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
            >
              {skipPending ? "Skipping..." : "Skip for now"}
            </button>
          ) : null}
        </div>
      </header>

      {error ? (
        <div
          role="alert"
          className="mb-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          Setup progress could not be loaded. The server may still be starting,
          or the connection details may be missing - the wizard works once the
          app can reach your tldw server.
        </div>
      ) : null}

      {stepError ? (
        <div
          role="alert"
          className="mb-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {stepError}
        </div>
      ) : null}

      {!isMultiUserServer ? (
        <SetupReadinessPanel
          status={setupReadinessStatus}
          loading={setupReadinessLoading}
          error={setupReadinessError}
          onRetry={refreshSetupReadiness}
        />
      ) : null}

      <div className="rounded-md border border-border bg-bg px-4 py-5 shadow-sm md:px-6">
        {activeStep === "setup_path" ? (
          <SetupPathStep onSelect={handlePathSelect} />
        ) : null}
        {activeStep === "privacy_security" ? (
          <PrivacySecurityStep
            metadata={metadata}
            onBack={() => setStep("setup_path")}
            onContinue={handlePrivacyContinue}
            saving={savingStep}
          />
        ) : null}
        {activeStep === "multi_user_exit" ? (
          <MultiUserExitPanel
            metadata={metadata}
            onBack={isMultiUserServer ? undefined : () => setStep("setup_path")}
            onSignIn={isMultiUserServer ? handleSignIn : undefined}
            loginPending={loginPending}
          />
        ) : null}
        {activeStep === "provider_setup" ? (
          <ProviderSetupStep
            providers={providerCatalog}
            initialSelection={providerSelection}
            savedProviders={providerSavedProviders}
            savedPayloadFingerprints={providerSavedPayloadFingerprints}
            savedDefaultProvider={providerSavedDefaultProvider}
            validationState={providerValidationState}
            providerEditRevisions={providerEditRevisions}
            onSaveProvider={saveProviderAndRefreshReadiness}
            onValidateProvider={validateProviderAndRefreshReadiness}
            onSavedProvidersChange={setProviderSavedProviders}
            onSavedPayloadFingerprintsChange={
              setProviderSavedPayloadFingerprints
            }
            onSavedDefaultProviderChange={setProviderSavedDefaultProvider}
            onValidationStateChange={setProviderValidationState}
            onProviderEditRevisionsChange={setProviderEditRevisions}
            onContinue={handleProviderContinue}
            onBack={() => setStep("privacy_security")}
          />
        ) : null}
        {activeStep === "ingest_defaults" ? (
          <IngestDefaultsStep
            saveIngestDefaults={saveIngestAndPublish}
            onContinue={() => setStep("audio_defaults")}
            onBack={() => setStep("provider_setup")}
          />
        ) : null}
        {activeStep === "audio_defaults" ? (
          <AudioSetupStep
            recommendations={audioRecommendations}
            saveAudioDefaults={saveAudioAndPublish}
            onContinue={() => setStep("optional_advanced")}
            onBack={() => setStep("ingest_defaults")}
          />
        ) : null}
        {activeStep === "optional_advanced" ? (
          <OptionalAdvancedStep
            saveOptionalAdvanced={saveAdvancedAndPublish}
            onContinue={() => setStep("mcp_tools")}
            onBack={() => setStep("audio_defaults")}
          />
        ) : null}
        {activeStep === "mcp_tools" ? (
          <McpToolsStep
            catalog={mcpToolsCatalog}
            initialStepData={
              state?.step_data?.mcp_tools ??
              initialState?.step_data?.mcp_tools ??
              null
            }
            loadCatalog={loadMcpToolsCatalog}
            applyMcpTools={applyMcpToolsAndPublish}
            validateMcpTools={validateMcpToolsAndPublish}
            skipPending={mcpToolsSkipPending || savingStep}
            onContinue={() => setStep("first_chat")}
            onBack={() => setStep("optional_advanced")}
            onSkip={() => {
              if (mcpToolsSkipPendingRef.current) return;
              mcpToolsSkipPendingRef.current = true;
              setMcpToolsSkipPending(true);
              void (async () => {
                try {
                  const nextState = await persistStep({
                    step: "mcp_tools",
                    data: { acknowledged: true, validation_state: "skipped" },
                  });
                  if (!nextState) return;
                  setStep("first_chat");
                } finally {
                  mcpToolsSkipPendingRef.current = false;
                  setMcpToolsSkipPending(false);
                }
              })();
            }}
          />
        ) : null}
        {activeStep === "first_chat" && providerSelection ? (
          <FirstChatStep
            provider={providerSelection.provider}
            model={providerSelection.model}
            verifyFirstChat={verifyFirstChatForHandoff}
            complete={completeAndPublish}
            onComplete={() => {
              onComplete?.();
            }}
            onBack={() => setStep("mcp_tools")}
            backLabel="Back to MCP tools"
            onEditProvider={() => setStep("provider_setup")}
            onSwitchProvider={() => setStep("provider_setup")}
            onCheckEndpoint={() => setStep("provider_setup")}
            onSkip={handleSkip}
            skipPending={skipPending}
          />
        ) : null}
      </div>
    </div>
  );
}

export default UnifiedSetupWizard;

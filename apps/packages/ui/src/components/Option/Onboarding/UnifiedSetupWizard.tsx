import React from "react";

import { PageAssistLoader } from "@/components/Common/PageAssistLoader";
import { useSetupReadinessSummary } from "@/hooks/useSetupReadinessSummary";
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding";
import type {
  FirstRunMetadata,
  FirstRunState,
  FirstRunStepUpdateRequest,
  SetupProviderSaveResponse,
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
import { FirstChatStep } from "./steps/FirstChatStep";
import { SetupReadinessPanel } from "./SetupReadinessPanel";

type WizardStep =
  | "setup_path"
  | "privacy_security"
  | "provider_setup"
  | "ingest_defaults"
  | "audio_defaults"
  | "optional_advanced"
  | "first_chat"
  | "multi_user_exit";

type SoloSetupPath = "docker" | "local";

type UnifiedSetupWizardProps = {
  initialState?: FirstRunState | null;
  initialMetadata?: FirstRunMetadata | null;
  onStateChange?: (state: FirstRunState) => void;
};

const setupPathToBackend = (path: SoloSetupPath) =>
  path === "docker" ? "docker_single_user" : "local_single_user";

const stepFromState = (state: FirstRunState | null): WizardStep => {
  const completed = new Set(state?.completed_steps ?? []);
  if (!completed.has("setup_path")) return "setup_path";
  if (!completed.has("privacy_security")) return "privacy_security";
  if (!completed.has("providers")) return "provider_setup";
  if (!completed.has("ingest_defaults")) return "ingest_defaults";
  if (!completed.has("audio_defaults")) return "audio_defaults";
  if (!completed.has("optional_advanced")) return "optional_advanced";
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
}: UnifiedSetupWizardProps = {}) {
  const {
    state,
    metadata,
    providerCatalog,
    audioRecommendations,
    loading,
    error,
    refresh,
    loadProviderCatalog,
    loadAudioRecommendations,
    saveStep,
    skip,
    saveProvider,
    validateProvider,
    saveIngestDefaults,
    saveAudioDefaults,
    saveOptionalAdvanced,
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
  } = useSetupReadinessSummary();
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
    React.useState<string | null>(() =>
      providerSelectionFromState(initialState)?.provider ?? null,
    );
  const [providerValidationState, setProviderValidationState] = React.useState<
    Record<string, ProviderValidationViewState>
  >({});
  const [providerEditRevisions, setProviderEditRevisions] = React.useState<
    Record<string, number>
  >({});
  const [savingStep, setSavingStep] = React.useState(false);
  const [stepError, setStepError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!state) return;
    setProviderSelection(
      (current) => current ?? providerSelectionFromState(state),
    );
  }, [state]);

  React.useEffect(() => {
    if (step !== "provider_setup" || providerCatalog.length > 0) return;
    void loadProviderCatalog().catch((err) => {
      console.error("Provider catalog could not be loaded", err);
      setStepError("Provider catalog could not be loaded. Try again.");
    });
  }, [loadProviderCatalog, providerCatalog.length, step]);

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

  const refreshParentState = React.useCallback(async () => {
    const nextState = await refresh().catch(() => null);
    if (nextState) onStateChange?.(nextState);
    return nextState;
  }, [onStateChange, refresh]);

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

  const completeAndPublish = React.useCallback(
    async (...args: Parameters<typeof complete>) => {
      const response = await complete(...args);
      await refreshParentState();
      return response;
    },
    [complete, refreshParentState],
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
          Solo onboarding
        </p>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold text-text">
              First-time setup
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-text-muted">
              Configure the minimum needed to reach a successful first chat.
            </p>
          </div>
          <button
            type="button"
            onClick={handleSkip}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
          >
            Skip for now
          </button>
        </div>
      </header>

      {error ? (
        <div
          role="alert"
          className="mb-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          Setup readiness could not be loaded.
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

      <SetupReadinessPanel
        status={setupReadinessStatus}
        loading={setupReadinessLoading}
        error={setupReadinessError}
        onRetry={refreshSetupReadinessStatus}
      />

      <div className="rounded-md border border-border bg-bg px-4 py-5 shadow-sm md:px-6">
        {step === "setup_path" ? (
          <SetupPathStep onSelect={handlePathSelect} />
        ) : null}
        {step === "privacy_security" ? (
          <PrivacySecurityStep
            metadata={metadata}
            onBack={() => setStep("setup_path")}
            onContinue={handlePrivacyContinue}
            saving={savingStep}
          />
        ) : null}
        {step === "multi_user_exit" ? (
          <MultiUserExitPanel
            metadata={metadata}
            onBack={() => setStep("setup_path")}
          />
        ) : null}
        {step === "provider_setup" ? (
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
        {step === "ingest_defaults" ? (
          <IngestDefaultsStep
            saveIngestDefaults={saveIngestAndPublish}
            onContinue={() => setStep("audio_defaults")}
            onBack={() => setStep("provider_setup")}
          />
        ) : null}
        {step === "audio_defaults" ? (
          <AudioSetupStep
            recommendations={audioRecommendations}
            saveAudioDefaults={saveAudioAndPublish}
            onContinue={() => setStep("optional_advanced")}
            onBack={() => setStep("ingest_defaults")}
          />
        ) : null}
        {step === "optional_advanced" ? (
          <OptionalAdvancedStep
            saveOptionalAdvanced={saveAdvancedAndPublish}
            onContinue={() => setStep("first_chat")}
            onBack={() => setStep("audio_defaults")}
          />
        ) : null}
        {step === "first_chat" && providerSelection ? (
          <FirstChatStep
            provider={providerSelection.provider}
            model={providerSelection.model}
            verifyFirstChat={verifyFirstChat}
            complete={completeAndPublish}
            onComplete={() => {
              void refreshParentState();
            }}
            onBack={() => setStep("provider_setup")}
          />
        ) : null}
      </div>
    </div>
  );
}

export default UnifiedSetupWizard;

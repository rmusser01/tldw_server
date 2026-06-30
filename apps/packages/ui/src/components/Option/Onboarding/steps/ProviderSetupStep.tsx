import React from "react";
import { CircleCheck, KeyRound, Server } from "lucide-react";

import type {
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse,
  SetupProviderValidationResponse,
} from "@/types/setup-onboarding";

export type ProviderSelection = {
  provider: string;
  model: string;
  credential_configured?: boolean;
};

type ProviderSetupStepProps = {
  providers: SetupProviderCatalogEntry[];
  initialSelection?: ProviderSelection | null;
  savedProviders?: SavedProviderState;
  savedPayloadFingerprints?: ProviderSavedPayloadFingerprintState;
  savedDefaultProvider?: string | null;
  validationState?: ProviderValidationState;
  providerEditRevisions?: ProviderEditRevisionState;
  onSaveProvider: (
    payload: SetupProviderSaveRequest,
  ) => Promise<SetupProviderSaveResponse>;
  onValidateProvider: (
    payload: SetupProviderSaveRequest,
  ) => Promise<SetupProviderValidationResponse>;
  onSavedProvidersChange?: React.Dispatch<
    React.SetStateAction<SavedProviderState>
  >;
  onSavedPayloadFingerprintsChange?: React.Dispatch<
    React.SetStateAction<ProviderSavedPayloadFingerprintState>
  >;
  onSavedDefaultProviderChange?: (savedDefaultProvider: string | null) => void;
  onValidationStateChange?: React.Dispatch<
    React.SetStateAction<ProviderValidationState>
  >;
  onProviderEditRevisionsChange?: React.Dispatch<
    React.SetStateAction<ProviderEditRevisionState>
  >;
  onContinue: (selection: ProviderSelection) => void;
  onBack?: () => void;
};

type ProviderFormValues = {
  apiKey: string;
  baseUrl: string;
  model: string;
};

type ProviderFieldName = "base-url" | "model";

type PendingProviderFocus = {
  providerKey: string;
  field: ProviderFieldName;
};

export type ProviderValidationViewState = {
  fingerprint: string;
  response: SetupProviderValidationResponse;
};

export type ProviderSavedPayloadFingerprintState = Record<string, string>;
type SavedProviderState = Record<string, SetupProviderSaveResponse>;
type ProviderValidationState = Record<string, ProviderValidationViewState>;
type ProviderEditRevisionState = Record<string, number>;

const emptyValues: ProviderFormValues = {
  apiKey: "",
  baseUrl: "",
  model: "",
};

const sortedProviders = (providers: SetupProviderCatalogEntry[]) =>
  [...providers].sort((a, b) => {
    const aScore =
      (a.recommended_for_first_chat ? 0 : 4) +
      (a.provider_type === "hosted_api_key" ? 0 : 2);
    const bScore =
      (b.recommended_for_first_chat ? 0 : 4) +
      (b.provider_type === "hosted_api_key" ? 0 : 2);
    if (aScore !== bScore) return aScore - bScore;
    return a.label.localeCompare(b.label);
  });

const validationCanGate = (
  response: SetupProviderValidationResponse | null | undefined,
) => {
  const statusCanGate =
    response?.status === "ready" || response?.status === "accepted";
  return Boolean(statusCanGate && (response?.can_gate_first_chat ?? true));
};

const FAILURE_LOCAL_PROVIDER_UNREACHABLE = "local_provider_unreachable";
const FAILURE_MODEL_DISCOVERY_UNAVAILABLE = "model_discovery_unavailable";

const validationStatusCopy = (
  response: SetupProviderValidationResponse,
): string => {
  if (response.status === "failed") {
    const details = [
      response.failure_category,
      response.message || "Provider validation failed.",
    ]
      .filter(Boolean)
      .join(": ");
    if (response.failure_category === FAILURE_LOCAL_PROVIDER_UNREACHABLE) {
      return [
        details || "Local provider endpoint is unreachable.",
        "Check that the local service is running, the host and port are correct, and the URL includes /v1.",
      ].join(" ");
    }
    return details;
  }
  if (response.status === "accepted") {
    if (
      response.failure_category === FAILURE_MODEL_DISCOVERY_UNAVAILABLE
    ) {
      return (
        response.message ||
        "Model discovery is unavailable. Enter the model name manually; first chat will verify it."
      );
    }
    if (response.message?.toLowerCase().includes("first chat verifies")) {
      return response.message;
    }
    return "Format accepted; first chat verifies this provider.";
  }
  if (response.status === "ready") {
    return response.message || "Provider validation is ready.";
  }
  return response.message || "Provider validation response received.";
};

const savedHostedCredentialFallbackValidation = (
  provider: SetupProviderCatalogEntry,
): SetupProviderValidationResponse => ({
  provider_key: provider.provider_key,
  status: "accepted",
  message: "Saved credentials are present; first chat verifies this provider.",
  models: [],
  validation_level: "local_syntax",
  can_gate_first_chat: true,
});

const hasBackendConfirmedSavedCredential = (
  saved: SetupProviderSaveResponse | null | undefined,
) => Boolean(saved?.credential_configured === true || saved?.masked_api_key);

const initialSavedProvider = (
  initialSelection: ProviderSelection,
): SetupProviderSaveResponse => ({
  provider_key: initialSelection.provider,
  status: "saved",
  credential_configured: Boolean(initialSelection.credential_configured),
  model: initialSelection.model,
  make_default: true,
});

const providerValuesEqual = (
  left: ProviderFormValues | undefined,
  right: ProviderFormValues,
) =>
  Boolean(left) &&
  left?.apiKey === right.apiKey &&
  left?.baseUrl === right.baseUrl &&
  left?.model === right.model;

const providerFieldRefKey = (
  providerKey: string,
  field: ProviderFieldName,
) => `${providerKey}:${field}`;

export function ProviderSetupStep({
  providers,
  initialSelection = null,
  savedProviders: controlledSavedProviders,
  savedPayloadFingerprints: controlledSavedPayloadFingerprints,
  savedDefaultProvider: controlledSavedDefaultProvider,
  validationState: controlledValidationState,
  providerEditRevisions: controlledProviderEditRevisions,
  onSaveProvider,
  onValidateProvider,
  onSavedProvidersChange,
  onSavedPayloadFingerprintsChange,
  onSavedDefaultProviderChange,
  onValidationStateChange,
  onProviderEditRevisionsChange,
  onContinue,
  onBack,
}: ProviderSetupStepProps) {
  const orderedProviders = React.useMemo(
    () => sortedProviders(providers),
    [providers],
  );
  const [selectedProviders, setSelectedProviders] = React.useState<Set<string>>(
    () =>
      new Set(initialSelection?.provider ? [initialSelection.provider] : []),
  );
  const [defaultProvider, setDefaultProvider] = React.useState(
    initialSelection?.provider ?? "",
  );
  const [values, setValues] = React.useState<
    Record<string, ProviderFormValues>
  >(() =>
    initialSelection?.provider
      ? {
          [initialSelection.provider]: {
            ...emptyValues,
            model: initialSelection.model,
          },
        }
      : {},
  );
  const [internalSavedProviders, setInternalSavedProviders] =
    React.useState<SavedProviderState>(() =>
      initialSelection?.provider
        ? {
            [initialSelection.provider]: {
              provider_key: initialSelection.provider,
              status: "saved",
              credential_configured: Boolean(
                initialSelection.credential_configured,
              ),
              model: initialSelection.model,
              make_default: true,
            },
          }
        : {},
    );
  const [
    internalSavedPayloadFingerprints,
    setInternalSavedPayloadFingerprints,
  ] = React.useState<ProviderSavedPayloadFingerprintState>({});
  const [internalSavedDefaultProvider, setInternalSavedDefaultProvider] =
    React.useState<string | null>(() => initialSelection?.provider ?? null);
  const [savingProvider, setSavingProvider] = React.useState(false);
  const [validatingProvider, setValidatingProvider] = React.useState<
    string | null
  >(null);
  const [internalValidationState, setInternalValidationState] =
    React.useState<ProviderValidationState>({});
  const [
    internalProviderEditRevisions,
    setInternalProviderEditRevisions,
  ] = React.useState<ProviderEditRevisionState>({});
  const [error, setError] = React.useState<string | null>(null);
  const providerFieldRefs = React.useRef<Map<string, HTMLInputElement>>(
    new Map(),
  );
  const [pendingProviderFocus, setPendingProviderFocus] =
    React.useState<PendingProviderFocus | null>(null);
  const savedProviders = controlledSavedProviders ?? internalSavedProviders;
  const savedPayloadFingerprints =
    controlledSavedPayloadFingerprints ?? internalSavedPayloadFingerprints;
  const savedDefaultProvider =
    controlledSavedDefaultProvider !== undefined
      ? controlledSavedDefaultProvider
      : internalSavedDefaultProvider;
  const validationState = controlledValidationState ?? internalValidationState;
  const providerEditRevisions =
    controlledProviderEditRevisions ?? internalProviderEditRevisions;

  const updateSavedProviders = React.useCallback(
    (updater: React.SetStateAction<SavedProviderState>) => {
      if (onSavedProvidersChange) {
        onSavedProvidersChange(updater);
      } else {
        setInternalSavedProviders(updater);
      }
    },
    [onSavedProvidersChange],
  );

  const updateSavedPayloadFingerprints = React.useCallback(
    (
      updater: React.SetStateAction<ProviderSavedPayloadFingerprintState>,
    ) => {
      if (onSavedPayloadFingerprintsChange) {
        onSavedPayloadFingerprintsChange(updater);
      } else {
        setInternalSavedPayloadFingerprints(updater);
      }
    },
    [onSavedPayloadFingerprintsChange],
  );

  const updateSavedDefaultProvider = React.useCallback(
    (next: string | null) => {
      if (onSavedDefaultProviderChange) {
        onSavedDefaultProviderChange(next);
      } else {
        setInternalSavedDefaultProvider(next);
      }
    },
    [onSavedDefaultProviderChange],
  );

  const updateValidationState = React.useCallback(
    (updater: React.SetStateAction<ProviderValidationState>) => {
      if (onValidationStateChange) {
        onValidationStateChange(updater);
      } else {
        setInternalValidationState(updater);
      }
    },
    [onValidationStateChange],
  );

  const updateProviderEditRevisions = React.useCallback(
    (updater: React.SetStateAction<ProviderEditRevisionState>) => {
      if (onProviderEditRevisionsChange) {
        onProviderEditRevisionsChange(updater);
      } else {
        setInternalProviderEditRevisions(updater);
      }
    },
    [onProviderEditRevisionsChange],
  );

  const registerProviderField = React.useCallback(
    (providerKey: string, field: ProviderFieldName) =>
      (element: HTMLInputElement | null) => {
        const key = providerFieldRefKey(providerKey, field);
        if (element) {
          providerFieldRefs.current.set(key, element);
        } else {
          providerFieldRefs.current.delete(key);
        }
      },
    [],
  );

  const focusProviderField = React.useCallback(
    (providerKey: string, field: ProviderFieldName) => {
      const element = providerFieldRefs.current.get(
        providerFieldRefKey(providerKey, field),
      );
      if (!element) return false;
      element.focus();
      return true;
    },
    [],
  );

  React.useEffect(() => {
    if (!pendingProviderFocus) return;
    if (
      focusProviderField(
        pendingProviderFocus.providerKey,
        pendingProviderFocus.field,
      )
    ) {
      setPendingProviderFocus(null);
    }
  }, [focusProviderField, pendingProviderFocus]);

  const currentValues = values[defaultProvider] ?? emptyValues;
  const defaultProviderConfig = orderedProviders.find(
    (provider) => provider.provider_key === defaultProvider,
  );
  const selectedDefaultModel = currentValues.model.trim();
  const providerFingerprint = React.useCallback(
    (
      provider: SetupProviderCatalogEntry,
      providerValues: ProviderFormValues,
      model: string | null,
      savedOverride?: SetupProviderSaveResponse,
    ) => {
      const saved = savedOverride ?? savedProviders[provider.provider_key];
      const baseUrl =
        provider.provider_type === "local_endpoint"
          ? providerValues.baseUrl.trim() || provider.default_base_url || ""
          : "";
      return JSON.stringify({
        provider_key: provider.provider_key,
        base_url: baseUrl,
        model: model || "",
        make_default: provider.provider_key === defaultProvider,
        edit_revision: providerEditRevisions[provider.provider_key] ?? 0,
        secret_present: Boolean(
          providerValues.apiKey.trim() ||
            hasBackendConfirmedSavedCredential(saved),
        ),
      });
    },
    [defaultProvider, providerEditRevisions, savedProviders],
  );
  const defaultValidation = defaultProviderConfig
    ? validationState[defaultProviderConfig.provider_key]
    : null;
  const defaultValidationFingerprint = defaultProviderConfig
    ? providerFingerprint(
        defaultProviderConfig,
        currentValues,
        selectedDefaultModel,
      )
    : "";
  const defaultValidationIsCurrent = Boolean(
    defaultValidation &&
      defaultValidation.fingerprint === defaultValidationFingerprint,
  );
  const defaultSaveIsCurrent = Boolean(
    savedPayloadFingerprints[defaultProvider] === defaultValidationFingerprint,
  );
  const canContinue = Boolean(
    defaultProvider &&
      selectedDefaultModel &&
      savedProviders[defaultProvider] &&
      savedDefaultProvider === defaultProvider &&
      defaultSaveIsCurrent &&
      defaultValidationIsCurrent &&
      validationCanGate(defaultValidation?.response),
  );
  const initialProviderKey = initialSelection?.provider ?? "";
  const seededInitialSavedProvider = React.useMemo(
    () => (initialSelection ? initialSavedProvider(initialSelection) : null),
    [
      initialSelection?.credential_configured,
      initialSelection?.model,
      initialSelection?.provider,
    ],
  );
  const initialProviderSavedResponse = initialProviderKey
    ? savedProviders[initialProviderKey]
    : undefined;
  const initialProviderSavedFingerprint = initialProviderKey
    ? savedPayloadFingerprints[initialProviderKey]
    : undefined;

  React.useEffect(() => {
    if (!initialSelection?.provider) return;
    setSelectedProviders((current) => {
      if (current.has(initialSelection.provider)) return current;
      return new Set([...current, initialSelection.provider]);
    });
    setDefaultProvider((current) => current || initialSelection.provider);
    setValues((current) => {
      const nextValues = {
        ...emptyValues,
        ...(current[initialSelection.provider] ?? {}),
        model:
          current[initialSelection.provider]?.model || initialSelection.model,
      };
      if (
        providerValuesEqual(current[initialSelection.provider], nextValues)
      ) {
        return current;
      }
      return {
        ...current,
        [initialSelection.provider]: nextValues,
      };
    });
  }, [initialSelection?.model, initialSelection?.provider]);

  React.useEffect(() => {
    if (!initialSelection?.provider || !seededInitialSavedProvider) return;
    if (initialProviderSavedResponse) return;
    updateSavedProviders((current) => {
      if (current[initialSelection.provider]) return current;
      return {
        ...current,
        [initialSelection.provider]: seededInitialSavedProvider,
      };
    });
  }, [
    initialProviderSavedResponse,
    initialSelection?.provider,
    seededInitialSavedProvider,
    updateSavedProviders,
  ]);

  React.useEffect(() => {
    if (!initialSelection?.provider || !seededInitialSavedProvider) return;
    if (initialProviderSavedFingerprint) return;
    if (defaultProvider !== initialSelection.provider) return;

    const provider = orderedProviders.find(
      (entry) => entry.provider_key === initialSelection.provider,
    );
    if (!provider) return;

    const currentProviderValues =
      values[initialSelection.provider] ?? emptyValues;
    const savedResponse =
      initialProviderSavedResponse ?? seededInitialSavedProvider;
    const savedModel = savedResponse.model || initialSelection.model;
    const seededStateStillMatchesCurrentPayload =
      currentProviderValues.model === savedModel &&
      currentProviderValues.apiKey.trim() === "" &&
      currentProviderValues.baseUrl.trim() === "" &&
      (providerEditRevisions[initialSelection.provider] ?? 0) === 0 &&
      savedResponse.make_default === true;
    if (!seededStateStillMatchesCurrentPayload) return;

    const fingerprint = providerFingerprint(
      provider,
      currentProviderValues,
      currentProviderValues.model,
      savedResponse,
    );
    updateSavedPayloadFingerprints((current) => {
      if (current[initialSelection.provider]) return current;
      return {
        ...current,
        [initialSelection.provider]: fingerprint,
      };
    });
  }, [
    defaultProvider,
    initialProviderSavedFingerprint,
    initialProviderSavedResponse,
    initialSelection?.model,
    initialSelection?.provider,
    orderedProviders,
    providerFingerprint,
    providerEditRevisions,
    seededInitialSavedProvider,
    updateSavedPayloadFingerprints,
    values,
  ]);

  const updateValues = (
    providerKey: string,
    patch: Partial<ProviderFormValues>,
    options: { invalidateValidation?: boolean } = {},
  ) => {
    setValues((current) => ({
      ...current,
      [providerKey]: {
        ...emptyValues,
        ...(current[providerKey] ?? {}),
        ...patch,
      },
    }));
    if (options.invalidateValidation) {
      updateProviderEditRevisions((current) => ({
        ...current,
        [providerKey]: (current[providerKey] ?? 0) + 1,
      }));
    }
  };

  const buildProviderPayload = (
    provider: SetupProviderCatalogEntry,
  ): SetupProviderSaveRequest => {
    const providerValues = values[provider.provider_key] ?? emptyValues;
    const apiKey = providerValues.apiKey.trim() || null;
    const baseUrl =
      provider.provider_type === "local_endpoint"
        ? providerValues.baseUrl.trim() || provider.default_base_url || null
        : null;
    const model =
      provider.provider_key === defaultProvider
        ? selectedDefaultModel
        : providerValues.model.trim() || null;

    return {
      provider_key: provider.provider_key,
      api_key: apiKey,
      base_url: baseUrl,
      model,
      make_default: provider.provider_key === defaultProvider,
    };
  };

  const validateProvider = async (provider: SetupProviderCatalogEntry) => {
    if (provider.provider_key === defaultProvider && !selectedDefaultModel) {
      setError("Default model is required before validation.");
      return;
    }
    setValidatingProvider(provider.provider_key);
    setError(null);
    try {
      const payload = buildProviderPayload(provider);
      const providerValues = values[provider.provider_key] ?? emptyValues;
      const saved = savedProviders[provider.provider_key];
      const response =
        provider.provider_type === "hosted_api_key" &&
        saved?.status === "saved" &&
        hasBackendConfirmedSavedCredential(saved) &&
        !payload.api_key
          ? savedHostedCredentialFallbackValidation(provider)
          : await onValidateProvider(payload);
      updateValidationState((current) => ({
        ...current,
        [provider.provider_key]: {
          fingerprint: providerFingerprint(
            provider,
            providerValues,
            payload.model || "",
          ),
          response,
        },
      }));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Provider could not be validated.",
      );
    } finally {
      setValidatingProvider(null);
    }
  };

  const toggleProvider = (provider: SetupProviderCatalogEntry) => {
    setError(null);
    setSelectedProviders((current) => {
      const next = new Set(current);
      if (next.has(provider.provider_key)) {
        next.delete(provider.provider_key);
      } else {
        next.add(provider.provider_key);
      }
      if (!defaultProvider || !next.has(defaultProvider)) {
        setDefaultProvider([...next][0] ?? "");
      }
      return next;
    });
  };

  const switchToHostedProvider = React.useCallback((fromProviderKey: string) => {
    const hostedProvider =
      orderedProviders.find((provider) => provider.provider_key === "openai") ??
      orderedProviders.find(
        (provider) =>
          provider.provider_type === "hosted_api_key" &&
          provider.recommended_for_first_chat,
      ) ??
      orderedProviders.find(
        (provider) => provider.provider_type === "hosted_api_key",
      );
    if (!hostedProvider) return;
    setError(null);
    setSelectedProviders((current) => {
      const next = new Set(current);
      if (fromProviderKey !== hostedProvider.provider_key) {
        next.delete(fromProviderKey);
      }
      next.add(hostedProvider.provider_key);
      return next;
    });
    setDefaultProvider(hostedProvider.provider_key);
  }, [orderedProviders]);

  const continueWithManualModel = React.useCallback(
    (providerKey: string) => {
      setError(null);
      setDefaultProvider(providerKey);
      setPendingProviderFocus({ providerKey, field: "model" });
      focusProviderField(providerKey, "model");
    },
    [focusProviderField],
  );

  const saveConfiguredProviders = async () => {
    if (!defaultProvider || !defaultProviderConfig || !selectedDefaultModel) {
      return;
    }
    setSavingProvider(true);
    setError(null);
    try {
      const selectedProviderEntries = orderedProviders.filter((provider) =>
        selectedProviders.has(provider.provider_key),
      );
      const responses: SetupProviderSaveResponse[] = [];
      const savedFingerprintEntries: Array<[string, string]> = [];
      for (const provider of selectedProviderEntries) {
        const payload = buildProviderPayload(provider);
        const providerValues = values[provider.provider_key] ?? emptyValues;
        if (
          provider.provider_type === "hosted_api_key" &&
          !payload.api_key &&
          !savedProviders[provider.provider_key]
        ) {
          throw new Error(
            `${provider.label} API key is required before saving.`,
          );
        }
        if (
          provider.provider_type === "local_endpoint" &&
          !payload.base_url &&
          !savedProviders[provider.provider_key]
        ) {
          throw new Error(
            `${provider.label} base URL is required before saving.`,
          );
        }
        const response = await onSaveProvider(payload);
        if (response.status === "failed") {
          throw new Error(
            response.message || `${provider.label} could not be saved.`,
          );
        }
        responses.push(response);
        savedFingerprintEntries.push([
          response.provider_key,
          providerFingerprint(
            provider,
            providerValues,
            payload.model || "",
            response,
          ),
        ]);
      }
      updateSavedProviders((current) => ({
        ...current,
        ...Object.fromEntries(
          responses.map((response) => [response.provider_key, response]),
        ),
      }));
      updateSavedPayloadFingerprints((current) => ({
        ...current,
        ...Object.fromEntries(savedFingerprintEntries),
      }));
      if (
        responses.some(
          (response) =>
            response.provider_key === defaultProvider &&
            response.status === "saved",
        )
      ) {
        updateSavedDefaultProvider(defaultProvider);
      }
      for (const response of responses) {
        updateValues(response.provider_key, { apiKey: "" });
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Provider could not be saved.",
      );
    } finally {
      setSavingProvider(false);
    }
  };

  if (orderedProviders.length === 0) {
    return (
      <section aria-labelledby="provider-setup-title" className="space-y-4">
        <div>
          <h2
            id="provider-setup-title"
            className="text-lg font-semibold text-text"
          >
            Chat provider
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            No setup provider catalog is available yet.
          </p>
        </div>
        <div className="flex flex-wrap justify-between gap-2">
          {onBack ? (
            <button
              type="button"
              onClick={onBack}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
            >
              Back
            </button>
          ) : null}
        </div>
      </section>
    );
  }

  return (
    <section aria-labelledby="provider-setup-title" className="space-y-5">
      <div>
        <h2
          id="provider-setup-title"
          className="text-lg font-semibold text-text"
        >
          Chat provider
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Add hosted keys or local OpenAI-compatible endpoints. Choose one
          default for the first chat.
        </p>
      </div>

      {error ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {error}
        </div>
      ) : null}

      <div className="grid gap-3 md:grid-cols-2">
        {orderedProviders.map((provider) => {
          const selected = selectedProviders.has(provider.provider_key);
          const providerValues = values[provider.provider_key] ?? emptyValues;
          const saved = savedProviders[provider.provider_key];
          const payloadModel =
            provider.provider_key === defaultProvider
              ? selectedDefaultModel
              : providerValues.model.trim() || null;
          const currentFingerprint = providerFingerprint(
            provider,
            providerValues,
            payloadModel,
          );
          const validation = validationState[provider.provider_key];
          const validationIsCurrent =
            validation?.fingerprint === currentFingerprint;
          const validationResponse = validationIsCurrent
            ? validation.response
            : null;
          const discoveredModels = validationResponse?.models ?? [];
          const localModelDiscoveryFallback = Boolean(
            provider.provider_type === "local_endpoint" &&
              !saved &&
              validationResponse?.status === "accepted" &&
              validationResponse.failure_category ===
                FAILURE_MODEL_DISCOVERY_UNAVAILABLE,
          );
          const localValidationFailed = Boolean(
            provider.provider_type === "local_endpoint" &&
              validationResponse?.status === "failed",
          );
          const showLocalRecoveryActions =
            localModelDiscoveryFallback || localValidationFailed;
          return (
            <div
              key={provider.provider_key}
              className="rounded-md border border-border bg-surface px-4 py-4"
            >
              <div className="flex items-start justify-between gap-3">
                <label className="flex items-start gap-3 text-sm font-semibold text-text">
                  <input
                    type="checkbox"
                    aria-label={`Select ${provider.label}`}
                    checked={selected}
                    onChange={() => toggleProvider(provider)}
                    className="mt-1"
                  />
                  <span>
                    {provider.label}
                    {provider.recommended_for_first_chat ? (
                      <span className="ml-2 rounded-sm bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary">
                        Recommended
                      </span>
                    ) : null}
                  </span>
                </label>
                {provider.provider_type === "hosted_api_key" ? (
                  <KeyRound
                    className="size-4 text-text-muted"
                    aria-hidden="true"
                  />
                ) : (
                  <Server
                    className="size-4 text-text-muted"
                    aria-hidden="true"
                  />
                )}
              </div>

              {selected ? (
                <div className="mt-4 space-y-3">
                  <label className="block text-sm font-medium text-text">
                    <span>{provider.label} API key</span>
                    <input
                      type="password"
                      value={providerValues.apiKey}
                      onChange={(event) =>
                        updateValues(
                          provider.provider_key,
                          {
                            apiKey: event.currentTarget.value,
                          },
                          { invalidateValidation: true },
                        )
                      }
                      className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                      placeholder={
                        provider.provider_type === "hosted_api_key"
                          ? "Paste API key"
                          : "Optional token"
                      }
                    />
                  </label>

                  {provider.provider_type === "local_endpoint" ? (
                    <div className="block text-sm font-medium text-text">
                      <label htmlFor={`provider-${provider.provider_key}-base-url`}>
                        {provider.label} base URL
                      </label>
                      <input
                        id={`provider-${provider.provider_key}-base-url`}
                        ref={registerProviderField(
                          provider.provider_key,
                          "base-url",
                        )}
                        aria-describedby={`provider-${provider.provider_key}-base-url-help`}
                        value={
                          providerValues.baseUrl ||
                          provider.default_base_url ||
                          ""
                        }
                        onChange={(event) =>
                          updateValues(
                            provider.provider_key,
                            {
                              baseUrl: event.currentTarget.value,
                            },
                            { invalidateValidation: true },
                          )
                        }
                        className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                        placeholder="http://127.0.0.1:11434/v1"
                      />
                      <p
                        id={`provider-${provider.provider_key}-base-url-help`}
                        className="mt-1 text-xs font-normal text-text-muted"
                      >
                        OpenAI-compatible base URL examples:
                        http://127.0.0.1:11434/v1 for Ollama or
                        http://127.0.0.1:8080/v1 for llama.cpp.
                      </p>
                    </div>
                  ) : null}

                  {defaultProvider === provider.provider_key ? (
                    <label className="block text-sm font-medium text-text">
                      <span>Default model</span>
                      <input
                        id={`provider-${provider.provider_key}-model`}
                        ref={registerProviderField(
                          provider.provider_key,
                          "model",
                        )}
                        value={providerValues.model}
                        onChange={(event) =>
                          updateValues(
                            provider.provider_key,
                            {
                              model: event.currentTarget.value,
                            },
                            { invalidateValidation: true },
                          )
                        }
                        className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                        placeholder="Model name"
                      />
                    </label>
                  ) : null}

                  <label className="flex items-center gap-2 text-sm text-text">
                    <input
                      type="radio"
                      name="first-run-default-provider"
                      checked={defaultProvider === provider.provider_key}
                      onChange={() => setDefaultProvider(provider.provider_key)}
                    />
                    Use as first chat default
                  </label>

                  <div className="space-y-2">
                    <button
                      type="button"
                      onClick={() => void validateProvider(provider)}
                      disabled={validatingProvider === provider.provider_key}
                      className="rounded-md border border-border bg-bg px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {validatingProvider === provider.provider_key
                        ? `Validating ${provider.label}...`
                        : `Validate ${provider.label}`}
                    </button>
                    {validationResponse ? (
                      <p
                        className={`text-sm ${
                          validationResponse.status === "failed"
                            ? "text-danger"
                            : "text-success"
                        }`}
                      >
                        {validationStatusCopy(validationResponse)}
                      </p>
                    ) : validation ? (
                      <p className="text-sm text-warning">
                        Provider validation changed. Validate again.
                      </p>
                    ) : (
                      <p className="text-sm text-text-muted">
                        Validation has not run for this configuration.
                      </p>
                    )}
                    {showLocalRecoveryActions ? (
                      <div className="rounded-md border border-border bg-bg px-3 py-3 text-sm text-text">
                        <p className="text-text-muted">
                          {localModelDiscoveryFallback
                            ? "Save this provider with the typed model. First chat will verify the model."
                            : "Check the endpoint details, retry validation, or switch providers to keep setup moving."}
                        </p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => void validateProvider(provider)}
                            disabled={
                              validatingProvider === provider.provider_key
                            }
                            className="rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Retry
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              focusProviderField(
                                provider.provider_key,
                                "base-url",
                              )
                            }
                            disabled={
                              validatingProvider === provider.provider_key
                            }
                            className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
                          >
                            Edit endpoint
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              switchToHostedProvider(provider.provider_key)
                            }
                            disabled={
                              validatingProvider === provider.provider_key
                            }
                            className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
                          >
                            Switch provider
                          </button>
                          {localModelDiscoveryFallback ? (
                            <button
                              type="button"
                              onClick={() =>
                                continueWithManualModel(provider.provider_key)
                              }
                              disabled={
                                validatingProvider === provider.provider_key
                              }
                              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
                            >
                              Continue with manual model
                            </button>
                          ) : null}
                        </div>
                      </div>
                    ) : null}
                    {discoveredModels.length > 0 ? (
                      <div className="space-y-1">
                        <p className="text-xs font-medium uppercase tracking-normal text-text-muted">
                          Discovered models
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {discoveredModels.map((model) => (
                            <button
                              key={model}
                              type="button"
                              onClick={() =>
                                updateValues(
                                  provider.provider_key,
                                  { model },
                                  { invalidateValidation: true },
                                )
                              }
                              className="rounded-sm border border-border bg-bg px-2 py-1 text-xs text-text hover:bg-surface2"
                            >
                              {model}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>

                  {saved ? (
                    <p className="flex items-center gap-2 text-sm text-success">
                      <CircleCheck className="size-4" aria-hidden="true" />
                      Saved
                      {saved.masked_api_key
                        ? ` as ${saved.masked_api_key}`
                        : ""}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap justify-between gap-2">
        {onBack ? (
          <button
            type="button"
            onClick={onBack}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
          >
            Back
          </button>
        ) : (
          <span />
        )}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={saveConfiguredProviders}
            disabled={
              savingProvider || !defaultProvider || !selectedDefaultModel
            }
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {savingProvider ? "Saving..." : "Save providers"}
          </button>
          <button
            type="button"
            disabled={!canContinue}
            onClick={() =>
              onContinue({
                provider: defaultProvider,
                model: selectedDefaultModel,
                credential_configured: hasBackendConfirmedSavedCredential(
                  savedProviders[defaultProvider],
                ),
              })
            }
            className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            Continue
          </button>
        </div>
      </div>
    </section>
  );
}

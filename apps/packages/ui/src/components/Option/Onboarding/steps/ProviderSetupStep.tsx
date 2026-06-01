import React from "react";
import { CircleCheck, KeyRound, Server } from "lucide-react";

import type {
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse,
} from "@/types/setup-onboarding";

export type ProviderSelection = {
  provider: string;
  model: string;
};

type ProviderSetupStepProps = {
  providers: SetupProviderCatalogEntry[];
  initialSelection?: ProviderSelection | null;
  onSaveProvider: (
    payload: SetupProviderSaveRequest,
  ) => Promise<SetupProviderSaveResponse>;
  onContinue: (selection: ProviderSelection) => void;
  onBack?: () => void;
};

type ProviderFormValues = {
  apiKey: string;
  baseUrl: string;
  model: string;
};

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

export function ProviderSetupStep({
  providers,
  initialSelection = null,
  onSaveProvider,
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
  const [savedProviders, setSavedProviders] = React.useState<
    Record<string, SetupProviderSaveResponse>
  >(() =>
    initialSelection?.provider
      ? {
          [initialSelection.provider]: {
            provider_key: initialSelection.provider,
            status: "saved",
            model: initialSelection.model,
            make_default: true,
          },
        }
      : {},
  );
  const [savingProvider, setSavingProvider] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const currentValues = values[defaultProvider] ?? emptyValues;
  const defaultProviderConfig = orderedProviders.find(
    (provider) => provider.provider_key === defaultProvider,
  );
  const selectedDefaultModel =
    currentValues.model.trim() || defaultProviderConfig?.model_field || "";
  const canContinue = Boolean(
    defaultProvider && selectedDefaultModel && savedProviders[defaultProvider],
  );

  React.useEffect(() => {
    if (!initialSelection?.provider) return;
    setSelectedProviders(
      (current) => new Set([...current, initialSelection.provider]),
    );
    setDefaultProvider((current) => current || initialSelection.provider);
    setValues((current) => ({
      ...current,
      [initialSelection.provider]: {
        ...emptyValues,
        ...(current[initialSelection.provider] ?? {}),
        model:
          current[initialSelection.provider]?.model || initialSelection.model,
      },
    }));
    setSavedProviders((current) => ({
      ...current,
      [initialSelection.provider]: current[initialSelection.provider] ?? {
        provider_key: initialSelection.provider,
        status: "saved",
        model: initialSelection.model,
        make_default: true,
      },
    }));
  }, [initialSelection?.model, initialSelection?.provider]);

  const updateValues = (
    providerKey: string,
    patch: Partial<ProviderFormValues>,
  ) => {
    setValues((current) => ({
      ...current,
      [providerKey]: {
        ...emptyValues,
        ...(current[providerKey] ?? {}),
        ...patch,
      },
    }));
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

  const saveConfiguredProviders = async () => {
    if (!defaultProvider || !defaultProviderConfig) return;
    setSavingProvider(true);
    setError(null);
    try {
      const selectedProviderEntries = orderedProviders.filter((provider) =>
        selectedProviders.has(provider.provider_key),
      );
      const responses: SetupProviderSaveResponse[] = [];
      for (const provider of selectedProviderEntries) {
        const providerValues = values[provider.provider_key] ?? emptyValues;
        const apiKey = providerValues.apiKey.trim() || null;
        const baseUrl =
          provider.provider_type === "local_endpoint"
            ? providerValues.baseUrl.trim() || provider.default_base_url || null
            : null;
        if (
          provider.provider_type === "hosted_api_key" &&
          !apiKey &&
          !savedProviders[provider.provider_key]
        ) {
          throw new Error(
            `${provider.label} API key is required before saving.`,
          );
        }
        if (
          provider.provider_type === "local_endpoint" &&
          !baseUrl &&
          !savedProviders[provider.provider_key]
        ) {
          throw new Error(
            `${provider.label} base URL is required before saving.`,
          );
        }
        const model =
          provider.provider_key === defaultProvider
            ? selectedDefaultModel
            : providerValues.model.trim() || null;
        const response = await onSaveProvider({
          provider_key: provider.provider_key,
          api_key: apiKey,
          base_url: baseUrl,
          model,
          make_default: provider.provider_key === defaultProvider,
        });
        if (response.status === "failed") {
          throw new Error(
            response.message || `${provider.label} could not be saved.`,
          );
        }
        responses.push(response);
      }
      setSavedProviders((current) => ({
        ...current,
        ...Object.fromEntries(
          responses.map((response) => [response.provider_key, response]),
        ),
      }));
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
                        updateValues(provider.provider_key, {
                          apiKey: event.currentTarget.value,
                        })
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
                    <label className="block text-sm font-medium text-text">
                      <span>{provider.label} base URL</span>
                      <input
                        value={
                          providerValues.baseUrl ||
                          provider.default_base_url ||
                          ""
                        }
                        onChange={(event) =>
                          updateValues(provider.provider_key, {
                            baseUrl: event.currentTarget.value,
                          })
                        }
                        className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                        placeholder="http://127.0.0.1:11434/v1"
                      />
                    </label>
                  ) : null}

                  {defaultProvider === provider.provider_key ? (
                    <label className="block text-sm font-medium text-text">
                      <span>Default model</span>
                      <input
                        value={providerValues.model}
                        onChange={(event) =>
                          updateValues(provider.provider_key, {
                            model: event.currentTarget.value,
                          })
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

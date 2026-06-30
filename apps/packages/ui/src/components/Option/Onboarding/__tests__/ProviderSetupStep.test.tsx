// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ProviderSetupStep } from "../steps/ProviderSetupStep";
import type {
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
} from "@/types/setup-onboarding";
import type { ProviderSelection } from "../steps/ProviderSetupStep";

const providers: SetupProviderCatalogEntry[] = [
  {
    provider_key: "openai",
    label: "OpenAI",
    provider_type: "hosted_api_key",
    supports_preflight: true,
    recommended_for_first_chat: true,
  },
  {
    provider_key: "ollama",
    label: "Ollama",
    provider_type: "local_endpoint",
    default_base_url: "http://127.0.0.1:11434/v1",
    supports_preflight: true,
    recommended_for_first_chat: false,
  },
];

const fillDefaultOpenAI = () => {
  fireEvent.click(screen.getByLabelText(/openai/i));
  fireEvent.change(screen.getByLabelText(/openai api key/i), {
    target: { value: "test-api-key-value" },
  });
  fireEvent.change(screen.getByLabelText(/default model/i), {
    target: { value: "gpt-4.1-mini" },
  });
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, reject, resolve };
};

const renderProviderStep = ({
  onSaveProvider = vi.fn().mockImplementation(async (payload) => ({
    provider_key: payload.provider_key,
    status: "saved",
    masked_api_key: payload.provider_key === "openai" ? "saved-key-present" : null,
    model: payload.model,
    make_default: payload.make_default,
  })),
  onValidateProvider = vi.fn().mockImplementation(async (payload) => ({
    provider_key: payload.provider_key,
    status: "accepted",
    message: "Provider credentials passed local syntax checks.",
    models: [],
    validation_level: "local_syntax",
    can_gate_first_chat: true,
  })),
  onContinue = vi.fn(),
  providerCatalog = providers,
  initialSelection = null,
}: {
  onSaveProvider?: (payload: SetupProviderSaveRequest) => Promise<any>;
  onValidateProvider?: (payload: SetupProviderSaveRequest) => Promise<any>;
  onContinue?: (selection: {
    provider: string;
    model: string;
    credential_configured?: boolean;
  }) => void;
  providerCatalog?: SetupProviderCatalogEntry[];
  initialSelection?: ProviderSelection | null;
} = {}) => {
  render(
    <ProviderSetupStep
      providers={providerCatalog}
      initialSelection={initialSelection}
      onSaveProvider={onSaveProvider}
      onValidateProvider={onValidateProvider}
      onContinue={onContinue}
    />,
  );

  return { onSaveProvider, onValidateProvider, onContinue };
};

describe("ProviderSetupStep", () => {
  it("lets users select multiple providers and saves every configured provider", async () => {
    const saveProvider = vi.fn().mockImplementation(async (payload) => ({
      provider_key: payload.provider_key,
      status: "saved",
      masked_api_key:
        payload.provider_key === "openai" ? "saved-key-present" : undefined,
    }));
    const onContinue = vi.fn();

    renderProviderStep({
      onSaveProvider: saveProvider,
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "openai",
        status: "accepted",
        models: [],
        validation_level: "local_syntax",
        can_gate_first_chat: true,
      }),
      onContinue,
    });

    fireEvent.click(screen.getByLabelText(/openai/i));
    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));

    await waitFor(() =>
      expect(saveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: "test-api-key-value",
          model: "gpt-4.1-mini",
          make_default: true,
        }),
      ),
    );
    expect(saveProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        provider_key: "ollama",
        base_url: "http://127.0.0.1:11434/v1",
        make_default: false,
      }),
    );
    expect(screen.getByText(/saved as saved-key-present/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onContinue).toHaveBeenCalledWith({
      provider: "openai",
      model: "gpt-4.1-mini",
      credential_configured: true,
    });
  });

  it("preselects the saved provider and model when resuming setup", () => {
    render(
      <ProviderSetupStep
        providers={[
          {
            provider_key: "openai",
            label: "OpenAI",
            provider_type: "hosted_api_key",
            supports_preflight: true,
            recommended_for_first_chat: true,
          },
        ]}
        initialSelection={{ provider: "openai", model: "gpt-4.1-mini" }}
        onSaveProvider={vi.fn()}
        onValidateProvider={vi.fn()}
        onContinue={vi.fn()}
      />,
    );

    expect(screen.getByLabelText(/select openai/i)).toBeChecked();
    expect(screen.getByLabelText(/default model/i)).toHaveValue("gpt-4.1-mini");
  });

  it("keeps continue disabled until the default provider is validated and saved", async () => {
    const { onContinue } = renderProviderStep();

    fillDefaultOpenAI();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);

    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onContinue).toHaveBeenCalledWith({
      provider: "openai",
      model: "gpt-4.1-mini",
      credential_configured: true,
    });
  });

  it("requires re-saving a reselected default provider after another default was saved", async () => {
    renderProviderStep({
      onSaveProvider: vi.fn().mockImplementation(async (payload) => ({
        provider_key: payload.provider_key,
        status: "saved",
        masked_api_key:
          payload.provider_key === "openai" ? "saved-key-present" : null,
        base_url: payload.base_url,
        model: payload.model,
        make_default: payload.make_default,
      })),
      onValidateProvider: vi.fn().mockImplementation(async (payload) => ({
        provider_key: payload.provider_key,
        status: payload.provider_key === "ollama" ? "ready" : "accepted",
        message:
          payload.provider_key === "ollama"
            ? "Provider validation is ready."
            : "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level:
          payload.provider_key === "ollama"
            ? "live_non_generative"
            : "local_syntax",
        can_gate_first_chat: true,
      })),
    });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/select openai/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "llama3.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText(/provider validation is ready/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/^Saved$/i);

    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.click(screen.getByLabelText(/openai/i));
    fireEvent.click(screen.getAllByLabelText(/use as first chat default/i)[0]);

    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled(),
    );
  });

  it("keeps validation valid after save clears the raw key and shows the masked key", async () => {
    renderProviderStep();

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));

    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByLabelText(/openai api key/i)).toHaveValue("");
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
  });

  it("allows accepted validation to satisfy the gate with first-chat-verifies copy", async () => {
    renderProviderStep({
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "openai",
        status: "accepted",
        message: "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level: "local_syntax",
        can_gate_first_chat: true,
      }),
    });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    expect(
      await screen.findByText(/format accepted; first chat verifies/i),
    ).toBeInTheDocument();
  });

  it("keeps continue disabled and shows categorized failure messages", async () => {
    renderProviderStep({
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "openai",
        status: "failed",
        failure_category: "auth_failed",
        message: "Local provider rejected the supplied credentials.",
        models: [],
        can_gate_first_chat: false,
      }),
    });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    expect(await screen.findByText(/auth_failed/i)).toBeInTheDocument();
    expect(
      screen.getByText(/rejected the supplied credentials/i),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("does not gate failed validation even when can_gate_first_chat is true", async () => {
    renderProviderStep({
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "openai",
        status: "failed",
        failure_category: "auth_failed",
        message: "Provider validation failed.",
        models: [],
        can_gate_first_chat: true,
      }),
    });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/auth_failed/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);

    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("saves non-default selected providers without validating them", async () => {
    const onValidateProvider = vi.fn().mockResolvedValue({
      provider_key: "openai",
      status: "accepted",
      models: [],
      validation_level: "local_syntax",
      can_gate_first_chat: true,
    });
    const onSaveProvider = vi.fn().mockImplementation(async (payload) => ({
      provider_key: payload.provider_key,
      status: "saved",
      masked_api_key: payload.provider_key === "openai" ? "saved-key-present" : null,
      model: payload.model,
      make_default: payload.make_default,
    }));
    renderProviderStep({ onSaveProvider, onValidateProvider });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));

    await waitFor(() => expect(onSaveProvider).toHaveBeenCalledTimes(2));
    expect(onValidateProvider).toHaveBeenCalledTimes(1);
    expect(onValidateProvider).toHaveBeenCalledWith(
      expect.objectContaining({ provider_key: "openai" }),
    );
  });

  it("preserves concurrent validation results for multiple providers", async () => {
    const openaiValidation = deferred<any>();
    const ollamaValidation = deferred<any>();
    const onValidateProvider = vi.fn((payload: SetupProviderSaveRequest) => {
      if (payload.provider_key === "openai") {
        return openaiValidation.promise;
      }
      if (payload.provider_key === "ollama") {
        return ollamaValidation.promise;
      }
      throw new Error(`Unexpected provider ${payload.provider_key}`);
    });
    renderProviderStep({ onValidateProvider });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await waitFor(() =>
      expect(screen.getByText(/validating openai/i)).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await waitFor(() => expect(onValidateProvider).toHaveBeenCalledTimes(2));

    ollamaValidation.resolve({
      provider_key: "ollama",
      status: "ready",
      message: "Ollama validation is ready.",
      models: [],
      validation_level: "live_non_generative",
      can_gate_first_chat: true,
    });
    await screen.findByText(/ollama validation is ready/i);

    openaiValidation.resolve({
      provider_key: "openai",
      status: "accepted",
      message: "OpenAI format accepted; first chat verifies this provider.",
      models: [],
      validation_level: "local_syntax",
      can_gate_first_chat: true,
    });

    expect(
      await screen.findByText(/openai format accepted/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/ollama validation is ready/i)).toBeInTheDocument();
  });

  it("shows discovered models while keeping manual model entry available", async () => {
    renderProviderStep({
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "ollama",
        status: "ready",
        message: "Local provider models were discovered.",
        models: ["llama3.1", "qwen2.5"],
        validation_level: "live_non_generative",
        can_gate_first_chat: true,
      }),
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/use as first chat default/i));
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "manual-local-model" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));

    expect(await screen.findByText("llama3.1")).toBeInTheDocument();
    expect(screen.getByText("qwen2.5")).toBeInTheDocument();
    expect(screen.getByLabelText(/default model/i)).toHaveValue(
      "manual-local-model",
    );
  });

  it("uses discovered model selections as real model values", async () => {
    const onValidateProvider = vi
      .fn()
      .mockResolvedValueOnce({
        provider_key: "ollama",
        status: "ready",
        message: "Local provider models were discovered.",
        models: ["llama3.1", "qwen2.5"],
        validation_level: "live_non_generative",
        can_gate_first_chat: true,
      })
      .mockResolvedValueOnce({
        provider_key: "ollama",
        status: "ready",
        message: "Provider validation is ready.",
        models: ["llama3.1", "qwen2.5"],
        validation_level: "live_non_generative",
        can_gate_first_chat: true,
      });
    const onSaveProvider = vi.fn().mockImplementation(async (payload) => ({
      provider_key: payload.provider_key,
      status: "saved",
      base_url: payload.base_url,
      model: payload.model,
      make_default: payload.make_default,
    }));
    renderProviderStep({
      onSaveProvider,
      onValidateProvider,
      providerCatalog: [
        {
          ...providers[1],
          model_field: "ollama_model",
        },
      ],
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "temporary-discovery-model" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText("llama3.1");

    fireEvent.click(screen.getByRole("button", { name: "qwen2.5" }));
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText(/provider validation is ready/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));

    await waitFor(() =>
      expect(onSaveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "ollama",
          model: "qwen2.5",
        }),
      ),
    );
    expect(onValidateProvider).toHaveBeenLastCalledWith(
      expect.objectContaining({
        provider_key: "ollama",
        model: "qwen2.5",
      }),
    );
  });

  it("does not use hosted catalog model field names as default model values", async () => {
    const onSaveProvider = vi.fn();
    const onValidateProvider = vi.fn();
    renderProviderStep({
      onSaveProvider,
      onValidateProvider,
      providerCatalog: [
        {
          ...providers[0],
          model_field: "openai_model",
        },
      ],
    });

    fireEvent.click(screen.getByLabelText(/openai/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    expect(onValidateProvider).not.toHaveBeenCalled();
    expect(onSaveProvider).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /save provider/i }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    expect(screen.queryByText("openai_model")).not.toBeInTheDocument();
  });

  it("does not use local catalog model field names as default model values", async () => {
    const onSaveProvider = vi.fn();
    const onValidateProvider = vi.fn();
    renderProviderStep({
      onSaveProvider,
      onValidateProvider,
      providerCatalog: [
        {
          ...providers[1],
          model_field: "ollama_model",
        },
      ],
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));

    expect(onValidateProvider).not.toHaveBeenCalled();
    expect(onSaveProvider).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /save provider/i }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    expect(screen.queryByText("ollama_model")).not.toBeInTheDocument();
  });

  it("invalidates default provider validation when the model changes", async () => {
    renderProviderStep();

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1" },
    });

    expect(screen.getByText(/validation changed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("invalidates default provider validation when the API key changes after save", async () => {
    renderProviderStep();

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByLabelText(/openai api key/i)).toHaveValue("");
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "replacement-api-key-value" },
    });

    expect(screen.getByText(/validation changed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("keeps resumed hosted model edits from continuing until the edited payload is saved", async () => {
    const onSaveProvider = vi.fn().mockImplementation(async (payload) => ({
      provider_key: payload.provider_key,
      status: "saved",
      credential_configured: true,
      model: payload.model,
      make_default: payload.make_default,
    }));
    const onContinue = vi.fn();
    renderProviderStep({
      initialSelection: {
        provider: "openai",
        model: "gpt-4.1-mini",
        credential_configured: true,
      },
      onSaveProvider,
      onContinue,
    });

    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/saved credentials are present/i);

    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    expect(onContinue).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() =>
      expect(onSaveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: null,
          model: "gpt-4.1",
          make_default: true,
        }),
      ),
    );

    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onContinue).toHaveBeenCalledWith({
      provider: "openai",
      model: "gpt-4.1",
      credential_configured: true,
    });
  });

  it("requires saving an API key edit after revalidation before continuing", async () => {
    const onContinue = vi.fn();
    renderProviderStep({ onContinue });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);

    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "replacement-api-key-value" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);

    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onContinue).toHaveBeenCalledWith({
      provider: "openai",
      model: "gpt-4.1-mini",
      credential_configured: true,
    });
  });

  it("invalidates local default provider validation when the base URL changes", async () => {
    renderProviderStep({
      onSaveProvider: vi.fn().mockImplementation(async (payload) => ({
        provider_key: payload.provider_key,
        status: "saved",
        base_url: payload.base_url,
        model: payload.model,
        make_default: payload.make_default,
      })),
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "ollama",
        status: "ready",
        message: "Provider validation is ready.",
        models: ["llama3.1"],
        validation_level: "live_non_generative",
        can_gate_first_chat: true,
      }),
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/use as first chat default/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "llama3.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText(/provider validation is ready/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved/i);
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();

    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11435/v1" },
    });

    expect(screen.getByText(/validation changed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("guides manual model entry when local model discovery is unavailable", async () => {
    const onContinue = vi.fn();
    const onSaveProvider = vi.fn().mockImplementation(async (payload) => ({
      provider_key: payload.provider_key,
      status: "saved",
      base_url: payload.base_url,
      model: payload.model,
      make_default: payload.make_default,
    }));
    renderProviderStep({
      onContinue,
      onSaveProvider,
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "ollama",
        status: "accepted",
        failure_category: "model_discovery_unavailable",
        message:
          "Model discovery is unavailable. Enter the model name manually; first chat will verify it.",
        models: [],
        validation_level: "live_endpoint_shape",
        can_gate_first_chat: true,
      }),
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/use as first chat default/i));
    expect(
      screen.getByText(/openai-compatible base url/i),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "manual-local-model" },
    });

    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));

    expect(
      await screen.findByText(/model discovery is unavailable/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/enter the model name manually/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /continue with manual model/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() =>
      expect(onSaveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "ollama",
          model: "manual-local-model",
          make_default: true,
        }),
      ),
    );
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));
    expect(onContinue).toHaveBeenCalledWith({
      provider: "ollama",
      model: "manual-local-model",
      credential_configured: false,
    });
  });

  it("shows local endpoint recovery actions when validation cannot reach the endpoint", async () => {
    const onValidateProvider = vi.fn().mockResolvedValue({
      provider_key: "ollama",
      status: "failed",
      failure_category: "local_provider_unreachable",
      message: "Local provider endpoint is unreachable.",
      models: [],
      can_gate_first_chat: false,
    });
    renderProviderStep({ onValidateProvider });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/use as first chat default/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:65535/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "llama3.2:3b" },
    });

    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));

    expect(
      await screen.findByText(/check that the local service is running/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^retry$/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /edit endpoint/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /switch provider/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /^retry$/i }));
    await waitFor(() => expect(onValidateProvider).toHaveBeenCalledTimes(2));
  });

  it("switches local endpoint recovery to the hosted provider", async () => {
    renderProviderStep({
      providerCatalog: [
        {
          provider_key: "anthropic",
          label: "Anthropic",
          provider_type: "hosted_api_key",
          supports_preflight: true,
          recommended_for_first_chat: true,
        },
        ...providers,
      ],
      onValidateProvider: vi.fn().mockResolvedValue({
        provider_key: "ollama",
        status: "failed",
        failure_category: "local_provider_unreachable",
        message: "Local provider endpoint is unreachable.",
        models: [],
        can_gate_first_chat: false,
      }),
    });

    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.click(screen.getByLabelText(/use as first chat default/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:65535/v1" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "llama3.2:3b" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText(/check that the local service is running/i);

    fireEvent.click(screen.getByRole("button", { name: /switch provider/i }));

    expect(screen.getByLabelText(/select openai/i)).toBeChecked();
    expect(screen.getByLabelText(/select anthropic/i)).not.toBeChecked();
    expect(screen.getByLabelText(/select ollama/i)).not.toBeChecked();
    expect(screen.getByLabelText(/default model/i)).toHaveValue("");
  });

  it("manual model fallback action promotes a non-default local provider", async () => {
    renderProviderStep({
      onValidateProvider: vi.fn().mockImplementation(async (payload) => ({
        provider_key: payload.provider_key,
        status: "accepted",
        failure_category:
          payload.provider_key === "ollama"
            ? "model_discovery_unavailable"
            : null,
        message:
          payload.provider_key === "ollama"
            ? "Model discovery is unavailable. Enter the model name manually; first chat will verify it."
            : "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level:
          payload.provider_key === "ollama"
            ? "live_endpoint_shape"
            : "local_syntax",
        can_gate_first_chat: true,
      })),
    });

    fillDefaultOpenAI();
    fireEvent.click(screen.getByLabelText(/ollama/i));
    fireEvent.change(screen.getByLabelText(/ollama base url/i), {
      target: { value: "http://127.0.0.1:11434/v1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate ollama/i }));
    await screen.findByText(/model discovery is unavailable/i);

    fireEvent.click(
      screen.getByRole("button", { name: /continue with manual model/i }),
    );

    expect(
      screen.getAllByLabelText(/use as first chat default/i)[1],
    ).toBeChecked();
    expect(screen.getByLabelText(/default model/i)).toHaveValue("");
    expect(screen.getByLabelText(/default model/i)).toHaveFocus();
  });
});

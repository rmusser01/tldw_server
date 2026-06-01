// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { FirstRunState } from "@/types/setup-onboarding";

const setupHookMocks = vi.hoisted(() => ({
  saveStep: vi.fn(),
  skip: vi.fn(),
  loadProviderCatalog: vi.fn(),
  loadAudioRecommendations: vi.fn(),
  saveProvider: vi.fn(),
  validateProvider: vi.fn(),
  saveIngestDefaults: vi.fn(),
  saveAudioDefaults: vi.fn(),
  saveOptionalAdvanced: vi.fn(),
  verifyFirstChat: vi.fn(),
  complete: vi.fn(),
  refresh: vi.fn(),
}));

const readinessHookMocks = vi.hoisted(() => ({
  refresh: vi.fn(),
}));

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

const initialStateForCompletedSteps = (
  completedSteps: string[],
): FirstRunState => ({
  status: "in_progress",
  completed_steps: completedSteps,
  skipped_steps: [],
  step_data: {
    providers: {
      acknowledged: true,
      default_provider: "openai",
      default_model: "gpt-4.1-mini",
      default_provider_credential_configured: true,
    },
  },
  acknowledged_steps: [],
  first_chat: { completed: false },
});

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false },
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://127.0.0.1:3000",
        api_origin: "http://127.0.0.1:8000",
        browser_access: "local",
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" },
    },
    providerCatalog: [
      {
        provider_key: "openai",
        label: "OpenAI",
        provider_type: "hosted_api_key",
        supports_preflight: true,
        recommended_for_first_chat: true,
      },
    ],
    audioRecommendations: [],
    loading: false,
    error: null,
    refresh: setupHookMocks.refresh,
    loadProviderCatalog: setupHookMocks.loadProviderCatalog,
    loadAudioRecommendations: setupHookMocks.loadAudioRecommendations,
    saveStep: setupHookMocks.saveStep,
    skip: setupHookMocks.skip,
    saveProvider: setupHookMocks.saveProvider,
    validateProvider: setupHookMocks.validateProvider,
    saveIngestDefaults: setupHookMocks.saveIngestDefaults,
    saveAudioDefaults: setupHookMocks.saveAudioDefaults,
    saveOptionalAdvanced: setupHookMocks.saveOptionalAdvanced,
    verifyFirstChat: setupHookMocks.verifyFirstChat,
    complete: setupHookMocks.complete,
  }),
}));

vi.mock("@/hooks/useSetupReadinessSummary", () => ({
  useSetupReadinessSummary: () => ({
    status: {
      readiness_status: "ready_with_warnings",
      lanes: [
        { lane_id: "chat", label: "Chat", status: "ready" },
        {
          lane_id: "embeddings_rag",
          label: "Embeddings/RAG",
          status: "not_configured",
        },
        { lane_id: "speech", label: "Speech", status: "skipped" },
      ],
      active_overlays: [],
      overlays: [],
    },
    loading: false,
    error: null,
    refresh: readinessHookMocks.refresh,
  }),
}));

describe("UnifiedSetupWizard", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    setupHookMocks.saveStep.mockReset();
    setupHookMocks.skip.mockReset();
    setupHookMocks.loadProviderCatalog.mockReset().mockResolvedValue([]);
    setupHookMocks.loadAudioRecommendations.mockReset().mockResolvedValue([]);
    setupHookMocks.saveProvider
      .mockReset()
      .mockImplementation(async (payload) => ({
        provider_key: payload.provider_key,
        status: "saved",
        masked_api_key: payload.api_key ? "saved-key-present" : null,
        credential_configured: true,
        model: payload.model,
        make_default: payload.make_default,
      }));
    setupHookMocks.validateProvider.mockReset().mockResolvedValue({
      provider_key: "openai",
      status: "accepted",
      message: "Format accepted; first chat verifies the provider.",
      models: [],
      validation_level: "local_syntax",
      can_gate_first_chat: true,
    });
    setupHookMocks.saveIngestDefaults.mockReset().mockResolvedValue({
      status: "saved",
      step: "ingest_defaults",
      requires_restart: false,
    });
    setupHookMocks.saveAudioDefaults.mockReset().mockResolvedValue({
      status: "saved",
      step: "audio_defaults",
      requires_restart: false,
    });
    setupHookMocks.saveOptionalAdvanced.mockReset().mockResolvedValue({
      status: "saved",
      step: "optional_advanced",
      requires_restart: false,
    });
    setupHookMocks.verifyFirstChat.mockReset().mockResolvedValue({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello.",
    });
    setupHookMocks.complete.mockReset().mockResolvedValue({
      success: true,
      message: "completed",
      requires_restart: false,
      install_plan_submitted: false,
    });
    setupHookMocks.refresh.mockReset().mockResolvedValue({
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false },
    });
    setupHookMocks.saveStep.mockResolvedValue({
      status: "in_progress",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false },
    });
    readinessHookMocks.refresh.mockReset().mockResolvedValue({
      readiness_status: "ready_with_warnings",
      lanes: [],
      active_overlays: [],
      overlays: [],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a focused first-run heading and setup path choices", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);

    expect(screen.getByTestId("setup-readiness-panel")).toBeInTheDocument();
    expect(screen.getByText("Embeddings/RAG")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /first-time setup/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /solo, docker/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /solo, local/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /multi-user/i }),
    ).toBeInTheDocument();
  });

  it("shows multi-user exit guidance instead of continuing solo wizard", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /multi-user/i }));

    expect(screen.getByText(/multi-user setup guide/i)).toBeInTheDocument();
  });

  it("requires privacy and security acknowledgement before provider setup", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }));

    expect(
      await screen.findByRole("heading", { name: /privacy and security/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByLabelText(/i understand/i));
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
  });

  it("does not advance past setup path if progress cannot be saved", async () => {
    setupHookMocks.saveStep.mockRejectedValueOnce(new Error("save failed"));
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /could not be saved/i,
      );
    });
    expect(
      screen.getByRole("heading", { name: /choose your setup path/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /privacy and security/i }),
    ).not.toBeInTheDocument();
  });

  it("does not advance to provider setup if privacy acknowledgement cannot be saved", async () => {
    setupHookMocks.saveStep
      .mockResolvedValueOnce({
        status: "in_progress",
        completed_steps: ["setup_path"],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: ["setup_path"],
        first_chat: { completed: false },
      })
      .mockRejectedValueOnce(new Error("save failed"));
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }));

    await screen.findByRole("heading", { name: /privacy and security/i });
    fireEvent.click(screen.getByLabelText(/i understand/i));
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /could not be saved/i,
      );
    });
    expect(
      screen.getByRole("heading", { name: /privacy and security/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /chat provider/i }),
    ).not.toBeInTheDocument();
  });

  it("reports skipped state to the parent route resolver", async () => {
    setupHookMocks.skip.mockResolvedValueOnce({
      status: "skipped",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false },
      skip_reason: "user_skip",
    });
    const onStateChange = vi.fn();
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard onStateChange={onStateChange} />);
    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }));

    await waitFor(() => {
      expect(onStateChange).toHaveBeenCalledWith(
        expect.objectContaining({ status: "skipped" }),
      );
    });
  });

  it("resumes at first chat when backend state includes the saved provider model", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: [
            "setup_path",
            "privacy_security",
            "providers",
            "ingest_defaults",
            "audio_defaults",
            "optional_advanced",
          ],
          skipped_steps: [],
          step_data: {
            providers: {
              acknowledged: true,
              default_provider: "openai",
              default_model: "gpt-4.1-mini",
            },
          },
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /first chat/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/openai \/ gpt-4.1-mini/i)).toBeInTheDocument();
  });

  it("resumes provider setup with the saved default provider selected", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: ["setup_path", "privacy_security"],
          skipped_steps: [],
          step_data: {
            providers: {
              acknowledged: true,
              default_provider: "openai",
              default_model: "gpt-4.1-mini",
              default_provider_credential_configured: true,
            },
          },
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /chat provider/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/select openai/i)).toBeChecked();
    expect(screen.getByLabelText(/default model/i)).toHaveValue("gpt-4.1-mini");
  });

  it("validates and saves the default provider before recording provider setup progress", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: ["setup_path", "privacy_security"],
          skipped_steps: [],
          step_data: {},
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    fireEvent.click(screen.getByLabelText(/select openai/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    await waitFor(() => {
      expect(setupHookMocks.validateProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: "test-api-key-value",
          model: "gpt-4.1-mini",
          make_default: true,
        }),
      );
    });
    await waitFor(() => {
      expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    await waitFor(() => {
      expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(2);
    });
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(setupHookMocks.saveStep).toHaveBeenCalledWith({
        step: "providers",
        data: {
          acknowledged: true,
          default_provider: "openai",
          default_model: "gpt-4.1-mini",
          default_provider_credential_configured: true,
        },
      });
    });
  });

  it("does not keep provider validation or save pending while readiness refresh is unresolved", async () => {
    const readinessRefreshes: Array<
      ReturnType<typeof createDeferred<Record<string, unknown>>>
    > = [];
    readinessHookMocks.refresh.mockImplementation(() => {
      const deferred = createDeferred<Record<string, unknown>>();
      readinessRefreshes.push(deferred);
      return deferred.promise;
    });
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: ["setup_path", "privacy_security"],
          skipped_steps: [],
          step_data: {},
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    fireEvent.click(screen.getByLabelText(/select openai/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /validate openai/i }))
        .toBeEnabled();
    });
    expect(screen.getByText(/first chat verifies/i)).toBeInTheDocument();
    expect(readinessRefreshes).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));

    await screen.findByText(/saved as saved-key-present/i);
    expect(screen.getByRole("button", { name: /save provider/i })).toBeEnabled();
    expect(readinessRefreshes).toHaveLength(2);

    for (const deferred of readinessRefreshes) {
      deferred.resolve({
        readiness_status: "ready_with_warnings",
        lanes: [],
        active_overlays: [],
        overlays: [],
      });
    }
  });

  it("refreshes readiness after ingest defaults are saved", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={initialStateForCompletedSteps([
          "setup_path",
          "privacy_security",
          "providers",
        ])}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(setupHookMocks.saveIngestDefaults).toHaveBeenCalledWith(
        expect.objectContaining({
          allow_local_file_ingest: false,
          chunking_profile: "balanced",
          metadata_mode: "automatic",
        }),
      );
    });
    await waitFor(() => {
      expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
    });
  });

  it("refreshes readiness after audio defaults are saved", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={initialStateForCompletedSteps([
          "setup_path",
          "privacy_security",
          "providers",
          "ingest_defaults",
        ])}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /audio, stt, and tts/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(setupHookMocks.saveAudioDefaults).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "skip",
          stt_provider: null,
          tts_provider: null,
          tts_voice: null,
        }),
      );
    });
    await waitFor(() => {
      expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
    });
  });

  it("refreshes readiness after optional advanced choices are saved", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={initialStateForCompletedSteps([
          "setup_path",
          "privacy_security",
          "providers",
          "ingest_defaults",
          "audio_defaults",
        ])}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /optional advanced setup/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(setupHookMocks.saveOptionalAdvanced).toHaveBeenCalledWith({
        rag: "defer",
        storage_paths: "defer",
      });
    });
    await waitFor(() => {
      expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
    });
  });

  it("does not refresh readiness after first chat completion", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={initialStateForCompletedSteps([
          "setup_path",
          "privacy_security",
          "providers",
          "ingest_defaults",
          "audio_defaults",
          "optional_advanced",
        ])}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /first chat/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await waitFor(() => {
      expect(setupHookMocks.complete).toHaveBeenCalledWith({
        acknowledged_steps: ["first_chat"],
      });
    });
    await waitFor(() => {
      expect(setupHookMocks.refresh).toHaveBeenCalledTimes(2);
    });
    expect(readinessHookMocks.refresh).not.toHaveBeenCalled();
  });

  it("refreshes readiness after a failed skip attempt", async () => {
    setupHookMocks.skip.mockRejectedValueOnce(new Error("skip failed"));
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /setup skip could not be saved/i,
      );
    });
    expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
  });

  it("wires the panel retry action to readiness refresh", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(<UnifiedSetupWizard />);
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(readinessHookMocks.refresh).toHaveBeenCalledTimes(1);
  });

  it("keeps validated provider gate available after back navigation", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: ["setup_path", "privacy_security"],
          skipped_steps: [],
          step_data: {},
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    fireEvent.click(screen.getByLabelText(/select openai/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    expect(
      await screen.findByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /back/i }));

    expect(
      await screen.findByRole("heading", { name: /chat provider/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/openai api key/i)).toHaveValue("");
    const validateCallsAfterBack =
      setupHookMocks.validateProvider.mock.calls.length;

    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    expect(
      await screen.findByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();
    expect(setupHookMocks.validateProvider).toHaveBeenCalledTimes(
      validateCallsAfterBack,
    );
  });

  it("revalidates saved hosted credentials after a model-only edit without re-pasting the key", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");
    setupHookMocks.validateProvider.mockImplementation(async (payload) => {
      if (!payload.api_key) {
        return {
          provider_key: "openai",
          status: "failed",
          failure_category: "provider_api_key_required",
          message: "Provider API key is required.",
          models: [],
          can_gate_first_chat: false,
        };
      }
      return {
        provider_key: "openai",
        status: "accepted",
        message: "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level: "local_syntax",
        can_gate_first_chat: true,
      };
    });

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: ["setup_path", "privacy_security"],
          skipped_steps: [],
          step_data: {},
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    fireEvent.click(screen.getByLabelText(/select openai/i));
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "test-api-key-value" },
    });
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    await screen.findByText(/first chat verifies/i);
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await screen.findByText(/saved as saved-key-present/i);
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(
      await screen.findByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /back/i }));
    expect(
      await screen.findByRole("heading", { name: /chat provider/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/openai api key/i)).toHaveValue("");
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1" },
    });
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    expect(
      await screen.findByText(/saved credentials are present/i),
    ).toBeInTheDocument();
    expect(setupHookMocks.validateProvider).toHaveBeenCalledTimes(1);
    expect(setupHookMocks.validateProvider).toHaveBeenLastCalledWith(
      expect.objectContaining({
        provider_key: "openai",
        api_key: "test-api-key-value",
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() => {
      expect(setupHookMocks.saveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: null,
          model: "gpt-4.1",
          make_default: true,
        }),
      );
    });

    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    expect(
      await screen.findByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();
  });

  it("resumes first chat and revalidates saved hosted credentials after provider edits", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");
    setupHookMocks.validateProvider.mockImplementation(async (payload) => {
      if (!payload.api_key) {
        return {
          provider_key: "openai",
          status: "failed",
          failure_category: "provider_api_key_required",
          message: "Provider API key is required.",
          models: [],
          can_gate_first_chat: false,
        };
      }
      return {
        provider_key: "openai",
        status: "accepted",
        message: "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level: "local_syntax",
        can_gate_first_chat: true,
      };
    });

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: [
            "setup_path",
            "privacy_security",
            "providers",
            "ingest_defaults",
            "audio_defaults",
            "optional_advanced",
          ],
          skipped_steps: [],
          step_data: {
            providers: {
              acknowledged: true,
              default_provider: "openai",
              default_model: "gpt-4.1-mini",
              default_provider_credential_configured: true,
            },
          },
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /first chat/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /back to providers/i }));
    expect(
      await screen.findByRole("heading", { name: /chat provider/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/openai api key/i)).toHaveValue("");
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1" },
    });

    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));
    expect(
      await screen.findByText(/saved credentials are present/i),
    ).toBeInTheDocument();
    expect(setupHookMocks.validateProvider).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /save provider/i }));
    await waitFor(() => {
      expect(setupHookMocks.saveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: null,
          model: "gpt-4.1",
          make_default: true,
        }),
      );
    });
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    expect(
      await screen.findByRole("heading", { name: /ingest defaults/i }),
    ).toBeInTheDocument();
  });

  it("does not infer saved hosted credentials when resumed state lacks the marker", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");
    setupHookMocks.validateProvider.mockImplementation(async (payload) => {
      if (!payload.api_key) {
        return {
          provider_key: "openai",
          status: "failed",
          failure_category: "provider_api_key_required",
          message: "Provider API key is required.",
          models: [],
          can_gate_first_chat: false,
        };
      }
      return {
        provider_key: "openai",
        status: "accepted",
        message: "Format accepted; first chat verifies the provider.",
        models: [],
        validation_level: "local_syntax",
        can_gate_first_chat: true,
      };
    });

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: [
            "setup_path",
            "privacy_security",
            "providers",
            "ingest_defaults",
            "audio_defaults",
            "optional_advanced",
          ],
          skipped_steps: [],
          step_data: {
            providers: {
              acknowledged: true,
              default_provider: "openai",
              default_model: "gpt-4.1-mini",
            },
          },
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /back to providers/i }));
    expect(
      await screen.findByRole("heading", { name: /chat provider/i }),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /validate openai/i }));

    expect(
      await screen.findByText(/provider_api_key_required/i),
    ).toBeInTheDocument();
    expect(setupHookMocks.validateProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        provider_key: "openai",
        api_key: null,
        model: "gpt-4.1",
      }),
    );
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("shows a non-blocking warning when audio recommendations fail to load", async () => {
    setupHookMocks.loadAudioRecommendations.mockRejectedValueOnce(
      new Error("recommendation failure"),
    );
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard");

    render(
      <UnifiedSetupWizard
        initialState={{
          status: "in_progress",
          completed_steps: [
            "setup_path",
            "privacy_security",
            "providers",
            "ingest_defaults",
          ],
          skipped_steps: [],
          step_data: {},
          acknowledged_steps: [],
          first_chat: { completed: false },
        }}
      />,
    );

    expect(
      await screen.findByText(/audio recommendations could not be loaded/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
  });
});

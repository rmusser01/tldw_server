import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Storage } from "@plasmohq/storage";
import { UnifiedSetupWizard } from "../UnifiedSetupWizard";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { useStoreMessageOption } from "@/store/option";
import type { FirstRunState } from "@/types/setup-onboarding";
import { TldwModelsService } from "@/services/tldw/TldwModels";
import { clearRuntimeAuthOverride } from "@/services/tldw/runtime-auth-override";
vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock(
  "@plasmohq/storage/hook",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook"),
);
const mocks = vi.hoisted(() => ({
  config: {
    serverUrl: "http://setup.test",
    authMode: "single-user",
    apiKey: "synthetic-key-a",
  },
  verify: vi.fn(),
  complete: vi.fn(),
  refresh: vi.fn(),
  models: vi.fn(),
  getConfig: vi.fn(),
  initialize: vi.fn(),
  getModels: vi.fn(),
}));
vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ setConfigPartial: vi.fn() }),
}));
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => "synthetic-owner",
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    initialize: mocks.initialize,
    getModels: mocks.getModels,
  },
}));
vi.mock("@/services/tldw", () => ({
  tldwModels: { getChatModels: mocks.models },
}));
vi.mock("@/hooks/useSetupReadinessSummary", () => ({
  useSetupReadinessSummary: () => ({
    status: null,
    loading: false,
    error: null,
    refresh: vi.fn(),
  }),
}));
const initial: FirstRunState = {
  status: "in_progress",
  completed_steps: [
    "setup_path",
    "privacy_security",
    "providers",
    "ingest_defaults",
    "audio_defaults",
    "optional_advanced",
    "mcp_tools",
  ],
  skipped_steps: [],
  acknowledged_steps: [],
  step_data: {
    providers: {
      default_provider: "custom_openai",
      default_model: "gemma-tested",
    },
    mcp_tools: { acknowledged: true },
  },
  first_chat: { completed: false },
};
vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: null,
    metadata: { auth_mode: "single_user" },
    providerCatalog: [],
    mcpToolsCatalog: null,
    audioRecommendations: [],
    loading: false,
    error: null,
    refresh: mocks.refresh,
    verifyFirstChat: mocks.verify,
    complete: mocks.complete,
    loadProviderCatalog: vi.fn().mockResolvedValue([]),
    loadMcpToolsCatalog: vi.fn(),
    loadAudioRecommendations: vi.fn(),
  }),
}));
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const ready = {
  status: "ready",
  provider: "custom_openai",
  model: "gemma-tested",
  response_text: "Verified hello",
};
const completed = {
  success: true,
  message: "Completed",
  requires_restart: false,
  install_plan_submitted: false,
};
let selectModel: ReturnType<typeof useSelectedModel>["setSelectedModel"];
function SharedOwner() {
  const owner = useSelectedModel();
  React.useEffect(() => {
    selectModel = owner.setSelectedModel;
  }, [owner.setSelectedModel]);
  return (
    <output aria-label="Chat model">
      {owner.selectedModel || "unselected"}
    </output>
  );
}
function Parent({
  publish,
  setupState = initial,
}: {
  publish?: () => void;
  setupState?: FirstRunState;
}) {
  const [done, setDone] = React.useState(false);
  return (
    <>
      <SharedOwner />
      {done ? (
        <p>Workspace ready</p>
      ) : (
        <UnifiedSetupWizard
          initialState={setupState}
          onStateChange={(state) => {
            if (state.status === "completed") {
              publish?.();
              setDone(true);
            }
          }}
        />
      )}
    </>
  );
}
const send = () =>
  fireEvent.click(screen.getByRole("button", { name: "Send test chat" }));

describe("verified setup model handoff through the real WebUI owner", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useStoreMessageOption.setState({ selectedModel: null });
    mocks.config = {
      serverUrl: "http://setup.test",
      authMode: "single-user",
      apiKey: "synthetic-key-a",
    };
    mocks.getConfig
      .mockReset()
      .mockImplementation(async () => ({ ...mocks.config }));
    mocks.verify.mockReset().mockResolvedValue(ready);
    mocks.complete.mockReset().mockResolvedValue(completed);
    mocks.refresh
      .mockReset()
      .mockResolvedValue({ ...initial, status: "completed" });
    mocks.models.mockReset().mockResolvedValue([
      { id: "gemma3:1b", provider: "ollama" },
      { id: "gemma-tested", provider: "custom-openai-api" },
    ]);
  });
  afterEach(() => vi.restoreAllMocks());
  it("offers provider reselection when anonymous resume redacts the saved model", () => {
    render(<Parent setupState={{
      ...initial,
      step_data: { ...initial.step_data, providers: { default_provider: "llamacpp" } },
    }} />);
    expect(document.querySelector("section[aria-labelledby='provider-setup-title']")).toBeInTheDocument();
  });
  it("finishes with the verified model before a browser key permits the protected catalog", async () => {
    const { TldwApiClient } = await vi.importActual<
      typeof import("@/services/tldw/TldwApiClient")
    >("@/services/tldw/TldwApiClient");
    clearRuntimeAuthOverride();
    window.sessionStorage.clear();
    await new Storage({ area: "local" }).set("tldwConfig", {
      serverUrl: "http://setup.test",
      authMode: "single-user",
      apiKey: "",
    });
    const client = new TldwApiClient();
    mocks.getConfig.mockImplementation(() => client.getConfig());
    mocks.initialize.mockImplementation(() => client.initialize());
    mocks.getModels.mockImplementation(() => client.getModels());
    const models = new TldwModelsService();
    mocks.models.mockImplementation(() => models.getChatModels(true));
    const network = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(
        new Error(
          "No protected network calls are expected during setup handoff",
        ),
      );
    expect((await client.getConfig())?.apiKey || "").toBe("");
    expect(await models.getChatModels(true)).toEqual([]);
    expect(mocks.getModels).not.toHaveBeenCalled();

    render(<Parent />);
    send();
    await screen.findByText("Workspace ready");
    expect(await new Storage().get("selectedModel")).toBe(
      "tldw:custom-openai-api:gemma-tested",
    );
    expect(network).not.toHaveBeenCalled();
    expect(mocks.getModels).not.toHaveBeenCalled();
    expect(mocks.models).not.toHaveBeenCalled();
  });
  it("persists the verified provider-qualified choice before the parent unmounts setup", async () => {
    let atPublication: string | null = null;
    render(
      <Parent
        publish={() => {
          atPublication = window.localStorage.getItem("selectedModel");
        }}
      />,
    );
    send();
    await screen.findByText("Workspace ready");
    expect(screen.getByLabelText("Chat model")).toHaveTextContent(
      "tldw:custom-openai-api:gemma-tested",
    );
    expect(atPublication).toBe(
      JSON.stringify("tldw:custom-openai-api:gemma-tested"),
    );
  });
  it.each(["tldw:chosen", '"tldw:chosen"', "  tldw:chosen  "])(
    "preserves the hydrated deliberate selection %s",
    async (stored) => {
      await new Storage().set("selectedModel", stored);
      render(<Parent />);
      send();
      await screen.findByText("Workspace ready");
      expect(useStoreMessageOption.getState().selectedModel).toBe(
        "tldw:chosen",
      );
    },
  );
  it("preserves a newer explicit choice while verification is pending", async () => {
    const pending = deferred<typeof ready>();
    mocks.verify.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.verify).toHaveBeenCalled());
    await act(async () => {
      await selectModel("tldw:newer");
      pending.resolve(ready);
    });
    await screen.findByText("Workspace ready");
    expect(useStoreMessageOption.getState().selectedModel).toBe("tldw:newer");
  });
  it("does not re-seed after a newer choice was deliberately cleared", async () => {
    const pending = deferred<typeof ready>();
    mocks.verify.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.verify).toHaveBeenCalled());
    await act(async () => {
      await selectModel("tldw:newer");
      await selectModel(null);
      pending.resolve(ready);
    });
    await screen.findByText("Workspace ready");
    expect(useStoreMessageOption.getState().selectedModel).toBeNull();
  });
  it.each(["missing", "ambiguous"])(
    "uses the verified pair despite a %s protected catalog target",
    async (kind) => {
      mocks.models.mockResolvedValue(
        kind === "missing"
          ? [{ id: "other", provider: "ollama" }]
          : [
              { id: "gemma-tested", provider: "custom-openai-api" },
              { id: "gemma-tested", provider: "custom_openai_api" },
            ],
      );
      render(<Parent />);
      send();
      await screen.findByText("Workspace ready");
      expect(await new Storage().get("selectedModel")).toBe(
        "tldw:custom-openai-api:gemma-tested",
      );
    },
  );
  it("retries rejected storage without repeating inference or acknowledged completion", async () => {
    const original = Storage.prototype.set;
    let fail = true;
    vi.spyOn(Storage.prototype, "set").mockImplementation(async function (
      this: Storage,
      key,
      value,
    ) {
      if (key === "selectedModel" && value && fail) {
        fail = false;
        throw new Error("Device storage unavailable");
      }
      return original.call(this, key, value);
    });
    render(<Parent />);
    send();
    expect(await screen.findByRole("alert")).toHaveTextContent(/storage/i);
    expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Finish setup" }));
    await screen.findByText("Workspace ready");
    expect(mocks.verify).toHaveBeenCalledTimes(1);
    expect(mocks.complete).toHaveBeenCalledTimes(1);
    expect(await new Storage().get("selectedModel")).toBe(
      "tldw:custom-openai-api:gemma-tested",
    );
  });
  it.each(["target", "key", "roundtrip", "unmount"])(
    "discards delayed completion after %s replacement",
    async (kind) => {
      const pending = deferred<typeof completed>();
      mocks.complete.mockReturnValueOnce(pending.promise);
      const view = render(<Parent />);
      send();
      await waitFor(() => expect(mocks.complete).toHaveBeenCalled());
      await act(async () => {
        if (kind === "unmount") view.unmount();
        else {
          if (kind === "target") mocks.config.serverUrl = "http://other.test";
          else mocks.config.apiKey = "synthetic-key-b";
          window.dispatchEvent(new CustomEvent("tldw:config-updated"));
          if (kind === "roundtrip") {
            mocks.config.apiKey = "synthetic-key-a";
            window.dispatchEvent(new CustomEvent("tldw:config-updated"));
          }
        }
        pending.resolve(completed);
      });
      expect(useStoreMessageOption.getState().selectedModel).toBeNull();
      expect(await new Storage().get("selectedModel")).toBeFalsy();
      expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
    },
  );
  it.each([null, "tldw:stored-choice"])(
    "waits for persisted preference hydration (%s)",
    async (stored) => {
      if (stored) await new Storage().set("selectedModel", stored);
      const barrier = deferred<void>();
      const original = Storage.prototype.get;
      vi.spyOn(Storage.prototype, "get").mockImplementation(async function (
        this: Storage,
        key,
      ) {
        const value = await original.call(this, key);
        if (key === "selectedModel") await barrier.promise;
        return value;
      });
      render(<Parent />);
      send();
      expect(mocks.verify).not.toHaveBeenCalled();
      await act(async () => {
        barrier.resolve();
      });
      await screen.findByText("Workspace ready");
      expect(useStoreMessageOption.getState().selectedModel).toBe(
        stored || "tldw:custom-openai-api:gemma-tested",
      );
    },
  );

  it.each(["failed", "throw"])(
    "does not publish a model after %s verification",
    async (outcome) => {
      if (outcome === "throw")
        mocks.verify.mockRejectedValueOnce(new Error("Provider unavailable"));
      else
        mocks.verify.mockResolvedValueOnce({
          ...ready,
          status: "failed",
          message: "Provider unavailable",
        });
      render(<Parent />);
      send();
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Provider unavailable",
      );
      expect(mocks.complete).not.toHaveBeenCalled();
      expect(await new Storage().get("selectedModel")).toBeFalsy();
    },
  );

  it("retries server completion without repeating a successful verification", async () => {
    mocks.complete.mockRejectedValueOnce(new Error("Completion unavailable"));
    render(<Parent />);
    send();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Completion unavailable",
    );
    fireEvent.click(screen.getByRole("button", { name: "Finish setup" }));
    await screen.findByText("Workspace ready");
    expect(mocks.verify).toHaveBeenCalledTimes(1);
    expect(mocks.complete).toHaveBeenCalledTimes(2);
  });

  it("does not replace a choice made while setup completion is pending", async () => {
    const pending = deferred<typeof completed>();
    mocks.complete.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.complete).toHaveBeenCalled());
    await act(async () => {
      await selectModel("tldw:later-choice");
      pending.resolve(completed);
    });
    await screen.findByText("Workspace ready");
    expect(await new Storage().get("selectedModel")).toBe("tldw:later-choice");
  });

  it("does not select the other provider's same-named model", async () => {
    mocks.models.mockResolvedValue([
      { id: "gemma-tested", provider: "ollama" },
      { id: "custom-openai-api:gemma-tested", provider: "custom-openai-api" },
    ]);
    render(<Parent />);
    send();
    await screen.findByText("Workspace ready");
    expect(await new Storage().get("selectedModel")).toBe(
      "tldw:custom-openai-api:gemma-tested",
    );
  });

  it("normalizes the setup spelling when the model catalog uses it too", async () => {
    mocks.models.mockResolvedValue([
      { id: "gemma-tested", provider: "custom_openai" },
    ]);
    render(<Parent />);
    send();
    await screen.findByText("Workspace ready");
    expect(await new Storage().get("selectedModel")).toBe(
      "tldw:custom-openai-api:gemma-tested",
    );
  });

  it.each([
    ["custom_openai", "custom-openai-api"],
    ["custom_openai_api", "custom-openai-api"],
    ["custom_openai2", "custom-openai-api-2"],
    ["custom_openai_api2", "custom-openai-api-2"],
    ["llamacpp", "llama.cpp"],
    ["koboldcpp", "kobold"],
    ["oobabooga", "ooba"],
    ["tabbyapi", "tabbyapi"],
  ])(
    "publishes the exact verified %s slot as %s without a catalog",
    async (provider, canonical) => {
      mocks.models.mockResolvedValue([]);
      mocks.verify.mockResolvedValue({
        ...ready,
        provider,
        model: "org/gemma:Q4",
      });
      render(
        <Parent
          setupState={{
            ...initial,
            step_data: {
              ...initial.step_data,
              providers: {
                default_provider: provider,
                default_model: "org/gemma:Q4",
              },
            },
          }}
        />,
      );
      send();
      await screen.findByText("Workspace ready");
      expect(await new Storage().get("selectedModel")).toBe(
        `tldw:${canonical}:org/gemma:Q4`,
      );
    },
  );

  it.each([
    { provider: "ollama", model: "gemma-tested" },
    { provider: "custom-openai-api-2", model: "gemma-tested" },
    { provider: "custom_openai", model: "another-model" },
  ])(
    "rejects a ready response that differs from the requested pair: %j",
    async (pair) => {
      mocks.verify.mockResolvedValue({ ...ready, ...pair });
      render(<Parent />);
      send();
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "did not match",
      );
      expect(mocks.complete).not.toHaveBeenCalled();
      expect(await new Storage().get("selectedModel")).toBeFalsy();
    },
  );

  it("keeps an unknown verified provider actionable without guessing a Chat route", async () => {
    mocks.verify.mockResolvedValue({
      ...ready,
      provider: "unrecognized-provider",
    });
    render(
      <Parent
        setupState={{
          ...initial,
          step_data: {
            ...initial.step_data,
            providers: {
              default_provider: "unrecognized-provider",
              default_model: ready.model,
            },
          },
        }}
      />,
    );
    send();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /provider|model/i,
    );
    expect(screen.getByRole("button", { name: "Finish setup" })).toBeEnabled();
    expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
    expect(await new Storage().get("selectedModel")).toBeFalsy();
  });

  it("refuses verification with unknown connection authority", async () => {
    mocks.getConfig.mockResolvedValue(null);
    render(<Parent />);
    send();
    expect(await screen.findByRole("alert")).toHaveTextContent(/connection/i);
    expect(mocks.verify).not.toHaveBeenCalled();
    expect(mocks.complete).not.toHaveBeenCalled();
  });

  it("does not publish completed state after replacement during the final refresh", async () => {
    const pending = deferred<FirstRunState>();
    mocks.refresh.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.refresh).toHaveBeenCalled());
    await act(async () => {
      mocks.config.serverUrl = "http://new.test";
      window.dispatchEvent(new CustomEvent("tldw:config-updated"));
      pending.resolve({ ...initial, status: "completed" });
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /connection changed/i,
    );
    expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
  });
});

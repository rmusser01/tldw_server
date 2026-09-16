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
}));
vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ setConfigPartial: vi.fn() }),
}));
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => "synthetic-owner",
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: mocks.getConfig },
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
    loadProviderCatalog: vi.fn(),
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
function Parent({ publish }: { publish?: () => void }) {
  const [done, setDone] = React.useState(false);
  return (
    <>
      <SharedOwner />
      {done ? (
        <p>Workspace ready</p>
      ) : (
        <UnifiedSetupWizard
          initialState={initial}
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
    "keeps completion actionable for a %s catalog target",
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
      expect(await screen.findByRole("alert")).toHaveTextContent(
        /model|selection/i,
      );
      expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
      expect(useStoreMessageOption.getState().selectedModel).toBeNull();
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

  it("does not replace a choice made while catalog resolution is pending", async () => {
    const pending = deferred<Array<{ id: string; provider: string }>>();
    mocks.models.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.models).toHaveBeenCalled());
    await act(async () => {
      await selectModel("tldw:later-choice");
      pending.resolve([{ id: "gemma-tested", provider: "custom-openai-api" }]);
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

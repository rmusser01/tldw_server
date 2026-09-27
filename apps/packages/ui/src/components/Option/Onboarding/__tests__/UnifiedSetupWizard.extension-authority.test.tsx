import React from "react";
import "@/i18n";
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
const mocks = vi.hoisted(() => {
  type Changes = Record<string, { oldValue?: unknown; newValue?: unknown }>;
  const listeners = new Set<(changes: Changes, area: string) => void>();
  const data = {
    local: new Map<string, unknown>(),
    sync: new Map<string, unknown>(),
    session: new Map<string, unknown>(),
  };
  const storage = Object.fromEntries(
    Object.entries(data).map(([area, values]) => [
      area,
      {
        get: async (keys: string[]) =>
          Object.fromEntries(
            keys
              .filter((key) => values.has(key))
              .map((key) => [key, values.get(key)]),
          ),
        set: async (entries: Record<string, unknown>) => {
          const changes: Changes = {};
          for (const [key, value] of Object.entries(entries)) {
            changes[key] = { oldValue: values.get(key), newValue: value };
            values.set(key, value);
          }
          for (const listener of listeners) listener(changes, area);
        },
        remove: async (keys: string[]) => {
          for (const key of keys) values.delete(key);
        },
        clear: async () => values.clear(),
      },
    ]),
  );
  Object.assign(storage, {
    onChanged: {
      addListener: (listener: (changes: Changes, area: string) => void) =>
        listeners.add(listener),
      removeListener: (listener: (changes: Changes, area: string) => void) =>
        listeners.delete(listener),
    },
  });
  Object.defineProperty(globalThis, "browser", {
    configurable: true,
    value: { storage },
  });
  return {
    data,
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
  };
});
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
function SharedOwner() {
  const owner = useSelectedModel();
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

describe("setup authority through actual extension storage", () => {
  beforeEach(() => {
    for (const values of Object.values(mocks.data)) values.clear();
    useStoreMessageOption.setState({ selectedModel: null });
    mocks.getConfig.mockImplementation(async () => ({ ...mocks.config }));
    mocks.verify.mockResolvedValue(ready);
    mocks.refresh.mockResolvedValue({ ...initial, status: "completed" });
    mocks.models.mockResolvedValue([
      { id: "gemma-tested", provider: "custom-openai-api" },
    ]);
  });
  afterEach(() => vi.restoreAllMocks());
  it("discards pending completion after local canonical authority changes A to B to A without window events", async () => {
    const configStorage = new Storage({ area: "local" });
    await configStorage.set("tldwConfig", mocks.config);
    const pending = deferred<typeof completed>();
    mocks.complete.mockReturnValueOnce(pending.promise);
    render(<Parent />);
    send();
    await waitFor(() => expect(mocks.complete).toHaveBeenCalled());
    await act(async () => {
      await configStorage.set("tldwConfig", {
        ...mocks.config,
        apiKey: "synthetic-key-b",
      });
      await configStorage.set("tldwConfig", mocks.config);
      pending.resolve(completed);
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /connection changed/i,
    );
    expect(screen.queryByText("Workspace ready")).not.toBeInTheDocument();
    expect(await new Storage().get("selectedModel")).toBeUndefined();
    expect(useStoreMessageOption.getState().selectedModel).toBeNull();
  });
});

// @vitest-environment jsdom
import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import {
  QueryClient,
  QueryClientProvider,
  useQueryClient,
} from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope";
import { ModerationPlaygroundShell } from "../ModerationPlaygroundShell";

const mocks = vi.hoisted(() => ({
  config: {
    serverUrl: "http://rules-a.test",
    authMode: "single-user",
    apiKey: "key-a",
  } as TldwConfig,
  loading: false,
  owner: "A" as "A" | "B",
  settings: vi.fn(),
  policy: vi.fn(),
  overrides: vi.fn(),
  test: vi.fn(),
}));
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: mocks.config,
    loading: false,
    authorityLoading: mocks.loading,
  }),
}));
vi.mock("@/services/prompt-local-owner", () => ({
  resolvePromptOwnerId: async () =>
    "recipe-owner:sha256:" + (mocks.owner === "A" ? "a" : "b").repeat(64),
}));
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }));
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({ uxState: "connected_ok" }),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key,
  }),
}));
vi.mock("@/store/milestones", () => ({
  useMilestoneStore: (select: (s: unknown) => unknown) =>
    select({ markMilestone: () => {} }),
}));
vi.mock("@/store/tutorials", () => ({
  useTutorialStore: (select: (s: unknown) => unknown) =>
    select({ startTutorial: () => {}, isCompleted: () => true }),
}));
vi.mock("@/services/moderation", () => ({
  getModerationSettings: (...args: unknown[]) => mocks.settings(...args),
  getEffectivePolicy: (...args: unknown[]) => mocks.policy(...args),
  listUserOverrides: (...args: unknown[]) => mocks.overrides(...args),
  testModeration: (...args: unknown[]) => mocks.test(...args),
  getUserOverride: vi.fn(),
  reloadModeration: vi.fn(),
}));
vi.mock("../PolicySettingsPanel", () => ({
  default: function PolicyProbe({
    settings,
  }: {
    settings: { draft: unknown; policyQuery: { data?: unknown } };
  }) {
    const queryClient = useQueryClient();
    return (
      <div>
        <output data-testid="rules-draft">
          {JSON.stringify(settings.draft)}
        </output>
        <output data-testid="rules-policy">
          {JSON.stringify(settings.policyQuery.data)}
        </output>
        <button
          onClick={() =>
            void queryClient.invalidateQueries({
              queryKey: ["moderation-policy"],
            })
          }
        >
          Invalidate fixture policy
        </button>
      </div>
    );
  },
}));
vi.mock("../TestSandboxPanel", () => ({
  default: ({
    tester,
  }: {
    tester: {
      history: unknown;
      runTestWith: (p: { text: string; phase: "input" }) => Promise<void>;
    };
  }) => (
    <div>
      <output data-testid="rules-history">
        {JSON.stringify(tester.history)}
      </output>
      <button
        onClick={() =>
          void tester.runTestWith({
            text: "PRIVATE_A_TEST_20261031",
            phase: "input",
          })
        }
      >
        Run fixture test
      </button>
    </div>
  ),
}));
const privateRule = (owner: string) => "PRIVATE_" + owner + "_RULE_20261031";
let client: QueryClient;
const shell = () => (
  <QueryClientProvider client={client}>
    <MemoryRouter>
      <ModerationPlaygroundShell />
    </MemoryRouter>
  </QueryClientProvider>
);
beforeEach(() => {
  vi.clearAllMocks();
  mocks.config = {
    serverUrl: "http://rules-a.test",
    authMode: "single-user",
    apiKey: "key-a",
  };
  mocks.loading = false;
  mocks.owner = "A";
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mocks.settings.mockImplementation(async () => ({
    categories_enabled: [privateRule(mocks.owner)],
    effective: {},
  }));
  mocks.policy.mockResolvedValue({});
  mocks.overrides.mockResolvedValue({ overrides: {} });
  mocks.test.mockResolvedValue({
    flagged: false,
    action: "pass",
    effective: {},
  });
  localStorage.setItem("moderation-playground-onboarded", "true");
});
afterEach(() => {
  cleanup();
  client.clear();
});
describe("Rules private authority state", () => {
  it("waits for canonical authority before reading rules", async () => {
    mocks.loading = true;
    render(shell());
    await act(async () => {});
    expect(mocks.settings).not.toHaveBeenCalled();
  });
  it("hides resident rules while authority is rehydrating", async () => {
    const view = render(shell());
    await waitFor(() =>
      expect(screen.getByTestId("rules-draft").textContent).toContain(
        privateRule("A"),
      ),
    );
    mocks.loading = true;
    view.rerender(shell());
    expect(
      screen.queryByTestId("rules-draft")?.textContent || "",
    ).not.toContain(privateRule("A"));
  });
  it.each(["server", "account"] as const)(
    "never shows cached A rules while the replacement %s loads",
    async (kind) => {
      const view = render(shell());
      await waitFor(() =>
        expect(screen.getByTestId("rules-draft").textContent).toContain(
          privateRule("A"),
        ),
      );
      mocks.settings.mockImplementation(() => new Promise(() => {}));
      mocks.owner = "B";
      mocks.config =
        kind === "server"
          ? { ...mocks.config, serverUrl: "http://rules-b.test" }
          : { ...mocks.config, apiKey: "key-b" };
      view.rerender(shell());
      await act(async () => {});
      expect(
        screen.queryByTestId("rules-draft")?.textContent || "",
      ).not.toContain(privateRule("A"));
    },
  );
  it("rejects a late A response after B owns Rules", async () => {
    let finish!: (v: unknown) => void;
    mocks.settings.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const view = render(shell());
    await waitFor(() => expect(mocks.settings).toHaveBeenCalledOnce());
    mocks.owner = "B";
    mocks.config = { ...mocks.config, apiKey: "key-b" };
    view.rerender(shell());
    await act(async () => {
      finish({ categories_enabled: [privateRule("A")], effective: {} });
    });
    await waitFor(() =>
      expect(screen.getByTestId("rules-draft").textContent).toContain(
        privateRule("B"),
      ),
    );
    expect(screen.getByTestId("rules-draft").textContent).not.toContain(
      privateRule("A"),
    );
  });
  it("retires private test history when a cookie principal changes without a new connection hash", async () => {
    mocks.config = {
      serverUrl: "http://rules-a.test",
      authMode: "multi-user",
      authSource: "cookie-session",
    };
    const boundary = deriveConnectionAuthorityId(mocks.config);
    render(shell());
    await screen.findByTestId("rules-draft");
    fireEvent.click(screen.getByRole("tab", { name: "Test Sandbox" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Run fixture test" }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("rules-history").textContent).toContain(
        "PRIVATE_A_TEST_20261031",
      ),
    );
    mocks.owner = "B";
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    expect(deriveConnectionAuthorityId(mocks.config)).toBe(boundary);
    expect(
      screen.queryByTestId("rules-history")?.textContent || "",
    ).not.toContain("PRIVATE_A_TEST_20261031");
  });
  it("does not borrow or clear the surrounding cache, including under Strict Mode", async () => {
    client.setQueryData(["moderation-settings"], {
      categories_enabled: ["OUTER_PRIVATE_RULE"],
      effective: {},
    });
    const view = render(<React.StrictMode>{shell()}</React.StrictMode>);
    await waitFor(() =>
      expect(screen.getByTestId("rules-draft").textContent).toContain(
        privateRule("A"),
      ),
    );
    expect(screen.getByTestId("rules-draft").textContent).not.toContain(
      "OUTER_PRIVATE_RULE",
    );
    const policyReads = mocks.policy.mock.calls.length;
    mocks.policy.mockResolvedValue({ revision: "AFTER_STRICT_REPLAY" });
    fireEvent.click(
      screen.getByRole("button", { name: "Invalidate fixture policy" }),
    );
    await waitFor(() =>
      expect(mocks.policy.mock.calls.length).toBeGreaterThan(policyReads),
    );
    await waitFor(() =>
      expect(screen.getByTestId("rules-policy").textContent).toContain(
        "AFTER_STRICT_REPLAY",
      ),
    );
    view.unmount();
    expect(client.getQueryData(["moderation-settings"])).toEqual({
      categories_enabled: ["OUTER_PRIVATE_RULE"],
      effective: {},
    });
  });
});

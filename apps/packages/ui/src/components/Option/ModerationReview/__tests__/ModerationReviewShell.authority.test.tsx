// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope";
import { ModerationReviewShell } from "../ModerationReviewShell";

const mocks = vi.hoisted(() => ({
  config: {
    serverUrl: "http://review-a.test",
    authMode: "single-user",
    apiKey: "key-a",
  } as TldwConfig,
  authorityLoading: false,
  ownerTag: "A" as "A" | "B" | null,
  localConfig: null as TldwConfig | null,
  configReads: 0,
  resolveOwner: vi.fn(),
  list: vi.fn(),
  detail: vi.fn(),
  decide: vi.fn(),
  undo: vi.fn(),
  bulk: vi.fn(),
  audit: vi.fn(),
}));
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config:
      mocks.localConfig && mocks.configReads++ % 2 === 1
        ? mocks.localConfig
        : mocks.config,
    loading: false,
    authorityLoading: mocks.authorityLoading,
  }),
}));
vi.mock("@/services/prompt-local-owner", () => ({
  resolvePromptOwnerId: (...args: unknown[]) => mocks.resolveOwner(...args),
}));
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }));
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: "connected_ok",
    hasCompletedFirstRun: true,
  }),
}));
vi.mock("@/services/moderation", () => ({
  listModerationReviewItems: (...args: unknown[]) => mocks.list(...args),
  getModerationReviewItem: (...args: unknown[]) => mocks.detail(...args),
  decideModerationReviewItem: (...args: unknown[]) => mocks.decide(...args),
  undoModerationReviewDecision: (...args: unknown[]) => mocks.undo(...args),
  bulkDecideModerationReviewItems: (...args: unknown[]) => mocks.bulk(...args),
  listModerationReviewAudit: (...args: unknown[]) => mocks.audit(...args),
}));
const row = (owner: "A" | "B") => ({
  id: "review-" + owner,
  status: "needs_review",
  phase: "input",
  created_at: "2026-10-31T00:00:00Z",
  severity: "low",
  excerpt: "PRIVATE_" + owner + "_REVIEW_20261031",
  safe_fields: {
    excerpt: true,
    context: true,
    matches: true,
    effective_policy: true,
  },
  context: {},
  effective_policy: {},
  matches: [],
});
const current = () =>
  mocks.config.authSource === "cookie-session"
    ? row(mocks.ownerTag === "B" ? "B" : "A")
    : row(
        mocks.config.serverUrl === "http://review-a.test" &&
          mocks.config.apiKey === "key-a"
          ? "A"
          : "B",
      );
const shell = () => (
  <MemoryRouter>
    <ModerationReviewShell />
  </MemoryRouter>
);
beforeEach(() => {
  vi.resetAllMocks();
  mocks.config = {
    serverUrl: "http://review-a.test",
    authMode: "single-user",
    apiKey: "key-a",
  };
  mocks.authorityLoading = false;
  mocks.ownerTag = "A";
  mocks.localConfig = null;
  mocks.configReads = 0;
  mocks.resolveOwner.mockImplementation(async () =>
    mocks.ownerTag === null
      ? null
      : "recipe-owner:sha256:" +
        (current().id === "review-A" ? "a" : "b").repeat(64),
  );
  mocks.list.mockImplementation(async () => ({
    items: [current()],
    total: 1,
    next_cursor: null,
  }));
  mocks.detail.mockImplementation(async (id: string) =>
    row(id === "review-A" ? "A" : "B"),
  );
  mocks.audit.mockResolvedValue({ events: [], next_cursor: null });
  window.localStorage.clear();
});
afterEach(cleanup);
describe("moderation canonical authority", () => {
  it("does not fetch review data before canonical authority resolves", async () => {
    mocks.authorityLoading = true;
    render(shell());
    await act(async () => {});
    expect(mocks.list).not.toHaveBeenCalled();
    expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
  });

  it("retires visible private review data during authority rehydration", async () => {
    const view = render(shell());
    await screen.findAllByText(row("A").excerpt);
    mocks.authorityLoading = true;
    view.rerender(shell());
    expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
  });

  it.each(["server", "account"] as const)(
    "retires private review data when the %s changes",
    async (kind) => {
      const view = render(shell());
      await screen.findAllByText(row("A").excerpt);
      mocks.config =
        kind === "server"
          ? { ...mocks.config, serverUrl: "http://review-b.test" }
          : { ...mocks.config, apiKey: "key-b" };
      view.rerender(shell());
      expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
      await screen.findAllByText(row("B").excerpt);
    },
  );

  it("ignores an old queue response after another account owns the screen", async () => {
    let finishOld!: (value: unknown) => void;
    mocks.list.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishOld = resolve;
        }),
    );
    const view = render(shell());
    await waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(1));
    mocks.config = { ...mocks.config, apiKey: "key-b" };
    view.rerender(shell());
    await act(async () => {
      finishOld({ items: [row("A")], total: 1, next_cursor: null });
    });
    expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
    await screen.findAllByText(row("B").excerpt);
  });

  it("keeps unowned legacy presets intact without exposing their private labels", async () => {
    const key = "tldw.moderationReview.filterPresets.v1",
      name = "PRIVATE_A_FILTER_20261031";
    const legacy = JSON.stringify([
      {
        name,
        filters: {
          status: "needs_review",
          category: "",
          severity: "",
          source_type: "",
          source_id: "",
          user_id: "Alice",
          q: "Private Alice query",
          sort: "newest",
          cursor: null,
        },
      },
    ]);
    window.localStorage.setItem(key, legacy);
    mocks.config = { ...mocks.config, apiKey: "key-b" };
    render(shell());
    await screen.findAllByText(row("B").excerpt);
    expect(screen.queryByRole("option", { name })).not.toBeInTheDocument();
    expect(window.localStorage.getItem(key)).toBe(legacy);
  });

  it("isolates cookie principals and their presets when the connection hash stays unchanged", async () => {
    mocks.config = {
      serverUrl: "http://review-a.test",
      authMode: "multi-user",
      authSource: "cookie-session",
    };
    const boundary = deriveConnectionAuthorityId(mocks.config);
    render(shell());
    await screen.findAllByText(row("A").excerpt);
    fireEvent.change(screen.getByLabelText("Preset name"), {
      target: { value: "COOKIE_A_PRESET" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save preset" }));
    expect(
      screen.getByRole("option", { name: "COOKIE_A_PRESET" }),
    ).toBeInTheDocument();
    mocks.ownerTag = "B";
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
    await screen.findAllByText(row("B").excerpt);
    expect(deriveConnectionAuthorityId(mocks.config)).toBe(boundary);
    expect(
      screen.queryByRole("option", { name: "COOKIE_A_PRESET" }),
    ).not.toBeInTheDocument();
    mocks.ownerTag = "A";
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    await screen.findAllByText(row("A").excerpt);
    expect(
      screen.getByRole("option", { name: "COOKIE_A_PRESET" }),
    ).toBeInTheDocument();
  });

  it("refuses mismatched canonical and local-owner boundaries", async () => {
    mocks.localConfig = { ...mocks.config, serverUrl: "http://review-b.test" };
    render(shell());
    await act(async () => {});
    expect(mocks.configReads).toBeGreaterThanOrEqual(2);
    expect(mocks.list).not.toHaveBeenCalled();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Resolving Moderation Review account",
    );
  });

  it("keeps an unavailable owner closed and offers a real Retry", async () => {
    mocks.resolveOwner.mockRejectedValueOnce(new Error("Owner unavailable"));
    render(shell());
    await screen.findByRole("alert");
    expect(mocks.list).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Retry account" }));
    await screen.findAllByText(row("A").excerpt);
  });

  it("never treats a resolved null owner as an authenticated preset owner", async () => {
    mocks.ownerTag = null;
    render(shell());
    await screen.findByRole("alert");
    expect(mocks.list).not.toHaveBeenCalled();
    expect(screen.queryAllByText(row("A").excerpt)).toHaveLength(0);
  });

  it("restores only the returning owner's saved filter preset", async () => {
    const view = render(shell());
    await screen.findAllByText(row("A").excerpt);
    fireEvent.change(screen.getByLabelText("Preset name"), {
      target: { value: "OWNER_A_PRESET" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save preset" }));
    mocks.config = { ...mocks.config, apiKey: "key-b" };
    view.rerender(shell());
    await screen.findAllByText(row("B").excerpt);
    expect(
      screen.queryByRole("option", { name: "OWNER_A_PRESET" }),
    ).not.toBeInTheDocument();
    mocks.config = { ...mocks.config, apiKey: "key-a" };
    view.rerender(shell());
    await screen.findAllByText(row("A").excerpt);
    expect(
      screen.getByRole("option", { name: "OWNER_A_PRESET" }),
    ).toBeInTheDocument();
    expect(
      Object.keys(window.localStorage).some(
        (key) => key.includes("key-a") || key.includes("key-b"),
      ),
    ).toBe(false);
  });
  it("deletes only the active owner's preset and preserves legacy bytes", async () => {
    const legacyKey = "tldw.moderationReview.filterPresets.v1";
    const legacyBytes =
      '[{"name":"UNOWNED_PRIVATE_LABEL","filters":{"q":"legacy query"}}]';
    window.localStorage.setItem(legacyKey, legacyBytes);
    const view = render(shell());
    await screen.findAllByText(row("A").excerpt);
    fireEvent.change(screen.getByLabelText("Preset name"), {
      target: { value: "OWNER_A_KEEP" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save preset" }));
    const ownerAKey = legacyKey + ":recipe-owner:sha256:" + "a".repeat(64);
    const ownerABytes = window.localStorage.getItem(ownerAKey);
    expect(ownerABytes).toContain("OWNER_A_KEEP");
    mocks.config = { ...mocks.config, apiKey: "key-b" };
    view.rerender(shell());
    await screen.findAllByText(row("B").excerpt);
    fireEvent.change(screen.getByLabelText("Preset name"), {
      target: { value: "OWNER_B_DELETE" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save preset" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Delete" }),
    );
    expect(
      screen.queryByRole("option", { name: "OWNER_B_DELETE" }),
    ).not.toBeInTheDocument();
    expect(
      window.localStorage.getItem(
        legacyKey + ":recipe-owner:sha256:" + "b".repeat(64),
      ),
    ).toBe("[]");
    expect(window.localStorage.getItem(ownerAKey)).toBe(ownerABytes);
    expect(window.localStorage.getItem(legacyKey)).toBe(legacyBytes);
  });
});

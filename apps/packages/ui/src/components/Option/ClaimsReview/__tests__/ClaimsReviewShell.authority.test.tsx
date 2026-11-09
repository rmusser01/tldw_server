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
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope";
import { ClaimsReviewShell } from "../ClaimsReviewShell";
const mocks = vi.hoisted(() => ({
  config: {
    serverUrl: "http://claims-cookie.test",
    authMode: "multi-user",
    authSource: "cookie-session",
  } as TldwConfig,
  owner: "A" as "A" | "B",
  resolve: vi.fn(),
  request: vi.fn(),
}));
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: mocks.config,
    loading: false,
    authorityLoading: false,
  }),
}));
vi.mock("@/services/prompt-local-owner", () => ({
  resolvePromptOwnerId: (...args: unknown[]) => mocks.resolve(...args),
}));
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.request(...args),
}));
const claim = (owner = mocks.owner) => ({
  id: owner === "A" ? 1 : 2,
  media_id: 11,
  claim_text: "PRIVATE_" + owner + "_CLAIM",
  chunk_index: 0,
  span_start: 0,
  span_end: 10,
  review_status: "pending",
  review_version: 1,
});
beforeEach(() => {
  vi.clearAllMocks();
  mocks.owner = "A";
  mocks.resolve.mockImplementation(
    async () =>
      "recipe-owner:sha256:" + (mocks.owner === "A" ? "a" : "b").repeat(64),
  );
  mocks.request.mockImplementation(async (init) =>
    String(init.path).includes("review-queue?")
      ? { items: [claim()], has_more: false, next_offset: null }
      : String(init.path).endsWith("/history")
        ? []
        : claim(),
  );
});
afterEach(cleanup);
describe("Claims cookie-principal ownership", () => {
  it("retires the previous principal's private claim and unsaved decision draft", async () => {
    const boundary = deriveConnectionAuthorityId(mocks.config);
    render(<ClaimsReviewShell />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Inspect claim 1" }),
    );
    fireEvent.change(await screen.findByLabelText("Review notes"), {
      target: { value: "PRIVATE_A_DECISION_DRAFT" },
    });
    mocks.owner = "B";
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    expect(deriveConnectionAuthorityId(mocks.config)).toBe(boundary);
    expect(screen.queryAllByText("PRIVATE_A_CLAIM")).toHaveLength(0);
    expect(
      screen.queryByDisplayValue("PRIVATE_A_DECISION_DRAFT"),
    ).not.toBeInTheDocument();
    await screen.findByText("PRIVATE_B_CLAIM");
  });
  it("rejects an old queue response after the cookie principal changes", async () => {
    let release!: (value: unknown) => void;
    mocks.request.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    render(<ClaimsReviewShell />);
    await waitFor(() => expect(release).toBeTypeOf("function"));
    mocks.owner = "B";
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
    await act(async () => {
      release({ items: [claim("A")], has_more: false, next_offset: null });
    });
    expect(screen.queryAllByText("PRIVATE_A_CLAIM")).toHaveLength(0);
    await screen.findByText("PRIVATE_B_CLAIM");
  });
  it("does not read private Claims while actual owner resolution is unavailable", async () => {
    mocks.resolve.mockRejectedValueOnce(new Error("Owner unavailable"));
    render(<ClaimsReviewShell />);
    await act(async () => {});
    expect(mocks.request).not.toHaveBeenCalled();
  });
  it("retries unavailable owner resolution before reading private Claims", async () => {
    mocks.resolve.mockRejectedValueOnce(new Error("Owner unavailable"));
    render(<ClaimsReviewShell />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Retry account" }),
    );
    await screen.findByText("PRIVATE_A_CLAIM");
    expect(mocks.resolve).toHaveBeenCalledTimes(2);
  });
  it("refuses a resolved null owner", async () => {
    mocks.resolve.mockResolvedValueOnce(null);
    render(<ClaimsReviewShell />);
    await screen.findByRole("button", { name: "Retry account" });
    expect(mocks.request).not.toHaveBeenCalled();
  });
});

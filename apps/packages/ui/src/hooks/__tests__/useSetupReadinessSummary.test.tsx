// @vitest-environment jsdom
import React from "react";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const readinessMocks = vi.hoisted(() => ({
  getSetupReadinessProfiles: vi.fn(),
  getSetupReadinessStatus: vi.fn(),
}));

vi.mock("@/services/tldw/setup-readiness", () => ({
  getSetupReadinessProfiles: readinessMocks.getSetupReadinessProfiles,
  getSetupReadinessStatus: readinessMocks.getSetupReadinessStatus,
}));

const statusPayload = {
  readiness_status: "ready_with_warnings",
  lane_ids: ["chat", "embeddings_rag", "speech"],
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
};

const profilesPayload = {
  setup_access: { mode: "first_run" },
  lane_ids: ["chat", "embeddings_rag", "speech"],
  lanes: statusPayload.lanes,
  active_overlays: [],
  overlays: [],
  profiles: [],
};

describe("useSetupReadinessSummary", () => {
  beforeEach(() => {
    vi.resetModules();
    readinessMocks.getSetupReadinessProfiles.mockReset().mockResolvedValue(
      profilesPayload,
    );
    readinessMocks.getSetupReadinessStatus.mockReset().mockResolvedValue(
      statusPayload,
    );
  });

  afterEach(() => {
    cleanup();
  });

  it("loads first-run setup readiness status on mount", async () => {
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.status).toEqual(statusPayload));
    expect(readinessMocks.getSetupReadinessStatus).toHaveBeenCalledWith({
      mode: "first-run",
    });
    expect(readinessMocks.getSetupReadinessProfiles).toHaveBeenCalledWith({
      mode: "first-run",
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("loads first-run setup readiness status under React StrictMode", async () => {
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );
    const StrictModeWrapper = ({ children }: { children: React.ReactNode }) => (
      <React.StrictMode>{children}</React.StrictMode>
    );

    const { result } = renderHook(() => useSetupReadinessSummary(), {
      wrapper: StrictModeWrapper,
    });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.status).toEqual(statusPayload);
    expect(result.current.error).toBeNull();
  });

  it("uses profile lanes when first-run status only reports lane ids", async () => {
    readinessMocks.getSetupReadinessStatus.mockResolvedValueOnce({
      readiness_status: "ready_with_warnings",
      lane_ids: ["chat", "embeddings_rag", "speech"],
      active_overlays: [],
      overlays: [],
    });
    readinessMocks.getSetupReadinessProfiles.mockResolvedValueOnce({
      ...profilesPayload,
      active_overlays: ["downloads_disabled"],
      overlays: ["network_unavailable"],
    });
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());

    await waitFor(() =>
      expect(result.current.status?.lanes).toEqual(statusPayload.lanes),
    );
    expect(result.current.status).toMatchObject({
      readiness_status: "ready_with_warnings",
      lane_ids: ["chat", "embeddings_rag", "speech"],
      active_overlays: ["downloads_disabled"],
      overlays: ["network_unavailable"],
    });
  });

  it("keeps authoritative status when profile enrichment fails", async () => {
    const authoritativeStatus = {
      readiness_status: "ready_with_warnings",
      lane_ids: ["chat"],
      active_overlays: [],
      overlays: [],
    };
    readinessMocks.getSetupReadinessStatus.mockResolvedValueOnce(
      authoritativeStatus,
    );
    readinessMocks.getSetupReadinessProfiles.mockRejectedValueOnce(
      new Error("profile enrichment failed"),
    );
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());

    await waitFor(() =>
      expect(result.current.status).toEqual(authoritativeStatus),
    );
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("exposes refresh to reload readiness status", async () => {
    const refreshedPayload = {
      ...statusPayload,
      readiness_status: "ready",
    };
    readinessMocks.getSetupReadinessStatus
      .mockResolvedValueOnce(statusPayload)
      .mockResolvedValueOnce(refreshedPayload);
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());
    await waitFor(() => expect(result.current.status).toEqual(statusPayload));

    await act(async () => {
      await result.current.refresh();
    });

    expect(result.current.status).toEqual(refreshedPayload);
    expect(readinessMocks.getSetupReadinessStatus).toHaveBeenCalledTimes(2);
    expect(readinessMocks.getSetupReadinessProfiles).toHaveBeenCalledTimes(2);
  });

  it("exposes sanitized fallback error copy on failure", async () => {
    readinessMocks.getSetupReadinessStatus.mockRejectedValueOnce(
      new Error("Traceback: stack secret"),
    );
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());

    await waitFor(() =>
      expect(result.current.error).toBe(
        "Setup readiness could not be loaded.",
      ),
    );
    expect(result.current.status).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("keeps newer refresh results when older requests resolve later", async () => {
    let resolveInitial: (value: typeof statusPayload) => void = () => undefined;
    let resolveOlder: (value: typeof statusPayload) => void = () => undefined;
    let resolveNewer: (value: typeof statusPayload) => void = () => undefined;
    const olderPayload = {
      ...statusPayload,
      readiness_status: "older_result",
    };
    const newerPayload = {
      ...statusPayload,
      readiness_status: "newer_result",
    };
    readinessMocks.getSetupReadinessStatus
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveInitial = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOlder = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveNewer = resolve;
          }),
      );
    const { useSetupReadinessSummary } = await import(
      "../useSetupReadinessSummary"
    );

    const { result } = renderHook(() => useSetupReadinessSummary());
    await act(async () => {
      resolveInitial(statusPayload);
    });
    await waitFor(() => expect(result.current.status).toEqual(statusPayload));

    let olderRefresh: Promise<unknown> = Promise.resolve();
    let newerRefresh: Promise<unknown> = Promise.resolve();
    await act(async () => {
      olderRefresh = result.current.refresh();
      newerRefresh = result.current.refresh();
    });
    await act(async () => {
      resolveNewer(newerPayload);
      await newerRefresh;
    });
    expect(result.current.status).toEqual(newerPayload);

    await act(async () => {
      resolveOlder(olderPayload);
      await olderRefresh;
    });

    expect(result.current.status).toEqual(newerPayload);
  });
});

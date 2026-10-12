import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DocumentWorkerClient } from "@/services/document-editing/worker-client";
import {
  createCheckResult,
  createDocumentRequest,
  createFormatResult,
} from "@/services/document-editing/types";
import { useDocumentDiagnostics } from "../hooks/useDocumentDiagnostics";

const mocks = vi.hoisted(() => ({
  client: null as DocumentWorkerClient | null,
}));
vi.mock("@/services/document-editing/app-service", () => ({
  getDocumentWorkerClient: () => mocks.client,
}));
const request = (revision = 0, document = "a", language = "json") =>
  createDocumentRequest({
    key: {
      authority_id: "alice",
      document_id: document,
      session_generation: 1,
      draft_revision: revision,
      language_revision: 0,
      request_id: revision,
    },
    text: '{"a":1}',
    language,
  });
const checked = (r: ReturnType<typeof request>) =>
  createCheckResult(r, {
    state: "checked",
    complete: true,
    diagnostics: [],
    truncated: false,
    format_allowed: true,
  });

describe("mounted document diagnostics", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mocks.client = new DocumentWorkerClient();
  });
  afterEach(() => {
    mocks.client?.close();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  it("retires successful status immediately and checks after 400ms inactivity", async () => {
    const check = vi
      .spyOn(mocks.client!, "check")
      .mockImplementation(async (r) => checked(r));
    const r = request();
    const view = renderHook(({ r }) => useDocumentDiagnostics(r, false), {
      initialProps: { r },
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(view.result.current.status).toBe("checked");
    expect(view.result.current.announce).toBe(false);
    act(() => view.result.current.checkNow());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(view.result.current.announce).toBe(true);
    view.rerender({ r: request(1) });
    expect(view.result.current.status).toBe("pending");
    expect(view.result.current.diagnostics).toEqual([]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(399);
    });
    expect(check).toHaveBeenCalledTimes(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(check).toHaveBeenCalledTimes(3);
    expect(view.result.current.announce).toBe(false);
  });
  it("cannot deliver an old result to another document or restored authority", async () => {
    let finish!: (value: ReturnType<typeof checked>) => void;
    vi.spyOn(mocks.client!, "check")
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      )
      .mockImplementation(async (r) => checked(r));
    const r = request();
    const view = renderHook(({ r }) => useDocumentDiagnostics(r, false), {
      initialProps: { r: r as typeof r | null },
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    view.rerender({ r: null });
    view.rerender({ r: request(0, "b") });
    await act(async () => {
      finish(checked(r));
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(view.result.current.result?.key.document_id).toBe("b");
  });
  it("defers initial and manual checks throughout native composition", async () => {
    const check = vi
      .spyOn(mocks.client!, "check")
      .mockImplementation(async (r) => checked(r));
    const r = request();
    const view = renderHook(
      ({ composing }) => useDocumentDiagnostics(r, composing),
      { initialProps: { composing: true } },
    );
    act(() => view.result.current.checkNow());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(check).not.toHaveBeenCalled();
    view.rerender({ composing: false });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(check).toHaveBeenCalledOnce();
  });
  it("does not announce a proposal as formatted when the native admission refuses it", async () => {
    const r = request();
    vi.spyOn(mocks.client!, "format").mockImplementation(async (r) =>
      createFormatResult({
        key: r.key,
        state: "formatted",
        replacement: '{\n  "a": 1\n}',
        check: checked(r),
      }),
    );
    const view = renderHook(() => useDocumentDiagnostics(r, false));
    let pending!: Promise<unknown>;
    act(() => {
      pending = view.result.current.format(() => false);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
      expect(await pending).toBeNull();
    });
    expect(view.result.current.status).toBe("blocked");
  });
});

describe("format authority retirement", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mocks.client = new DocumentWorkerClient();
  });
  afterEach(() => {
    mocks.client?.close();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  it.each(["edit", "composition", "inactive", "navigation"])(
    "refuses a late format after %s",
    async (reason) => {
      let finish!: (value: ReturnType<typeof createFormatResult>) => void;
      const r = request();
      vi.spyOn(mocks.client!, "format").mockImplementation(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      );
      const apply = vi.fn(() => true);
      const view = renderHook(
        ({ r, composing }) => useDocumentDiagnostics(r, composing),
        { initialProps: { r: r as typeof r | null, composing: false } },
      );
      let pending!: Promise<unknown>;
      act(() => {
        pending = view.result.current.format(apply);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      view.rerender({
        r:
          reason === "inactive"
            ? null
            : reason === "navigation"
              ? request(0, "b")
              : reason === "edit"
                ? request(1)
                : r,
        composing: reason === "composition",
      });
      await act(async () => {
        finish(
          createFormatResult({
            key: r.key,
            state: "formatted",
            replacement: "{}",
            check: checked(r),
          }),
        );
        expect(await pending).toBeNull();
      });
      expect(apply).not.toHaveBeenCalled();
      expect(view.result.current.status).not.toBe("formatted");
    },
  );
});

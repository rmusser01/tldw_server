import { afterEach, expect, it, vi } from "vitest";
import { requestFor } from "./request-fixture";
import {
  createCheckResult,
  createFormatResult,
  type DocumentRequest,
} from "../types";

function controlled() {
  const requests: DocumentRequest[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const check = async (request: DocumentRequest) => {
    requests.push(request);
    if (requests.length === 1) await gate;
    return createCheckResult(request, {
      state: "checked",
      complete: true,
      diagnostics: [],
      truncated: false,
      format_allowed: true,
    });
  };
  return {
    requests,
    release: () => release(),
    check,
    cancel: vi.fn(),
    format: async (request: DocumentRequest) =>
      createFormatResult({
        key: request.key,
        state: "unchanged",
        replacement: null,
        check: await check(request),
      }),
  };
}
afterEach(() => vi.useRealTimers());

it("debounces400ms and keeps only latest pending revision", async () => {
  const { DocumentCheckScheduler } = await import("../scheduler");
  vi.useFakeTimers();
  const worker = controlled();
  const results = vi.fn();
  const scheduler = new DocumentCheckScheduler(worker, results);
  scheduler.submit(requestFor("{}", "json", { draft_revision: 1 }));
  await vi.advanceTimersByTimeAsync(399);
  expect(worker.requests).toHaveLength(0);
  await vi.advanceTimersByTimeAsync(1);
  for (let revision = 2; revision < 50; revision++)
    scheduler.submit(requestFor("{}", "json", { draft_revision: revision }));
  worker.release();
  await vi.advanceTimersByTimeAsync(400);
  expect(worker.requests.map((request) => request.key.draft_revision)).toEqual([
    1, 49,
  ]);
  expect(results).toHaveBeenCalledTimes(1);
  scheduler.close();
});

it("retires identical keys permanently and settles superseded formatting", async () => {
  const { DocumentCheckScheduler } = await import("../scheduler");
  vi.useFakeTimers();
  const worker = controlled();
  const scheduler = new DocumentCheckScheduler(worker, vi.fn());
  const request = requestFor("{}");
  scheduler.submit(request, true);
  await vi.advanceTimersByTimeAsync(0);
  const format = scheduler.format(request);
  scheduler.retire();
  scheduler.submit(request, true);
  worker.release();
  await vi.advanceTimersByTimeAsync(500);
  expect(await format).toBeNull();
  expect(worker.requests).toHaveLength(1);
  expect(worker.cancel).toHaveBeenCalledOnce();
  scheduler.close();
});

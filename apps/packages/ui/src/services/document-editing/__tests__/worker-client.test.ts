import { describe, expect, it, vi } from "vitest";
import { requestFor } from "./request-fixture";

describe("worker boundary", () => {
  it.each([
    ["x".repeat(1024 * 1024 + 1), "resource_limited"],
    ["\ud800", "unsupported"],
  ])("refuses unsupported input before dispatch", async (source, state) => {
    const { DocumentWorkerClient } = await import("../worker-client");
    const construct = vi.fn(() => {
      throw new Error("must not start");
    });
    vi.stubGlobal("Worker", construct);
    const client = new DocumentWorkerClient();
    const result = await client.check(requestFor(source));
    expect(result.state).toBe(state);
    expect(result.complete).toBe(false);
    expect(construct).not.toHaveBeenCalled();
    client.close();
    vi.unstubAllGlobals();
  });
  it("rejects an excerpt and contains startup failure", async () => {
    const { DocumentWorkerClient } = await import("../worker-client");
    const client = new DocumentWorkerClient();
    expect(
      (await client.check(requestFor("{}", "json", { is_excerpt: true })))
        .state,
    ).toBe("unsupported");
    vi.stubGlobal(
      "Worker",
      class {
        constructor() {
          throw new Error("PRIVATE_SOURCE");
        }
      },
    );
    const result = await client.check(requestFor("{}"));
    expect(result.state).toBe("failed");
    expect(JSON.stringify(result)).not.toContain("PRIVATE_SOURCE");
    client.close();
    vi.unstubAllGlobals();
  });
});

it("bounds escaped JSON frames without lowering the source limit", async () => {
  const { admit, encode } = await import("../worker-client");
  const request = requestFor("\x00".repeat(1024 * 1024));
  expect(admit(request)).toBeNull();
  expect(encode({ operation: "check", request }).byteLength).toBeGreaterThan(
    6 * 1024 * 1024,
  );
});

it("validates reply identity, types, ranges and completion without trusting prose", async () => {
  const { decodeResult } = await import("../worker-client");
  const { checkDocument } = await import("../engine");
  const request = requestFor("{}");
  const base = checkDocument(request);
  for (const override of [
    { key: {} },
    { key: { ...base.key, draft_revision: true } },
    { complete: 1 },
    { truncated: true },
    { diagnostics: Array(101).fill({}) },
    { state: "other" },
    {
      diagnostics: [
        {
          code: "syntax",
          severity: "error",
          message: "PRIVATE_SOURCE",
          start: 0,
          end: 3,
        },
      ],
    },
  ])
    expect(() =>
      decodeResult(request, "check", { ...base, ...override }),
    ).toThrow();
  const result = decodeResult(request, "check", {
    ...base,
    format_allowed: false,
    diagnostics: [
      {
        code: "syntax",
        severity: "error",
        message: "PRIVATE_SOURCE",
        start: 0,
        end: 1,
      },
    ],
  });
  expect(JSON.stringify(result)).not.toContain("PRIVATE_SOURCE");
  expect(() =>
    decodeResult(request, "format", {
      key: request.key,
      state: "formatted",
      replacement: "\ud800",
      check: base,
    }),
  ).toThrow();
});

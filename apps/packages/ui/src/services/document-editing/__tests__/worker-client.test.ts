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

it.each([
  ["json", false, "resource_limited"],
  ["json", true, "unsupported"],
  ["future-mode", false, "unsupported"],
] as const)(
  "refuses %s/excerpt=%s without scanning rejected source or starting a worker",
  async (language, isExcerpt, state) => {
    const { DocumentWorkerClient } = await import("../worker-client");
    const source = "x".repeat(1024 * 1024 + 1);
    const request = requestFor(source, language, { is_excerpt: isExcerpt });
    const factory = vi.fn(() => {
      throw new Error("Worker must not start");
    });
    const client = new DocumentWorkerClient(factory);
    const iterator = String.prototype[Symbol.iterator];
    const codePointAt = String.prototype.codePointAt;
    const charCodeAt = String.prototype.charCodeAt;
    const encode = TextEncoder.prototype.encode;
    const visited: string[] = [];
    vi.spyOn(String.prototype, Symbol.iterator).mockImplementation(function (
      this: string,
    ) {
      if (String(this) === source) visited.push("iterator");
      return iterator.call(this);
    });
    vi.spyOn(String.prototype, "codePointAt").mockImplementation(function (
      this: string,
      offset: number,
    ) {
      if (String(this) === source) visited.push("codePointAt");
      return codePointAt.call(this, offset);
    });
    vi.spyOn(String.prototype, "charCodeAt").mockImplementation(function (
      this: string,
      offset: number,
    ) {
      if (String(this) === source) visited.push("charCodeAt");
      return charCodeAt.call(this, offset);
    });
    vi.spyOn(TextEncoder.prototype, "encode").mockImplementation(function (
      this: TextEncoder,
      input?: string,
    ) {
      if (input === source) visited.push("encode");
      return encode.call(this, input);
    });
    try {
      expect((await client.check(request)).state).toBe(state);
      expect((await client.format(request)).state).toBe(
        state === "unsupported" ? "blocked" : state,
      );
      expect(factory).not.toHaveBeenCalled();
      expect(visited).toEqual([]);
    } finally {
      client.close();
      vi.restoreAllMocks();
    }
  },
);

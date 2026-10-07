import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  createCheckResult,
  createDiagnostic,
  createFormatResult,
  type CheckResult,
} from "../types";
import { requestFor } from "./request-fixture";

const checked = (
  request = requestFor("{}"),
  overrides: Partial<Omit<CheckResult, "key">> = {},
) =>
  createCheckResult(request, {
    state: "checked",
    complete: true,
    diagnostics: [],
    truncated: false,
    format_allowed: true,
    ...overrides,
  });

describe("document contracts", () => {
  it.each([
    [-1, 0],
    [2, 1],
  ])("rejects range %i:%i", (start, end) => {
    expect(() =>
      createDiagnostic({
        code: "syntax",
        severity: "error",
        message: "Invalid",
        start,
        end,
      }),
    ).toThrow();
  });
  it("keeps independent immutable request keys and future modes", () => {
    const first = requestFor("x", "future", {
      draft_revision: 2,
      newline: "\r\n",
    });
    expect(first.key.request_id).not.toBe(requestFor("x").key.request_id);
    expect(first.language).toBe("future");
    expect(first.key.draft_revision).toBe(2);
    expect(Object.isFrozen(first.key)).toBe(true);
    expect(Object.isFrozen(first)).toBe(true);
  });
  it.each([
    { complete: false },
    { truncated: true },
    { state: "unsupported" as const },
    {
      diagnostics: [
        {
          code: "syntax",
          severity: "error" as const,
          message: "Invalid",
          start: 0,
          end: 1,
        },
      ],
    },
    {
      diagnostics: [
        {
          code: "duplicate_key",
          severity: "warning" as const,
          message: "Duplicate",
          start: 0,
          end: 1,
        },
      ],
    },
  ])("rejects false formatting admission %j", (fields) => {
    expect(() => checked(requestFor("{}"), fields)).toThrow();
  });
  it.each(["unsupported", "resource_limited", "failed"] as const)(
    "rejects complete %s",
    (state) => {
      expect(() =>
        checked(requestFor("{}"), { state, format_allowed: false }),
      ).toThrow();
    },
  );
  it("refuses excerpt success and ranges beyond scalar source length", () => {
    expect(() =>
      checked(requestFor("{}", "json", { is_excerpt: true }), {
        format_allowed: false,
      }),
    ).toThrow();
    expect(() =>
      checked(requestFor("😀"), {
        format_allowed: false,
        diagnostics: [
          {
            code: "syntax",
            severity: "error",
            message: "Invalid",
            start: 0,
            end: 2,
          },
        ],
      }),
    ).toThrow();
  });
  it.each(["blocked", "unchanged", "resource_limited", "failed"] as const)(
    "rejects replacement on %s",
    (state) => {
      const req = requestFor("{}");
      expect(() =>
        createFormatResult({
          key: req.key,
          state,
          replacement: "{ }",
          check: checked(req),
        }),
      ).toThrow();
    },
  );
  it("requires authoritative check and matching identity", () => {
    const req = requestFor("{}");
    expect(() =>
      createFormatResult({
        key: req.key,
        state: "formatted",
        replacement: "{ }",
        check: checked(req, { format_allowed: false }),
      }),
    ).toThrow();
    expect(() =>
      createFormatResult({
        key: requestFor("{}").key,
        state: "formatted",
        replacement: "{ }",
        check: checked(req),
      }),
    ).toThrow();
    expect(() =>
      createFormatResult({
        key: req.key,
        state: "formatted",
        replacement: null,
        check: checked(req),
      }),
    ).toThrow();
    expect(
      createFormatResult({
        key: { ...req.key },
        state: "formatted",
        replacement: "",
        check: checked(req),
      }).replacement,
    ).toBe("");
    expect(
      createFormatResult({
        key: req.key,
        state: "unchanged",
        replacement: null,
        check: checked(req),
      }).state,
    ).toBe("unchanged");
  });
  it("validates shared fixture schema without parsing source", () => {
    const raw = readFileSync(
      "src/services/document-editing/__tests__/fixtures/v1/cases.json",
    );
    expect(createHash("sha256").update(raw).digest("hex")).toBe(
      "fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7",
    );
    const cases: {
      id: string;
      language: string;
      text: string;
      valid: boolean;
      codes: string[];
      format: string;
      must_keep: string[];
    }[] = JSON.parse(raw.toString());
    expect(new Set(cases.map((c) => c.id)).size).toBe(cases.length);
    for (const c of cases) {
      expect(Object.keys(c).sort()).toEqual(
        [
          "id",
          "language",
          "text",
          "valid",
          "codes",
          "format",
          "must_keep",
        ].sort(),
      );
      expect(["json", "jsonl", "yaml"]).toContain(c.language);
      expect(typeof c.id).toBe("string");
      expect(c.id.length).toBeGreaterThan(0);
      expect(typeof c.text).toBe("string");
      expect(
        Array.isArray(c.codes) &&
          c.codes.every((code) => typeof code === "string"),
      ).toBe(true);
      expect(
        Array.isArray(c.must_keep) &&
          c.must_keep.every((token) => typeof token === "string"),
      ).toBe(true);
      expect(typeof c.valid).toBe("boolean");
      expect(["allowed", "blocked"]).toContain(c.format);
      for (const token of c.must_keep) expect(c.text).toContain(token);
      if (c.codes.includes("unsupported")) {
        expect(c.valid).toBe(true);
        expect(c.codes).not.toContain("syntax");
        expect(c.format).toBe("blocked");
      }
      if (!c.valid || c.codes.length) expect(c.format).toBe("blocked");
    }
    expect(cases.find((c) => c.id === "json-escaped-duplicate")?.text).toBe(
      '{"a":1,"\\u0061":2}',
    );
    expect(cases.find((c) => c.id === "jsonl-blank")?.text).toBe("{}\n\n{}\n");
    expect(cases.find((c) => c.id === "jsonl-empty")?.text).toBe("");
    console.log(
      "fixture v1 SHA-256:",
      createHash("sha256").update(raw).digest("hex"),
    );
  });
});

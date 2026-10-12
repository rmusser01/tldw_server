import { expect, it, vi } from "vitest";
import { checkDocument, formatDocument } from "../engine";
import { requestFor } from "./request-fixture";
import cases from "./fixtures/v1/cases.json";

it("formats source without rounding a JSON number", () => {
  const result = formatDocument(requestFor('{"n":9007199254740993,"z":-0}'));
  expect(result.state).toBe("formatted");
  expect(result.replacement).toContain("9007199254740993");
  expect(result.replacement).toContain('"z": -0');
  expect(formatDocument(requestFor(result.replacement!)).state).toBe(
    "unchanged",
  );
});
it.each(cases.filter((c) => c.language !== "yaml"))("shared $id", (c) => {
  const req = requestFor(c.text, c.language);
  const check = checkDocument(req);
  expect(check.state).toBe("checked");
  expect(!check.diagnostics.some((d) => d.severity === "error")).toBe(c.valid);
  expect([...new Set(check.diagnostics.map((d) => d.code))]).toEqual(c.codes);
  expect(check.format_allowed).toBe(c.format === "allowed");
  const result = formatDocument(req);
  if (c.format === "blocked") expect(result.state).toBe("blocked");
  else {
    expect(["formatted", "unchanged"]).toContain(result.state);
    const output = result.replacement ?? c.text;
    for (const token of c.must_keep) expect(output).toContain(token);
    expect(formatDocument(requestFor(output, c.language)).state).toBe(
      "unchanged",
    );
  }
});
it.each(['{"a":1,"\\u0061":2}', '{"__proto__":1,"__proto__":2}'])(
  "compares decoded own keys: %s",
  (text) => {
    expect(checkDocument(requestFor(text)).diagnostics).toEqual([
      expect.objectContaining({
        code: "duplicate_key",
        severity: "warning",
        start: text.indexOf(",") + 1,
        end: text.lastIndexOf(":"),
      }),
    ]);
  },
);
it.each([
  "01",
  "+1",
  ".1",
  "1.",
  "1e",
  "undefined",
  "[1,]",
  '"bad\tstring"',
  "/*x*/1",
])("rejects strict JSON %s", (text) => {
  expect(
    checkDocument(requestFor(text)).diagnostics.some(
      (d) => d.code === "syntax",
    ),
  ).toBe(true);
  expect(formatDocument(requestFor(text)).replacement).toBeNull();
});
it("compacts JSONL independently with exact CRLF and source tokens", () => {
  const text =
    ' { "n" : 1e400, "s" : "\\u0061", "v":-0 } \r\n [ 9007199254740993, 0.1234567890123456789 ]';
  const expected =
    '{"n":1e400,"s":"\\u0061","v":-0}\r\n[9007199254740993,0.1234567890123456789]';
  expect(
    formatDocument(requestFor(text, "jsonl", { newline: "\r\n" })).replacement,
  ).toBe(expected);
});
it.each([true, false])(
  "refuses visible or metadata JSONL BOM %s",
  (metadata) => {
    const check = checkDocument(
      requestFor(metadata ? "{}" : "\ufeff{}", "jsonl", { has_bom: metadata }),
    );
    expect(check.diagnostics.some((d) => d.code === "syntax")).toBe(true);
    expect(check.format_allowed).toBe(false);
  },
);
it("preserves JSON final-newline presence and requested newline", () => {
  expect(
    formatDocument(requestFor('{"a":1}\r\n', "json", { newline: "\r\n" }))
      .replacement,
  ).toBe('{\r\n  "a": 1\r\n}\r\n');
});
it.each(["json", "yaml"])(
  "bounds %s size and depth before construction",
  (language) => {
    expect(
      checkDocument(requestFor('"' + "😀".repeat(262144) + '"', language))
        .state,
    ).toBe("resource_limited");
    expect(
      checkDocument(
        requestFor("[".repeat(101) + "0" + "]".repeat(101), language),
      ).state,
    ).toBe("resource_limited");
    expect(
      checkDocument(
        requestFor("[".repeat(100) + "0" + "]".repeat(100), language),
      ).state,
    ).toBe("checked");
  },
);
it("caps diagnostics without advertising completeness", () => {
  const check = checkDocument(requestFor("\n".repeat(102), "jsonl"));
  expect(check.diagnostics).toHaveLength(100);
  expect(check.truncated).toBe(true);
  expect(check.complete).toBe(false);
  expect(check.format_allowed).toBe(false);
});
it.each([null, "text", "markdown", "future", "__proto__"])(
  "does not parse unsupported %s",
  (language) => {
    expect(checkDocument(requestFor("not json", language)).state).toBe(
      "unsupported",
    );
  },
);
it("refuses excerpts and honors a cooperative deadline", () => {
  expect(
    checkDocument(requestFor("{}", "json", { is_excerpt: true })).state,
  ).toBe("unsupported");
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => (now += 1001));
  expect(checkDocument(requestFor("{}")).state).toBe("resource_limited");
  vi.restoreAllMocks();
});
it("preserves lexemes across generated nested inputs", () => {
  for (let i = 0; i < 80; i++) {
    const text =
      '{ "a' +
      i +
      '" : [1e400,-0,9007199254740993,{"s":"\\u0061😀"}],"b":true}';
    const result = formatDocument(requestFor(text));
    expect(result.state).toBe("formatted");
    expect(result.replacement).toContain('"s": "\\u0061😀"');
    expect(formatDocument(requestFor(result.replacement!)).state).toBe(
      "unchanged",
    );
  }
});
it("catches a formatter changing an authoritative source token", async () => {
  const parser = await import("jsonc-parser");
  vi.spyOn(parser, "applyEdits").mockReturnValue('{"n":9007199254740992}');
  const result = formatDocument(requestFor('{"n":9007199254740993}'));
  expect(result.state).toBe("failed");
  expect(result.replacement).toBeNull();
  expect(result.check.diagnostics[0].message).not.toContain("9007199254740993");
  vi.restoreAllMocks();
});
it("fails safely on an unexpected parser exception", async () => {
  const parser = await import("jsonc-parser");
  vi.spyOn(parser, "visit").mockImplementation(() => {
    throw new Error("private_source_secret");
  });
  const result = checkDocument(requestFor("{}"));
  expect(result.state).toBe("failed");
  expect(JSON.stringify(result)).not.toContain("private_source_secret");
  vi.restoreAllMocks();
});

it.each(["json", "yaml"])(
  "independently refuses unchecked %s format above byte/depth limits",
  (language) => {
    for (const text of [
      '"' + "😀".repeat(262144) + '"',
      "[".repeat(101) + "0" + "]".repeat(101),
    ]) {
      const request = requestFor(text, language);
      const result = formatDocument(request);
      expect(result.state).toBe("resource_limited");
      expect(result.replacement).toBeNull();
      expect(result.check.format_allowed).toBe(false);
      expect(request.text).toBe(text);
    }
  },
);

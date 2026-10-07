import { expect, it } from "vitest";
import { scalarOffset } from "../source-positions";
import { checkDocument } from "../engine";
import { requestFor } from "./request-fixture";
it("counts emoji once, combining marks separately, and tabs/CRLF literally", () => {
  const source = "😀e\u0301\t\r\nx";
  expect([0, 2, 3, 4, 5, 6, 7, 8].map((n) => scalarOffset(source, n))).toEqual([
    0, 1, 2, 3, 4, 5, 6, 7,
  ]);
});
it.each([
  ['{"😀":1,"é":\t?}', 14, 15],
  ['{"😀":', 5, 5],
  ['{}\r\n{"😀":?}\r\n', 9, 10],
] as const)("reports exact scalar positions for %s", (text, start, end) => {
  const result = checkDocument(
    requestFor(text, text.startsWith("{}") ? "jsonl" : "json"),
  );
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({ code: "syntax", start, end }),
  );
});
it("points to the repeated YAML key after emoji, CRLF, and a combining mark", () => {
  const result = checkDocument(
    requestFor("emoji: 😀\r\né: 1\r\né: 2\r\n", "yaml"),
  );
  expect(result.diagnostics).toContainEqual(
    expect.objectContaining({ code: "duplicate_key", start: 17, end: 19 }),
  );
});

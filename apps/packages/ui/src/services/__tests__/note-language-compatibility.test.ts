import { describe, expect, it } from "vitest";
import { decodeNotesLanguageCapability } from "../note-language-capability";

describe("explicit Notes language contract", () => {
  it.each([
    null,
    undefined,
    {},
    [],
    true,
    { content_language: "true" },
    { content_language: 1 },
    { content_language: false },
  ])("refuses inferred support: %j", (value) => {
    expect(decodeNotesLanguageCapability(value)).toEqual({
      available: false,
      reason: "unsupported_server",
    });
  });
  it("accepts the reviewed authenticated server response", () => {
    expect(decodeNotesLanguageCapability({ content_language: true })).toEqual({
      available: true,
      reason: null,
    });
  });
});

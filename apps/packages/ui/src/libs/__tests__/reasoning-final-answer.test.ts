import { describe, expect, it } from "vitest";
import {
  isReasoningOnlyResponse,
  parseReasoning,
  removeReasoning,
  replaceThinkTagToEM,
} from "../reasoning";

describe("settled reasoning without a final answer", () => {
  it.each([
    "<think>Thinking</think>",
    "<think>Thinking",
    "<reason>Thinking</reason><thought>More thinking</thought>",
    "<THINK></THINK>"
  ])("requires recovery for %s", content => {
    expect(isReasoningOnlyResponse(content)).toBe(true)
  })

  it.each([
    "<think>Thinking</think>Final answer",
    "Final answer<think>Thinking</think>",
    "Ordinary prose",
    "<thinker>Ordinary prose</thinker>",
    ""
  ])("leaves normal or unclassified content alone: %s", content => {
    expect(isReasoningOnlyResponse(content)).toBe(false)
  })
})


describe("nested reasoning boundaries", () => {
  const aliases = ["think", "reason", "reasoning", "thought"];
  it.each(aliases.flatMap((outer) => aliases.map((inner) => [outer, inner])))(
    "keeps nested %s/%s trace out of the final answer",
    (outer, inner) => {
      const content = `<${outer}>outer <${inner}>inner</${inner}> tail</${outer}>FINAL`;
      expect(
        parseReasoning(content)
          .filter((part) => part.type === "text")
          .map((part) => part.content),
      ).toEqual(["FINAL"]);
    },
  );
  it("does not invent a final answer from nested reasoning tail", () => {
    expect(
      isReasoningOnlyResponse("<think>outer <think>inner</think> tail</think>"),
    ).toBe(true);
  });
  it("keeps unfinished nested reasoning out of final-only output", () => {
    expect(
      removeReasoning("PREFIX <think>outer <reason>inner</reason> unfinished"),
    ).toBe("PREFIX");
  });
  it("preserves surrounding whitespace when removing nested reasoning", () => {
    expect(
      removeReasoning(
        "before <think>outer <think>inner</think> tail</think> after",
      ),
    ).toBe("before  after");
  });
  it("does not insert whitespace around adjacent final text around nested reasoning", () => {
    expect(
      removeReasoning("A<think>outer <think>inner</think> tail</think>B"),
    ).toBe("AB");
  });
  it("recognizes nested mixed-case aliases", () => {
    expect(
      removeReasoning("<THINK>outer <Reason>inner</rEaSoN> tail</think>FINAL"),
    ).toBe("FINAL");
  });
  it("keeps the entire nested trace inside HTML emphasis", () => {
    expect(
      replaceThinkTagToEM(
        "<think>outer <reason>inner</reason> tail</think>FINAL",
      ),
    ).toBe(
      '<em style="font-style: italic; font-size: 0.9em; margin-bottom: 1em;">outer inner tail</em>\n\nFINAL',
    );
  });
});


describe("orphan reasoning delimiter boundaries", () => {
  it.each(["think", "reason", "reasoning", "thought"])(
    "excludes orphan %s closing markup from final text",
    (tag) => {
      expect(
        parseReasoning(`<${tag}>brief check</${tag}>FINAL</${tag}>`)
          .filter((part) => part.type === "text")
          .map((part) => part.content),
      ).toEqual(["FINAL"]);
    },
  );
  it.each(["think", "reason", "reasoning", "thought"])(
    "removes orphan %s delimiters from final-only output",
    (tag) => {
      expect(removeReasoning(`before </${tag}> after`)).toBe("before  after");
    },
  );
  it.each(["think", "reason", "reasoning", "thought"])(
    "does not treat orphan %s tokens as a final answer",
    (tag) => {
      expect(isReasoningOnlyResponse(`</${tag}>`)).toBe(true);
    },
  );
  it("preserves unrecognized closing markup as ordinary text", () => {
    expect(removeReasoning("literal </thinker> text")).toBe(
      "literal </thinker> text",
    );
  });
});

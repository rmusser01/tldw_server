import { describe, expect, it } from "vitest";
import { parseGeneratedBoardDraft } from "../kanban-generation";

const valid = () => ({
  board: { name: "Rowan", description: "Original source" },
  lists: [
    {
      name: "To Do",
      cards: [{ title: "Calibrate Selene", description: "Friday at 18:00" }],
    },
  ],
});
const response = (content: string) => ({
  choices: [
    {
      finish_reason: "stop",
      message: {
        content,
        reasoning_content: "Provider metadata is not draft content",
      },
    },
  ],
});

describe("generated Kanban draft admission", () => {
  it.each(["length", "content_filter"])(
    "refuses otherwise-valid JSON after %s termination",
    (finishReason) => {
      const reply = {
        choices: [
          {
            finish_reason: finishReason,
            message: { content: JSON.stringify(valid()) },
          },
        ],
      };
      expect(() => parseGeneratedBoardDraft(reply)).toThrow(
        "invalid board draft",
      );
    },
  );
  it.each([false, true])(
    "accepts a complete strict draft (fenced=%s)",
    (fenced) => {
      const draft = valid();
      const body = JSON.stringify(draft);
      expect(
        parseGeneratedBoardDraft(
          response(fenced ? "```json\n" + body + "\n```" : body),
        ),
      ).toEqual(draft);
    },
  );
  it("normalizes names while preserving actual description text", () => {
    const draft = valid();
    draft.board.name = "  Rowan  ";
    draft.lists[0].cards[0].description = "Line 1\nLine 2 <literal>";
    expect(
      parseGeneratedBoardDraft(response(JSON.stringify(draft))),
    ).toMatchObject({
      board: { name: "Rowan" },
      lists: [{ cards: [{ description: "Line 1\nLine 2 <literal>" }] }],
    });
  });
  it.each([
    "identity",
    "empty-lists",
    "empty-cards",
    "too-many-lists",
    "too-many-cards",
    "bad-title",
  ])("rejects %s", (change) => {
    const draft: {
      board: Record<string, unknown>;
      lists: { name: string; cards: Record<string, unknown>[] }[];
    } = valid();
    if (change === "identity") draft.lists[0].cards[0].id = "foreign";
    if (change === "empty-lists") draft.lists = [];
    if (change === "empty-cards") draft.lists[0].cards = [];
    if (change === "too-many-lists")
      draft.lists = Array.from({ length: 21 }, () => valid().lists[0]);
    if (change === "too-many-cards")
      draft.lists = [200, 200, 101].map((count) => ({
        name: "List",
        cards: Array.from({ length: count }, () => ({ title: "Card" })),
      }));
    if (change === "bad-title") draft.lists[0].cards[0].title = false;
    expect(() =>
      parseGeneratedBoardDraft(response(JSON.stringify(draft))),
    ).toThrow("invalid board draft");
  });
  it.each([
    "",
    "not json",
    "<think>reasoning</think>" + JSON.stringify(valid()),
    JSON.stringify(valid()) + " trailing commentary",
  ])("refuses incomplete or contaminated text", (content) => {
    expect(() => parseGeneratedBoardDraft(response(content))).toThrow(
      "invalid board draft",
    );
  });
  it("enforces the UTF-8 size cap before admitting output", () => {
    const draft = valid();
    draft.lists[0].cards = Array.from({ length: 27 }, () => ({
      title: "Card",
      description: "🦉".repeat(1200),
    }));
    const body = JSON.stringify(draft);
    expect(body.length).toBeLessThan(128_000);
    expect(new TextEncoder().encode(body).length).toBeGreaterThan(128_000);
    expect(() => parseGeneratedBoardDraft(response(body))).toThrow(
      "invalid board draft",
    );
  });
});

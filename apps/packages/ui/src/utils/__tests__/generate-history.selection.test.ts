import { describe, expect, it, vi } from "vitest"
vi.mock("@/db/dexie/models", () => ({ isCustomModel: () => false }))
import { generateHistory } from "../generate-history"

describe("canonical selected history", () => {
  it("preserves ordered images and same-text rows with distinct identities", () => {
    const history = generateHistory(
      [
        {
          id: "u1",
          role: "user",
          content: "same",
          images: ["data:first", "data:second"],
          image: "obsolete"
        },
        { id: "u2", role: "user", content: "same", image: "data:singular" }
      ],
      "test",
      { versioned: true }
    )
    expect(history.map((row) => row.content)).toEqual([
      [
        { type: "image_url", image_url: "data:first" },
        { type: "image_url", image_url: "data:second" },
        { type: "text", text: "same" }
      ],
      [
        { type: "image_url", image_url: "data:singular" },
        { type: "text", text: "same" }
      ]
    ])
  })
  it("keeps canonical tool roles and tool-call-only assistant rows", () => {
    const calls = [
      {
        id: "c1",
        type: "function",
        function: { name: "lookup", arguments: "{}" }
      }
    ]
    const history = generateHistory(
      [
        { id: "a1", role: "assistant", content: "", tool_calls: calls },
        { id: "t1", role: "tool", content: "found", tool_call_id: "c1" },
        { id: "s1", role: "system", content: "filtered" },
        {
          id: "i1",
          role: "assistant",
          content: "filtered",
          messageType: "image:assistant"
        }
      ],
      "test",
      { versioned: true }
    )
    expect(history.slice(0, 2).map((row) => row._getType())).toEqual([
      "ai",
      "tool"
    ])
    expect(history[0].additional_kwargs.tool_calls).toEqual(calls)
    expect(history[1]).toMatchObject({ tool_call_id: "c1", content: "found" })
    expect(history.some((row) => row._getType() === "system")).toBe(false)
  })
  it("rejects unsupported assistant images and unknown roles before dispatch", () => {
    expect(() =>
      generateHistory(
        [{ role: "assistant", content: "image", images: ["data:image"] }],
        "test",
        { versioned: true }
      )
    ).toThrow("unsupported_history_message_images")
    expect(() =>
      generateHistory([{ role: "observer", content: "hidden" }], "test", {
        versioned: true
      })
    ).toThrow("unsupported_history_message_role")
  })
  it("rejects function rows and missing tool identities explicitly", () => {
    expect(() =>
      generateHistory(
        [{ id: "f1", role: "function", content: "found" }],
        "test",
        { versioned: true }
      )
    ).toThrow("unsupported_history_function_message")
    expect(() =>
      generateHistory([{ id: "t1", role: "tool", content: "found" }], "test", {
        versioned: true
      })
    ).toThrow("missing_history_tool_call_id")
  })
})

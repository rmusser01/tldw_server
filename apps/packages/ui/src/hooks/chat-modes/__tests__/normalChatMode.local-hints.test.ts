import { describe, expect, it } from "vitest"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { generateHistory } from "@/utils/generate-history"
import { IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE } from "@/utils/image-generation-chat"

// Execute the actual canonical projection block against a declared capture.
// Existing mounted H1 suites qualify owner/admission/transport separately.
const source = readFileSync(
  path.resolve(__dirname, "../normalChatMode.ts"),
  "utf8"
)
const ast = ts.createSourceFile(
  "normalChatMode.ts",
  source,
  ts.ScriptTarget.Latest,
  true
)
let declaration: ts.VariableDeclaration | undefined
function visit(node: ts.Node) {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "selected" &&
    node.initializer?.getText(ast).startsWith("historyTurn.capture.rows.map")
  )
    declaration = node
  ts.forEachChild(node, visit)
}
visit(ast)
if (!declaration)
  throw new Error("Canonical selected-history projection missing")
const compiled = ts.transpileModule(
  "return " + declaration.initializer!.getText(ast),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
).outputText
const project = new Function("historyTurn", compiled)
const fixture = (kind: "native" | "local") => ({
  owner: { kind },
  capture: {
    rows: [
      { id: "u", role: "user" },
      { id: "a", role: "assistant" }
    ],
    selected_content: [
      { message: "Question", images: [] },
      {
        message: "Canonical answer",
        images: [],
        extra_metadata: {
          local_history: {
            messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
          }
        }
      }
    ]
  }
})

describe("canonical provider history and owned local hints", () => {
  it("does not let native private wrapper metadata skip canonical assistant content", () => {
    const turn = fixture("native"),
      before = structuredClone(turn)
    const selected = project(turn)
    expect(selected[1].messageType).toBeUndefined()
    expect(
      generateHistory(selected, "gpt-4o-mini", { versioned: true }).map(
        (row) => row.content
      )
    ).toEqual([
      [{ type: "text", text: "Question" }],
      [{ type: "text", text: "Canonical answer" }]
    ])
    expect(turn).toEqual(before)
  })
  it("preserves the existing local image-event exclusion", () => {
    const selected = project(fixture("local"))
    expect(selected[1].messageType).toBe(
      IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
    )
    expect(
      generateHistory(selected, "gpt-4o-mini", { versioned: true }).map(
        (row) => row.content
      )
    ).toEqual([[{ type: "text", text: "Question" }]])
  })
})

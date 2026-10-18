import "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import { describe, expect, it } from "vitest"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { excludeLocalRagDiagnostics } from "@/utils/local-rag-diagnostic"
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
const dependencies = new Map<string, ts.VariableDeclaration>()
let declaration: ts.VariableDeclaration | undefined
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && ["displayMessages", "inferenceIds"].includes(node.name.getText(ast)))
    dependencies.set(node.name.getText(ast), node)
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "selected" &&
    node.initializer?.getText(ast).startsWith("historyTurn.capture.rows.map")
  )
    declaration = node
  ts.forEachChild(node, visit)
}
visit(ast)
if (!declaration || ["displayMessages", "inferenceIds"].some(name => !dependencies.get(name)?.initializer))
  throw new Error("Canonical selected-history projection missing")
const compiled = ts.transpileModule(
  ["displayMessages", "inferenceIds"].map(name => "const " + dependencies.get(name)!.getText(ast) + ";").join("\n") + "\nreturn " + declaration.initializer!.getText(ast),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
).outputText
const runProjection = new Function("historyTurn", "formatSelectedHistory", "excludeLocalRagDiagnostics", compiled)
const project = (turn: ReturnType<typeof fixture>) => runProjection(turn, formatSelectedHistory, excludeLocalRagDiagnostics)
const fixture = (kind: "native" | "local") => ({
  owner: { kind },
  capture: {
    view: { conversation_id: "fixture" },
    snapshot: { owner_key: kind === "local" ? "local-history-v1:fixture" : "native-fixture", nodes: [] },
    rows: [
      { id: "u", role: "user", parent_id: null },
      { id: "a", role: "assistant", parent_id: "u" }
    ],
    selected_content: [
      { id: "u", message: "Question", images: [], extra_metadata: {} },
      {
        id: "a",
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
  it.each(["native", "local"] as const)("filters undispatched diagnostic pairs only for %s history", kind => {
    const turn = fixture(kind)
    const diagnostic = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    for (const row of turn.capture.selected_content)
      row.extra_metadata = { local_history: { generationInfo: diagnostic } } as typeof row.extra_metadata
    const before = structuredClone(turn)
    const selected = project(turn)
    expect(selected.map((row: { id: string }) => row.id)).toEqual(kind === "local" ? [] : ["u", "a"])
    expect(turn).toEqual(before)
  })

})

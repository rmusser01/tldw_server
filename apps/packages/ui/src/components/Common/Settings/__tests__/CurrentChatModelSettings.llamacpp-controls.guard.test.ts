import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("CurrentChatModelSettings llama.cpp controls guard", () => {
  it("keeps llama.cpp controls on the modal form/save path", () => {
    const sourcePath = path.resolve(
      __dirname,
      "../CurrentChatModelSettings.tsx"
    )
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('"llamaThinkingBudgetTokens"')
    expect(source).toContain('"llamaGrammarMode"')
    expect(source).toContain('"llamaGrammarId"')
    expect(source).toContain('"llamaGrammarInline"')
    expect(source).toContain('"llamaGrammarOverride"')
    expect(source).toContain('Form.useWatch("llamaThinkingBudgetTokens", form)')
    expect(source).toContain('Form.useWatch("llamaGrammarMode", form)')
    expect(source).toContain('Form.useWatch("llamaGrammarId", form)')
    expect(source).toContain('Form.useWatch("llamaGrammarInline", form)')
    expect(source).toContain('Form.useWatch("llamaGrammarOverride", form)')
    expect(source).toContain('Form.useWatch("extraBody", form)')
    expect(source).toContain("handleLlamaControlChange")
    expect(source).toContain("settingsScope?: string | null")
    expect(source).toContain("explicitSettingsScope")
    expect(source).toContain("selectedModelSettingsScope")
    expect(source).toContain("latestSettingsState.activeSettingsScope")
    expect(source).toContain("updateScopedSetting(targetSettingsScope")
    expect(source).toContain("setActiveSettingsScope(targetSettingsScope)")
    expect(source).toContain("thinkingBudget={llamaThinkingBudgetTokens}")
    expect(source).toContain("grammarMode={llamaGrammarMode}")
    expect(source).toContain("grammarId={llamaGrammarId}")
    expect(source).toContain("grammarInline={llamaGrammarInline}")
    expect(source).toContain("grammarOverride={llamaGrammarOverride}")
    expect(source).toContain("extraBody={llamaExtraBody}")
    expect(source).toContain("onChange={handleLlamaControlChange}")
  })
})

import { describe, expect, it } from "vitest"
import {
  MAX_IMPORT_FILE_BYTES,
  parseCharacterImportPreview
} from "../utils"

const makeFile = (
  name: string,
  type: string,
  size: number,
  content = "{}"
): File => {
  const file = new File([content], name, { type })
  Object.defineProperty(file, "size", { value: size })
  return file
}

describe("parseCharacterImportPreview size cap", () => {
  it("rejects a file larger than MAX_IMPORT_FILE_BYTES before reading it", async () => {
    const file = makeFile(
      "huge.json",
      "application/json",
      MAX_IMPORT_FILE_BYTES + 1
    )

    const preview = await parseCharacterImportPreview(file, 0)

    expect(preview.parseError?.key).toBe(
      "settings:manageCharacters.import.previewTooLarge"
    )
    expect(preview.fieldCount).toBe(0)
  })

  it("does not flag a within-limit file as too large", async () => {
    const file = makeFile(
      "small.json",
      "application/json",
      256,
      JSON.stringify({ name: "Ada", description: "A researcher" })
    )

    const preview = await parseCharacterImportPreview(file, 0)

    expect(preview.parseError?.key).not.toBe(
      "settings:manageCharacters.import.previewTooLarge"
    )
  })
})

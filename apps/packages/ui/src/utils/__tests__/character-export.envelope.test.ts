import { beforeEach, afterEach, describe, expect, it, vi } from "vitest"
import {
  embedMetadataInPNG,
  exportCharacterToJSON,
  exportCharactersToJSON
} from "../character-export"

const data = {
  name: "Rowan",
  description: "café\n",
  extensions: { tldw: { generation: { temperature: 0.4 } } }
}
const card = {
  spec: "chara_card_v3" as const,
  spec_version: "3.0" as const,
  data
}
const createObjectURL = vi.fn((_blob: Blob) => "blob:test")
const readBlob = (blob: Blob, binary = false): Promise<string | ArrayBuffer> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string | ArrayBuffer)
    reader.onerror = reject
    if (binary) reader.readAsArrayBuffer(blob)
    else reader.readAsText(blob)
  })

beforeEach(() => {
  vi.stubGlobal(
    "URL",
    class extends URL {
      static createObjectURL = createObjectURL
      static revokeObjectURL = vi.fn()
    }
  )
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
  createObjectURL.mockClear()
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("Character card export envelope", () => {
  it.each([data, card])("exports one complete v3 envelope", async (input) => {
    exportCharacterToJSON(input)
    expect(
      JSON.parse((await readBlob(createObjectURL.mock.calls[0][0])) as string)
    ).toEqual(card)
  })
  it("normalizes raw and server-enveloped cards in bulk export", async () => {
    exportCharactersToJSON([data, card])
    expect(
      JSON.parse((await readBlob(createObjectURL.mock.calls[0][0])) as string)
    ).toEqual([card, card])
  })
  it.each([data, card])(
    "embeds the same single envelope in PNG metadata",
    async (input) => {
      const png = new Uint8Array(33)
      png.set([137, 80, 78, 71, 13, 10, 26, 10])
      png[11] = 13
      png.set([73, 72, 68, 82], 12)
      const blob = await embedMetadataInPNG(png.buffer, input)
      const bytes = new Uint8Array((await readBlob(blob, true)) as ArrayBuffer)
      const length = new DataView(bytes.buffer).getUint32(33)
      const text = new TextDecoder().decode(bytes.slice(41, 41 + length))
      expect(text.startsWith("chara\0")).toBe(true)
      const json = decodeURIComponent(escape(atob(text.slice(6))))
      expect(JSON.parse(json)).toEqual(card)
    }
  )
})

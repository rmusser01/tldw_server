import { describe, expect, it } from "vitest"
import JSZip from "jszip"
import { LocalProvider } from "../LocalProvider"

describe("LocalProvider", () => {
  it("initializes directory mode and returns blob nodes", async () => {
    const provider = new LocalProvider()
    const file = new File(["const a=1"], "src/a.ts", { type: "text/plain" })
    await provider.initialize({
      source: "directory",
      files: { 0: file, length: 1 } as unknown as FileList
    })
    const tree = await provider.fetchTree("local://directory")
    expect(tree.some((node) => node.path.includes("a.ts"))).toBe(true)
  })
})

const files = (...items: File[]) =>
  Object.assign(items, { item: (i: number) => items[i] }) as unknown as FileList
const textFile = (text: string, name = "README.md") => {
  const file = new File([text], name)
  Object.defineProperty(file, "text", { value: async () => text })
  return file
}

describe("LocalProvider export boundaries", () => {
  it("rejects oversized directory files before reading them", async () => {
    const provider = new LocalProvider()
    const file = textFile("x")
    Object.defineProperty(file, "size", { value: 1024 * 1024 + 1 })
    await provider.initialize({ source: "directory", files: files(file) })
    await expect(
      provider.fetchFile((await provider.fetchTree("local://directory"))[0])
    ).rejects.toThrow(/1 MiB|too large/i)
  })

  it("rejects binary bytes rather than exporting corrupted text", async () => {
    const provider = new LocalProvider()
    await provider.initialize({
      source: "directory",
      files: files(textFile("hello\0world"))
    })
    await expect(
      provider.fetchFile((await provider.fetchTree("local://directory"))[0])
    ).rejects.toThrow(/binary|text/i)
  })

  it.each(["../outside.md", "/outside.md", "C:\\outside.md"])(
    "rejects unsafe directory path %s",
    async (path) => {
      const provider = new LocalProvider()
      const file = textFile("public")
      Object.defineProperty(file, "webkitRelativePath", { value: path })
      await expect(
        provider.initialize({ source: "directory", files: files(file) })
      ).rejects.toThrow(/path/i)
    }
  )

  it("rejects ZIP traversal before JSZip sanitization hides it", async () => {
    const zip = new JSZip().file("../outside.md", "public outside sentinel")
    const archive = new File(
      [await zip.generateAsync({ type: "uint8array" })],
      "unsafe.zip"
    )
    await expect(
      new LocalProvider().initialize({ source: "zip", zipFile: archive })
    ).rejects.toThrow(/path/i)
  })

  it("rejects ZIP symlink metadata", async () => {
    const zip = new JSZip().file("link.md", "../public-outside.md", {
      unixPermissions: 0o120777
    })
    const archive = new File(
      [await zip.generateAsync({ type: "uint8array", platform: "UNIX" })],
      "link.zip"
    )
    await expect(
      new LocalProvider().initialize({ source: "zip", zipFile: archive })
    ).rejects.toThrow(/symlink|symbolic/i)
  })
})

describe("LocalProvider bounded ZIP recovery", () => {
  it("exports complete Unicode text from a valid ZIP after rejecting a bad ZIP", async () => {
    const provider = new LocalProvider()
    await expect(
      provider.initialize({
        source: "zip",
        zipFile: new File(["invalid"], "bad.zip")
      })
    ).rejects.toThrow()
    const text = "Selene — café 星\n"
    const zip = new JSZip().file("README.md", text)
    await provider.initialize({
      source: "zip",
      zipFile: new File(
        [await zip.generateAsync({ type: "uint8array" })],
        "valid.zip"
      )
    })
    const tree = await provider.fetchTree("local://zip")
    expect(await provider.fetchFile(tree[0])).toMatchObject({
      path: "README.md",
      text
    })
  })
  it("bounds decompressed ZIP content, including compressible input", async () => {
    const zip = new JSZip().file("large.md", "x".repeat(1024 * 1024 + 1))
    const provider = new LocalProvider()
    await provider.initialize({
      source: "zip",
      zipFile: new File(
        [
          await zip.generateAsync({
            type: "uint8array",
            compression: "DEFLATE"
          })
        ],
        "small.zip"
      )
    })
    await expect(
      provider.fetchFile((await provider.fetchTree("local://zip"))[0])
    ).rejects.toThrow(/1 MiB/)
  })
})

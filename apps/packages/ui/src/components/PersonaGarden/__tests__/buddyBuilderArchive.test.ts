import { describe, expect, it, vi } from "vitest"

import {
  BUDDY_IMPORT_ARCHIVE_ACCEPT,
  getBuddyImportArchiveFileError,
  isBuddyImportArchiveFile
} from "../buddyBuilderArchive"

const t = vi.fn((key: string, options?: { defaultValue?: string }) =>
  options?.defaultValue ?? key
)

describe("buddyBuilderArchive", () => {
  it("accepts native Persona Visual pack archives with normal zip media types", () => {
    const file = new File(["zip"], "pack.tldw-persona-vpack", {
      type: "application/zip"
    })

    expect(isBuddyImportArchiveFile(file)).toBe(true)
    expect(getBuddyImportArchiveFileError(file, t)).toBeNull()
  })

  it("accepts Codex and Petdex zip archives with generic browser media types", () => {
    const zip = new File(["zip"], "pet.zip", {
      type: "application/octet-stream"
    })
    const compressed = new File(["zip"], "pet.zip", {
      type: "application/x-zip-compressed"
    })

    expect(isBuddyImportArchiveFile(zip)).toBe(true)
    expect(isBuddyImportArchiveFile(compressed)).toBe(true)
  })

  it("rejects unsupported extensions before preview", () => {
    const file = new File(["image"], "pet.png", { type: "image/png" })

    expect(isBuddyImportArchiveFile(file)).toBe(false)
    expect(getBuddyImportArchiveFileError(file, t)).toContain(
      ".tldw-persona-vpack"
    )
  })

  it("rejects unsupported media types before preview", () => {
    const file = new File(["zip"], "pet.zip", { type: "image/png" })

    expect(isBuddyImportArchiveFile(file)).toBe(false)
    expect(getBuddyImportArchiveFileError(file, t)).toContain(
      "supported zip media type"
    )
  })

  it("keeps the file input accept string explicit", () => {
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain(".tldw-persona-vpack")
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain(".zip")
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain("application/zip")
  })
})

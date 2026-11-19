import { describe, expect, it } from "vitest"
import {
  createUrlQueueItems,
  getEligibleQueueItems,
  getQueueItemExclusionReason,
  isValidQueueUrl,
  validateQueueItem
} from "../queue-items"
import type { WizardQueueItem } from "../types"
const DOCUMENT_LIMIT = 250 * 1024 ** 2
const AV_LIMIT = 1024 ** 3

const fileItem = (id: string, size = 4): WizardQueueItem => ({
  id,
  fileName: "source.txt",
  file: new File(["test"], "source.txt"),
  detectedType: "document",
  icon: "File",
  fileSize: size,
  validation: { valid: true }
})

describe("queue eligibility", () => {
  it("shares the same file name/size duplicate definition and permits explicit repetition", () => {
    const queue = [fileItem("first"), fileItem("second"), fileItem("larger", 5)]
    expect(getEligibleQueueItems(queue).map((item) => item.id)).toEqual([
      "first",
      "larger"
    ])
    queue[1].processAgain = true
    expect(getEligibleQueueItems(queue)).toHaveLength(3)
  })
  it("dedupes normalized URLs and promotes the remaining item after removal", () => {
    const queue = createUrlQueueItems(
      "https://EXAMPLE.com/a/?utm_source=news#top\nhttps://example.com/a"
    )
    expect(getEligibleQueueItems(queue)).toHaveLength(1)
    expect(getEligibleQueueItems(queue.slice(1))).toHaveLength(1)
  })
  it("explains exclusions and does not let excluded playlist items block new URLs", () => {
    const queue = createUrlQueueItems(
      "https://example.com/a\nhttps://example.com/a\ninvalid\nhttps://example.com/c"
    )
    queue[0].playlist = { duplicateStatus: "duplicate_existing" }
    queue[3].conferenceOverride = { selected: false }
    expect(
      queue.map((item) => getQueueItemExclusionReason(item, queue))
    ).toEqual(["duplicate", null, "invalid", "unselected"])
    queue[0].conferenceOverride = {
      selected: true,
      duplicatePolicy: "overwrite"
    }
    expect(getEligibleQueueItems(queue).map((item) => item.id)).toEqual([
      queue[0].id
    ])
  })
  it.each([
    ["pdf", "scan.pdf", "application/pdf", DOCUMENT_LIMIT],
    ["ebook", "sample.epub", "application/epub+zip", DOCUMENT_LIMIT],
    ["document", "notes.docx", "", DOCUMENT_LIMIT],
    ["audio", "sample.ogg", "audio/ogg", AV_LIMIT],
    ["video", "sample.mkv", "", AV_LIMIT]
  ] as const)("enforces the %s boundary from the actual File", (detectedType, name, type, limit) => {
    const file = new File(["sample"], name, { type })
    const item = { ...fileItem("selected"), file, fileName: name, detectedType }
    Object.defineProperty(file, "size", { value: limit, configurable: true })
    expect(validateQueueItem(item).valid).toBe(true)
    Object.defineProperty(file, "size", { value: limit + 1 })
    expect(validateQueueItem(item).valid).toBe(false)
    expect(getEligibleQueueItems([item])).toEqual([])
  })

  it("validates file size and unsupported types without allowing process-again to bypass validation", () => {
    const item = fileItem("large", DOCUMENT_LIMIT + 1)
    Object.defineProperty(item.file!, "size", { value: item.fileSize })
    expect(validateQueueItem(item).valid).toBe(false)
    item.detectedType = "unknown"
    item.validation = validateQueueItem(item)
    item.processAgain = true
    expect(getEligibleQueueItems([item])).toEqual([])
  })
  it.each([
    "",
    "ftp://example.com/a",
    "chrome://settings",
    "https://example.com/a,https://example.com/b",
    "https://example.com/a b"
  ])("rejects %s", (url) => expect(isValidQueueUrl(url)).toBe(false))
})

import { describe, expect, it } from "vitest"
import {
  createUrlQueueItems,
  getEligibleQueueItems,
  getQueueItemExclusionReason,
  isValidQueueUrl,
  validateQueueItem
} from "../queue-items"
import type { WizardQueueItem } from "../types"
import { QUICK_INGEST_MAX_FILE_SIZE } from "../constants"

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
  it("validates file size and unsupported types without allowing process-again to bypass validation", () => {
    const item = fileItem("large", QUICK_INGEST_MAX_FILE_SIZE + 1)
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

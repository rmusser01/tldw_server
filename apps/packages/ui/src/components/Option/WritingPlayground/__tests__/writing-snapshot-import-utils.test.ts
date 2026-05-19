import { describe, expect, it } from "vitest"
import {
  getSnapshotImportInvalidationKeys,
  resolveSnapshotImportAction,
  type SnapshotReplaceConfirmLabels
} from "../writing-snapshot-import-utils"

const LABELS: SnapshotReplaceConfirmLabels = {
  title: "Replace existing writing data?",
  body: "This will replace current sessions, templates, and themes with the imported snapshot.",
  action: "Choose file",
  cancel: "Cancel"
}

describe("writing snapshot import utils", () => {
  it("adds the active session detail key for merge imports", () => {
    expect(getSnapshotImportInvalidationKeys("merge", "session-1")).toEqual([
      ["writing-sessions"],
      ["writing-templates"],
      ["writing-themes"],
      ["writing-session", "session-1"]
    ])
  })

  it("keeps replace imports on collection invalidation only", () => {
    expect(getSnapshotImportInvalidationKeys("replace", "session-1")).toEqual([
      ["writing-sessions"],
      ["writing-templates"],
      ["writing-themes"]
    ])
  })

  it("opens picker directly for merge mode", () => {
    expect(resolveSnapshotImportAction("merge", LABELS)).toEqual({
      type: "open-picker",
      mode: "merge"
    })
  })

  it("requires confirmation for replace mode", () => {
    expect(resolveSnapshotImportAction("replace", LABELS)).toEqual({
      type: "confirm-replace",
      mode: "replace",
      title: LABELS.title,
      content: LABELS.body,
      okText: LABELS.action,
      cancelText: LABELS.cancel,
      danger: true
    })
  })
})

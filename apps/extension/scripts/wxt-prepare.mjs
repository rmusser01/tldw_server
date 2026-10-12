import { prepareMonacoAssets } from "../../scripts/prepare-monaco-assets.mjs"

prepareMonacoAssets()

import { prepare } from "wxt"

const skipPrepare = ["1", "true", "yes", "y", "on"].includes(
  String(process.env.SKIP_WXT_PREPARE || "").toLowerCase().trim()
)

if (skipPrepare) {
  console.log("Skipping wxt prepare (SKIP_WXT_PREPARE=1).")
  process.exit(0)
}

const dumpActiveHandles = () => {
  const handles = typeof process._getActiveHandles === "function"
    ? process._getActiveHandles()
    : []
  const requests = typeof process._getActiveRequests === "function"
    ? process._getActiveRequests()
    : []

  const summarize = (items) =>
    items.reduce((acc, item) => {
      const name = item?.constructor?.name || "Unknown"
      acc[name] = (acc[name] || 0) + 1
      return acc
    }, {})

  console.log("Active handles:", summarize(handles))
  console.log("Active requests:", summarize(requests))

  const verbose =
    String(process.env.WXT_PREPARE_DEBUG_HANDLES || "")
      .toLowerCase()
      .trim() === "verbose"

  if (verbose) {
    handles.forEach((handle, idx) => {
      const name = handle?.constructor?.name || "Unknown"
      const details = {}
      if (handle?.path) details.path = handle.path
      if (handle?.fd != null) details.fd = handle.fd
      if (handle?._idleTimeout != null) details.idleTimeout = handle._idleTimeout
      console.log(`Handle[${idx}] ${name}`, details)
    })
  }
}

const run = async () => {
  try {
    await prepare({ root: process.cwd() })
    if (process.env.WXT_PREPARE_DEBUG_HANDLES) {
      dumpActiveHandles()
    }
    // Force exit to avoid hanging on open handles after prepare completes.
    process.exit(0)
  } catch (err) {
    console.error("wxt prepare failed:", err)
    process.exit(1)
  }
}

run()

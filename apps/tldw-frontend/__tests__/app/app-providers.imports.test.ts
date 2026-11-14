import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("AppProviders imports", () => {
  it("keeps notification lifecycle dependencies inside the provider-owned layout", () => {
    const source = readFileSync(
      path.join(process.cwd(), "components", "AppProviders.tsx"),
      "utf8"
    )

    expect(source).not.toContain("NotificationToastBridge")
    const layout = readFileSync(path.join(process.cwd(), "components", "layout", "WebLayout.tsx"), "utf8")
    expect(layout).toMatch(/<NotificationLifecycleProvider[^>]*>\s*<NotificationToastBridge \/>/)
  })
})

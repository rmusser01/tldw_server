import { describe, expect, it } from "vitest"
import { safeExternalUrl } from "../safe-external-url"

describe("safeExternalUrl", () => {
  it("allows http and https URLs", () => {
    expect(safeExternalUrl("https://x")).toBe("https://x")
    expect(safeExternalUrl("http://example.com/a?b=c#d")).toBe(
      "http://example.com/a?b=c#d"
    )
  })

  it("allows mailto URLs", () => {
    expect(safeExternalUrl("mailto:a@b")).toBe("mailto:a@b")
  })

  it("allows relative paths and anchors", () => {
    expect(safeExternalUrl("/foo/bar")).toBe("/foo/bar")
    expect(safeExternalUrl("./rel")).toBe("./rel")
    expect(safeExternalUrl("../up")).toBe("../up")
    expect(safeExternalUrl("#section")).toBe("#section")
  })

  it("rejects javascript: URLs", () => {
    expect(safeExternalUrl("javascript:alert(1)")).toBeNull()
    expect(safeExternalUrl("JavaScript:alert(1)")).toBeNull()
    expect(safeExternalUrl("  javascript:alert(1)")).toBeNull()
  })

  it("rejects control-char obfuscated schemes", () => {
    // A tab inside the scheme (`java\tscript:`) is stripped by the browser at
    // click time, so it must be neutralized before scheme matching.
    expect(safeExternalUrl("java\tscript:alert(1)")).toBeNull()
    expect(safeExternalUrl("java\nscript:alert(1)")).toBeNull()
    expect(safeExternalUrl("javascript:alert(1)")).toBeNull()
  })

  it("rejects other dangerous schemes", () => {
    expect(safeExternalUrl("data:text/html,<script>alert(1)</script>")).toBeNull()
    expect(safeExternalUrl("vbscript:msgbox(1)")).toBeNull()
    expect(safeExternalUrl("file:///etc/passwd")).toBeNull()
  })

  it("rejects empty and non-string input", () => {
    expect(safeExternalUrl("")).toBeNull()
    expect(safeExternalUrl("   ")).toBeNull()
    expect(safeExternalUrl(null)).toBeNull()
    expect(safeExternalUrl(undefined)).toBeNull()
    expect(safeExternalUrl(42)).toBeNull()
  })
})

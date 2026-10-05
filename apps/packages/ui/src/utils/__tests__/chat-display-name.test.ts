import { describe, expect, it } from "vitest"
import { replaceUserDisplayNamePlaceholders } from "../chat-display-name"

describe("user and character display names", () => {
  it("renders user and character placeholders with distinct identities and literal dollar names", () => {
    expect(replaceUserDisplayNamePlaceholders("{{user}} / {{ username }} / {{CHAR}} / {{ character }}", "Viewer $&", "Guide $'"))
      .toBe("Viewer $& / Viewer $& / Guide $' / Guide $'")
  })
  it("leaves character placeholders unresolved when only the user identity is known", () => {
    expect(replaceUserDisplayNamePlaceholders("{{user}} meets {{char}}", "Viewer")).toBe("Viewer meets {{char}}")
  })
  it("renders a known character even when the user name is absent", () => {
    expect(replaceUserDisplayNamePlaceholders("{{user}} meets {{char}}", "", "Guide")).toBe("{{user}} meets Guide")
  })
})

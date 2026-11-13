import { afterEach, describe, expect, it } from "vitest"
import { holdDocumentTitle } from "../document-title"

// G14 follow-up: a component that names the tab for a while (an admin page,
// a live voice chat) gives the old title back only if nobody has renamed the
// tab since; the route title hook (XP-19) renames it on every route change.
describe("holdDocumentTitle", () => {
  afterEach(() => {
    document.title = ""
  })

  it("names the tab and gives the old title back", () => {
    document.title = "Chat | tldw"
    const release = holdDocumentTitle("🎤 Listening - Chat")
    expect(document.title).toBe("🎤 Listening - Chat")
    release()
    expect(document.title).toBe("Chat | tldw")
  })

  it("leaves a title someone else set in the meantime", () => {
    document.title = "Admin | tldw"
    const release = holdDocumentTitle("Server Admin · Admin · tldw")
    document.title = "Notes | tldw"
    release()
    expect(document.title).toBe("Notes | tldw")
  })
})

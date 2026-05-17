import { describe, expect, it } from "vitest"

import { detectPlaylistPreflightCandidate, detectTypeFromUrl } from "../AddContentStep"
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads"

describe("detectTypeFromUrl", () => {
  it("detects supported media hosts from exact hosts and subdomains", () => {
    expect(detectTypeFromUrl("https://www.youtube.com/watch?v=123")).toBe("video")
    expect(detectTypeFromUrl("https://player.vimeo.com/video/42")).toBe("video")
    expect(detectTypeFromUrl("https://m.soundcloud.com/example/track")).toBe("audio")
    expect(detectTypeFromUrl("https://open.spotify.com/track/abc")).toBe("audio")
  })

  it("does not trust lookalike or suffix-appended hosts", () => {
    expect(detectTypeFromUrl("https://youtube.com.evil.test/watch?v=123")).toBe("web")
    expect(detectTypeFromUrl("https://evil-youtube.com/watch?v=123")).toBe("web")
    expect(detectTypeFromUrl("https://soundcloud.com.evil.test/track")).toBe("web")
  })

  it("identifies YouTube playlist URLs as preflight candidates", () => {
    expect(
      detectPlaylistPreflightCandidate(
        "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
      )
    ).toBe(true)
    expect(
      detectPlaylistPreflightCandidate("https://www.youtube.com/playlist?list=PLtest")
    ).toBe(true)
  })

  it("does not identify single videos or lookalike hosts as playlist preflight candidates", () => {
    expect(detectPlaylistPreflightCandidate("https://www.youtube.com/watch?v=abc123")).toBe(false)
    expect(
      detectPlaylistPreflightCandidate("https://youtube.com.evil.test/watch?v=abc&list=PLtest")
    ).toBe(false)
  })
})

describe("Quick Ingest URL normalization", () => {
  it("normalizes common tracking variants for queue duplicate checks", () => {
    expect(
      normalizeUrlForDedupe("https://EXAMPLE.com/a/?utm_source=x#frag")
    ).toBe("https://example.com/a")
  })

  it("canonicalizes youtu.be links to watch URLs", () => {
    expect(normalizeUrlForDedupe("https://youtu.be/abc123?t=30")).toBe(
      "https://www.youtube.com/watch?v=abc123"
    )
  })
})

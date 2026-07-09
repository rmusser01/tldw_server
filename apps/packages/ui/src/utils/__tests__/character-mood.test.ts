import { describe, expect, it } from "vitest"
import {
  detectCharacterMood,
  getCharacterMoodImagesFromExtensions,
  mergeCharacterMoodImagesIntoExtensions,
  normalizeCharacterMoodLabel,
  removeCharacterMoodImage,
  resolveCharacterMoodImageUrl,
  upsertCharacterMoodImage
} from "../character-mood"

const TINY_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5n4QAAAABJRU5ErkJggg=="

describe("character mood utilities", () => {
  it("normalizes known mood aliases", () => {
    expect(normalizeCharacterMoodLabel("JOYFUL")).toBe("happy")
    expect(normalizeCharacterMoodLabel("unknown")).toBeNull()
  })

  it("detects excited mood from energetic text", () => {
    const detected = detectCharacterMood({
      assistantText: "Wow, this is amazing! Let's go!",
      userText: "Can you celebrate this win with me?"
    })

    expect(detected.label).toBe("excited")
    expect(detected.confidence).toBeGreaterThan(0.6)
    expect(detected.topic).toBeTruthy()
  })

  it("falls back to neutral when there are no strong mood signals", () => {
    const detected = detectCharacterMood({
      assistantText: "Here is the summary of the API response payload."
    })

    expect(detected.label).toBe("neutral")
    expect(detected.confidence).toBeGreaterThan(0.3)
    expect(detected.confidence).toBeLessThanOrEqual(0.72)
  })

  it("reads and merges mood images under extensions.tldw.mood_images", () => {
    const initial = {
      tldw: {
        prompt_preset: "default",
        mood_images: {
          happy: `data:image/png;base64,${TINY_PNG_BASE64}`
        }
      }
    }

    const merged = mergeCharacterMoodImagesIntoExtensions(initial, {
      happy: `data:image/png;base64,${TINY_PNG_BASE64}`,
      sad: TINY_PNG_BASE64
    })

    const moodImages = getCharacterMoodImagesFromExtensions(merged)
    expect(moodImages.happy).toMatch(/^data:image\/png;base64,/)
    expect(moodImages.sad).toMatch(/^data:image\/png;base64,/)
    expect((merged as any).tldw.prompt_preset).toBe("default")
  })

  it("reads arbitrary safe expression states from canonical mood images", () => {
    const images = getCharacterMoodImagesFromExtensions({
      tldw: {
        mood_images: {
          smirk: TINY_PNG_BASE64,
          "joy-soft": "https://example.test/joy.png"
        }
      }
    })

    expect(images.smirk).toMatch(/^data:image\/png;base64,/)
    expect(images["joy-soft"]).toBe("https://example.test/joy.png")
  })

  it("prefers canonical mood images as a whole map over legacy aliases", () => {
    const images = getCharacterMoodImagesFromExtensions({
      tldw: {
        mood_images: { happy: "https://example.test/happy.png" },
        moodImages: { sad: "https://example.test/sad.png" }
      },
      mood_images: { angry: "https://example.test/angry.png" }
    })

    expect(Object.keys(images)).toEqual(["happy"])
  })

  it("writes canonical mood images and removes legacy aliases", () => {
    const merged = mergeCharacterMoodImagesIntoExtensions(
      {
        tldw: {
          moodImages: { sad: "https://example.test/sad.png" },
          prompt_preset: "roleplay"
        },
        mood_images: { angry: "https://example.test/angry.png" },
        moodImages: { confused: "https://example.test/confused.png" }
      },
      { smirk: "https://example.test/smirk.png" }
    )

    expect((merged as any).tldw.mood_images).toEqual({
      smirk: "https://example.test/smirk.png"
    })
    expect((merged as any).tldw.moodImages).toBeUndefined()
    expect((merged as any).mood_images).toBeUndefined()
    expect((merged as any).moodImages).toBeUndefined()
    expect((merged as any).tldw.prompt_preset).toBe("roleplay")
  })

  it("removes mood image keys when saving an empty map", () => {
    const merged = mergeCharacterMoodImagesIntoExtensions(
      { tldw: { mood_images: { happy: "https://example.test/happy.png" } } },
      {}
    )

    expect((merged as any).tldw).toBeUndefined()
  })

  it("resolves custom emote image states without expanding classifier labels", () => {
    const extensions = {
      tldw: {
        mood_images: {
          smug: TINY_PNG_BASE64,
          "thinking-hard": TINY_PNG_BASE64,
          "../../bad": TINY_PNG_BASE64
        }
      }
    }

    expect(resolveCharacterMoodImageUrl({ extensions }, "smug")).toMatch(/^data:image\/png;base64,/)
    expect(resolveCharacterMoodImageUrl({ extensions }, "thinking hard")).toMatch(/^data:image\/png;base64,/)
    expect(resolveCharacterMoodImageUrl({ extensions }, "../../bad")).toBe("")
    expect(normalizeCharacterMoodLabel("smug")).toBeNull()
  })

  it("resolves legacy mood aliases after custom emote lookup misses", () => {
    const extensions = {
      tldw: {
        mood_images: {
          happy: "https://example.test/happy.png"
        }
      }
    }

    expect(resolveCharacterMoodImageUrl({ extensions }, "joy")).toBe(
      "https://example.test/happy.png"
    )
  })

  it("upserts and removes mood images", () => {
    const withImage = upsertCharacterMoodImage({}, "happy", TINY_PNG_BASE64)
    const imageAfterUpsert = resolveCharacterMoodImageUrl(
      { extensions: withImage },
      "happy"
    )
    expect(imageAfterUpsert).toMatch(/^data:image\/png;base64,/)

    const withoutImage = removeCharacterMoodImage(withImage, "happy")
    const imageAfterRemove = resolveCharacterMoodImageUrl(
      { extensions: withoutImage },
      "happy"
    )
    expect(imageAfterRemove).toBe("")
  })

  it("upserts and removes custom expression state images", () => {
    const withImage = upsertCharacterMoodImage({}, "smirk", TINY_PNG_BASE64)
    expect(resolveCharacterMoodImageUrl({ extensions: withImage }, "smirk")).toMatch(
      /^data:image\/png;base64,/
    )

    const withoutImage = removeCharacterMoodImage(withImage, "smirk")
    expect(resolveCharacterMoodImageUrl({ extensions: withoutImage }, "smirk")).toBe("")
  })
})

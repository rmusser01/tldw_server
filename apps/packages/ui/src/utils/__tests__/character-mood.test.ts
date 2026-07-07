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
})

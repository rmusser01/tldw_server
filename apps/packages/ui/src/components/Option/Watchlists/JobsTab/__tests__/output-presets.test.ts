import { describe, expect, it } from "vitest"
import { applyOutputPresetToPrefs } from "../output-presets"

describe("applyOutputPresetToPrefs", () => {
  it("replaces known output, delivery, and audio fields while preserving unknown advanced prefs", () => {
    const applied = applyOutputPresetToPrefs({
      baseOutputPrefs: {
        template: {
          default_name: "old",
          default_version: 4,
          experimental_renderer: "keep"
        },
        deliveries: {
          email: { enabled: true, recipients: ["old@example.com"] },
          webhook: { url: "https://hooks.example.com/watchlists" }
        },
        auto_output: {
          enabled: true,
          type: "briefing_markdown",
          custom_flag: "keep"
        },
        generate_audio: true,
        audio_voice: "alloy",
        target_audio_minutes: 12,
        raw_advanced: { preserve: true }
      },
      presetOutputPrefs: {
        template: {
          default_name: "podcast_script",
          default_format: "md"
        },
        deliveries: {
          chatbook: { enabled: true, title: "Daily Brief" }
        },
        generate_audio: false,
        preset_custom: { copied: true }
      }
    })

    expect(applied).toEqual({
      template: {
        default_name: "podcast_script",
        default_format: "md",
        experimental_renderer: "keep"
      },
      deliveries: {
        chatbook: { enabled: true, title: "Daily Brief" },
        webhook: { url: "https://hooks.example.com/watchlists" }
      },
      auto_output: { custom_flag: "keep" },
      generate_audio: false,
      raw_advanced: { preserve: true },
      preset_custom: { copied: true }
    })
  })

  it("removes legacy scalar nested output groups when the preset omits them", () => {
    const applied = applyOutputPresetToPrefs({
      baseOutputPrefs: {
        template: "legacy-template-name",
        deliveries: "legacy-delivery-target",
        raw_advanced: { preserve: true }
      },
      presetOutputPrefs: {
        generate_audio: true,
        audio_voice: "nova"
      }
    })

    expect(applied).toEqual({
      generate_audio: true,
      audio_voice: "nova",
      raw_advanced: { preserve: true }
    })
  })

  it("deep clones raw advanced values without stringifying non-JSON browser values", () => {
    const observedAt = new Date("2026-05-23T01:00:00.000Z")
    const applied = applyOutputPresetToPrefs({
      baseOutputPrefs: {
        raw_advanced: { observedAt }
      },
      presetOutputPrefs: {
        generate_audio: false
      }
    })

    const rawAdvanced = applied.raw_advanced as { observedAt: Date }
    expect(rawAdvanced.observedAt).toEqual(observedAt)
    expect(rawAdvanced.observedAt).not.toBe(observedAt)
  })
})

import { describe, expect, it } from "vitest"

import {
  buildSttModelOptions,
  buildTtsReadinessItems,
  describeCapabilityValue
} from "../audio-readiness"

describe("audio readiness helpers", () => {
  it("enriches STT model ids with static labels and bounded health metadata", () => {
    const options = buildSttModelOptions({
      catalog: {
        categories: {
          "Whisper Models": [
            {
              value: "whisper-small",
              label: "Whisper Small",
              description: "Balanced speed/accuracy"
            }
          ],
          "VibeVoice-ASR": [
            {
              value: "vibevoice-asr",
              label: "VibeVoice-ASR",
              description: "Long-form ASR with diarization + hotwords"
            }
          ]
        },
        all_models: ["whisper-small", "vibevoice-asr"]
      },
      healthByModel: {
        "whisper-small": {
          available: true,
          usable: true,
          on_demand: false,
          message: "Model is ready",
          provider: "whisper"
        }
      }
    })

    expect(options).toEqual([
      expect.objectContaining({
        id: "vibevoice-asr",
        label: "VibeVoice-ASR",
        category: "VibeVoice-ASR",
        availability: "unknown",
        capabilities: expect.objectContaining({
          diarization: "unknown"
        })
      }),
      expect.objectContaining({
        id: "whisper-small",
        label: "Whisper Small",
        category: "Whisper Models",
        availability: "ready",
        provider: "whisper",
        readinessMessage: "Model is ready",
        sources: expect.objectContaining({
          availability: "health",
          label: "static_catalog"
        })
      })
    ])
  })

  it("prefers STT capability summary metadata when the endpoint is available", () => {
    const options = buildSttModelOptions({
      catalog: {
        categories: {
          "Whisper Models": [
            {
              value: "whisper-small",
              label: "Whisper Small",
              description: "Balanced speed/accuracy"
            }
          ]
        },
        all_models: ["whisper-small"]
      },
      capabilitySummary: {
        models: [
          {
            id: "whisper-small",
            label: "Whisper Small",
            description: "Balanced speed/accuracy",
            category: "Whisper Models",
            provider: "faster-whisper",
            availability: "ready",
            availability_source: "health",
            capabilities: {
              batch: "supported",
              streaming: "supported",
              diarization: "supported",
              timestamps: "supported",
              segments: "supported"
            },
            sources: {
              availability: "health",
              batch: "provider",
              streaming: "provider",
              diarization: "provider",
              timestamps: "response_schema",
              segments: "response_schema"
            },
            message: "Ready"
          }
        ]
      },
      healthByModel: {
        "whisper-small": {
          available: false,
          usable: false,
          message: "Stale fallback"
        }
      }
    })

    expect(options).toEqual([
      expect.objectContaining({
        id: "whisper-small",
        provider: "faster-whisper",
        availability: "ready",
        readinessMessage: "Ready",
        capabilities: {
          batch: "supported",
          streaming: "supported",
          diarization: "supported",
          timestamps: "supported",
          segments: "supported"
        },
        sources: expect.objectContaining({
          availability: "health",
          timestamps: "response_schema"
        })
      })
    ])
  })

  it("attributes capability-only labels and descriptions to the capability response", () => {
    const options = buildSttModelOptions({
      capabilitySummary: {
        models: [
          {
            id: "remote-asr",
            label: "Remote ASR",
            description: "Hosted model metadata",
            availability: "ready",
            capabilities: {}
          }
        ]
      }
    })

    expect(options[0]).toEqual(
      expect.objectContaining({
        label: "Remote ASR",
        description: "Hosted model metadata",
        sources: expect.objectContaining({
          label: "response_schema",
          description: "response_schema"
        })
      })
    )
  })

  it("keeps unsupported and unknown capability labels distinct", () => {
    expect(describeCapabilityValue("supported")).toEqual({
      label: "Supported",
      tone: "success"
    })
    expect(describeCapabilityValue("unsupported")).toEqual({
      label: "Unsupported",
      tone: "error"
    })
    expect(describeCapabilityValue("unknown")).toEqual({
      label: "Unknown",
      tone: "default"
    })
  })

  it("builds TTS readiness that treats Browser preview as a no-setup fallback", () => {
    const items = buildTtsReadinessItems({
      provider: "browser",
      hasAudio: false,
      providersInfo: null
    })

    expect(items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "browser-preview",
          label: "Browser preview",
          state: "ready",
          detail: "Available in this browser without server setup."
        })
      ])
    )
  })

  it("builds TTS readiness that flags missing ElevenLabs credentials", () => {
    const items = buildTtsReadinessItems({
      provider: "elevenlabs",
      hasAudio: true,
      providersInfo: null,
      elevenLabsApiKey: "   "
    })

    expect(items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "elevenlabs-credentials",
          label: "ElevenLabs setup",
          state: "blocked",
          detail: "API key required before generation."
        })
      ])
    )
  })

  it("does not require server audio for ElevenLabs after a key is saved", () => {
    const items = buildTtsReadinessItems({
      provider: "elevenlabs",
      hasAudio: false,
      providersInfo: null,
      elevenLabsApiKey: "sk_test_key"
    })

    expect(items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "elevenlabs-credentials",
          label: "ElevenLabs setup",
          state: "ready",
          detail: "API key saved. Voices and models load directly from ElevenLabs."
        })
      ])
    )
    expect(items).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "tts-server-audio"
        })
      ])
    )
  })
})

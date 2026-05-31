import { describe, expect, it } from "vitest"

import { resolveAudioCapturePlan } from "@/audio"

describe("resolveAudioCapturePlan", () => {
  it("forces browser dictation off when a non-default mic is selected", () => {
    const requestedSource = { sourceKind: "mic_device", deviceId: "usb-1" } as const
    const plan = resolveAudioCapturePlan({
      featureGroup: "dictation",
      requestedSource,
      requestedSpeechPath: "browser_dictation",
      capabilities: {
        browserDictationSupported: true,
        serverDictationSupported: true,
        liveVoiceSupported: true,
        secureContextAvailable: true
      }
    })

    expect(plan.requestedSource).toEqual(requestedSource)
    expect(plan.resolvedSource).toEqual(requestedSource)
    expect(plan.resolvedDeviceId).toBe("usb-1")
    expect(plan.requestedSourceKind).toBe("mic_device")
    expect(plan.resolvedSpeechPath).toBe("server_dictation")
    expect(plan.reason).toBe("browser_dictation_incompatible_with_selected_source")
  })
})

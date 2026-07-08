export type AudioSourceKind =
  | "default_mic"
  | "mic_device"
  | "tab_audio"
  | "system_audio"

export type AudioFeatureGroup =
  | "dictation"
  | "live_voice"
  | "speech_playground"

export type AudioCaptureOwner =
  | AudioFeatureGroup
  | "voice_chat"
  | "push_to_talk"
  | "captions"

export type AudioSpeechPath =
  | "browser_dictation"
  | "server_dictation"
  | "live_voice_stream"
  | "speech_playground_recording"

export type AudioCaptureReason =
  | "resolved_as_requested"
  | "browser_dictation_incompatible_with_selected_source"
  | "selected_source_unavailable"

export type AudioCaptureRequestedSource = {
  sourceKind: AudioSourceKind
  deviceId?: string | null
}

export type AudioCapturePlanCapabilities = {
  browserDictationSupported: boolean
  serverDictationSupported: boolean
  liveVoiceSupported: boolean
  secureContextAvailable: boolean
}

export type ResolveAudioCapturePlanInput = {
  featureGroup: AudioFeatureGroup
  requestedSource: AudioCaptureRequestedSource
  requestedSpeechPath: AudioSpeechPath
  capabilities: AudioCapturePlanCapabilities
}

export type ResolvedAudioCapturePlan = {
  requestedSource: AudioCaptureRequestedSource
  resolvedSource: AudioCaptureRequestedSource
  resolvedDeviceId: string | null
  requestedSourceKind: AudioSourceKind
  resolvedSourceKind: AudioSourceKind
  requestedSpeechPath: AudioSpeechPath
  resolvedSpeechPath: AudioSpeechPath
  reason: AudioCaptureReason
}

export type AudioPresetKind = "tts" | "stt" | "speech"

export type AudioPreset = {
  id: string
  owner_user_id: string
  kind: AudioPresetKind
  name: string
  description?: string | null
  favorite: boolean
  is_default: boolean
  config: Record<string, unknown>
  capability_assumptions: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type AudioPresetListResponse = {
  items: AudioPreset[]
  total: number
  limit: number
  offset: number
}

export type AudioPresetCreatePayload = {
  kind: AudioPresetKind
  name: string
  description?: string | null
  favorite?: boolean
  is_default?: boolean
  config?: Record<string, unknown>
  capability_assumptions?: Record<string, unknown>
}

export type AudioPresetUpdatePayload = Partial<
  Pick<
    AudioPresetCreatePayload,
    "name" | "description" | "favorite" | "is_default" | "config" | "capability_assumptions"
  >
>

export type AudioPresetValidationWarning = {
  code: string
  message: string
  field?: string | null
}

export type AudioPresetValidationResponse = {
  preset: AudioPreset
  valid: boolean
  warnings: AudioPresetValidationWarning[]
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  AudioPreset,
  AudioPresetCreatePayload,
  AudioPresetKind,
  AudioPresetListResponse,
  AudioPresetUpdatePayload,
  AudioPresetValidationResponse
} from "@/types/audio-presets"

export const audioPresetsQueryKey = (kind?: AudioPresetKind) => [
  "audio-presets",
  kind || "all"
]

type UseAudioPresetsOptions = {
  kind?: AudioPresetKind
  enabled?: boolean
}

type UseAudioPresetsResult = {
  presets: AudioPreset[]
  total: number
  loading: boolean
  error: unknown
  refetch: () => Promise<unknown>
  createPreset: (payload: AudioPresetCreatePayload) => Promise<AudioPreset>
  updatePreset: (
    presetId: string,
    payload: AudioPresetUpdatePayload
  ) => Promise<AudioPreset>
  deletePreset: (presetId: string) => Promise<void>
  validatePreset: (presetId: string) => Promise<AudioPresetValidationResponse>
  creating: boolean
  updating: boolean
  deleting: boolean
  validating: boolean
}

export function useAudioPresets(
  options: UseAudioPresetsOptions = {}
): UseAudioPresetsResult {
  const { kind, enabled = true } = options
  const queryClient = useQueryClient()
  const queryKey = audioPresetsQueryKey(kind)

  const invalidatePresets = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey }),
      queryClient.invalidateQueries({ queryKey: audioPresetsQueryKey() })
    ])
  }

  const presetsQuery = useQuery<AudioPresetListResponse>({
    queryKey,
    queryFn: () => tldwClient.listAudioPresets({ kind }),
    enabled
  })

  const createMutation = useMutation({
    mutationFn: (payload: AudioPresetCreatePayload) =>
      tldwClient.createAudioPreset(payload),
    onSuccess: invalidatePresets
  })

  const updateMutation = useMutation({
    mutationFn: ({
      presetId,
      payload
    }: {
      presetId: string
      payload: AudioPresetUpdatePayload
    }) => tldwClient.updateAudioPreset(presetId, payload),
    onSuccess: invalidatePresets
  })

  const deleteMutation = useMutation({
    mutationFn: (presetId: string) => tldwClient.deleteAudioPreset(presetId),
    onSuccess: invalidatePresets
  })

  const validateMutation = useMutation({
    mutationFn: (presetId: string) => tldwClient.validateAudioPreset(presetId)
  })

  return {
    presets: presetsQuery.data?.items ?? [],
    total: presetsQuery.data?.total ?? 0,
    loading: presetsQuery.isLoading,
    error: presetsQuery.error,
    refetch: presetsQuery.refetch,
    createPreset: createMutation.mutateAsync,
    updatePreset: (presetId, payload) =>
      updateMutation.mutateAsync({ presetId, payload }),
    deletePreset: deleteMutation.mutateAsync,
    validatePreset: validateMutation.mutateAsync,
    creating: createMutation.isPending,
    updating: updateMutation.isPending,
    deleting: deleteMutation.isPending,
    validating: validateMutation.isPending
  }
}

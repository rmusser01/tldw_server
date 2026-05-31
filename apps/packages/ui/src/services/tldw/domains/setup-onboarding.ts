import { bgRequest } from "@/services/background-proxy"
import type {
  AudioDefaultsRequest,
  AudioRecommendationsResponse,
  FirstChatVerifyRequest,
  FirstChatVerifyResponse,
  FirstRunCompleteRequest,
  FirstRunMetadata,
  FirstRunSkipRequest,
  FirstRunState,
  FirstRunStepSaveResponse,
  FirstRunStepUpdateRequest,
  IngestDefaultsRequest,
  OptionalAdvancedRequest,
  SetupCompleteResponse,
  SetupProviderCatalogResponse,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse,
  SetupProviderValidationResponse
} from "@/types/setup-onboarding"

const jsonHeaders = { "Content-Type": "application/json" }

export const setupOnboardingMethods = {
  async getFirstRunState(): Promise<FirstRunState> {
    return await bgRequest<FirstRunState>({
      path: "/api/v1/setup/first-run/state",
      method: "GET",
      noAuth: true
    })
  },

  async updateFirstRunState(
    payload: FirstRunStepUpdateRequest
  ): Promise<FirstRunState> {
    return await bgRequest<FirstRunState>({
      path: "/api/v1/setup/first-run/state",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async getFirstRunMetadata(): Promise<FirstRunMetadata> {
    return await bgRequest<FirstRunMetadata>({
      path: "/api/v1/setup/first-run/metadata",
      method: "GET",
      noAuth: true
    })
  },

  async skipFirstRun(payload: FirstRunSkipRequest = {}): Promise<FirstRunState> {
    return await bgRequest<FirstRunState>({
      path: "/api/v1/setup/first-run/skip",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async getSetupProviderCatalog(): Promise<SetupProviderCatalogResponse> {
    return await bgRequest<SetupProviderCatalogResponse>({
      path: "/api/v1/setup/first-run/providers/catalog",
      method: "GET",
      noAuth: true
    })
  },

  async saveSetupProvider(
    payload: SetupProviderSaveRequest
  ): Promise<SetupProviderSaveResponse> {
    return await bgRequest<SetupProviderSaveResponse>({
      path: "/api/v1/setup/first-run/providers",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async validateSetupProvider(
    payload: SetupProviderSaveRequest
  ): Promise<SetupProviderValidationResponse> {
    return await bgRequest<SetupProviderValidationResponse>({
      path: "/api/v1/setup/first-run/providers/validate",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async saveIngestDefaults(
    payload: IngestDefaultsRequest
  ): Promise<FirstRunStepSaveResponse> {
    return await bgRequest<FirstRunStepSaveResponse>({
      path: "/api/v1/setup/first-run/ingest-defaults",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async saveAudioDefaults(
    payload: AudioDefaultsRequest
  ): Promise<FirstRunStepSaveResponse> {
    return await bgRequest<FirstRunStepSaveResponse>({
      path: "/api/v1/setup/first-run/audio-defaults",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async getSetupAudioRecommendations(): Promise<AudioRecommendationsResponse> {
    return await bgRequest<AudioRecommendationsResponse>({
      path: "/api/v1/setup/audio/recommendations",
      method: "GET",
      noAuth: true
    })
  },

  async saveOptionalAdvanced(
    payload: OptionalAdvancedRequest
  ): Promise<FirstRunStepSaveResponse> {
    return await bgRequest<FirstRunStepSaveResponse>({
      path: "/api/v1/setup/first-run/optional-advanced",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async verifyFirstRunChat(
    payload: FirstChatVerifyRequest
  ): Promise<FirstChatVerifyResponse> {
    return await bgRequest<FirstChatVerifyResponse>({
      path: "/api/v1/setup/first-run/first-chat",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  },

  async completeFirstRun(
    payload: FirstRunCompleteRequest = {}
  ): Promise<SetupCompleteResponse> {
    return await bgRequest<SetupCompleteResponse>({
      path: "/api/v1/setup/first-run/complete",
      method: "POST",
      headers: jsonHeaders,
      noAuth: true,
      body: payload
    })
  }
}

export type SetupOnboardingMethods = typeof setupOnboardingMethods

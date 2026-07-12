export const createGroundedClaimVerification = () => ({
  verdict: "grounded" as const,
  metadata: {
    generation_provider: "openai",
    generation_model: "gpt-4o-mini",
    verification_provider: "openai",
    verification_model: "gpt-4o-mini",
    verification_llm_is_default: true
  }
})

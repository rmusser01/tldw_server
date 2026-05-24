export const PARITY_SUMMARY_ARTIFACT_ID = "artifact-parity-summary"

export const PARITY_SUMMARY_ARTIFACT = {
  id: PARITY_SUMMARY_ARTIFACT_ID,
  type: "summary" as const,
  title: "Parity Summary",
  status: "completed" as const,
  content: "Deterministic summary content for workspace parity checks.",
  createdAtIso: "2026-03-04T12:00:00.000Z"
}

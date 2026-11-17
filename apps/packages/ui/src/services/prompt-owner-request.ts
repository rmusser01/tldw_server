import { apiSend, type ApiSendPayload } from "@/services/api-send"

/** Explicit prompt requests dispatch only under their captured authenticated owner. */
export async function requestOwnedPrompt<T>(payload: ApiSendPayload, expectedOwnerId: string | null): Promise<T> {
  if (!expectedOwnerId) throw new Error("Prompt owner is unavailable")
  const response = await apiSend<T>({ ...payload, recipePersistence: { mode: "owner", expectedOwnerId } })
  if (!response.ok) throw new Error(response.error || "Prompt request failed")
  return response.data as T
}

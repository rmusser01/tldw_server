import { beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { createEvaluation, listEvaluations } from "@/services/evaluations"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))

beforeEach(() => vi.clearAllMocks())

describe("canonical evaluation collection routes", () => {
  it.each([
    [undefined, "/api/v1/evaluations/"],
    [{ limit: 20, after: "eval a/b", eval_type: "exact_match" },
      "/api/v1/evaluations/?limit=20&after=eval+a%2Fb&eval_type=exact_match"],
  ] as const)("lists without a redirect and preserves filters %j", async (params, path) => {
    await listEvaluations(params)
    expect(apiSend).toHaveBeenCalledWith({ path, method: "GET" })
  })

  it("creates without a redirect and preserves the body and idempotency key", async () => {
    const payload = { name: "exact", eval_type: "exact_match", eval_spec: { case_sensitive: true } }
    await createEvaluation(payload, { idempotencyKey: "evaluation-create-once" })
    expect(apiSend).toHaveBeenCalledWith({
      path: "/api/v1/evaluations/", method: "POST", body: payload,
      headers: { "Idempotency-Key": "evaluation-create-once" },
    })
  })
})

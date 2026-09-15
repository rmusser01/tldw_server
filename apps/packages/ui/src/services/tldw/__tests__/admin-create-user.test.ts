import { describe, expect, it, vi } from "vitest"
import { adminMethods } from "../domains/admin"
import { bgRequest } from "@/services/background-proxy"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))

describe("admin user creation", () => {
  it("uses the authenticated admin endpoint with the entered account fields", async () => {
    const payload = { username: "alice", email: "alice@example.com", password: "strong-test-password", role: "user" as const }
    const user = { id: 2, username: "alice", role: "user" }
    vi.mocked(bgRequest).mockResolvedValue(user)
    expect(await adminMethods.createAdminUser(payload)).toEqual(user)
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/admin/users", method: "POST",
      headers: { "Content-Type": "application/json" }, body: payload
    })
  })
})

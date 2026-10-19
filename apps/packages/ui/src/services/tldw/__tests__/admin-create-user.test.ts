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

describe("admin workspace captured requests", () => {
  const scope = {
    configSnapshot: { serverUrl: "https://captured.test", authMode: "multi-user" as const },
    abortSignal: new AbortController().signal,
    headers: { "X-TLDW-Expected-User-ID": "42" }
  }
  it.each([
    ["create user", () => adminMethods.createAdminUser({ username: "alice", email: "alice@example.test", password: "test-only-password", role: "user" }, scope)],
    ["list users", () => adminMethods.listAdminUsers({ page: 1 }, scope)],
    ["update user", () => adminMethods.updateAdminUser(2, { is_active: false, reason: "Test captured account" }, scope)],
    ["reset password", () => adminMethods.resetAdminUserPassword(2, { temporary_password: "test-only-password", reason: "Test captured account" }, scope)],
    ["list roles", () => adminMethods.listAdminRoles(scope)],
    ["create role", () => adminMethods.createAdminRole("reader", "Read only", scope)],
    ["delete role", () => adminMethods.deleteAdminRole(2, scope)]
  ])("binds %s to its captured account and cancellation", async (_name, request) => {
    vi.mocked(bgRequest).mockClear()
    await request()
    expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({ ...scope, headers: expect.objectContaining(scope.headers) }))
    const sent = vi.mocked(bgRequest).mock.calls[0][0]
    if (sent.body !== undefined) expect(sent.headers).toHaveProperty("Content-Type", "application/json")
  })
})

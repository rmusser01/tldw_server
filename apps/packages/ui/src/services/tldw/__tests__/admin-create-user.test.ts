import { describe, expect, it, vi } from "vitest";
import { adminMethods } from "../domains/admin";
import { bgRequest } from "@/services/background-proxy";

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }));

describe("admin user creation", () => {
  it("uses the authenticated admin endpoint with the entered account fields", async () => {
    const payload = {
      username: "alice",
      email: "alice@example.com",
      password: "strong-test-password",
      role: "user" as const,
    };
    const user = { id: 2, username: "alice", role: "user" };
    vi.mocked(bgRequest).mockResolvedValue(user);
    expect(await adminMethods.createAdminUser(payload)).toEqual(user);
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/admin/users",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  });
});

describe("admin key authority transport", () => {
  it.each(["users", "list", "create", "revoke", "rotate"])(
    "preserves the immutable snapshot and abort signal for %s",
    async (operation) => {
      const configSnapshot = {
        serverUrl: "https://admin.example.test",
        authMode: "multi-user" as const,
        accessToken: "fixture-token",
      };
      const abortSignal = new AbortController().signal;
      const options = { configSnapshot, abortSignal };
      const payload = { name: "test key" };
      const expected =
        operation === "users"
          ? { path: "/api/v1/admin/users?limit=100", method: "GET" }
          : operation === "list"
            ? { path: "/api/v1/admin/users/11/api-keys", method: "GET" }
            : operation === "create"
              ? {
                  path: "/api/v1/admin/users/11/api-keys",
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: payload,
                }
              : operation === "revoke"
                ? {
                    path: "/api/v1/admin/users/11/api-keys/101",
                    method: "DELETE",
                  }
                : {
                    path: "/api/v1/admin/users/11/api-keys/101/rotate",
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: {},
                  };
      if (operation === "users")
        await adminMethods.listAdminUsers({ limit: 100 }, options);
      else if (operation === "list")
        await adminMethods.listUserApiKeys(11, options);
      else if (operation === "create")
        await adminMethods.createUserApiKey(11, payload, options);
      else if (operation === "revoke")
        await adminMethods.revokeUserApiKey(11, 101, options);
      else await adminMethods.rotateUserApiKey(11, 101, options);
      expect(bgRequest).toHaveBeenLastCalledWith({
        ...expected,
        configSnapshot,
        abortSignal,
      });
      const sent = vi.mocked(bgRequest).mock.calls.at(-1)![0];
      expect(sent.configSnapshot).toBe(configSnapshot);
      expect(sent.abortSignal).toBe(abortSignal);
    },
  );
});

describe("RBAC and membership authority transport", () => {
  const configSnapshot = { serverUrl: "https://admin.example.test", authMode: "multi-user" as const, accessToken: "fixture-token" };
  const options = { configSnapshot, abortSignal: new AbortController().signal };

  it("uses the supported ID-bearing role matrix", async () => {
    await adminMethods.getRolePermissionMatrix(options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/roles/matrix", method: "GET", ...options });
  });

  it("keeps member filters separate from the immutable authority options", async () => {
    await adminMethods.listOrgMembers(71, { role: "member", status: "active" }, options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/orgs/71/members?role=member&status=active", method: "GET", ...options });
  });

  it("retains the optional role description before request options", async () => {
    await adminMethods.createAdminRole("Scoped reviewer", undefined, options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/roles", method: "POST", headers: { "Content-Type": "application/json" }, body: { name: "Scoped reviewer", description: undefined }, ...options });
  });

  it("binds role assignment to both the target user and role", async () => {
    await adminMethods.assignUserRole(11, 31, options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/users/11/roles/31", method: "POST", headers: { "Content-Type": "application/json" }, body: {}, ...options });
  });

  it("binds team creation and member updates to their exact parent IDs", async () => {
    await adminMethods.createTeam(71, { name: "Scoped team" }, options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/orgs/71/teams", method: "POST", headers: { "Content-Type": "application/json" }, body: { name: "Scoped team" }, ...options });
    await adminMethods.updateTeamMemberRole(81, 11, { role: "viewer" }, options);
    expect(bgRequest).toHaveBeenLastCalledWith({ path: "/api/v1/admin/teams/81/members/11", method: "PATCH", headers: { "Content-Type": "application/json" }, body: { role: "viewer" }, ...options });
  });
});

import { describe, expect, it, vi } from "vitest";

import {
  deriveRecipePersistenceOwner,
  type RecipePersistenceOwnerMaterial,
} from "@/services/recipe-persistence-owner";
import * as chatSurfaceScope from "@/services/chat-surface-scope";

const base: RecipePersistenceOwnerMaterial = {
  effectiveBase: "https://recipes.example.test/api",
  authMode: "multi-user",
  authSource: "manual_bearer",
  orgId: "1",
  principalKind: "user",
  principal: "alice",
};

const owner = (
  material: RecipePersistenceOwnerMaterial,
  credentialRevisionMaterial = "token-a",
) => deriveRecipePersistenceOwner(material, credentialRevisionMaterial);

describe("recipe persistence owner contract", () => {
  it.each([
    [
      "auth source",
      { authSource: "manual_api_key" },
      { authSource: "runtime_api_key" },
    ],
    ["organization", { orgId: "1" }, { orgId: "2" }],
    [
      "effective base",
      { effectiveBase: "https://a.test" },
      { effectiveBase: "https://b.test" },
    ],
    ["principal", { principal: "alice" }, { principal: "bob" }],
  ] as const)("separates %s", (_label, left, right) => {
    expect(owner({ ...base, ...left }).ownerId).not.toBe(
      owner({ ...base, ...right }).ownerId,
    );
  });

  it("keeps owner stable but rotates authorization on same-subject token refresh", () => {
    expect(owner(base, "token-a").ownerId).toBe(owner(base, "token-b").ownerId);
    expect(owner(base, "token-a").authorizationRevision).not.toBe(
      owner(base, "token-b").authorizationRevision,
    );
  });

  it.each([
    "manual_api_key",
    "runtime_api_key",
    "manual_bearer",
    "cookie_session",
  ] as const)("derives an opaque view for %s", (authSource) => {
    const view = owner({ ...base, authSource });

    expect(view).toEqual({
      ownerId: expect.stringMatching(/^recipe-owner:sha256:[0-9a-f]{64}$/),
      authorizationRevision: expect.stringMatching(
        /^recipe-authorization:sha256:[0-9a-f]{64}$/,
      ),
    });
  });

  it("normalizes default ports and trailing slashes", () => {
    expect(
      owner({ ...base, effectiveBase: "https://recipes.example.test:443/api/" })
        .ownerId,
    ).toBe(
      owner({ ...base, effectiveBase: "https://recipes.example.test/api" })
        .ownerId,
    );
  });

  it("separates active deployment base paths", () => {
    expect(
      owner({ ...base, effectiveBase: "https://recipes.example.test/deploy-a" })
        .ownerId,
    ).not.toBe(
      owner({ ...base, effectiveBase: "https://recipes.example.test/deploy-b" })
        .ownerId,
    );
  });

  it("does not expose credentials or the custom endpoint", () => {
    const endpoint = "https://custom.internal.test/deployment";
    const credential = "do-not-expose-this-token";
    const view = owner({ ...base, effectiveBase: endpoint }, credential);
    const serializedView = JSON.stringify(view);

    expect(serializedView).not.toContain(endpoint);
    expect(serializedView).not.toContain(credential);
  });

  it.each(["", "relative/path", "ftp://recipes.example.test", "https://"])(
    "rejects malformed effective base %j",
    (effectiveBase) => {
      expect(() => owner({ ...base, effectiveBase })).toThrow(/effectiveBase/);
    },
  );

  it("rejects invalid owner enums and an empty principal", () => {
    expect(() =>
      owner({ ...base, authMode: "unknown" as "multi-user" }),
    ).toThrow(/authMode/);
    expect(() =>
      owner({ ...base, authSource: "unknown" as "manual_bearer" }),
    ).toThrow(/authSource/);
    expect(() =>
      owner({ ...base, principalKind: "unknown" as "user" }),
    ).toThrow(/principalKind/);
    expect(() => owner({ ...base, principal: " " })).toThrow(/principal/);
  });

  it("does not call the global chat surface scope function", () => {
    const scopeSpy = vi.spyOn(chatSurfaceScope, "buildChatSurfaceScopeKey");

    owner(base);

    expect(scopeSpy).not.toHaveBeenCalled();
  });
});

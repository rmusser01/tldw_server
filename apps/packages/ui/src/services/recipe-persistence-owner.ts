import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

export type RecipePersistenceAuthSource =
  "manual_api_key" | "runtime_api_key" | "manual_bearer" | "cookie_session";

export type RecipePersistenceOwnerMaterial = Readonly<{
  effectiveBase: string;
  authMode: "single-user" | "multi-user";
  authSource: RecipePersistenceAuthSource;
  orgId: string | null;
  principalKind: "user" | "api_key";
  principal: string;
}>;

export type RecipePersistenceOwnerView = Readonly<{
  ownerId: string;
  authorizationRevision: string;
}>;

const authModes = new Set(["single-user", "multi-user"]);
const authSources = new Set<RecipePersistenceAuthSource>([
  "manual_api_key",
  "runtime_api_key",
  "manual_bearer",
  "cookie_session",
]);
const principalKinds = new Set(["user", "api_key"]);

const digest = (domain: string, value: string): string =>
  bytesToHex(sha256(utf8ToBytes(`${domain}\0${value}`)));

const normalizeEffectiveBase = (effectiveBase: string): string => {
  if (typeof effectiveBase !== "string" || !effectiveBase.trim()) {
    throw new Error("effectiveBase must be an absolute HTTP(S) URL");
  }

  let url: URL;
  try {
    url = new URL(effectiveBase.trim());
  } catch {
    throw new Error("effectiveBase must be an absolute HTTP(S) URL");
  }

  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    !url.hostname ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("effectiveBase must be an absolute HTTP(S) URL");
  }

  url.pathname = url.pathname.replace(/\/+$/, "") || "/";
  return url.toString();
};

const validateMaterial = (material: RecipePersistenceOwnerMaterial): void => {
  if (!authModes.has(material.authMode)) {
    throw new Error("authMode must be single-user or multi-user");
  }
  if (!authSources.has(material.authSource)) {
    throw new Error("authSource is not supported");
  }
  if (!principalKinds.has(material.principalKind)) {
    throw new Error("principalKind must be user or api_key");
  }
  if (typeof material.principal !== "string" || !material.principal.trim()) {
    throw new Error("principal must be non-empty");
  }
};

export function deriveRecipePersistenceOwner(
  material: RecipePersistenceOwnerMaterial,
  credentialRevisionMaterial: string,
): RecipePersistenceOwnerView {
  validateMaterial(material);
  if (typeof credentialRevisionMaterial !== "string") {
    throw new Error("credentialRevisionMaterial must be a string");
  }

  const canonical = JSON.stringify([
    "recipe-persistence-owner-v1",
    normalizeEffectiveBase(material.effectiveBase),
    material.authMode,
    material.authSource,
    material.orgId ?? null,
    material.principalKind,
    material.principalKind === "api_key"
      ? digest("recipe-api-key-principal-v1", material.principal)
      : material.principal,
  ]);

  return {
    ownerId: `recipe-owner:sha256:${digest(
      "recipe-persistence-owner-id-v1",
      canonical,
    )}`,
    authorizationRevision: `recipe-authorization:sha256:${digest(
      "recipe-persistence-authorization-revision-v1",
      `${canonical}\0${credentialRevisionMaterial}`,
    )}`,
  };
}

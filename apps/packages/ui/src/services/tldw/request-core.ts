import { resolveRequestTimeout } from "@/utils/request-timeout";
import type { ApiSendResponse } from "@/services/api-send";
import type { RecipeDeliveryReceipt } from "@/services/recipe-persistence-registry";
import {
  isCookieSessionBrowserTransport,
  resolveAdvancedRequestTransportGuard,
} from "@/services/tldw/browser-networking";
import type { PathOrUrl } from "@/services/tldw/openapi-guard";
import {
  type RecipeDispatchAuthority,
  type RecipePersistenceDispatch,
  type RecipePersistenceRequestPolicy,
  type RecipeRequestSnapshotResolution,
  type RecipeRequestTransportSnapshot,
  isUnsafeMethod,
  resolveBrowserRequestTransport,
  resolveRecipeRequestSnapshot,
} from "@/services/tldw/recipe-request-snapshot";
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override";
import {
  createServicePromptScopeChangedError,
  isRequestConfigScopeChangedError,
  servicePromptTargetsMatch,
} from "@/services/tldw/service-prompt-scope-error";
import { deriveScopedUserId } from "@/utils/media-navigation-scope";
import {
  ABSOLUTE_URL_BLOCK_ERROR,
  type AllowlistWarnHooks,
  isAbsoluteUrlAllowlisted as guardIsAbsoluteUrlAllowlisted,
  isSameOriginAbsoluteUrlForConfiguredServer as guardIsSameOriginAbsoluteUrlForConfiguredServer,
} from "@/utils/absolute-url-guard";
import { isPlaceholderApiKey } from "@/utils/api-key";
import { formatErrorMessage } from "@/utils/format-error-message";
import { isExplicitRequestCancellation } from "@/services/request-events";

export {
  isUnsafeMethod,
  resolveBrowserRequestTransport,
  type BrowserRequestTransport,
} from "@/services/tldw/recipe-request-snapshot";

/** Stop detaches a caller without cancelling an account-owned token rotation. */
export const waitForAuthRefresh = async (
  refresh: () => Promise<void>,
  signal?: AbortSignal,
): Promise<void> => {
  signal?.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const stopped = () => {
      signal?.removeEventListener("abort", stopped);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", stopped, { once: true });
    void Promise.resolve().then(() => {
      signal?.throwIfAborted();
      return refresh();
    }).then(() => {
      signal?.removeEventListener("abort", stopped);
      resolve();
    }, error => {
      signal?.removeEventListener("abort", stopped);
      reject(error);
    });
    if (signal?.aborted) stopped();
  });
};

export type TldwRequestPayload = {
  path: PathOrUrl;
  method?: string;
  headers?: Record<string, string>;
  body?: any;
  noAuth?: boolean;
  timeoutMs?: number;
  abortSignal?: AbortSignal;
  responseType?: "json" | "text" | "arrayBuffer";
  maxResponseBytes?: number;
  recipePersistence?: RecipePersistenceRequestPolicy;
};

type TldwConfigLike = Record<string, any> | null | undefined;

/**
 * Credential strategy for transport Stage B (ADR-065 / TASK-13526): lets a
 * runtime (the web shell, in Stage B2) supply its own auth headers instead
 * of the default tldwConfig-driven resolution. Default (undefined) keeps the
 * exact current behavior.
 */
export type TldwCredentialResolution =
  | { headers: Record<string, string> }
  | { error: { status: number; error: string } };

export type TldwCredentialContext = {
  cfg: TldwConfigLike;
  method: string;
  isUnsafeMethod: boolean;
  cookieSession: boolean;
  hostedMode: boolean;
  shouldSkipAuth: boolean;
  runtimeApiKey: string;
  existingHeaders: Record<string, string>;
};

type TldwRequestRuntime = {
  getConfig: () => Promise<TldwConfigLike>;
  refreshAuth?: (signal?: AbortSignal) => Promise<void>;
  fetchFn?: typeof fetch;
  useRuntimeAuthOverride?: boolean;
  getAuthenticatedPrincipal?: (
    snapshot: RecipeRequestTransportSnapshot,
  ) => Promise<string | number | null>;
  dispatchAuthority?: RecipeDispatchAuthority;
  requestScopeSignal?: AbortSignal;
  assertRequestCurrent?: () => void;
  releaseRequest?: () => void;
  /**
   * Optional credential strategy (Stage B). When present, the default
   * auth-header resolution in performTldwRequest is replaced by this
   * callback's result. The callback owns stripping/replacing the
   * authorization / x-api-key / x-csrf-token headers it needs to.
   */
  resolveCredentials?: (
    context: TldwCredentialContext,
  ) => TldwCredentialResolution;
};

export const readBrowserCookie = (name: string): string | null => {
  if (typeof document === "undefined") return null;
  const prefix = `${encodeURIComponent(name)}=`;
  for (const part of String(document.cookie || "").split(";")) {
    const cookie = part.trim();
    if (!cookie.startsWith(prefix)) continue;
    try {
      return decodeURIComponent(cookie.slice(prefix.length));
    } catch {
      return null;
    }
  }
  return null;
};

const REQUEST_LOG_PREFIX = "[tldw:request]";
const malformedConfigServerUrlWarnings = new Set<string>();
const malformedAllowlistEntryWarnings = new Set<string>();

const normalizeKnownPathQuirks = (path: PathOrUrl): PathOrUrl => {
  if (typeof path !== "string") return path;
  // Some callers still build media listing URLs as `/api/v1/media/?...`.
  // Certain proxies treat that as a distinct route and return 404.
  return path.replace("/api/v1/media/?", "/api/v1/media?") as PathOrUrl;
};

export const deriveRequestTimeout = (
  cfg: TldwConfigLike,
  path: PathOrUrl,
  override?: number,
): number =>
  resolveRequestTimeout(
    cfg,
    String(normalizeKnownPathQuirks(path) || ""),
    override,
  );

export const parseRetryAfter = (headerValue?: string | null): number | null => {
  if (!headerValue) return null;
  const asNumber = Number(headerValue);
  if (!Number.isNaN(asNumber)) {
    return Math.max(0, asNumber * 1000);
  }
  const asDate = Date.parse(headerValue);
  if (!Number.isNaN(asDate)) {
    return Math.max(0, asDate - Date.now());
  }
  return null;
};

const warnMalformedServerUrl = (raw: string, error: unknown) => {
  const key = raw.trim();
  if (!key || malformedConfigServerUrlWarnings.has(key)) return;
  malformedConfigServerUrlWarnings.add(key);
  console.warn(
    `${REQUEST_LOG_PREFIX} Invalid configured serverUrl: ${key}`,
    error,
  );
};

const warnMalformedAllowlistEntry = (raw: string, error: unknown) => {
  const key = raw.trim();
  if (!key || malformedAllowlistEntryWarnings.has(key)) return;
  malformedAllowlistEntryWarnings.add(key);
  console.warn(
    `${REQUEST_LOG_PREFIX} Invalid absoluteUrlAllowlist entry: ${key}`,
    error,
  );
};

// The origin-allowlist / same-origin primitives live in the canonical
// utils/absolute-url-guard module. request-core keeps its once-per-value
// malformed-config warnings, so it passes those diagnostics through as hooks;
// the actual allowlist/same-origin logic is not duplicated here.
const requestCoreAllowlistWarnHooks: AllowlistWarnHooks = {
  onMalformedServerUrl: warnMalformedServerUrl,
  onMalformedAllowlistEntry: warnMalformedAllowlistEntry,
};

const isSameOriginAbsoluteUrlForConfiguredServer = (
  absoluteUrl: string,
  cfg: TldwConfigLike,
): boolean =>
  guardIsSameOriginAbsoluteUrlForConfiguredServer(
    absoluteUrl,
    cfg,
    requestCoreAllowlistWarnHooks,
  );

const isAbsoluteUrlAllowlisted = (
  absoluteUrl: string,
  cfg: TldwConfigLike,
): boolean =>
  guardIsAbsoluteUrlAllowlisted(
    absoluteUrl,
    cfg,
    requestCoreAllowlistWarnHooks,
  );

export const tldwRequest = async (
  payload: TldwRequestPayload,
  runtime: TldwRequestRuntime,
): Promise<ApiSendResponse> => {
  const assertCurrent = () => runtime.assertRequestCurrent?.();
  const dispatch = payload.recipePersistence
    ? { value: { state: "not_dispatched", actualOwnerId: null } as RecipePersistenceDispatch,
        receipt: undefined as RecipeDeliveryReceipt | undefined }
    : undefined;
  const resultAuthority = () => dispatch ? {
    recipePersistence: dispatch.value,
    ...(dispatch.receipt ? { recipeDelivery: dispatch.receipt } : {}),
  } : {};
  try {
    const response = await performTldwRequest({
      ...payload,
      abortSignal: runtime.requestScopeSignal
        ? payload.abortSignal
          ? AbortSignal.any([payload.abortSignal, runtime.requestScopeSignal])
          : runtime.requestScopeSignal
        : payload.abortSignal,
    }, {
      ...runtime,
      getConfig: async () => {
        assertCurrent();
        const config = await runtime.getConfig();
        assertCurrent();
        return config;
      },
    }, dispatch);
    assertCurrent();
    return { ...response, ...resultAuthority() };
  } catch (error) {
    const failure = runtime.requestScopeSignal?.aborted &&
      isRequestConfigScopeChangedError(runtime.requestScopeSignal.reason)
      ? runtime.requestScopeSignal.reason
      : error;
    if (!dispatch) throw failure;
    return {
      ok: false,
      status: isRequestConfigScopeChangedError(failure) ? 412 : 0,
      error: formatErrorMessage(failure, "Request failed"),
      ...(isRequestConfigScopeChangedError(failure)
        ? { data: (failure as { details?: unknown }).details }
        : {}),
      ...resultAuthority(),
    };
  } finally {
    runtime.releaseRequest?.();
  }
};

const readBoundedArrayBuffer = async (response: Response, maxBytes: number, signal: AbortSignal): Promise<ArrayBuffer> => {
  const tooLarge = () => Object.assign(new Error('Response exceeds the size limit.'), { code: 'RESPONSE_TOO_LARGE' });
  if (Number(response.headers.get('content-length')) > maxBytes) {
    void response.body?.cancel().catch(() => undefined);
    throw tooLarge();
  }
  if (!response.body) return new ArrayBuffer(0);
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => undefined); };
  signal.addEventListener('abort', cancel, { once: true });
  let buffer = new Uint8Array(0);
  let length = 0;
  try {
    if (signal.aborted) cancel();
    signal.throwIfAborted();
    while (true) {
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      const nextLength = length + value.byteLength;
      if (nextLength > maxBytes) {
        cancel();
        throw tooLarge();
      }
      if (nextLength > buffer.length) {
        const grown = new Uint8Array(Math.min(maxBytes, Math.max(nextLength, buffer.length * 2, 64 * 1024)));
        grown.set(buffer.subarray(0, length));
        buffer = grown;
      }
      buffer.set(value, length);
      length = nextLength;
    }
    return buffer.buffer.slice(0, length);
  } finally {
    signal.removeEventListener('abort', cancel);
    reader.releaseLock();
  }
};

/**
 * Stage B (ADR-065 / TASK-13526): pure pre-flight resolution shared by
 * performTldwRequest and (in transport Stage B2) the web shell's raw-response
 * transport. Returns the resolved URL, seed headers, transport diagnostics,
 * and the config context the auth/recipe stages need — or the same
 * guard-failure envelope performTldwRequest has always returned.
 */
export const resolveTldwRequestPlan = async (
  payload: TldwRequestPayload,
  runtime: TldwRequestRuntime,
): Promise<
  | {
      ok: true;
      plan: {
        method: string;
        body: TldwRequestPayload["body"];
        abortSignal?: AbortSignal;
        responseType?: "json" | "text" | "arrayBuffer";
        maxResponseBytes?: number;
        overrideTimeoutMs?: number;
        noAuth: boolean;
        cfg: TldwConfigLike;
        fetchFn: typeof fetch;
        normalizedPath: string | undefined;
        isAbsolute: boolean;
        sameOriginAbsoluteUrl: boolean;
        hostedMode: boolean;
        cookieSession: boolean;
        url: string;
        headers: Record<string, string>;
        isBinaryBody: (value: any) => boolean;
        pageOrigin: string | null;
        shouldSkipAuth: boolean;
      };
    }
  | { ok: false; status: number; error: string }
> => {
  const {
    path,
    method = "GET",
    headers = {},
    body,
    noAuth = false,
    timeoutMs: overrideTimeoutMs,
    abortSignal,
    responseType,
    maxResponseBytes,
  } = payload || {};
  // Extension IPC payloads are runtime input despite the TypeScript signature.
  // Coercing an array/object later would bypass the absolute-URL guard.
  if (typeof path !== "string") {
    return { ok: false, status: 400, error: "Request path must be a string" };
  }
  if (maxResponseBytes !== undefined && (!Number.isSafeInteger(maxResponseBytes) || maxResponseBytes < 1)) {
    return { ok: false, status: 400, error: 'Invalid response size limit.' };
  }
  const normalizedPath = normalizeKnownPathQuirks(path);
  const fetchFn = runtime.fetchFn || fetch;
  const resolvedConfig = await runtime.getConfig();
  const cfg = resolvedConfig ? { ...resolvedConfig } : resolvedConfig;
  const isAbsolute =
    typeof normalizedPath === "string" && /^https?:/i.test(normalizedPath);
  const absolutePath = isAbsolute ? String(normalizedPath) : "";
  const transport =
    !isAbsolute && typeof normalizedPath === "string"
      ? resolveBrowserRequestTransport({
          config: cfg,
          path: String(normalizedPath),
        })
      : null;
  const hostedMode = transport?.mode === "hosted";
  const advancedTransportGuard = resolveAdvancedRequestTransportGuard({
    transport,
    hasConfiguredServerUrl: Boolean(cfg?.serverUrl),
    isAbsolute,
  });
  const sameOriginAbsoluteUrl =
    isAbsolute && isSameOriginAbsoluteUrlForConfiguredServer(absolutePath, cfg);
  const pageOrigin =
    typeof window === "undefined"
      ? null
      : String(window.location?.origin || "");
  const samePageOriginAbsoluteUrl =
    isAbsolute && pageOrigin
      ? isSameOriginAbsoluteUrlForConfiguredServer(absolutePath, {
          serverUrl: pageOrigin,
        })
      : false;
  const absoluteCookieTransport =
    isAbsolute && sameOriginAbsoluteUrl && samePageOriginAbsoluteUrl
      ? resolveBrowserRequestTransport({
          config: cfg,
          path: absolutePath,
          pageOrigin,
        })
      : null;
  const cookieSession = isCookieSessionBrowserTransport({
    authMode: cfg?.authMode,
    authSource: cfg?.authSource,
    transportMode: absoluteCookieTransport?.mode || transport?.mode,
    transportKind: absoluteCookieTransport?.kind || transport?.kind,
    pageOrigin,
  });
  if (
    isAbsolute &&
    !sameOriginAbsoluteUrl &&
    !isAbsoluteUrlAllowlisted(absolutePath, cfg)
  ) {
    return {
      ok: false,
      status: 400,
      error: ABSOLUTE_URL_BLOCK_ERROR,
    };
  }
  if (advancedTransportGuard.isUnconfigured) {
    return { ok: false, status: 400, error: "tldw server not configured" };
  }
  if (!normalizedPath) {
    return { ok: false, status: 400, error: "Request path is required" };
  }
  let url = isAbsolute
    ? normalizedPath
    : transport?.url || String(normalizedPath);
  if (!isAbsolute && transport?.kind === "same-origin" && pageOrigin) {
    try {
      // Browsers interpret //host and /\host as cross-origin URLs, even
      // though neither is classified as an absolute HTTP URL above.
      if (new URL(url, pageOrigin).origin !== pageOrigin) {
        return { ok: false, status: 400, error: ABSOLUTE_URL_BLOCK_ERROR };
      }
    } catch {
      return { ok: false, status: 400, error: ABSOLUTE_URL_BLOCK_ERROR };
    }
  }
  const shouldSkipAuth = noAuth || (isAbsolute && !sameOriginAbsoluteUrl);
  const h: Record<string, string> = { ...(headers || {}) };
  const hasContentType = Object.keys(h).some(
    (key) => key.toLowerCase() === "content-type",
  );
  const isBinaryBody = (value: any) => {
    if (!value || typeof value !== "object") return false;
    if (typeof FormData !== "undefined" && value instanceof FormData)
      return true;
    if (typeof Blob !== "undefined" && value instanceof Blob) return true;
    if (
      typeof URLSearchParams !== "undefined" &&
      value instanceof URLSearchParams
    ) {
      return true;
    }
    if (typeof ArrayBuffer !== "undefined") {
      if (value instanceof ArrayBuffer) return true;
      if (ArrayBuffer.isView?.(value)) return true;
    }
    return false;
  };
  if (
    body != null &&
    !hasContentType &&
    typeof body !== "string" &&
    !isBinaryBody(body)
  ) {
    h["Content-Type"] = "application/json";
  }
  return {
    ok: true,
    plan: {
      method,
      body,
      abortSignal,
      responseType,
      maxResponseBytes,
      overrideTimeoutMs,
      noAuth,
      cfg,
      fetchFn,
      normalizedPath,
      isAbsolute,
      sameOriginAbsoluteUrl,
      hostedMode,
      cookieSession,
      url,
      headers: h,
      isBinaryBody,
      pageOrigin,
      shouldSkipAuth,
    },
  };
};

const performTldwRequest = async (
  payload: TldwRequestPayload,
  runtime: TldwRequestRuntime,
  dispatch?: {
    value: RecipePersistenceDispatch;
    receipt?: RecipeDeliveryReceipt;
  },
): Promise<ApiSendResponse> => {
  const resolvedPlan = await resolveTldwRequestPlan(payload, runtime);
  if (resolvedPlan.ok !== true) {
    return {
      ok: false,
      status: resolvedPlan.status,
      error: resolvedPlan.error,
    };
  }
  const {
    method,
    body,
    abortSignal,
    responseType,
    maxResponseBytes,
    overrideTimeoutMs,
    noAuth,
    cfg,
    fetchFn,
    normalizedPath,
    isAbsolute,
    sameOriginAbsoluteUrl,
    hostedMode,
    cookieSession,
    url: resolvedUrl,
    headers: h,
    isBinaryBody,
    pageOrigin,
    shouldSkipAuth,
  } = resolvedPlan.plan;
  let url = resolvedUrl;

  const runtimeApiKey =
    runtime.useRuntimeAuthOverride === false
      ? ""
      : String(getRuntimeSingleUserApiKeyOverride() || "").trim();
  let recipeSnapshot: RecipeRequestSnapshotResolution | null = null;
  if (payload.recipePersistence) {
    const snapshotInput = {
      config: cfg,
      path: String(normalizedPath),
      method,
      headers: h,
      noAuth,
      runtimeApiKey,
      csrfToken: readBrowserCookie("csrf_token"),
      pageOrigin,
      absoluteAuthAllowed: sameOriginAbsoluteUrl,
      cookieSessionTransport: cookieSession,
    };
    recipeSnapshot = resolveRecipeRequestSnapshot(snapshotInput);
    const needsAuthenticatedPrincipal =
      recipeSnapshot.snapshot.credentials === "same-origin" ||
      Object.keys(recipeSnapshot.snapshot.headers).some(
        (key) => key.toLowerCase() === "authorization",
      );
    if (needsAuthenticatedPrincipal && runtime.getAuthenticatedPrincipal) {
      const authenticatedPrincipal = await runtime.getAuthenticatedPrincipal(
        recipeSnapshot.snapshot,
      );
      recipeSnapshot = resolveRecipeRequestSnapshot({
        ...snapshotInput,
        authenticatedPrincipalId: authenticatedPrincipal,
      });
    }
    url = recipeSnapshot.snapshot.url as PathOrUrl;
    for (const key of Object.keys(h)) delete h[key];
    Object.assign(h, recipeSnapshot.snapshot.headers);
    if (recipeSnapshot.authenticationError) {
      return {
        ok: false,
        status: recipeSnapshot.authenticationError.status,
        error: recipeSnapshot.authenticationError.error,
      };
    }

    if (payload.recipePersistence.mode === "require") {
      const expectedOwnerId = payload.recipePersistence.expectedOwnerId;
      const localId = payload.recipePersistence.localId;
      const validExpectedOwner =
        typeof expectedOwnerId === "string" &&
        /^recipe-owner:sha256:[0-9a-f]{64}$/.test(expectedOwnerId);
      const validLocalId =
        typeof localId === "string" &&
        localId.length > 0 &&
        localId === localId.trim();
      if (
        !validExpectedOwner ||
        !validLocalId ||
        !recipeSnapshot.view ||
        recipeSnapshot.view.ownerId !== expectedOwnerId ||
        !runtime.dispatchAuthority
      ) {
        return {
          ok: false,
          status: 412,
          error: "Request persistence owner is unavailable or changed",
        };
      }
    }
  } else if (runtime.resolveCredentials) {
    // Stage B credential strategy (ADR-065): the runtime owns auth headers.
    const resolution = runtime.resolveCredentials({
      cfg,
      method,
      isUnsafeMethod: isUnsafeMethod(method),
      cookieSession,
      hostedMode,
      shouldSkipAuth,
      runtimeApiKey,
      existingHeaders: h,
    });
    if ("error" in resolution) {
      return { ok: false, status: resolution.error.status, error: resolution.error.error };
    }
    for (const k of Object.keys(h)) {
      const kl = k.toLowerCase();
      if (
        kl === "x-api-key" ||
        kl === "authorization" ||
        kl === "x-csrf-token"
      ) {
        delete h[k];
      }
    }
    Object.assign(h, resolution.headers);
  } else if (cookieSession) {
    for (const k of Object.keys(h)) {
      const kl = k.toLowerCase();
      if (
        kl === "x-api-key" ||
        kl === "authorization" ||
        kl === "x-csrf-token"
      ) {
        delete h[k];
      }
    }
    if (!shouldSkipAuth && isUnsafeMethod(method)) {
      const csrfToken = readBrowserCookie("csrf_token");
      if (csrfToken) h["X-CSRF-Token"] = csrfToken;
    }
  } else if (!shouldSkipAuth) {
    for (const k of Object.keys(h)) {
      const kl = k.toLowerCase();
      if (kl === "x-api-key" || kl === "authorization") delete h[k];
    }
    if (!hostedMode) {
      if (runtimeApiKey && !isPlaceholderApiKey(runtimeApiKey)) {
        h["X-API-KEY"] = runtimeApiKey;
      } else if (cfg?.authMode === "single-user") {
        const key = (cfg?.apiKey || "").trim();
        if (!key) {
          if (runtimeApiKey && isPlaceholderApiKey(runtimeApiKey)) {
            return {
              ok: false,
              status: 401,
              error:
                "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings -> tldw server before continuing.",
            };
          }
          return {
            ok: false,
            status: 401,
            error:
              "Add or update your API key in Settings -> tldw server, then try again.",
          };
        }
        if (isPlaceholderApiKey(key)) {
          return {
            ok: false,
            status: 401,
            error:
              "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings -> tldw server before continuing.",
          };
        }
        h["X-API-KEY"] = key;
      } else if (cfg?.authMode === "multi-user") {
        const token = (cfg?.accessToken || "").trim();
        if (token) h["Authorization"] = `Bearer ${token}`;
        else {
          return {
            ok: false,
            status: 401,
            error: "Not authenticated. Please login under Settings > tldw.",
          };
        }
      }
    }
    if (cfg?.orgId) {
      h["X-TLDW-Org-Id"] = String(cfg.orgId);
    }
  }

  const controller = new AbortController();
  let retryController: AbortController | null = null;
  const timeoutMs = deriveRequestTimeout(
    cfg,
    (normalizedPath ?? "") as Parameters<typeof deriveRequestTimeout>[1],
    Number(overrideTimeoutMs),
  );
  const onAbort = () => {
    try {
      controller.abort();
      retryController?.abort();
    } catch {}
  };
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let retryTimeoutId: ReturnType<typeof setTimeout> | null = null;

  try {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    if (abortSignal) {
      if (abortSignal.aborted) {
        controller.abort();
      } else {
        abortSignal.addEventListener("abort", onAbort, { once: true });
      }
    }

    const resolvedBody =
      body == null
        ? undefined
        : typeof body === "string" || isBinaryBody(body)
          ? body
          : JSON.stringify(body);

    // lgtm[js/request-forgery]: url is same-origin/configured-server transport or an allowlisted absolute URL checked above.
    controller.signal.throwIfAborted();
    if (dispatch && payload.recipePersistence) {
      const actualOwnerId = recipeSnapshot?.view?.ownerId ?? null;
      if (payload.recipePersistence.mode === "require") {
        const receipt = await runtime.dispatchAuthority!.markDispatched(
          payload.recipePersistence.localId,
          actualOwnerId!,
        );
        if (receipt) dispatch.receipt = receipt;
      }
      dispatch.value = { state: "dispatched", actualOwnerId };
    }
    controller.signal.throwIfAborted();
    let resp = await fetchFn(url, {
      redirect: "error",
      method,
      headers: h,
      body: resolvedBody,
      signal: controller.signal,
      ...(recipeSnapshot
        ? recipeSnapshot.snapshot.credentials
          ? { credentials: recipeSnapshot.snapshot.credentials }
          : {}
        : cookieSession
          ? { credentials: "same-origin" as const }
          : {}),
    });
    // Headers have arrived; fetch() resolves before the body is read. Re-arm the
    // timeout so the body read below is bounded too — otherwise a server that
    // sends headers then stalls hangs forever despite the "timeout".
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    if (
      !shouldSkipAuth &&
      !hostedMode &&
      resp.status === 401 &&
      cfg?.authMode === "multi-user" &&
      cfg?.refreshToken &&
      runtime.refreshAuth
    ) {
      // The first request is finished; stop its timer before refreshing/retrying.
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      try {
        await waitForAuthRefresh(() => runtime.refreshAuth!(abortSignal), abortSignal);
      } catch (refreshError) {
        if (abortSignal?.aborted) throw refreshError;
        if (isRequestConfigScopeChangedError(refreshError)) {
          throw refreshError;
        }
        const failure = refreshError as Partial<ApiSendResponse> | null;
        return {
          ok: false,
          status: typeof failure?.status === "number" ? failure.status : 0,
          error: formatErrorMessage(refreshError, "Unable to refresh session."),
          code: failure?.code,
          headers: failure?.headers,
          retryAfterMs: failure?.retryAfterMs,
        };
      }
      if (abortSignal?.aborted) {
        const abortError = new Error(
          "Request was aborted during token refresh.",
        );
        abortError.name = "AbortError";
        throw abortError;
      }
      const updated = await runtime.getConfig();
      let retryHeaders: Record<string, string>;
      let retryCredentials: RequestCredentials | undefined;
      if (recipeSnapshot) {
        const updatedConfig = updated ? { ...updated } : updated;
        const updatedSnapshotInput = {
          config: updatedConfig,
          path: String(normalizedPath),
          method,
          headers: h,
          noAuth,
          runtimeApiKey:
            runtime.useRuntimeAuthOverride === false
              ? ""
              : String(getRuntimeSingleUserApiKeyOverride() || "").trim(),
          csrfToken: readBrowserCookie("csrf_token"),
          pageOrigin,
          absoluteAuthAllowed: sameOriginAbsoluteUrl,
          cookieSessionTransport: cookieSession,
        };
        let updatedSnapshot =
          resolveRecipeRequestSnapshot(updatedSnapshotInput);
        const needsAuthenticatedPrincipal =
          updatedSnapshot.snapshot.credentials === "same-origin" ||
          Object.keys(updatedSnapshot.snapshot.headers).some(
            (key) => key.toLowerCase() === "authorization",
          );
        if (needsAuthenticatedPrincipal && runtime.getAuthenticatedPrincipal) {
          const authenticatedPrincipal =
            await runtime.getAuthenticatedPrincipal(updatedSnapshot.snapshot);
          updatedSnapshot = resolveRecipeRequestSnapshot({
            ...updatedSnapshotInput,
            authenticatedPrincipalId: authenticatedPrincipal,
          });
        }
        if (
          updatedSnapshot.authenticationError ||
          !recipeSnapshot.view?.ownerId ||
          !updatedSnapshot.view?.ownerId ||
          updatedSnapshot.snapshot.url !== recipeSnapshot.snapshot.url ||
          updatedSnapshot.snapshot.effectiveBase !==
            recipeSnapshot.snapshot.effectiveBase ||
          updatedSnapshot.view?.ownerId !== recipeSnapshot.view?.ownerId
        ) {
          return {
            ok: false,
            status: 412,
            error: "Request persistence owner changed before retry",
          };
        }
        retryHeaders = { ...updatedSnapshot.snapshot.headers };
        retryCredentials = updatedSnapshot.snapshot.credentials;
      } else {
        if (
          !servicePromptTargetsMatch(cfg || {}, updated || {}) ||
          deriveScopedUserId({
            userId: null,
            authMode: cfg?.authMode,
            accessToken: cfg?.accessToken,
          }) !==
            deriveScopedUserId({
              userId: null,
              authMode: updated?.authMode,
              accessToken: updated?.accessToken,
            })
        ) {
          throw createServicePromptScopeChangedError();
        }
        retryHeaders = { ...h };
        for (const k of Object.keys(retryHeaders)) {
          const kl = k.toLowerCase();
          if (kl === "authorization" || kl === "x-api-key")
            delete retryHeaders[k];
        }
        if (updated?.accessToken) {
          retryHeaders["Authorization"] = `Bearer ${updated.accessToken}`;
        }
      }
      retryController = new AbortController();
      const activeRetryController = retryController;
      if (abortSignal?.aborted) {
        const abortError = new Error("Request was aborted before retry.");
        abortError.name = "AbortError";
        throw abortError;
      }
      retryTimeoutId = setTimeout(
        () => activeRetryController.abort(),
        timeoutMs,
      );
      // lgtm[js/request-forgery]: retry reuses the same validated URL from the initial request.
      resp = await fetchFn(url, {
        redirect: "error",
        method,
        headers: retryHeaders,
        // Reuse the binary-aware serialization from the first attempt. A plain
        // JSON.stringify here corrupts FormData/Blob uploads into "{}".
        body: resolvedBody,
        signal: activeRetryController.signal,
        ...(retryCredentials ? { credentials: retryCredentials } : {}),
      });
      // Re-arm so the retry body read is bounded as well.
      if (retryTimeoutId) clearTimeout(retryTimeoutId);
      retryTimeoutId = setTimeout(
        () => activeRetryController.abort(),
        timeoutMs,
      );
    }

    const headersOut: Record<string, string> = {};
    try {
      resp.headers.forEach((value, key) => {
        headersOut[key] = value;
      });
    } catch {}

    const retryAfterMs = parseRetryAfter(resp.headers?.get?.("retry-after"));
    const contentType = resp.headers.get("content-type") || "";
    let data: any = null;
    const handleBodyReadError = (error: unknown) => {
      if (isExplicitRequestCancellation(error)) throw error;
      return null;
    };
    const readDefaultBody = async () => {
      if (contentType.includes("application/json")) {
        return await resp.json().catch(handleBodyReadError);
      }
      return await resp.text().catch(handleBodyReadError);
    };
    if (responseType === "arrayBuffer") {
      if (maxResponseBytes !== undefined) {
        const buffer = await readBoundedArrayBuffer(resp, maxResponseBytes, retryController?.signal ?? controller.signal);
        if (resp.ok) data = buffer;
        else {
          const text = new TextDecoder().decode(buffer);
          if (contentType.includes('application/json')) {
            try { data = JSON.parse(text); } catch { data = null; }
          } else data = text;
        }
      } else data = resp.ok ? await resp.arrayBuffer().catch(handleBodyReadError) : await readDefaultBody();
    } else if (responseType === "json") {
      data = await resp.json().catch(handleBodyReadError);
    } else if (responseType === "text") {
      data = await resp.text().catch(handleBodyReadError);
    } else {
      data = await readDefaultBody();
    }

    if (!resp.ok) {
      let detail: unknown = undefined;
      if (typeof data === "object" && data) {
        const raw = data.detail ?? data.error ?? data.message;
        // FastAPI validation errors return detail as an array
        if (Array.isArray(raw)) {
          detail = raw
            .map((item: any) =>
              typeof item === "string"
                ? item
                : typeof item?.msg === "string"
                  ? item.msg
                  : JSON.stringify(item),
            )
            .join("; ");
        } else if (raw !== undefined && raw !== null) {
          detail = raw;
        }
      }
      const errorMessage = formatErrorMessage(
        detail !== undefined && detail !== null
          ? detail
          : resp.statusText || `HTTP ${resp.status}`,
        `HTTP ${resp.status}`,
      );
      return {
        ok: false,
        status: resp.status,
        error: errorMessage,
        data,
        headers: headersOut,
        retryAfterMs,
      };
    }

    return {
      ok: true,
      status: resp.status,
      data,
      headers: headersOut,
      retryAfterMs,
    };
  } catch (e: any) {
    if (e?.code === 'RESPONSE_TOO_LARGE') {
      return { ok: false, status: 0, error: e.message, code: 'RESPONSE_TOO_LARGE' };
    }
    // Internal deadlines also raise AbortError; only the caller's signal marks
    // deliberate cancellation that should suppress connection/error feedback.
    if (abortSignal?.aborted) {
      return {
        ok: false,
        status: 0,
        error: "Request aborted.",
        code: "REQUEST_ABORTED",
      };
    }
    if (controller.signal.aborted || retryController?.signal.aborted) {
      return {
        ok: false,
        status: 0,
        error: "Request timed out.",
        code: "REQUEST_TIMEOUT",
      };
    }
    if (isRequestConfigScopeChangedError(e)) throw e;
    if (isExplicitRequestCancellation(e)) {
      return { ok: false, status: 0, error: formatErrorMessage(e, "Request aborted."), code: "REQUEST_ABORTED" };
    }
    return {
      ok: false,
      status: 0,
      error: formatErrorMessage(e, "Network error"),
    };
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    if (retryTimeoutId) {
      clearTimeout(retryTimeoutId);
    }
    if (abortSignal) {
      try {
        abortSignal.removeEventListener("abort", onAbort);
      } catch {}
    }
  }
};

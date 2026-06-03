// Non-secret placeholder used by local single-user E2E server fixtures only.
export const DEFAULT_E2E_API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY";

const EXPLICIT_API_KEY_ENV_KEYS = [
  "TLDW_API_KEY",
  "TLDW_E2E_API_KEY",
  "SINGLE_USER_API_KEY",
] as const;

type E2eAuthEnvironment = Record<string, string | undefined>;

export const resolveExplicitE2eApiKey = (
  env: E2eAuthEnvironment = process.env
): string | undefined => {
  for (const key of EXPLICIT_API_KEY_ENV_KEYS) {
    const value = env[key];
    if (value?.trim()) return value.trim();
  }
  return undefined;
};

export const isLocalE2eServerUrl = (serverUrl: string): boolean => {
  try {
    const { hostname, protocol } = new URL(serverUrl);
    if (protocol === "file:") return true;
    const normalizedHost = hostname.toLowerCase().replace(/^\[|\]$/g, "");
    return (
      normalizedHost === "localhost" ||
      normalizedHost.endsWith(".localhost") ||
      normalizedHost === "127.0.0.1" ||
      normalizedHost === "::1" ||
      normalizedHost === "0.0.0.0"
    );
  } catch {
    return false;
  }
};

export const resolveE2eApiKey = ({
  serverUrl,
  env = process.env,
}: {
  serverUrl: string;
  env?: E2eAuthEnvironment;
}): string => {
  const explicitApiKey = resolveExplicitE2eApiKey(env);
  if (explicitApiKey) return explicitApiKey;

  if (!isLocalE2eServerUrl(serverUrl)) {
    throw new Error(
      [
        `Remote E2E server URL "${serverUrl}" is configured without an explicit API key.`,
        `Set ${EXPLICIT_API_KEY_ENV_KEYS.join(", ")} before running against non-local servers.`,
      ].join(" ")
    );
  }

  return DEFAULT_E2E_API_KEY;
};

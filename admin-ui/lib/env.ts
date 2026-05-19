import { z } from 'zod';

/**
 * Build-time environment schema.
 * NEXT_PUBLIC_* vars are baked into the client JS bundle at build time and are
 * NOT available as runtime env vars in a Docker container.  Validate these
 * during development (where they live in .env / .env.local).
 */
const buildEnvSchema = z.object({
  NEXT_PUBLIC_API_URL: z
    .string()
    .min(1, 'NEXT_PUBLIC_API_URL is required')
    .url('NEXT_PUBLIC_API_URL must be a valid URL'),
  NEXT_PUBLIC_API_VERSION: z.string().default('v1'),
  NEXT_PUBLIC_DEFAULT_AUTH_MODE: z
    .enum(['password', 'apikey'])
    .default('password'),
  NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN: z
    .string()
    .transform((v) => v === 'true')
    .default(false),
  NEXT_PUBLIC_BILLING_ENABLED: z
    .string()
    .transform((v) => v === 'true')
    .default(false),
});

/**
 * Runtime environment schema.
 * Server-side secrets that MUST be present when the container starts.
 */
const runtimeEnvSchema = z.object({
  JWT_SECRET_KEY: z.string().min(1, 'JWT_SECRET_KEY is required'),
  JWT_ALGORITHM: z.string().default('HS256'),
  JWT_SECONDARY_SECRET: z.string().optional(),
});

export type AppEnv = z.infer<typeof buildEnvSchema>;
export type RuntimeEnv = z.infer<typeof runtimeEnvSchema>;

let cachedEnv: AppEnv | null = null;
let cachedRuntimeEnv: RuntimeEnv | null = null;

/**
 * Validate build-time (NEXT_PUBLIC_*) environment variables.
 * Use in development where these vars are available via .env files.
 */
export function validateEnv(): AppEnv {
  if (cachedEnv) return cachedEnv;

  const result = buildEnvSchema.safeParse(process.env);
  if (!result.success) {
    const errors = result.error.issues
      .map((i) => `  ${i.path.join('.')}: ${i.message}`)
      .join('\n');
    throw new Error(`Environment validation failed:\n${errors}`);
  }

  cachedEnv = result.data;
  return cachedEnv;
}

/**
 * Validate runtime (server-side) environment variables.
 * Use at production startup to fail fast on missing secrets.
 */
export function validateRuntimeEnv(): RuntimeEnv {
  if (cachedRuntimeEnv) return cachedRuntimeEnv;

  const result = runtimeEnvSchema.safeParse(process.env);
  if (!result.success) {
    const errors = result.error.issues
      .map((i) => `  ${i.path.join('.')}: ${i.message}`)
      .join('\n');
    throw new Error(`Runtime environment validation failed:\n${errors}`);
  }

  cachedRuntimeEnv = result.data;
  return cachedRuntimeEnv;
}

import path from 'path';
import { fileURLToPath } from 'url';
import { validateNetworkingConfig } from './scripts/validate-networking-config.mjs';

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const paTesseractPath = path.resolve(__dirname, 'node_modules/pa-tesseract.js');
const repoWorkspaceRoot = path.resolve(__dirname, '../..');
const backendRuntimeWatchIgnoreRoots = [
  'Databases',
  'tldw_Server_API/Databases',
  'tldw_Server_API/Logs',
  'logs',
].map((root) => path.resolve(repoWorkspaceRoot, root).split(path.sep).join('/'));
const backendRuntimeWatchIgnorePatterns = backendRuntimeWatchIgnoreRoots.map(
  (root) => `${root}/**`
);
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const backendRuntimeWatchIgnoreSource = backendRuntimeWatchIgnoreRoots
  .map((root) => `^${escapeRegExp(root)}(?:/|$)`)
  .join('|');
const liveTierDistDir = process.env.TLDW_NEXT_DIST_DIR;
if (
  liveTierDistDir &&
  !/^\.next-live-tier-[A-Za-z0-9._-]+$/.test(liveTierDistDir)
) {
  throw new Error(
    'TLDW_NEXT_DIST_DIR must be a direct .next-live-tier-* child directory'
  );
}
const {
  deploymentMode,
  internalApiOrigin: validatedInternalApiOrigin
} = validateNetworkingConfig(process.env);
const internalApiOrigin = validatedInternalApiOrigin.replace(/\/$/, '');

// Content-Security-Policy (defense-in-depth for DOM-XSS; TASK-12093 + H1 follow-up).
// Locks script sources to the app origin, forbids plugins (object-src) and <base>
// hijacking (base-uri), and blocks framing of the app (frame-ancestors).
//
// H1 follow-up: script-src drops 'unsafe-inline'. Arbitrary inline <script> and
// javascript: URIs are now blocked by the browser itself. The one trusted inline
// script — the _document theme bootstrap — is allowlisted by its SHA-256 hash.
//
// Why a hash and NOT a per-request nonce: this is a Next.js pages-router app with
// automatic static optimization (see .next/server/pages/*.html). Prerendered HTML
// cannot carry a per-request nonce, so a middleware/nonce policy would blank out
// every statically optimized page. A hash is identical at build time and request
// time, needs no middleware, and cannot mismatch — the correct fit here.
//
// The only executable inline script Next emits is this theme bootstrap (verified
// against the prerendered HTML). Framework code loads as external /_next/static
// chunks (covered by 'self'); __NEXT_DATA__ is type="application/json", a data
// island browsers never execute, so script-src does not gate it. 'strict-dynamic'
// is intentionally NOT used: it would cause CSP3 browsers to ignore 'self' and
// block those parser-inserted external chunks.
//
// IMPORTANT: if you edit THEME_BOOTSTRAP_SCRIPT in pages/_document.tsx you MUST
// regenerate the hash below or the theme bootstrap will be blocked (theme flash;
// app still functions). Regenerate from apps/tldw-frontend/ with:
//   node -e "const s=require('fs').readFileSync('pages/_document.tsx','utf8').match(/THEME_BOOTSTRAP_SCRIPT = \`([\s\S]*?)\`;/)[1];console.log('sha256-'+require('crypto').createHash('sha256').update(s).digest('base64'))"
//
// Notes:
//   - 'unsafe-eval' is RETAINED: OCR/tokenizer WASM and some runtime deps may
//     rely on it, and this could not be verified end-to-end in a browser as part
//     of this change. Narrowing to 'wasm-unsafe-eval' (or dropping eval entirely)
//     is a tracked follow-up that needs manual per-feature browser verification.
//   - blob: is required for Web Workers (OCR/diff/tokenizer) and blob: iframe
//     previews (CodeBlock/PDF split view).
//   - img-/media-/connect-src stay broad so external thumbnails, remote
//     backends, and realtime-audio WebSockets keep working.
const themeBootstrapScriptHash =
  "'sha256-0MRPvgBCfD2/IgaYuIOc/QNI8hETlwrwO53VUw3LEkM='";
const contentSecurityPolicy = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  `script-src 'self' 'unsafe-eval' blob: ${themeBootstrapScriptHash}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https: http:",
  "font-src 'self' data:",
  "media-src 'self' data: blob: https: http:",
  "connect-src 'self' https: http: ws: wss: data: blob:",
  "worker-src 'self' blob:",
  "frame-src 'self' blob: data: https: http:"
].join('; ');

const nextConfig = {
  reactStrictMode: true,
  reactCompiler: false,
  ...(liveTierDistDir ? { distDir: liveTierDistDir } : {}),
  // Preserve backend API paths exactly in quickstart mode. FastAPI routes such as
  // POST /api/v1/chats/ are slash-sensitive and otherwise bounce through redirects
  // before the same-origin rewrite reaches the real backend.
  skipTrailingSlashRedirect: true,
  // Support loopback-origin access during local dev, e.g. 127.0.0.1 -> localhost.
  allowedDevOrigins: ['localhost', '127.0.0.1', '[::1]'],
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: contentSecurityPolicy,
          },
          // Block MIME sniffing (defense against content-type confusion XSS).
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          // Trim referrer leakage on cross-origin navigations.
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          // Legacy clickjacking guard; complements CSP frame-ancestors 'none'.
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          // Least-privilege browser features. Mic is allowed for same-origin
          // audio capture (realtime transcription/dictation); camera and
          // geolocation are unused and denied.
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(self), geolocation=()',
          },
        ],
      },
    ];
  },
  async redirects() {
    return [
      {
        source: '/chat/settings',
        destination: '/settings/chat',
        permanent: false,
      },
    ];
  },
  async rewrites() {
    if (deploymentMode !== 'quickstart' || !internalApiOrigin) {
      return [];
    }

    return [
      {
        source: '/health',
        destination: `${internalApiOrigin}/health`,
      },
      {
        source: '/api/v1/media',
        destination: `${internalApiOrigin}/api/v1/media/`,
      },
      {
        source: '/api/:path*/',
        destination: `${internalApiOrigin}/api/:path*/`,
      },
      {
        source: '/api/:path*',
        destination: `${internalApiOrigin}/api/:path*`,
      },
    ];
  },
  // Emit a standalone server bundle for distribution via Docker or tarball artifacts.
  output: 'standalone',
  // Skip TypeScript errors during build - packages/ui was developed with
  // Vite/WXT type definitions that Next.js doesn't provide.
  // Runtime works correctly; these are type-definition mismatches.
  typescript: {
    ignoreBuildErrors: true,
  },
  turbopack: {
    root: repoWorkspaceRoot,
    // Keep Turbopack aliases aligned with shared UI + web shims.
    resolveAlias: {
      '@tldw/ui': '../packages/ui/src',
      '@': '../packages/ui/src',
      '~': '../packages/ui/src',
      '@web': '.',
      'pa-tesseract.js': './node_modules/pa-tesseract.js',
      'react-router-dom': './extension/shims/react-router-dom.tsx',
      '@plasmohq/storage': './extension/shims/plasmo-storage.ts',
      '@plasmohq/storage/hook': './extension/shims/plasmo-storage-hook.tsx',
      'wxt/browser': './extension/shims/wxt-browser.ts',
    },
  },
  // Ensure Next resolves the correct monorepo root when multiple lockfiles exist.
  outputFileTracingRoot: repoWorkspaceRoot,
  transpilePackages: ['@tldw/ui'],
  webpack: (config) => {
    // Support extension-aligned aliases + shims
    config.resolve.alias['@tldw/ui'] = path.resolve(__dirname, '../packages/ui/src');
    config.resolve.alias['@'] = path.resolve(__dirname, '../packages/ui/src');
    config.resolve.alias['~'] = path.resolve(__dirname, '../packages/ui/src');
    config.resolve.alias['@web'] = path.resolve(__dirname, '.');
    config.resolve.alias['pa-tesseract.js'] = paTesseractPath;
    config.resolve.alias['react-router-dom'] = path.resolve(
      __dirname,
      'extension/shims/react-router-dom.tsx'
    );
    config.resolve.alias['@plasmohq/storage'] = path.resolve(
      __dirname,
      'extension/shims/plasmo-storage.ts'
    );
    config.resolve.alias['@plasmohq/storage/hook'] = path.resolve(
      __dirname,
      'extension/shims/plasmo-storage-hook.tsx'
    );
    config.resolve.alias['wxt/browser'] = path.resolve(
      __dirname,
      'extension/shims/wxt-browser.ts'
    );
    const existingIgnored = config.watchOptions?.ignored;
    if (existingIgnored instanceof RegExp) {
      config.watchOptions = {
        ...config.watchOptions,
        ignored: new RegExp(
          `(?:${existingIgnored.source})|(?:${backendRuntimeWatchIgnoreSource})`,
          existingIgnored.flags.replace(/[gy]/g, '')
        ),
      };
      return config;
    }
    const normalizedExistingIgnored = (
      Array.isArray(existingIgnored)
        ? existingIgnored
        : existingIgnored == null
          ? []
          : [existingIgnored]
    ).filter((item) => typeof item === 'string' && item.trim());
    config.watchOptions = {
      ...config.watchOptions,
      ignored: [
        ...normalizedExistingIgnored,
        ...backendRuntimeWatchIgnorePatterns,
      ],
    };
    return config;
  },
};

// Sentry wrapping (optional — only active when NEXT_PUBLIC_SENTRY_DSN is set)
let withSentryConfig = (/** @type {import('next').NextConfig} */ c) => c;
if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
  try {
    const sentry = await import('@sentry/nextjs');
    withSentryConfig = sentry.withSentryConfig;
  } catch (error) {
    console.warn('[next.config] Skipping Sentry integration:', error)
  }
}

export default withSentryConfig(nextConfig, {
  silent: true,
  disableLogger: true,
});

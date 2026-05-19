import path from 'path';
import { fileURLToPath } from 'url';
import { validateNetworkingConfig } from './scripts/validate-networking-config.mjs';

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const paTesseractPath = path.resolve(__dirname, 'node_modules/pa-tesseract.js');
const {
  deploymentMode,
  internalApiOrigin: validatedInternalApiOrigin
} = validateNetworkingConfig(process.env);
const internalApiOrigin = validatedInternalApiOrigin.replace(/\/$/, '');

const nextConfig = {
  reactStrictMode: true,
  reactCompiler: false,
  // Preserve backend API paths exactly in quickstart mode. FastAPI routes such as
  // POST /api/v1/chats/ are slash-sensitive and otherwise bounce through redirects
  // before the same-origin rewrite reaches the real backend.
  skipTrailingSlashRedirect: true,
  // Support loopback-origin access during local dev, e.g. 127.0.0.1 -> localhost.
  allowedDevOrigins: ['localhost', '127.0.0.1', '[::1]'],
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
  outputFileTracingRoot: path.resolve(__dirname, '../..'),
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

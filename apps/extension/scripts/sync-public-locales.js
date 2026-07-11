const fs = require('fs');
const path = require('path');

const root = process.cwd();
// Go up one directory from apps/extension to apps where packages/ui is
const sharedSrc = path.join(root, '..', 'packages', 'ui', 'src');
const assetsBase = path.join(sharedSrc, 'assets', 'locale');
const publicBase = path.join(sharedSrc, 'public', '_locales');
// Map public locale folder -> assets locale folder.
// Expected keys: folder names under src/public/_locales.
// Expected values: folder names under src/assets/locale.
const localeMap = {
  ja: 'ja-JP',
  zh_TW: 'zh-TW',
  zh_CN: 'zh',
};
const defaultLocaleForFiles = 'en';
// Default to all JSON locale files from assets; pass filenames as args to limit scope.
const defaultFiles = fs
  .readdirSync(path.join(assetsBase, defaultLocaleForFiles))
  .filter((file) => file.endsWith('.json'))
  .sort();

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const backup = args.includes('--backup');
const targetFiles = args.filter((arg) => arg !== '--dry-run' && arg !== '--backup');
const files = targetFiles.length ? targetFiles : defaultFiles;

function normalizeMessage(value, key, invalid) {
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    console.error(`[sync-public-locales] Arrays are not supported: ${key}`);
    invalid.push(key);
    return null;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    console.warn(`[sync-public-locales] Coercing ${key} to string.`);
    return String(value);
  }
  invalid.push(key);
  return null;
}

function toChromeMessageKey(key) {
  return String(key).replace(/[^A-Za-z0-9_]/g, '_');
}

// Flatten nested assets locales into Chrome i18n keys: section.subkey -> section_subkey.
// Only string-ish leaves are written as { message: string } entries.
function flatten(obj, prefix = '', out = {}, invalid = [], origins = {}, sourcePath = '') {
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [key, value] of Object.entries(obj)) {
      const segment = toChromeMessageKey(key);
      const next = prefix ? `${prefix}_${segment}` : segment;
      const nextSourcePath = sourcePath ? `${sourcePath}.${key}` : key;
      flatten(value, next, out, invalid, origins, nextSourcePath);
    }
  } else {
    const normalized = normalizeMessage(obj, prefix, invalid);
    if (normalized !== null) {
      if (
        Object.prototype.hasOwnProperty.call(out, prefix) &&
        origins[prefix] !== sourcePath
      ) {
        throw new Error(
          `[sync-public-locales] Key collision: ${origins[prefix]} and ${sourcePath} both map to ${prefix}`
        );
      }
      origins[prefix] = sourcePath;
      out[prefix] = normalized;
    }
  }
  return out;
}

function syncPublicLocales() {
  const locales = fs
    .readdirSync(publicBase, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  for (const locale of locales) {
    const assetLocale = localeMap[locale] || locale;
    for (const filename of files) {
      const assetPath = path.join(assetsBase, assetLocale, filename);
      if (!fs.existsSync(assetPath)) {
        continue;
      }
      const assets = JSON.parse(fs.readFileSync(assetPath, 'utf8'));
      const invalidLeaves = [];
      const flat = flatten(assets, '', {}, invalidLeaves);
      if (invalidLeaves.length) {
        console.warn(
          `[sync-public-locales] Skipped non-string values in ${assetPath}: ${invalidLeaves.join(
            ', '
          )}`
        );
        process.exitCode = 1;
      }
      const output = Object.fromEntries(
        Object.entries(flat).map(([key, value]) => [key, { message: value }])
      );
      const outPath = path.join(publicBase, locale, filename);
      // NOTE: This rewrites public locale files. Asset-derived keys overwrite existing values.
      let merged = output;
      let existingRaw = null;
      if (fs.existsSync(outPath)) {
        existingRaw = fs.readFileSync(outPath, 'utf8');
        const existing = JSON.parse(existingRaw);
        const publicOnlyKeys = Object.keys(existing).filter((key) => !(key in output) && /^[A-Za-z0-9_]+$/.test(key));
        const invalidPublicOnlyKeys = Object.keys(existing).filter((key) => !(key in output) && !/^[A-Za-z0-9_]+$/.test(key));
        if (publicOnlyKeys.length) {
          console.warn(
            `[sync-public-locales] Preserving ${publicOnlyKeys.length} public-only keys in ${outPath}.`
          );
        }
        if (invalidPublicOnlyKeys.length) {
          console.warn(
            `[sync-public-locales] Dropping ${invalidPublicOnlyKeys.length} invalid Chrome i18n keys in ${outPath}: ${invalidPublicOnlyKeys.join(
              ', '
            )}`
          );
        }
        merged = { ...existing, ...output };
        for (const key of invalidPublicOnlyKeys) {
          delete merged[key];
        }
      }
      const nextJson = JSON.stringify(merged, null, 2) + '\n';
      if (existingRaw !== null && existingRaw === nextJson) {
        continue;
      }
      if (dryRun) {
        console.log(`[sync-public-locales] DRY RUN: would write ${outPath}`);
        continue;
      }
      if (backup && existingRaw !== null) {
        fs.writeFileSync(`${outPath}.bak`, existingRaw);
      }
      fs.writeFileSync(outPath, nextJson);
    }
  }
}

if (require.main === module) {
  syncPublicLocales();
}

module.exports = { flatten, syncPublicLocales, toChromeMessageKey };

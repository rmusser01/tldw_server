#!/usr/bin/env node
/**
 * Verify that:
 *   1) All ClientPath entries in src/services/tldw/openapi-guard.ts
 *      exist in the current OpenAPI spec.
 *   2) The MEDIA_ADD_SCHEMA_FALLBACK field list remains a subset of the
 *      /api/v1/media/add request schema in that spec.
 *
 * This is a lightweight, build-time safety net to catch drift between
 * the extension's manually-maintained API types and the server's
 * OpenAPI spec, without bundling the 1.4MB JSON into the runtime.
 *
 * Spec source resolution order:
 *   1) apps/extension/openapi.json when maintainers keep a local snapshot
 *   2) A generated spec from tldw_Server_API.app.main via app.openapi()
 *      using the repo Python environment and a local-only synthetic API key
 *
 * Usage:
 *   npm run verify:openapi
 *   bun run verify:openapi
 */

import childProcess from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import url from 'node:url'

const require = createRequire(import.meta.url)

const __dirname = path.dirname(url.fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')
const workspaceRoot = path.resolve(root, '..', '..')
const sharedSrc = path.resolve(root, '../packages/ui/src')

const guardFile = path.join(sharedSrc, 'services/tldw/openapi-guard.ts')
const fallbackFile = path.join(sharedSrc, 'services/tldw/fallback-schemas.ts')
const specFile = path.join(root, 'openapi.json')
const GENERATED_OPENAPI_KEY = 'verify-openapi-local-key-1234567890'
const STRICT_VERIFY = process.env.TLDW_VERIFY_OPENAPI_STRICT === '1'
const KNOWN_MISSING_CLIENT_PATHS = new Map([
  [
    '/api/v1/media/bulk/keyword-update',
    'Optional bulk keyword update optimization; the UI falls back to per-item PATCH /api/v1/media/{media_id}/keywords on 404.'
  ],
  [
    '/api/v1/media/statistics',
    'Legacy media statistics client surface; the current OSS backend does not publish this route.'
  ],
  [
    '/api/v1/billing/plans',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/subscription',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/usage',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/invoices',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/subscription/cancel',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/subscription/resume',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/checkout',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ],
  [
    '/api/v1/billing/portal',
    'Public billing routes are intentionally absent from the OSS backend; see tldw_Server_API/tests/Billing/test_billing_public_api_removed.py.'
  ]
])
const GENERATED_OPENAPI_SCRIPT = `
import json
import os
import sys

os.environ["AUTH_MODE"] = "single_user"
os.environ["SINGLE_USER_API_KEY"] = "${GENERATED_OPENAPI_KEY}"

from tldw_Server_API.app.main import app

json.dump(app.openapi(), sys.stdout)
`

function normalizePath(p) {
  if (!p) return ''
  let v = String(p).trim()
  if (!v.startsWith('/')) v = '/' + v
  // Parameter names are not semantically meaningful for route matching.
  v = v.replace(/\{[^/]+\}/g, '{param}')
  // Strip trailing slashes to tolerate /path vs /path/
  v = v.replace(/\/+$/, '')
  return v || '/'
}

function resolveGitCommonDir() {
  const result = childProcess.spawnSync(
    'git',
    ['-C', workspaceRoot, 'rev-parse', '--git-common-dir'],
    {
      encoding: 'utf8'
    }
  )

  if (result.error || result.status !== 0) {
    return null
  }

  const output = result.stdout.trim()
  if (!output) {
    return null
  }

  return path.resolve(workspaceRoot, output)
}

function loadJsonFromFile(file) {
  try {
    const text = fs.readFileSync(file, 'utf8')
    return JSON.parse(text)
  } catch (err) {
    console.error(`Failed to read or parse OpenAPI spec at ${file}:`, err)
    process.exit(1)
  }
}

function loadSpecFromBackend() {
  const gitCommonDir = resolveGitCommonDir()
  const pythonCandidates = [
    gitCommonDir ? path.join(path.dirname(gitCommonDir), '.venv', 'bin', 'python') : null,
    path.join(workspaceRoot, '.venv', 'bin', 'python'),
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'bin', 'python') : null,
    'python3',
    'python'
  ].filter(Boolean)

  const failures = []

  for (const candidate of pythonCandidates) {
    const result = childProcess.spawnSync(candidate, ['-c', GENERATED_OPENAPI_SCRIPT], {
      cwd: workspaceRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        AUTH_MODE: 'single_user',
        SINGLE_USER_API_KEY: GENERATED_OPENAPI_KEY,
        PYTHONWARNINGS: process.env.PYTHONWARNINGS || 'ignore'
      },
      maxBuffer: 32 * 1024 * 1024
    })

    if (result.error) {
      if (result.error.code === 'ENOENT') {
        continue
      }
      failures.push(`${candidate}: ${result.error.message}`)
      continue
    }

    if (result.status !== 0) {
      const detail = (result.stderr || result.stdout || `exit ${result.status}`).trim()
      failures.push(`${candidate}: ${detail}`)
      continue
    }

    try {
      return {
        json: JSON.parse(result.stdout),
        sourceLabel: `generated from tldw_Server_API.app.main via ${candidate}`
      }
    } catch (err) {
      failures.push(`${candidate}: generated output was not valid JSON (${err})`)
    }
  }

  console.error('Unable to load an OpenAPI spec for verification.')
  console.error(`- Expected snapshot: ${specFile}`)
  console.error('- Snapshot was not present, so backend generation was attempted.')
  console.error('- Generation requires a working project Python environment.')
  if (failures.length > 0) {
    console.error('\nGeneration attempts:')
    for (const failure of failures) {
      console.error(`  - ${failure}`)
    }
  }
  process.exit(1)
}

function loadOpenApiSpec() {
  if (fs.existsSync(specFile)) {
    return {
      json: loadJsonFromFile(specFile),
      sourceLabel: specFile
    }
  }

  return loadSpecFromBackend()
}

function loadSpecPaths(specJson) {
  const paths = specJson && typeof specJson === 'object' && specJson.paths
  if (!paths || typeof paths !== 'object') {
    console.error('OpenAPI spec does not contain a valid "paths" object')
    process.exit(1)
  }
  return new Set(Object.keys(paths).map(normalizePath))
}

export function findBalancedArrayLiteralEnd(src, arrStart) {
  let depth = 0
  let quote = null
  let inLineComment = false
  let inBlockComment = false

  for (let index = arrStart; index < src.length; index += 1) {
    const char = src[index]
    const next = src[index + 1]

    if (inLineComment) {
      if (char === '\n') {
        inLineComment = false
      }
      continue
    }

    if (inBlockComment) {
      if (char === '*' && next === '/') {
        inBlockComment = false
        index += 1
      }
      continue
    }

    if (quote) {
      if (char === '\\') {
        index += 1
        continue
      }
      if (char === quote) {
        quote = null
      }
      continue
    }

    if (char === '/' && next === '/') {
      inLineComment = true
      index += 1
      continue
    }

    if (char === '/' && next === '*') {
      inBlockComment = true
      index += 1
      continue
    }

    if (char === "'" || char === '"' || char === '`') {
      quote = char
      continue
    }

    if (char === '[') {
      depth += 1
      continue
    }

    if (char === ']') {
      depth -= 1
      if (depth === 0) {
        return index
      }
    }
  }

  return -1
}

function extractClientPathsFromTypeScript(src) {
  let ts
  try {
    ts = require('typescript')
  } catch {
    return []
  }

  const sourceFile = ts.createSourceFile(
    guardFile,
    src,
    ts.ScriptTarget.Latest,
    /* setParentNodes */ true,
    ts.ScriptKind.TS
  )

  for (const statement of sourceFile.statements) {
    if (!ts.isTypeAliasDeclaration(statement) || statement.name?.text !== 'ClientPath') continue
    if (!ts.isUnionTypeNode(statement.type)) return []

    const paths = []
    for (const typeNode of statement.type.types) {
      const node = ts.isParenthesizedTypeNode(typeNode) ? typeNode.type : typeNode
      if (ts.isLiteralTypeNode(node) && ts.isStringLiteral(node.literal)) {
        paths.push(node.literal.text)
      }
    }
    return paths
  }

  return []
}

function extractFallbackFieldNamesFromTypeScript(src) {
  let ts
  try {
    ts = require('typescript')
  } catch {
    return []
  }

  const sourceFile = ts.createSourceFile(
    fallbackFile,
    src,
    ts.ScriptTarget.Latest,
    /* setParentNodes */ true,
    ts.ScriptKind.TS
  )

  for (const statement of sourceFile.statements) {
    if (!ts.isVariableStatement(statement)) continue

    for (const declaration of statement.declarationList.declarations) {
      if (!ts.isIdentifier(declaration.name)) continue
      if (declaration.name.text !== 'MEDIA_ADD_SCHEMA_FALLBACK') continue
      if (!declaration.initializer || !ts.isArrayLiteralExpression(declaration.initializer)) {
        return []
      }

      const names = []
      for (const element of declaration.initializer.elements) {
        if (!ts.isObjectLiteralExpression(element)) continue
        for (const property of element.properties) {
          if (!ts.isPropertyAssignment(property)) continue
          const name = property.name
          if (!ts.isIdentifier(name) || name.text !== 'name') continue
          const initializer = property.initializer
          if (ts.isStringLiteral(initializer) || ts.isNoSubstitutionTemplateLiteral(initializer)) {
            names.push(initializer.text)
          }
        }
      }

      return names
    }
  }

  return []
}

function loadMediaAddProperties(json) {
  const mediaPath = json.paths && json.paths['/api/v1/media/add']
  if (!mediaPath || !mediaPath.post || !mediaPath.post.requestBody) {
    console.error('OpenAPI spec does not contain /api/v1/media/add with a POST requestBody')
    process.exit(1)
  }
  const content = mediaPath.post.requestBody.content
  if (!content || !content['multipart/form-data']) {
    console.error(
      'OpenAPI spec does not define multipart/form-data content for /api/v1/media/add requestBody'
    )
    process.exit(1)
  }
  const schema = content['multipart/form-data'].schema
  if (!schema || typeof schema !== 'object') {
    console.error(
      'OpenAPI spec multipart/form-data schema for /api/v1/media/add requestBody is missing or invalid'
    )
    process.exit(1)
  }

  let properties
  if (schema.$ref) {
    const refPrefix = '#/components/schemas/'
    if (!schema.$ref.startsWith(refPrefix)) {
      console.error(`Unsupported $ref format for /api/v1/media/add schema: ${schema.$ref}`)
      process.exit(1)
    }
    const name = schema.$ref.slice(refPrefix.length)
    const comp = json.components && json.components.schemas && json.components.schemas[name]
    if (!comp || !comp.properties) {
      console.error(
        `Referenced schema ${name} for /api/v1/media/add does not define a properties object`
      )
      process.exit(1)
    }
    properties = comp.properties
  } else if (schema.properties) {
    properties = schema.properties
  } else {
    console.error(
      'OpenAPI spec /api/v1/media/add schema does not define a properties object or $ref'
    )
    process.exit(1)
  }

  if (!properties || typeof properties !== 'object') {
    console.error(
      'OpenAPI spec /api/v1/media/add properties object is missing or not an object after resolution'
    )
    process.exit(1)
  }

  return new Set(Object.keys(properties))
}

function extractClientPaths() {
  if (!fs.existsSync(guardFile)) {
    console.error(`openapi-guard.ts not found at ${guardFile}`)
    process.exit(1)
  }
  const src = fs.readFileSync(guardFile, 'utf8')
  const astPaths = extractClientPathsFromTypeScript(src)
  if (astPaths.length > 0) return [...new Set(astPaths)]

  const startRegex = /export\s+type\s+ClientPath\s*=\s*/m
  const startMatch = startRegex.exec(src)
  if (!startMatch) {
    console.error('Could not locate "export type ClientPath" in openapi-guard.ts')
    process.exit(1)
  }

  const afterStart = startMatch.index + startMatch[0].length
  const rest = src.slice(afterStart)

  // ClientPath is expected to appear before the next top-level export.
  const endMatch = /^\s*export\s/m.exec(rest)
  const typeRhs = endMatch ? rest.slice(0, endMatch.index) : rest

  // Strip comments so blank/comment lines can't truncate parsing and we don't
  // accidentally pick up string literals from commented-out examples.
  const withoutComments = typeRhs
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '')

  const paths = []
  const unionLiteralRegex = /(?:^|\|)\s*(["'])([^"'\\]*(?:\\.[^"'\\]*)*)\1/gm
  for (const match of withoutComments.matchAll(unionLiteralRegex)) {
    paths.push(match[2])
  }

  if (paths.length === 0) {
    console.error('No ClientPath entries were parsed from openapi-guard.ts')
    process.exit(1)
  }

  return [...new Set(paths)]
}

export function extractFallbackFieldNamesFromSource(src) {
  const astNames = extractFallbackFieldNamesFromTypeScript(src)
  if (astNames.length > 0) return astNames

  const marker = 'export const MEDIA_ADD_SCHEMA_FALLBACK'
  const start = src.indexOf(marker)
  if (start === -1) return []

  const arrStart = src.indexOf('[', start)
  if (arrStart === -1) return []

  const arrEnd = findBalancedArrayLiteralEnd(src, arrStart)
  if (arrEnd === -1) return []

  const block = src.slice(arrStart, arrEnd + 1)
  const names = []
  const re = /name:\s*['"]([^'"]+)['"]/g
  for (const match of block.matchAll(re)) {
    names.push(match[1])
  }

  return names
}

function extractFallbackFieldNames() {
  if (!fs.existsSync(fallbackFile)) {
    console.error(`fallback-schemas.ts not found at ${fallbackFile}`)
    process.exit(1)
  }

  const src = fs.readFileSync(fallbackFile, 'utf8')
  const names = extractFallbackFieldNamesFromSource(src)

  if (names.length === 0) {
    console.error('No MEDIA_ADD_SCHEMA_FALLBACK entries were parsed from fallback-schemas.ts')
    process.exit(1)
  }

  return names
}

function verifyClientPaths(specJson, sourceLabel) {
  const specPaths = loadSpecPaths(specJson)
  const clientPaths = extractClientPaths()

  const missing = []
  for (const p of clientPaths) {
    const norm = normalizePath(p)
    if (!specPaths.has(norm)) {
      missing.push({ path: p, normalized: norm })
    }
  }

  const knownMissing = []
  const unexpectedMissing = []
  for (const item of missing) {
    const reason = KNOWN_MISSING_CLIENT_PATHS.get(item.path)
    if (reason) {
      knownMissing.push({ ...item, reason })
    } else {
      unexpectedMissing.push(item)
    }
  }

  if (knownMissing.length > 0) {
    const prefix = STRICT_VERIFY ? '❌' : '⚠️'
    console.warn(
      `${prefix} Reviewed client-path exceptions outside the current OSS OpenAPI contract (${sourceLabel}):`
    )
    for (const item of knownMissing) {
      console.warn(`  - ${item.path}: ${item.reason}`)
    }
    if (STRICT_VERIFY) {
      unexpectedMissing.push(...knownMissing)
    }
  }

  if (unexpectedMissing.length > 0) {
    console.error(`❌ ClientPath entries missing from OpenAPI spec (${sourceLabel}):`)
    for (const m of unexpectedMissing) {
      console.error(`  - ${m.path} (normalized: ${m.normalized})`)
    }
    console.error(
      '\nEither update ClientPath in src/services/tldw/openapi-guard.ts, ' +
        'or refresh the OpenAPI source used by verify:openapi.'
    )
    process.exit(1)
  }

  console.log(
    `✅ Verified ${clientPaths.length} ClientPath entries against OpenAPI spec (${sourceLabel}); all paths are present.`
  )
  if (knownMissing.length > 0 && !STRICT_VERIFY) {
    console.log(
      `ℹ️ verify:openapi allowed ${knownMissing.length} reviewed exception path(s). Set TLDW_VERIFY_OPENAPI_STRICT=1 to fail on them.`
    )
  }
}

function verifyMediaAddFallback(specJson, sourceLabel) {
  const specProps = loadMediaAddProperties(specJson)
  const fallbackNames = extractFallbackFieldNames()

  const missing = fallbackNames.filter((name) => !specProps.has(name))

  if (missing.length > 0) {
    console.error(
      `❌ MEDIA_ADD_SCHEMA_FALLBACK contains field names not present in /api/v1/media/add (${sourceLabel}):`
    )
    for (const name of missing) {
      console.error(`  - ${name}`)
    }
    console.error(
      '\nEither update MEDIA_ADD_SCHEMA_FALLBACK in src/services/tldw/fallback-schemas.ts, ' +
        'or refresh the OpenAPI source used by verify:openapi.'
    )
    process.exit(1)
  }

  console.log(
    `✅ Verified ${fallbackNames.length} MEDIA_ADD_SCHEMA_FALLBACK fields against /api/v1/media/add (${sourceLabel}); all names are present.`
  )
}

function main() {
  const { json: specJson, sourceLabel } = loadOpenApiSpec()
  verifyClientPaths(specJson, sourceLabel)
  verifyMediaAddFallback(specJson, sourceLabel)
}

const isEntrypoint =
  process.argv[1] && path.resolve(process.argv[1]) === url.fileURLToPath(import.meta.url)

if (isEntrypoint) {
  main()
}

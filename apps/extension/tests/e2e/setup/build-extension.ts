import { execSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'path'

export default async function globalSetup() {
  // If a built chrome extension already exists, skip rebuilding.
  const projectRoot = path.resolve(__dirname, '..', '..', '..')
  const builtChromeCandidates = [
    path.resolve(projectRoot, 'build/chrome-mv3'),
    path.resolve(projectRoot, '.output/chrome-mv3')
  ]
  const isBuiltChromeDir = (dir: string) => {
    if (!dir || !fs.existsSync(dir)) {
      return false
    }

    const manifestPath = path.join(dir, 'manifest.json')
    const backgroundPath = path.join(dir, 'background.js')
    const optionsPath = path.join(dir, 'options.html')
    const sidepanelPath = path.join(dir, 'sidepanel.html')

    return (
      fs.existsSync(manifestPath) &&
      fs.existsSync(backgroundPath) &&
      (fs.existsSync(optionsPath) || fs.existsSync(sidepanelPath))
    )
  }
  const builtChromePath =
    builtChromeCandidates.find((candidate) => isBuiltChromeDir(candidate)) ||
    builtChromeCandidates.find((candidate) => fs.existsSync(candidate)) ||
    builtChromeCandidates[0]
  const forceBuildChrome =
    process.env.FORCE_BUILD_CHROME === '1' ||
    process.env.FORCE_BUILD_CHROME === 'true'

  const getLatestMtime = (dir: string): number => {
    let latest = 0
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true })
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name)
        if (entry.isDirectory()) {
          latest = Math.max(latest, getLatestMtime(fullPath))
        } else if (entry.isFile()) {
          const stat = fs.statSync(fullPath)
          if (stat.mtimeMs > latest) latest = stat.mtimeMs
        }
      }
    } catch {
      return latest
    }
    return latest
  }

  const getFileMtime = (filePath: string): number => {
    try {
      return fs.statSync(filePath).mtimeMs
    } catch {
      return 0
    }
  }

  const isDevBuild = (dir: string) => {
    const optionsPath = path.join(dir, 'options.html')
    if (!fs.existsSync(optionsPath)) return false
    const html = fs.readFileSync(optionsPath, 'utf8')
    return (
      html.includes('http://localhost:') ||
      html.includes('/@vite/client') ||
      html.includes('virtual:wxt-html-plugins')
    )
  }

  if (
    fs.existsSync(builtChromePath) &&
    isBuiltChromeDir(builtChromePath) &&
    !forceBuildChrome &&
    !isDevBuild(builtChromePath)
  ) {
    const buildStamp = path.join(builtChromePath, 'manifest.json')
    const buildMtime = getFileMtime(buildStamp)
    const latestSourceMtime = Math.max(
      getLatestMtime(path.resolve(projectRoot, 'src')),
      getLatestMtime(path.resolve(projectRoot, '../packages/ui/src')),
      getFileMtime(path.resolve(projectRoot, 'wxt.config.ts')),
      getFileMtime(path.resolve(projectRoot, 'package.json')),
      getFileMtime(path.resolve(projectRoot, 'tailwind.config.js')),
      getFileMtime(path.resolve(projectRoot, 'tsconfig.json'))
    )
    if (buildMtime && latestSourceMtime <= buildMtime) {
      // Still apply test-only host permissions if configured.
      applyTestHostPermissions(projectRoot)
      return
    }
  }

  const buildCommands: Array<{ command: string; env?: NodeJS.ProcessEnv }> = [
    { command: 'npm run build:chrome:prod' },
    { command: 'bun run build:chrome:prod' },
    {
      command: 'node scripts/build-with-profile.mjs --browser=chrome',
      env: { ...process.env, TLDW_BUILD_PROFILE: 'production' }
    }
  ]

  const attemptFailures: Array<{ command: string; error: unknown }> = []
  for (const { command, env } of buildCommands) {
    try {
      execSync(command, {
        stdio: 'inherit',
        cwd: projectRoot,
        env
      })
      attemptFailures.length = 0
      break
    } catch (error) {
      attemptFailures.push({ command, error })
    }
  }

  if (attemptFailures.length > 0) {
    throw new AggregateError(
      attemptFailures.map(({ command, error }) =>
        new Error(`Build attempt failed: ${command}`, { cause: error as Error })
      ),
      'All extension build attempts failed.'
    )
  }

  applyTestHostPermissions(projectRoot)
}

function applyTestHostPermissions(projectRoot: string) {
  const rawUrl =
    process.env.TLDW_E2E_SERVER_URL || process.env.LDW_E2E_SERVER_URL
  if (!rawUrl) return
  let originPattern = ''
  try {
    const url = new URL(rawUrl)
    originPattern = `${url.origin}/*`
  } catch {
    return
  }

  const candidates = [
    path.resolve(projectRoot, 'build/chrome-mv3/manifest.json'),
    path.resolve(projectRoot, '.output/chrome-mv3/manifest.json')
  ]
  for (const manifestPath of candidates) {
    if (!fs.existsSync(manifestPath)) continue
    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
      const hostPerms = new Set<string>(
        Array.isArray(manifest.host_permissions) ? manifest.host_permissions : []
      )
      hostPerms.add(originPattern)
      manifest.host_permissions = Array.from(hostPerms)
      fs.writeFileSync(manifestPath, JSON.stringify(manifest))
    } catch {
      // ignore manifest patch failures
    }
  }
}

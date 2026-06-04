import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const repoRoot = path.resolve(appDir, "..", "..")

const readRepoFile = (relativePath: string) =>
  readFileSync(path.join(repoRoot, relativePath), "utf8")

describe("PR 916 review follow-ups", () => {
  it("keeps the WebUI Docker build aligned with the reviewed networking expectations", () => {
    const dockerfile = readRepoFile("Dockerfiles/Dockerfile.webui")
    const compose = readRepoFile("Dockerfiles/docker-compose.webui.yml")

    expect(dockerfile).toContain("COPY apps/extension/package.json /app/apps/extension/package.json")
    expect(dockerfile).toContain("COPY apps/extension/scripts/wxt-prepare.mjs")
    expect(dockerfile).toContain("COPY apps/packages/voice-assistant-sdk/package.json")
    expect(dockerfile).toContain("ENV SKIP_WXT_PREPARE=1")
    expect(dockerfile).not.toContain("pkg.workspaces=")
    expect(dockerfile).toContain("RUN bun install --frozen-lockfile --cwd /app/apps")
    expect(dockerfile).toContain("WORKDIR /app/apps/tldw-frontend")
    expect(dockerfile).toContain("RUN bun /app/apps/tldw-frontend/scripts/validate-networking-config.mjs")
    expect(dockerfile).toContain("bun run build:prod")
    expect(dockerfile).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart")
    expect(dockerfile).toContain("TLDW_INTERNAL_API_ORIGIN=http://app:8000")
    expect(compose).toContain("NEXT_PUBLIC_API_BASE_URL: ${NEXT_PUBLIC_API_BASE_URL:-}")
    expect(compose).toContain("- NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL:-}")
  })

  it("waits for healthy backing services in the host-storage compose stack", () => {
    const compose = readRepoFile("Dockerfiles/docker-compose.host-storage.yml")

    expect(compose).toContain("depends_on:")
    expect(compose).toContain("postgres:")
    expect(compose).toContain("redis:")
    expect(compose).toContain("condition: service_healthy")
  })

  it("removes machine-local absolute paths from the new audio setup docs", () => {
    const cpuGuide = readRepoFile("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md")
    const gpuGuide = readRepoFile("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md")

    expect(cpuGuide).not.toContain("/Users/macbook-dev/Documents/GitHub/tldw_server2")
    expect(gpuGuide).not.toContain("/Users/macbook-dev/Documents/GitHub/tldw_server2")
  })
})

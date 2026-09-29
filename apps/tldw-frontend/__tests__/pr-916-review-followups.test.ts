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
    expect(dockerfile).toContain("RUN node /app/apps/tldw-frontend/scripts/validate-networking-config.mjs")
    expect(dockerfile).toContain("npm run build:prod")
    expect(dockerfile).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart")
    expect(dockerfile).toContain("TLDW_INTERNAL_API_ORIGIN=http://app:8000")
    expect(compose).toContain("NEXT_PUBLIC_API_BASE_URL: ${NEXT_PUBLIC_API_BASE_URL:-}")
    expect(compose).toContain("- NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL:-}")
    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain(
      "- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}"
    )
    expect(compose).toContain(
      "- TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}"
    )
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

  it("documents Docker WebUI runtime auth instead of rebuild-only public key auth", () => {
    const readme = readRepoFile("README.md")
    const troubleshooting = readRepoFile("Docs/Getting_Started/TROUBLESHOOTING.md")
    const profile = readRepoFile("Docs/Getting_Started/Profile_Docker_Single_User.md")
    const publishedProfile = readRepoFile(
      "Docs/Published/Getting_Started/Profile_Docker_Single_User.md"
    )
    const dockerfilesReadme = readRepoFile("Dockerfiles/README.md")

    expect(readme).toContain(
      "reads `AUTH_MODE` and `SINGLE_USER_API_KEY` at runtime"
    )
    expect(readme).toContain(
      "Do not use it for the normal Docker quickstart path."
    )
    expect(troubleshooting).toContain(
      "Docker single-user WebUI quickstart uses runtime auth bootstrap"
    )
    expect(troubleshooting).not.toContain(
      "set it in `.env` and rebuild the WebUI"
    )
    expect(profile).toContain(
      "The WebUI image is not tied to a specific single-user API key."
    )
    expect(publishedProfile).toContain(
      "The WebUI image is not tied to a specific single-user API key."
    )
    expect(dockerfilesReadme).toContain(
      "Single-user WebUI auth is runtime-configured."
    )
  })
})

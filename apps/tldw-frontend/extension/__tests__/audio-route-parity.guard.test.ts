import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("audio route parity", () => {
  it("keeps extension STT mapped to the dedicated shared STT playground", () => {
    const webRoute = fs.readFileSync(
      path.resolve(__dirname, "../../../packages/ui/src/routes/option-stt.tsx"),
      "utf8"
    )
    const extRoute = fs.readFileSync(
      path.resolve(__dirname, "../routes/option-stt.tsx"),
      "utf8"
    )

    expect(webRoute).toContain("SttPlaygroundPage")
    expect(extRoute).toContain("SttPlaygroundPage")
    expect(extRoute).toContain("RouteErrorBoundary")
    expect(extRoute).toContain('routeId="stt"')
    expect(extRoute).toContain('routeLabel="STT Playground"')
    expect(extRoute).not.toContain("SpeechPlaygroundPage")
    expect(extRoute).not.toContain('initialMode="speak"')
  })

  it("keeps extension TTS locked to the dedicated listen-mode workflow", () => {
    const webRoute = fs.readFileSync(
      path.resolve(__dirname, "../../../packages/ui/src/routes/option-tts.tsx"),
      "utf8"
    )
    const extRoute = fs.readFileSync(
      path.resolve(__dirname, "../routes/option-tts.tsx"),
      "utf8"
    )

    expect(webRoute).toContain('lockedMode="listen"')
    expect(webRoute).toContain("hideModeSwitcher")
    expect(webRoute).not.toContain('initialMode="listen"')
    expect(extRoute).toContain('lockedMode="listen"')
    expect(extRoute).toContain("hideModeSwitcher")
    expect(extRoute).not.toContain('initialMode="listen"')
  })
})

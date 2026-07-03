import React from "react"
import { render, fireEvent, waitFor, act } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { message } from "antd"
import { AvatarField, MAX_AVATAR_IMAGE_BYTES } from "../AvatarField"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getImageBackends: vi.fn().mockResolvedValue([]),
    createImageArtifact: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

const makeFile = (name: string, type: string, size: number): File => {
  const file = new File(["x"], name, { type })
  Object.defineProperty(file, "size", { value: size })
  return file
}

describe("AvatarField upload size cap", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("rejects an avatar larger than the cap and does not emit its base64", async () => {
    const onChange = vi.fn()
    const errorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => ({}) as any)

    const { container } = render(
      <AvatarField value={{ mode: "upload" }} onChange={onChange} />
    )

    const input = container.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement
    expect(input).toBeTruthy()

    const bigFile = makeFile("big.png", "image/png", MAX_AVATAR_IMAGE_BYTES + 1)
    await act(async () => {
      fireEvent.change(input, { target: { files: [bigFile] } })
    })

    await waitFor(() => expect(errorSpy).toHaveBeenCalled())
    expect(
      errorSpy.mock.calls.some(
        ([msg]) => typeof msg === "string" && msg.includes("too large")
      )
    ).toBe(true)
    // The oversized image's base64 must never reach onChange (would autosave to
    // localStorage and risk QuotaExceededError).
    expect(onChange).not.toHaveBeenCalledWith(
      expect.objectContaining({ base64: expect.any(String) })
    )
  })
})

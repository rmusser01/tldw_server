import { describe, expect, it } from "vitest"
import { arrayBufferToBase64, base64ToArrayBuffer } from "@/utils/compress"

/**
 * W5 Stage 2 (WP-59): arrayBufferToBase64 must produce output byte-identical
 * to the reference btoa semantics for arbitrary inputs while converting in
 * 8 KB slices instead of per-byte string concatenation. These cases pin the
 * wire format (STT audio frames, voice-chat hooks) across the slice seam.
 */

// Deterministic PRNG (mulberry32): property coverage must never flake.
const mulberry32 = (seed: number) => {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// The pre-chunking implementation's semantics, kept as the reference.
const referenceBase64 = (bytes: Uint8Array): string => {
  let binary = ""
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return globalThis.btoa(binary)
}

const randomBytes = (size: number, seed: number): Uint8Array => {
  const rand = mulberry32(seed)
  const bytes = new Uint8Array(size)
  for (let i = 0; i < size; i++) {
    bytes[i] = Math.floor(rand() * 256)
  }
  return bytes
}

const SIZES = [0, 1, 7, 8191, 8192, 8193, 65536] as const

describe("arrayBufferToBase64 chunked conversion", () => {
  it.each(SIZES)(
    "is byte-identical to the btoa reference for random inputs of size %d",
    (size) => {
      const trials = size > 8193 ? 3 : 25
      for (let trial = 0; trial < trials; trial++) {
        const bytes = randomBytes(size, 0x574f3539 ^ size ^ (trial * 2654435761))
        expect(arrayBufferToBase64(bytes.buffer)).toBe(referenceBase64(bytes))
      }
    }
  )

  it.each(SIZES)(
    "round-trips through base64ToArrayBuffer for random inputs of size %d",
    (size) => {
      const bytes = randomBytes(size, 0x13525461 ^ size)
      const decoded = new Uint8Array(
        base64ToArrayBuffer(arrayBufferToBase64(bytes.buffer))
      )
      expect(decoded).toHaveLength(bytes.length)
      expect(decoded).toEqual(bytes)
    }
  )
})

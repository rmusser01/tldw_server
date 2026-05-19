import type { RGBTriple } from "./types"
import { parseRgbTriple } from "./contrast"
import { hexToRgbTriple } from "./conversion"

// ---------------------------------------------------------------------------
// OKLCH color space helpers (private)
// ---------------------------------------------------------------------------

/**
 * Convert a single sRGB channel (0-255) to linear light.
 */
const srgbToLinear = (c: number): number => {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
}

/**
 * Convert a linear-light value back to sRGB (0-255), clamped.
 */
const linearToSrgb = (c: number): number => {
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
  return Math.round(Math.min(255, Math.max(0, s * 255)))
}

/**
 * Convert RGB (0-255 each) to OKLCH. Returns [L, C, H] where
 * L is 0-1, C is chroma (~0-0.5), H is hue in degrees 0-360.
 */
const rgbToOklch = (r: number, g: number, b: number): [number, number, number] => {
  // sRGB -> linear RGB
  const lr = srgbToLinear(r)
  const lg = srgbToLinear(g)
  const lb = srgbToLinear(b)

  // Linear RGB -> LMS (using the OKLab M1 matrix)
  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb

  // Cube root for perceptual linearity
  const lc = Math.cbrt(l)
  const mc = Math.cbrt(m)
  const sc = Math.cbrt(s)

  // LMS cube-root -> OKLab (M2 matrix)
  const L = 0.2104542553 * lc + 0.7936177850 * mc - 0.0040720468 * sc
  const a = 1.9779984951 * lc - 2.4285922050 * mc + 0.4505937099 * sc
  const bLab = 0.0259040371 * lc + 0.7827717662 * mc - 0.8086757660 * sc

  // OKLab -> OKLCH
  const C = Math.sqrt(a * a + bLab * bLab)
  let H = (Math.atan2(bLab, a) * 180) / Math.PI
  if (H < 0) H += 360

  return [L, C, H]
}

/**
 * Convert OKLCH back to RGB (0-255 each).
 * L is 0-1, C is chroma, H is hue in degrees.
 */
const oklchToRgb = (L: number, C: number, H: number): [number, number, number] => {
  // OKLCH -> OKLab
  const hRad = (H * Math.PI) / 180
  const a = C * Math.cos(hRad)
  const bLab = C * Math.sin(hRad)

  // OKLab -> LMS cube-root (inverse of M2)
  const lc = L + 0.3963377774 * a + 0.2158037573 * bLab
  const mc = L - 0.1055613458 * a - 0.0638541728 * bLab
  const sc = L - 0.0894841775 * a - 1.2914855480 * bLab

  // Cube to get LMS
  const l = lc * lc * lc
  const m = mc * mc * mc
  const s = sc * sc * sc

  // LMS -> linear RGB (inverse of M1)
  const lr =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
  const lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
  const lb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)]
}

// ---------------------------------------------------------------------------
// Helper: format an RGB triple as a space-separated string
// ---------------------------------------------------------------------------

const toTriple = (r: number, g: number, b: number): RGBTriple => `${r} ${g} ${b}`

// ---------------------------------------------------------------------------
// Surface palette derivation
// ---------------------------------------------------------------------------

export interface SurfacePalette {
  bg: RGBTriple
  surface: RGBTriple
  surface2: RGBTriple
  elevated: RGBTriple
}

/**
 * Derive a set of four surface colors from a tint hex color.
 *
 * The tint is converted to OKLCH, its chroma is clamped to a very low value
 * (max 0.02) so the surfaces carry only a subtle hue, and then lightness
 * offsets are applied to produce background, surface, surface2, and elevated
 * shades appropriate for the requested color scheme.
 */
export function deriveSurfacePalette(tintHex: string, isDark: boolean): SurfacePalette {
  const triple = hexToRgbTriple(tintHex)
  const [r, g, b] = parseRgbTriple(triple)
  const [baseL, baseC, baseH] = rgbToOklch(r, g, b)

  // Clamp chroma for subtle tinting
  const c = Math.min(baseC, 0.02)

  // Lightness offsets per surface tier
  const offsets = isDark
    ? { bg: 0, surface: 0.03, surface2: 0.05, elevated: 0.07 }
    : { bg: 0, surface: 0.03, surface2: 0.04, elevated: 0.05 }

  const make = (offset: number): RGBTriple => {
    const L = Math.min(1, Math.max(0, baseL + offset))
    const [cr, cg, cb] = oklchToRgb(L, c, baseH)
    return toTriple(cr, cg, cb)
  }

  return {
    bg: make(offsets.bg),
    surface: make(offsets.surface),
    surface2: make(offsets.surface2),
    elevated: make(offsets.elevated),
  }
}

// ---------------------------------------------------------------------------
// Border-radius derivation
// ---------------------------------------------------------------------------

export interface RadiiResult {
  radiusSm: number
  radiusMd: number
  radiusLg: number
  radiusXl: number
  buttonStyle: "square" | "rounded" | "pill"
}

/**
 * Derive border-radius values and a button style from a 0-100 roundness slider.
 *
 * Linear interpolation between three stops:
 *   0%  -> radiusSm=0,  radiusMd=0,  radiusLg=2,  radiusXl=4   (square)
 *   50% -> radiusSm=2,  radiusMd=6,  radiusLg=8,  radiusXl=12  (rounded)
 *  100% -> radiusSm=6,  radiusMd=12, radiusLg=18, radiusXl=24  (pill)
 */
export function deriveRadii(roundness: number): RadiiResult {
  const t = Math.min(100, Math.max(0, roundness))

  const lerp = (a: number, b: number, fraction: number): number =>
    Math.round(a + (b - a) * fraction)

  let radiusSm: number
  let radiusMd: number
  let radiusLg: number
  let radiusXl: number
  let buttonStyle: "square" | "rounded" | "pill"

  if (t <= 50) {
    // Interpolate between 0% and 50%
    const f = t / 50
    radiusSm = lerp(0, 2, f)
    radiusMd = lerp(0, 6, f)
    radiusLg = lerp(2, 8, f)
    radiusXl = lerp(4, 12, f)
    buttonStyle = t < 25 ? "square" : "rounded"
  } else {
    // Interpolate between 50% and 100%
    const f = (t - 50) / 50
    radiusSm = lerp(2, 6, f)
    radiusMd = lerp(6, 12, f)
    radiusLg = lerp(8, 18, f)
    radiusXl = lerp(12, 24, f)
    buttonStyle = t >= 75 ? "pill" : "rounded"
  }

  return { radiusSm, radiusMd, radiusLg, radiusXl, buttonStyle }
}

// ---------------------------------------------------------------------------
// Shadow derivation
// ---------------------------------------------------------------------------

export interface ShadowResult {
  shadowSm: string
  shadowMd: string
}

/**
 * Derive CSS box-shadow values from a 0-100 intensity slider.
 *
 * The shadow opacity is scaled by the intensity value. Dark mode receives
 * 60% of the light mode opacity to avoid overwhelming dark surfaces.
 * Returns "none" for both shadows when intensity is 0.
 */
export function deriveShadows(intensity: number, isDark: boolean): ShadowResult {
  const t = Math.min(100, Math.max(0, intensity))

  if (t === 0) {
    return { shadowSm: "none", shadowMd: "none" }
  }

  // Base opacities at full intensity (100)
  const baseSm = 0.12
  const baseMd = 0.20

  const darkFactor = isDark ? 0.6 : 1
  const scale = (t / 100) * darkFactor

  const opSm = Math.round(baseSm * scale * 100) / 100
  const opMd = Math.round(baseMd * scale * 100) / 100

  return {
    shadowSm: `0 1px 2px rgba(0,0,0,${opSm})`,
    shadowMd: `0 4px 12px rgba(0,0,0,${opMd})`,
  }
}

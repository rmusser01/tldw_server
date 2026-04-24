import { describe, expect, it } from "vitest"
import {
  getSettingLabel,
  getSettingDescription,
  getSettingTechnicalNote
} from "../worldBookLabelUtils"

describe("worldBookLabelUtils", () => {
  describe("getSettingLabel", () => {
    it("returns friendly labels by default for all known keys", () => {
      expect(getSettingLabel("scan_depth", false)).toBe("Messages to search")
      expect(getSettingLabel("token_budget", false)).toBe("Context size limit")
      expect(getSettingLabel("recursive_scanning", false)).toBe("Chain matching")
    })

    it("returns technical labels when showTechnical is true", () => {
      expect(getSettingLabel("scan_depth", true)).toBe("Scan Depth")
      expect(getSettingLabel("token_budget", true)).toBe("Token Budget")
      expect(getSettingLabel("recursive_scanning", true)).toBe("Recursive Scanning")
    })

    it("returns the key as fallback for unknown settings", () => {
      expect(getSettingLabel("unknown_setting", false)).toBe("unknown_setting")
      expect(getSettingLabel("unknown_setting", true)).toBe("unknown_setting")
    })
  })

  describe("getSettingDescription", () => {
    it("returns user-friendly description by default", () => {
      expect(getSettingDescription("scan_depth", false)).toMatch(/how far back/i)
    })

    it("returns technical description when showTechnical is true", () => {
      expect(getSettingDescription("scan_depth", true)).toMatch(/scan_depth/i)
    })

    it("describes token budget in tokens rather than characters", () => {
      expect(getSettingDescription("token_budget", true)).toMatch(/tokens/i)
      expect(getSettingDescription("token_budget", true)).not.toMatch(/maximum characters/i)
      expect(getSettingTechnicalNote("token_budget")).toMatch(/tokens/i)
    })

    it("returns empty string for unknown key", () => {
      expect(getSettingDescription("unknown_setting", false)).toBe("")
      expect(getSettingDescription("unknown_setting", true)).toBe("")
    })
  })

  describe("getSettingTechnicalNote", () => {
    it("returns API field name and range for known keys", () => {
      expect(getSettingTechnicalNote("scan_depth")).toMatch(/1-20/)
      expect(getSettingTechnicalNote("token_budget")).toMatch(/50-5000/)
    })

    it("returns empty string for unknown key", () => {
      expect(getSettingTechnicalNote("unknown_setting")).toBe("")
    })
  })
})

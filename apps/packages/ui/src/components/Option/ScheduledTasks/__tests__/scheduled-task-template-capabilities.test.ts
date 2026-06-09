import { describe, expect, it } from "vitest"

import {
  REQUIRED_INGEST_AVAILABILITY_GATES,
  REQUIRED_WATCH_AVAILABILITY_GATES,
  buildScheduledTaskTemplateCapability,
  getMissingAvailabilityGates,
  resolveTemplateCapabilityState
} from "../scheduled-task-template-capabilities"

describe("scheduled task template capabilities", () => {
  it("requires preview before Watch can be available", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
        (gate) => gate !== "source_preview"
      )
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
    expect(getMissingAvailabilityGates("watch", capability)).toContain("source_preview")
  })

  it("keeps Watch limited when gates pass but no creation adapter is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
  })

  it("allows Watch only when every Watch gate and the creation adapter guard pass", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("available")
  })

  it("does not require notification gate for Ingest availability once creation is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("ingest", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_INGEST_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("ingest", capability)).toBe("available")
  })
})

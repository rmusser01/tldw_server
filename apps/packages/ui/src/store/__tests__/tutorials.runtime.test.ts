import { beforeEach, describe, expect, it } from "vitest"
import { useTutorialStore } from "../tutorials"

beforeEach(() => useTutorialStore.getState().resetProgress())

describe("tutorial run ownership", () => {
  it("gives a same-ID restart a distinct transient run", () => {
    useTutorialStore.getState().startTutorial("playground-basics")
    const first = useTutorialStore.getState().activeRun
    useTutorialStore.getState().startTutorial("playground-basics")
    const second = useTutorialStore.getState().activeRun
    expect(first).toBeDefined()
    expect(second).not.toBe(first)
    expect(second?.id).not.toBe(first?.id)
  })

  it("refuses every stale progress/completion/end write after a same-ID restart", () => {
    useTutorialStore.getState().startTutorial("playground-basics")
    const first = useTutorialStore.getState().activeRun!
    useTutorialStore.getState().startTutorial("playground-basics")
    const second = useTutorialStore.getState().activeRun
    useTutorialStore.getState().setStepIndex(2)
    useTutorialStore.getState().setStepIndex(7, first)
    useTutorialStore.getState().markComplete("playground-basics", first)
    useTutorialStore.getState().endTutorial(first)
    expect(useTutorialStore.getState()).toMatchObject({
      activeTutorialId: "playground-basics",
      activeRun: second,
      activeStepIndex: 2,
      completedTutorials: []
    })
  })

  it.each(["endTutorial", "resetProgress"] as const)(
    "%s retires the current run while a restart gets a new identity",
    (action) => {
      useTutorialStore.getState().startTutorial("playground-basics")
      const first = useTutorialStore.getState().activeRun!
      useTutorialStore.getState()[action]()
      expect(useTutorialStore.getState().activeRun).toBeNull()
      useTutorialStore.getState().startTutorial("playground-basics")
      expect(useTutorialStore.getState().activeRun?.id).not.toBe(first.id)
    }
  )

  it("persists only progress, never DOM/source predicates or run identity", () => {
    const target = document.createElement("button")
    const focusReturn = { target, route: "/chat", isCurrent: () => true }
    useTutorialStore.getState().startTutorial("playground-basics", focusReturn)
    expect(useTutorialStore.getState().activeRun?.focusReturn).toBe(focusReturn)
    useTutorialStore.getState().markComplete("playground-basics")
    const saved = JSON.parse(localStorage.getItem("tldw-tutorials")!)
    expect(saved.state).toEqual({
      completedTutorials: ["playground-basics"],
      seenPromptPages: []
    })
  })
})

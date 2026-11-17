import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { useActorStore, actorSceneForTarget } from "../actor"
import { createDefaultActorSettings } from "@/types/actor"
import type { ActorSettingsTarget } from "@/utils/actor-settings-target"

const owned = (key: string, isCurrent = () => true): ActorSettingsTarget => ({
  key,
  accountKey: "a".repeat(64),
  isCurrent,
  profileIsCurrent: isCurrent
})
const target = owned("qualified-one")
const scene = {
  ...createDefaultActorSettings(),
  isEnabled: true,
  notes: "Applied scene."
}
const proposal = () => ({
  target,
  settings: scene,
  preview: "Owned preview.",
  tokenCount: 3
})
const subscriptions: Array<() => void> = []
afterEach(() => {
  subscriptions.splice(0).forEach((stop) => stop())
})
beforeEach(() => useActorStore.getState().reset())

describe("atomic owned Applied scene projection", () => {
  it("publishes settings and matching preview in one store notification", () => {
    const seen: unknown[] = []
    const stop = useActorStore.subscribe((state) =>
      seen.push(actorSceneForTarget(state.applied, target))
    )
    subscriptions.push(stop)
    expect(useActorStore.getState().applyScene(proposal())).toBe(true)
    stop()
    expect(seen).toEqual([proposal()])
  })
  it("masks a different account/address target", () => {
    useActorStore.getState().applyScene(proposal())
    expect(
      actorSceneForTarget(
        useActorStore.getState().applied,
        owned("qualified-two")
      )
    ).toBeNull()
  })
  it("masks a retired Applied owner even when a replacement has the same locator", () => {
    let current = true
    useActorStore
      .getState()
      .applyScene({ ...proposal(), target: owned(target.key, () => current) })
    current = false
    expect(
      actorSceneForTarget(useActorStore.getState().applied, target)
    ).toBeNull()
  })
  it("refuses a retired proposed destination without replacing Applied", () => {
    useActorStore.getState().applyScene(proposal())
    const original = useActorStore.getState().applied
    expect(
      useActorStore
        .getState()
        .applyScene({ ...proposal(), target: owned("retired", () => false) })
    ).toBe(false)
    expect(useActorStore.getState().applied).toBe(original)
  })
  it("freezes the saved proposal so later staged edits cannot mutate Applied", () => {
    const draft = structuredClone(scene)
    useActorStore.getState().applyScene({ ...proposal(), settings: draft })
    draft.notes = "Later staged edit."
    expect(
      actorSceneForTarget(useActorStore.getState().applied, target)?.settings
        .notes
    ).toBe(scene.notes)
  })
  it("does not acknowledge a projection retired by a synchronous subscriber", () => {
    let current = true
    const stop = useActorStore.subscribe(() => {
      current = false
    })
    subscriptions.push(stop)
    expect(
      useActorStore
        .getState()
        .applyScene({ ...proposal(), target: owned(target.key, () => current) })
    ).toBe(false)
    stop()
  })
  it("clears the owned projection on explicit reset", () => {
    useActorStore.getState().applyScene(proposal())
    useActorStore.getState().reset()
    expect(
      actorSceneForTarget(useActorStore.getState().applied, target)
    ).toBeNull()
  })
})

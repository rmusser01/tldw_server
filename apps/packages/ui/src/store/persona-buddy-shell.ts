import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

import type { PersonaBuddyPositionBucket } from "@/types/persona-buddy"

export type PersonaBuddyShellPosition = { x: number; y: number }

export type PersonaBuddyShellBounds = {
  viewportWidth: number
  viewportHeight: number
  shellWidth?: number
  shellHeight?: number
  margin?: number
}

type PersonaBuddyShellPositions = Record<
  PersonaBuddyPositionBucket,
  PersonaBuddyShellPosition
>

type PersonaBuddyShellStoreState = {
  isOpen: boolean
  positions: PersonaBuddyShellPositions
  setOpen: (open: boolean) => void
  toggleOpen: () => void
  resetSessionState: () => void
  setPosition: (
    bucket: PersonaBuddyPositionBucket,
    position: PersonaBuddyShellPosition
  ) => void
  getPosition: (
    bucket: PersonaBuddyPositionBucket
  ) => PersonaBuddyShellPosition
  resetPosition: (bucket: PersonaBuddyPositionBucket) => void
}

export const PERSONA_BUDDY_SHELL_STORAGE_KEY = "tldw-persona-buddy-shell"

export const DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS: PersonaBuddyShellPositions = {
  "web-desktop": { x: 1120, y: 640 },
  "sidepanel-desktop": { x: 1120, y: 640 }
}

const DEFAULT_SHELL_MARGIN = 16

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)

const isPositionRecord = (
  value: unknown
): value is PersonaBuddyShellPosition =>
  Boolean(
    value &&
      typeof value === "object" &&
      isFiniteNumber((value as PersonaBuddyShellPosition).x) &&
      isFiniteNumber((value as PersonaBuddyShellPosition).y)
  )

const clonePosition = (
  position: PersonaBuddyShellPosition
): PersonaBuddyShellPosition => ({
  x: position.x,
  y: position.y
})

export const getDefaultPersonaBuddyShellPosition = (
  bucket: PersonaBuddyPositionBucket
): PersonaBuddyShellPosition =>
  clonePosition(DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS[bucket])

export const resolvePersonaBuddyShellPosition = (
  positions: Partial<PersonaBuddyShellPositions> | null | undefined,
  bucket: PersonaBuddyPositionBucket
): PersonaBuddyShellPosition => {
  const candidate = positions?.[bucket]
  if (isPositionRecord(candidate)) {
    return clonePosition(candidate)
  }
  return getDefaultPersonaBuddyShellPosition(bucket)
}

const normalizePersonaBuddyShellPositions = (
  positions: Partial<PersonaBuddyShellPositions> | null | undefined
): PersonaBuddyShellPositions => ({
  "web-desktop": resolvePersonaBuddyShellPosition(positions, "web-desktop"),
  "sidepanel-desktop": resolvePersonaBuddyShellPosition(
    positions,
    "sidepanel-desktop"
  )
})

export const resetPersonaBuddyShellPositionBucket = (
  positions: Partial<PersonaBuddyShellPositions> | null | undefined,
  bucket: PersonaBuddyPositionBucket
): PersonaBuddyShellPositions => ({
  ...normalizePersonaBuddyShellPositions(positions),
  [bucket]: getDefaultPersonaBuddyShellPosition(bucket)
})

const clampNumber = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max)

export const clampPersonaBuddyShellPosition = (
  position: PersonaBuddyShellPosition | null | undefined,
  bucket: PersonaBuddyPositionBucket,
  bounds?: PersonaBuddyShellBounds | null
): PersonaBuddyShellPosition => {
  const resolved = isPositionRecord(position)
    ? clonePosition(position)
    : getDefaultPersonaBuddyShellPosition(bucket)

  if (!bounds) {
    return resolved
  }

  const viewportWidth = bounds.viewportWidth
  const viewportHeight = bounds.viewportHeight
  if (!isFiniteNumber(viewportWidth) || !isFiniteNumber(viewportHeight)) {
    return resolved
  }

  const shellWidth = isFiniteNumber(bounds.shellWidth) ? bounds.shellWidth : 0
  const shellHeight = isFiniteNumber(bounds.shellHeight) ? bounds.shellHeight : 0
  const margin =
    isFiniteNumber(bounds.margin) && bounds.margin >= 0
      ? bounds.margin
      : DEFAULT_SHELL_MARGIN

  const maxX = Math.max(margin, viewportWidth - shellWidth - margin)
  const maxY = Math.max(margin, viewportHeight - shellHeight - margin)

  return {
    x: clampNumber(resolved.x, margin, maxX),
    y: clampNumber(resolved.y, margin, maxY)
  }
}

export const usePersonaBuddyShellStore =
  createWithEqualityFn<PersonaBuddyShellStoreState>()(
    persist(
      (set, get) => ({
        isOpen: false,
        positions: normalizePersonaBuddyShellPositions(undefined),
        setOpen: (open) => set({ isOpen: open }),
        toggleOpen: () => set({ isOpen: !get().isOpen }),
        resetSessionState: () => set({ isOpen: false }),
        setPosition: (bucket, position) =>
          set((state) => ({
            positions: {
              ...state.positions,
              [bucket]: isPositionRecord(position)
                ? clonePosition(position)
                : getDefaultPersonaBuddyShellPosition(bucket)
            }
          })),
        getPosition: (bucket) =>
          resolvePersonaBuddyShellPosition(get().positions, bucket),
        resetPosition: (bucket) =>
          set((state) => ({
            positions: resetPersonaBuddyShellPositionBucket(
              state.positions,
              bucket
            )
          }))
      }),
      {
        name: PERSONA_BUDDY_SHELL_STORAGE_KEY,
        storage: createJSONStorage(() =>
          typeof window !== "undefined" ? localStorage : createMemoryStorage()
        ),
        partialize: (state) => ({
          positions: state.positions
        })
      }
    )
  )

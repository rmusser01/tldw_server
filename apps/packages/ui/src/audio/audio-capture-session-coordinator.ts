import type { AudioCaptureOwner } from "./source-types"

export type AudioCaptureSessionClaim = {
  ownerBeforeClaim: AudioCaptureOwner | null
  ownerAfterClaim: AudioCaptureOwner
}

export type AudioCaptureSessionRelease = {
  ownerBeforeRelease: AudioCaptureOwner | null
  released: boolean
}

export type AudioCaptureSessionCoordinator = {
  claim: (owner: AudioCaptureOwner) => AudioCaptureSessionClaim
  release: (owner: AudioCaptureOwner) => AudioCaptureSessionRelease
  getActiveOwner: () => AudioCaptureOwner | null
}

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

export const createAudioCaptureSessionCoordinator = (
  initialOwner: AudioCaptureOwner | null = null
): AudioCaptureSessionCoordinator => {
  let activeOwner: AudioCaptureOwner | null = initialOwner

  return {
    claim(owner) {
      const ownerBeforeClaim = activeOwner
      activeOwner = owner

      return {
        ownerBeforeClaim,
        ownerAfterClaim: owner
      }
    },
    release(owner) {
      const ownerBeforeRelease = activeOwner
      const shouldRelease = ownerBeforeRelease === owner

      if (shouldRelease) {
        activeOwner = null
      }

      return {
        ownerBeforeRelease,
        released: shouldRelease
      }
    },
    getActiveOwner() {
      return activeOwner
    }
  }
}

export const getGlobalAudioCaptureSessionCoordinator =
  (): AudioCaptureSessionCoordinator => {
    const globalState = globalThis as typeof globalThis & {
      [AUDIO_CAPTURE_COORDINATOR_KEY]?: AudioCaptureSessionCoordinator
    }

    if (!globalState[AUDIO_CAPTURE_COORDINATOR_KEY]) {
      globalState[AUDIO_CAPTURE_COORDINATOR_KEY] =
        createAudioCaptureSessionCoordinator()
    }

    return globalState[AUDIO_CAPTURE_COORDINATOR_KEY]
  }

export const createAudioCaptureBusyError = (
  activeOwner: AudioCaptureOwner
): Error => new Error(`Audio capture is already active for ${activeOwner}`)

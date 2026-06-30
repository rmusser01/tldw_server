/**
 * OnboardingWizard
 *
 * Re-exports OnboardingConnectForm as the sole onboarding implementation.
 * The legacy multi-step wizard was removed as part of the FTUE audit (2026-03-30).
 */

import React from 'react'
import { OnboardingConnectForm } from './OnboardingConnectForm'
import type { OnboardingEntryIntent } from '@/utils/onboarding-route-intent'

type Props = {
  entryIntent?: OnboardingEntryIntent | null
  onFinish?: () => void
  returnTo?: string | null
}

export const OnboardingWizard: React.FC<Props> = ({
  entryIntent,
  onFinish,
  returnTo
}) => {
  return (
    <OnboardingConnectForm
      entryIntent={entryIntent}
      onFinish={onFinish}
      returnTo={returnTo}
    />
  )
}

export default OnboardingWizard

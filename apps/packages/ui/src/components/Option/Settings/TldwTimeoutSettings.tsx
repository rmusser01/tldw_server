import {
  Segmented,
  Input,
  Collapse,
  Button,
  Tag
} from "antd"
import React from "react"
import type { TFunction } from "i18next"
import type { MessageInstance } from "antd/es/message/interface"

export type TimeoutPresetKey = "balanced" | "extended"

export type TimeoutValues = {
  request: number
  stream: number
  chatRequest: number
  chatStartup: number
  chatStream: number
  ragRequest: number
  media: number
  upload: number
}

export const TIMEOUT_PRESETS: Record<TimeoutPresetKey, TimeoutValues> = {
  balanced: {
    request: 10,
    stream: 15,
    chatRequest: 10,
    chatStartup: 10,
    chatStream: 15,
    ragRequest: 10,
    media: 60,
    upload: 60
  },
  extended: {
    request: 20,
    stream: 30,
    chatRequest: 20,
    chatStartup: 20,
    chatStream: 30,
    ragRequest: 20,
    media: 90,
    upload: 90
  }
}

export const determinePreset = (values: TimeoutValues): TimeoutPresetKey | "custom" => {
  for (const [key, presetValues] of Object.entries(TIMEOUT_PRESETS) as [TimeoutPresetKey, typeof TIMEOUT_PRESETS[TimeoutPresetKey]][]) {
    const matches =
      presetValues.request === values.request &&
      presetValues.stream === values.stream &&
      presetValues.chatRequest === values.chatRequest &&
      presetValues.chatStartup === values.chatStartup &&
      presetValues.chatStream === values.chatStream &&
      presetValues.ragRequest === values.ragRequest &&
      presetValues.media === values.media &&
      presetValues.upload === values.upload
    if (matches) {
      return key
    }
  }
  return "custom"
}

export const parseSeconds = (value: string, fallback: number) => {
  const parsed = parseInt(value, 10)
  if (Number.isNaN(parsed)) {
    return fallback
  }
  return Math.max(1, parsed)
}

export type TldwTimeoutSettingsProps = {
  t: TFunction
  message: MessageInstance
  requestTimeoutSec: number
  setRequestTimeoutSec: (value: number) => void
  streamIdleTimeoutSec: number
  setStreamIdleTimeoutSec: (value: number) => void
  chatRequestTimeoutSec: number
  setChatRequestTimeoutSec: (value: number) => void
  chatStartupTimeoutSec: number
  setChatStartupTimeoutSec: (value: number) => void
  chatStreamIdleTimeoutSec: number
  setChatStreamIdleTimeoutSec: (value: number) => void
  ragRequestTimeoutSec: number
  setRagRequestTimeoutSec: (value: number) => void
  mediaRequestTimeoutSec: number
  setMediaRequestTimeoutSec: (value: number) => void
  uploadRequestTimeoutSec: number
  setUploadRequestTimeoutSec: (value: number) => void
  timeoutPreset: TimeoutPresetKey | "custom"
  setTimeoutPreset: (preset: TimeoutPresetKey | "custom") => void
}

export const TldwTimeoutSettings = ({
  t,
  message,
  requestTimeoutSec,
  setRequestTimeoutSec,
  streamIdleTimeoutSec,
  setStreamIdleTimeoutSec,
  chatRequestTimeoutSec,
  setChatRequestTimeoutSec,
  chatStartupTimeoutSec,
  setChatStartupTimeoutSec,
  chatStreamIdleTimeoutSec,
  setChatStreamIdleTimeoutSec,
  ragRequestTimeoutSec,
  setRagRequestTimeoutSec,
  mediaRequestTimeoutSec,
  setMediaRequestTimeoutSec,
  uploadRequestTimeoutSec,
  setUploadRequestTimeoutSec,
  timeoutPreset,
  setTimeoutPreset
}: TldwTimeoutSettingsProps) => {
  const applyTimeoutPreset = (preset: TimeoutPresetKey) => {
    const presetValues = TIMEOUT_PRESETS[preset]
    setRequestTimeoutSec(presetValues.request)
    setStreamIdleTimeoutSec(presetValues.stream)
    setChatRequestTimeoutSec(presetValues.chatRequest)
    setChatStartupTimeoutSec(presetValues.chatStartup)
    setChatStreamIdleTimeoutSec(presetValues.chatStream)
    setRagRequestTimeoutSec(presetValues.ragRequest)
    setMediaRequestTimeoutSec(presetValues.media)
    setUploadRequestTimeoutSec(presetValues.upload)
    setTimeoutPreset(preset)
  }

  const currentTimeouts = (): TimeoutValues => ({
    request: requestTimeoutSec,
    stream: streamIdleTimeoutSec,
    chatRequest: chatRequestTimeoutSec,
    chatStartup: chatStartupTimeoutSec,
    chatStream: chatStreamIdleTimeoutSec,
    ragRequest: ragRequestTimeoutSec,
    media: mediaRequestTimeoutSec,
    upload: uploadRequestTimeoutSec
  })

  const updateAndDeterminePreset = (override: Partial<TimeoutValues>) => {
    const next = { ...currentTimeouts(), ...override }
    setTimeoutPreset(determinePreset(next))
  }

  return (
    <div id="tldw-settings-timeouts" className="scroll-mt-24">
      <Collapse
        className="mt-4"
        items={[
        {
          key: 'adv',
          label: t('settings:tldw.advancedTimeouts'),
          children: (
            <div className="space-y-3">
              <div className="flex flex-col gap-2">
                <span className="text-sm font-medium">
                  {t('settings:tldw.timeoutPresetLabel')}
                </span>
                <div className="flex flex-wrap items-center gap-3">
                  <Segmented
                    value={timeoutPreset === 'extended' ? 'extended' : 'balanced'}
                    onChange={(value) => applyTimeoutPreset(value as TimeoutPresetKey)}
                    options={[
                      {
                        label: t('settings:tldw.timeoutPresetBalanced'),
                        value: 'balanced'
                      },
                      {
                        label: t('settings:tldw.timeoutPresetExtended'),
                        value: 'extended'
                      }
                    ]}
                  />
                  {timeoutPreset === 'custom' && (
                    <Tag color="default">
                      {t('settings:tldw.timeoutPresetCustom')}
                    </Tag>
                  )}
                </div>
                <span className="text-xs text-text-subtle">
                  {t('settings:tldw.timeoutPresetHint')}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.requestTimeout')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={requestTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.request
                      )
                      setRequestTimeoutSec(newValue)
                      updateAndDeterminePreset({ request: newValue })
                    }}
                    placeholder="10"
                    suffix="s"
                  />
                  <div className="text-xs text-text-subtle mt-1">
                    {t('settings:tldw.hints.requestTimeout', {
                      defaultValue:
                        'Abort initial requests if no response within this time. Default: {{seconds}}s.',
                      seconds: TIMEOUT_PRESETS.balanced.request
                    })}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.streamingIdle')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={streamIdleTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.stream
                      )
                      setStreamIdleTimeoutSec(newValue)
                      updateAndDeterminePreset({ stream: newValue })
                    }}
                    placeholder="15"
                    suffix="s"
                  />
                  <div className="text-xs text-text-subtle mt-1">
                    {t('settings:tldw.hints.streamingIdle', {
                      defaultValue:
                        'Abort streaming if no updates received within this time. Default: {{seconds}}s.',
                      seconds: TIMEOUT_PRESETS.balanced.stream
                    })}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.chatRequest')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={chatRequestTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.chatRequest
                      )
                      setChatRequestTimeoutSec(newValue)
                      updateAndDeterminePreset({ chatRequest: newValue })
                    }}
                    suffix="s"
                  />
                  <div className="text-xs text-text-subtle mt-1">
                    {t('settings:tldw.hints.chatRequest', {
                      defaultValue:
                        'Applies to non-stream chat request timeout handling. Use startup timeout below for first visible token.',
                    })}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.chatStartup', {
                      defaultValue: 'Chat startup timeout'
                    })}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={chatStartupTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.chatStartup
                      )
                      setChatStartupTimeoutSec(newValue)
                      updateAndDeterminePreset({ chatStartup: newValue })
                    }}
                    suffix="s"
                  />
                  <div className="text-xs text-text-subtle mt-1">
                    {t('settings:tldw.hints.chatStartup', {
                      defaultValue:
                        'Abort streaming if no visible assistant output arrives within this time. Default: {{seconds}}s.',
                      seconds: TIMEOUT_PRESETS.balanced.chatStartup
                    })}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.chatStreamIdle')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={chatStreamIdleTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.chatStream
                      )
                      setChatStreamIdleTimeoutSec(newValue)
                      updateAndDeterminePreset({ chatStream: newValue })
                    }}
                    suffix="s"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.ragRequest')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={ragRequestTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.ragRequest
                      )
                      setRagRequestTimeoutSec(newValue)
                      updateAndDeterminePreset({ ragRequest: newValue })
                    }}
                    suffix="s"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.mediaRequest')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={mediaRequestTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.media
                      )
                      setMediaRequestTimeoutSec(newValue)
                      updateAndDeterminePreset({ media: newValue })
                    }}
                    suffix="s"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {t('settings:tldw.uploadRequest')}
                  </label>
                  <Input
                    type="number"
                    min={1}
                    value={uploadRequestTimeoutSec}
                    onChange={(e) => {
                      const newValue = parseSeconds(
                        e.target.value,
                        TIMEOUT_PRESETS.balanced.upload
                      )
                      setUploadRequestTimeoutSec(newValue)
                      updateAndDeterminePreset({ upload: newValue })
                    }}
                    suffix="s"
                  />
                </div>
              </div>
              <div className="flex justify-end">
                <Button
                  onClick={() => {
                    applyTimeoutPreset('balanced')
                    message.success(t('settings:tldw.resetDone'))
                  }}
                >
                  {t('settings:tldw.reset')}
                </Button>
              </div>
            </div>
          )
        }
        ]}
      />
    </div>
  )
}

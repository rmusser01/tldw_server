import React from "react"
import { Alert, Button, Checkbox, Input, InputNumber, Space, Switch, Tag } from "antd"
import { useTranslation } from "react-i18next"

import { SplashOverlay } from "@/components/Common/SplashScreen"
import type { SplashCard } from "@/components/Common/SplashScreen/engine/types"
import {
  DEFAULT_SPLASH_CARD_NAMES,
  SPLASH_CARDS,
  randomSplashCard
} from "@/data/splash-cards"
import { randomSplashMessage } from "@/data/splash-messages"
import { useSetting } from "@/hooks/useSetting"
import {
  SPLASH_DISABLED_SETTING,
  SPLASH_ENABLED_CARD_NAMES_SETTING,
  SPLASH_DURATION_SECONDS_MAX,
  SPLASH_DURATION_SECONDS_MIN,
  SPLASH_DURATION_SECONDS_SETTING
} from "@/services/settings/ui-settings"

const arraysEqual = (a: string[], b: string[]) =>
  a.length === b.length && a.every((value, index) => value === b[index])

const SplashCardListItem = ({
  card,
  enabled,
  onToggle,
  onPreview
}: {
  card: SplashCard
  enabled: boolean
  onToggle: (checked: boolean) => void
  onPreview: () => void
}) => {
  const { t } = useTranslation("settings")
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-3">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0 space-y-1">
          <Checkbox checked={enabled} onChange={(event) => onToggle(event.target.checked)}>
            <span className="font-medium text-text">{card.name}</span>
          </Checkbox>
          <div className="pl-6 text-xs text-text-muted space-y-1">
            <div>
              {t("splashSettings.effect", {
                defaultValue: "Effect: {{effect}}",
                effect: card.effect || t("splashSettings.effectNone", "None")
              })}
            </div>
            <div>
              {t("splashSettings.asciiArt", {
                defaultValue: "ASCII art: {{name}}",
                name: card.asciiArt || "default_splash"
              })}
            </div>
          </div>
        </div>
        <Button size="small" onClick={onPreview}>
          {t("splashSettings.preview", "Preview")}
        </Button>
      </div>
    </div>
  )
}

export const SplashSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const [splashDisabled, setSplashDisabled] = useSetting(SPLASH_DISABLED_SETTING)
  const [enabledCardNames, setEnabledCardNames] = useSetting(
    SPLASH_ENABLED_CARD_NAMES_SETTING
  )
  const [durationSeconds, setDurationSeconds] = useSetting(
    SPLASH_DURATION_SECONDS_SETTING
  )

  const [search, setSearch] = React.useState("")
  const [preview, setPreview] = React.useState<{
    card: SplashCard
    message: string
  } | null>(null)

  const enabledCardNamesOrdered = React.useMemo(() => {
    const selected = new Set(enabledCardNames)
    return DEFAULT_SPLASH_CARD_NAMES.filter((name) => selected.has(name))
  }, [enabledCardNames])

  React.useEffect(() => {
    if (arraysEqual(enabledCardNamesOrdered, enabledCardNames)) return
    void setEnabledCardNames(enabledCardNamesOrdered)
  }, [enabledCardNames, enabledCardNamesOrdered, setEnabledCardNames])

  const enabledCardSet = React.useMemo(
    () => new Set(enabledCardNamesOrdered),
    [enabledCardNamesOrdered]
  )

  const filteredCards = React.useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return SPLASH_CARDS
    return SPLASH_CARDS.filter((card) => {
      const effect = card.effect ? String(card.effect).toLowerCase() : ""
      const art = card.asciiArt ? String(card.asciiArt).toLowerCase() : ""
      return (
        card.name.toLowerCase().includes(query) ||
        effect.includes(query) ||
        art.includes(query)
      )
    })
  }, [search])

  const openPreview = React.useCallback((card: SplashCard) => {
    setPreview({
      card: {
        ...card,
        duration: Math.round(durationSeconds * 1000)
      },
      message: card.subtitle || randomSplashMessage()
    })
  }, [durationSeconds])

  const handleToggleCard = React.useCallback(
    (name: string, checked: boolean) => {
      void setEnabledCardNames((previous) => {
        const selected = new Set(
          previous.filter((entry) => DEFAULT_SPLASH_CARD_NAMES.includes(entry))
        )
        if (checked) selected.add(name)
        else selected.delete(name)
        return DEFAULT_SPLASH_CARD_NAMES.filter((entry) => selected.has(entry))
      })
    },
    [setEnabledCardNames]
  )

  const handlePreviewRandomEnabled = React.useCallback(() => {
    if (enabledCardNamesOrdered.length === 0) return
    const selected = randomSplashCard({ enabledNames: enabledCardNamesOrdered })
    openPreview(selected)
  }, [enabledCardNamesOrdered, openPreview])

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("settings:splashSettings.title", "Splash screens")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "settings:splashSettings.subtitle",
            "Control login splash behavior, choose allowed cards, and preview any splash effect."
          )}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <div className="panel-card p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <div className="text-sm font-medium text-text">
              {t(
                "settings:splashSettings.featureLabel",
                "Enable splash screens after successful login"
              )}
            </div>
            <p className="text-xs text-text-muted">
              {t(
                "settings:splashSettings.featureHelp",
                "When disabled, login success will not display splash overlays."
              )}
            </p>
          </div>
          <Switch
            checked={!splashDisabled}
            onChange={(checked) => {
              void setSplashDisabled(!checked)
            }}
            aria-label={t(
              "settings:splashSettings.featureLabel",
              "Enable splash screens after successful login"
            )}
          />
        </div>

        <div className="mt-4 border-t border-border pt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <div className="text-sm font-medium text-text">
              {t(
                "settings:splashSettings.durationLabel",
                "Display duration"
              )}
            </div>
            <p className="text-xs text-text-muted">
              {t(
                "settings:splashSettings.durationHelp",
                "Choose how long the splash overlay stays visible before it auto-closes."
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <InputNumber
              min={SPLASH_DURATION_SECONDS_MIN}
              max={SPLASH_DURATION_SECONDS_MAX}
              step={1}
              precision={0}
              value={durationSeconds}
              onChange={(value) => {
                if (typeof value !== "number" || !Number.isFinite(value)) return
                const normalized = Math.min(
                  SPLASH_DURATION_SECONDS_MAX,
                  Math.max(SPLASH_DURATION_SECONDS_MIN, Math.round(value))
                )
                void setDurationSeconds(normalized)
              }}
              aria-label={t(
                "settings:splashSettings.durationLabel",
                "Display duration"
              )}
            />
            <span className="text-xs text-text-muted">
              {t("settings:splashSettings.secondsUnit", "seconds")}
            </span>
          </div>
        </div>
      </div>

      <div className="panel-card p-4 space-y-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <Space wrap>
            <Button onClick={() => void setEnabledCardNames(DEFAULT_SPLASH_CARD_NAMES)}>
              {t("settings:splashSettings.actions.enableAll", "Enable all")}
            </Button>
            <Button onClick={() => void setEnabledCardNames([])}>
              {t("settings:splashSettings.actions.disableAll", "Disable all")}
            </Button>
            <Button
              type="primary"
              disabled={enabledCardNamesOrdered.length === 0}
              onClick={handlePreviewRandomEnabled}
            >
              {t(
                "settings:splashSettings.actions.previewRandom",
                "Preview random enabled"
              )}
            </Button>
          </Space>
          <Tag color={enabledCardNamesOrdered.length > 0 ? "blue" : "red"}>
            {t("settings:splashSettings.enabledCount", {
              defaultValue: "{{count}} enabled",
              count: enabledCardNamesOrdered.length
            })}
          </Tag>
        </div>

        {enabledCardNamesOrdered.length === 0 && (
          <Alert
            type="warning"
            showIcon
            title={t(
              "settings:splashSettings.emptySelectionWarning",
              "No splash cards are enabled. Enable at least one card to show splash screens."
            )}
          />
        )}

        <Input
          allowClear
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t(
            "settings:splashSettings.searchPlaceholder",
            "Search splash cards by name or effect"
          )}
        />

        <div>
          <h3 className="text-sm font-semibold text-text">
            {t("settings:splashSettings.cardsHeading", "Available splash cards")}
          </h3>
          <p className="text-xs text-text-muted">
            {t(
              "settings:splashSettings.cardsHint",
              "Use checkboxes to control which cards can appear."
            )}
          </p>
        </div>

        <div className="max-h-[30rem] overflow-y-auto pr-1 space-y-2">
          {filteredCards.length === 0 ? (
            <div className="text-sm text-text-muted">
              {t(
                "settings:splashSettings.searchNoResults",
                "No splash cards match your search."
              )}
            </div>
          ) : (
            filteredCards.map((card) => (
              <SplashCardListItem
                key={card.name}
                card={card}
                enabled={enabledCardSet.has(card.name)}
                onToggle={(checked) => handleToggleCard(card.name, checked)}
                onPreview={() => openPreview(card)}
              />
            ))
          )}
        </div>
      </div>

      {preview ? (
        <SplashOverlay
          card={preview.card}
          message={preview.message}
          onDismiss={() => setPreview(null)}
        />
      ) : null}
    </div>
  )
}

import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import i18next from "i18next"
import { describe, expect, it } from "vitest"
import { formatWatchlistOccurrenceDate } from "../OverviewTab/LatestBriefing"

type NestedLocale = Record<string, unknown>
type ExtensionLocale = Record<string, { message?: unknown }>

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")
const canonicalLocales = [
  "ar", "da", "de", "es", "fa", "fr", "it", "ja-JP", "ko", "ml",
  "no", "pt-BR", "ru", "sv", "uk", "zh", "zh-TW"
] as const
const publicAliases = { ja: "ja-JP", zh_CN: "zh", zh_TW: "zh-TW" } as const
const requiredLatestRuntimeKeys = [
  "actions.downloadAudio",
  "actions.downloadAudioAria",
  "actions.retryExactAria",
  "actions.reviewScript",
  "actions.reviewScriptAria",
  "delivery.chatbookDestination",
  "delivery.confirmRetry",
  "delivery.confirmRetryAria",
  "delivery.recipientCount",
  "delivery.reviewAcknowledgement",
  "delivery.reviewSettings",
  "delivery.reviewTitle",
  "playback.accessError",
  "playback.missing",
  "scriptAccessError",
  "scriptLoading",
  "scriptMissing",
  "recovery.renderText",
  "recovery.persistText",
  "recovery.composeScript",
  "recovery.persistScript",
  "recovery.generateAudio",
  "recovery.persistAudio"
] as const
const requiredRecipientPluralKeys = [
  "delivery.recipientCount_zero",
  "delivery.recipientCount_one",
  "delivery.recipientCount_two",
  "delivery.recipientCount_few",
  "delivery.recipientCount_many",
  "delivery.recipientCount_other"
] as const
const requiredOutputRuntimeKeys = [
  "artifactMissing",
  "artifactAuthError",
  "artifactAccessError"
] as const
const runtimeCorrections: Record<string, Record<string, string>> = {
  en: {
    noUpdates: "No qualifying updates were found. A status {{noun}} was saved.",
    reviewRetry: "Review and retry",
    playAria: "Play {{name}}",
    pauseAria: "Pause {{name}}",
    resumeAria: "Resume {{name}}"
  },
  ar: {
    noUpdates: "لم يتم العثور على تحديثات مؤهلة. تم حفظ {{noun}} للحالة.",
    reviewRetry: "مراجعة وإعادة المحاولة",
    playAria: "تشغيل {{name}}",
    pauseAria: "إيقاف {{name}} مؤقتًا",
    resumeAria: "استئناف {{name}}"
  },
  da: {
    noUpdates: "Der blev ikke fundet nogen kvalificerende opdateringer. En status-{{noun}} blev gemt.",
    reviewRetry: "Gennemgå og prøv igen",
    playAria: "Afspil {{name}}",
    pauseAria: "Sæt {{name}} på pause",
    resumeAria: "Genoptag {{name}}"
  },
  de: {
    noUpdates: "Es wurden keine passenden Aktualisierungen gefunden. Ein Status-{{noun}} wurde gespeichert.",
    reviewRetry: "Prüfen und erneut versuchen",
    playAria: "{{name}} abspielen",
    pauseAria: "{{name}} pausieren",
    resumeAria: "{{name}} fortsetzen"
  },
  es: {
    noUpdates: "No se encontraron actualizaciones que cumplieran los criterios. Se guardó un {{noun}} de estado.",
    reviewRetry: "Revisar y reintentar",
    playAria: "Reproducir {{name}}",
    pauseAria: "Pausar {{name}}",
    resumeAria: "Reanudar {{name}}"
  },
  fa: {
    noUpdates: "هیچ به‌روزرسانی واجد شرایطی یافت نشد. یک {{noun}} وضعیت ذخیره شد.",
    reviewRetry: "بررسی و تلاش دوباره",
    playAria: "پخش {{name}}",
    pauseAria: "مکث {{name}}",
    resumeAria: "ادامه {{name}}"
  },
  fr: {
    noUpdates: "Aucune mise à jour correspondante n’a été trouvée. Un {{noun}} de statut a été enregistré.",
    reviewRetry: "Vérifier et réessayer",
    playAria: "Lire {{name}}",
    pauseAria: "Mettre {{name}} en pause",
    resumeAria: "Reprendre {{name}}"
  },
  it: {
    noUpdates: "Non sono stati trovati aggiornamenti idonei. È stato salvato un {{noun}} di stato.",
    reviewRetry: "Verifica e riprova",
    playAria: "Riproduci {{name}}",
    pauseAria: "Metti in pausa {{name}}",
    resumeAria: "Riprendi {{name}}"
  },
  "ja-JP": {
    noUpdates: "条件に一致する更新はありませんでした。ステータス用の{{noun}}を保存しました。",
    reviewRetry: "確認して再試行",
    playAria: "{{name}}を再生",
    pauseAria: "{{name}}を一時停止",
    resumeAria: "{{name}}を再開"
  },
  ko: {
    noUpdates: "조건에 맞는 업데이트를 찾지 못했습니다. 상태 {{noun}}을(를) 저장했습니다.",
    reviewRetry: "검토 후 다시 시도",
    playAria: "{{name}} 재생",
    pauseAria: "{{name}} 일시 중지",
    resumeAria: "{{name}} 다시 재생"
  },
  ml: {
    noUpdates: "യോഗ്യമായ അപ്‌ഡേറ്റുകളൊന്നും കണ്ടെത്തിയില്ല. ഒരു സ്റ്റാറ്റസ് {{noun}} സംരക്ഷിച്ചു.",
    reviewRetry: "പരിശോധിച്ച് വീണ്ടും ശ്രമിക്കുക",
    playAria: "{{name}} പ്ലേ ചെയ്യുക",
    pauseAria: "{{name}} താൽക്കാലികമായി നിർത്തുക",
    resumeAria: "{{name}} പുനരാരംഭിക്കുക"
  },
  no: {
    noUpdates: "Ingen kvalifiserende oppdateringer ble funnet. En status-{{noun}} ble lagret.",
    reviewRetry: "Se gjennom og prøv igjen",
    playAria: "Spill av {{name}}",
    pauseAria: "Sett {{name}} på pause",
    resumeAria: "Fortsett {{name}}"
  },
  "pt-BR": {
    noUpdates: "Nenhuma atualização qualificada foi encontrada. Um {{noun}} de status foi salvo.",
    reviewRetry: "Revisar e tentar novamente",
    playAria: "Reproduzir {{name}}",
    pauseAria: "Pausar {{name}}",
    resumeAria: "Retomar {{name}}"
  },
  ru: {
    noUpdates: "Подходящих обновлений не найдено. Сохранён статусный материал «{{noun}}».",
    reviewRetry: "Проверить и повторить",
    playAria: "Воспроизвести {{name}}",
    pauseAria: "Приостановить {{name}}",
    resumeAria: "Продолжить {{name}}"
  },
  sv: {
    noUpdates: "Inga kvalificerande uppdateringar hittades. En status-{{noun}} sparades.",
    reviewRetry: "Granska och försök igen",
    playAria: "Spela upp {{name}}",
    pauseAria: "Pausa {{name}}",
    resumeAria: "Återuppta {{name}}"
  },
  uk: {
    noUpdates: "Відповідних оновлень не знайдено. Збережено статусний матеріал «{{noun}}».",
    reviewRetry: "Перевірити й повторити",
    playAria: "Відтворити {{name}}",
    pauseAria: "Призупинити {{name}}",
    resumeAria: "Продовжити {{name}}"
  },
  zh: {
    noUpdates: "未找到符合条件的更新。已保存状态{{noun}}。",
    reviewRetry: "检查后重试",
    playAria: "播放 {{name}}",
    pauseAria: "暂停 {{name}}",
    resumeAria: "继续播放 {{name}}"
  },
  "zh-TW": {
    noUpdates: "找不到符合條件的更新。已儲存狀態{{noun}}。",
    reviewRetry: "檢查後重試",
    playAria: "播放 {{name}}",
    pauseAria: "暫停 {{name}}",
    resumeAria: "繼續播放 {{name}}"
  }
}

const readNested = (locale: string): NestedLocale => JSON.parse(readFileSync(
  path.resolve(srcRoot, `assets/locale/${locale}/watchlists.json`),
  "utf8"
)) as NestedLocale
const readPublic = (locale: string): ExtensionLocale => JSON.parse(readFileSync(
  path.resolve(srcRoot, `public/_locales/${locale}/watchlists.json`),
  "utf8"
)) as ExtensionLocale

const flatten = (value: unknown, prefix: string[] = []): Record<string, string> => {
  if (typeof value === "string") return { [prefix.join("_")]: value }
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return Object.entries(value as Record<string, unknown>).reduce((all, [key, nested]) => ({
    ...all,
    ...flatten(nested, [...prefix, key])
  }), {})
}

const latestCopy = (locale: NestedLocale): Record<string, string> => flatten(
  (locale.overview as Record<string, unknown> | undefined)?.latest,
  ["overview", "latest"]
)
const placeholders = (value: string) => [...value.matchAll(/{{\s*([^}]+?)\s*}}/g)]
  .map((match) => match[1])
  .sort()
const nestedString = (value: unknown, path: string): string | undefined => {
  let current = value
  for (const segment of path.split(".")) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return typeof current === "string" ? current : undefined
}

const latestComponentSource = readFileSync(path.resolve(
  srcRoot,
  "components/Option/Watchlists/OverviewTab/LatestBriefing.tsx"
), "utf8")
const outputComponentSource = readFileSync(path.resolve(
  srcRoot,
  "components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx"
), "utf8")

describe("Watchlists Latest briefing locale contract", () => {
  const english = latestCopy(readNested("en"))
  const englishKeys = Object.keys(english).sort()

  it.each(canonicalLocales)("%s ships the complete translated Latest contract", (locale) => {
    const translated = latestCopy(readNested(locale))
    expect(englishKeys.length).toBeGreaterThan(50)
    expect(Object.keys(translated).sort()).toEqual(englishKeys)
    expect(englishKeys.filter((key) => translated[key] !== english[key]).length)
      .toBeGreaterThan(englishKeys.length * 0.8)
    for (const key of englishKeys) {
      expect(translated[key]?.trim(), `${locale}:${key}`).toBeTruthy()
      expect(placeholders(translated[key]), `${locale}:${key}`).toEqual(placeholders(english[key]))
    }
  })

  it.each(canonicalLocales)("%s mirrors Latest copy into the extension locale", (locale) => {
    const nested = latestCopy(readNested(locale))
    const extension = readPublic(locale)
    for (const [key, value] of Object.entries(nested)) {
      expect(extension[key]?.message, `${locale}:${key}`).toBe(value)
    }
  })

  it.each(Object.entries(publicAliases))("%s mirrors its canonical Latest locale", (alias, canonical) => {
    const aliasMessages = readPublic(alias)
    const canonicalMessages = readPublic(canonical)
    for (const key of Object.keys(latestCopy(readNested(canonical)))) {
      expect(aliasMessages[key]?.message, `${alias}:${key}`).toBe(canonicalMessages[key]?.message)
    }
  })

  it.each(["en", ...canonicalLocales])("%s contains every runtime Latest key with English placeholder parity", (locale) => {
    const resource = readNested(locale)
    const english = readNested("en")
    for (const key of [...requiredLatestRuntimeKeys, ...requiredRecipientPluralKeys]) {
      const value = nestedString(resource, `overview.latest.${key}`)
      const englishValue = nestedString(english, `overview.latest.${key}`)
      expect(value?.trim(), `${locale}:overview.latest.${key}`).toBeTruthy()
      expect(placeholders(value || ""), `${locale}:overview.latest.${key}`)
        .toEqual(placeholders(englishValue || ""))
    }
    const confirmation = nestedString(resource, "overview.latest.delivery.unknownConfirmation") || ""
    expect(placeholders(confirmation), `${locale}:unknownConfirmation`)
      .toEqual(["adapter", "destination"])
  })

  it("keeps the runtime manifest aligned with component translation references", () => {
    for (const key of requiredLatestRuntimeKeys.filter((key) => !key.startsWith("recovery."))) {
      expect(latestComponentSource, `LatestBriefing:${key}`)
        .toContain(`watchlists:overview.latest.${key}`)
    }
    expect(latestComponentSource).toContain("watchlists:overview.latest.recovery.${entry[0]}")
    for (const key of requiredOutputRuntimeKeys) {
      expect(outputComponentSource, `OutputPreviewDrawer:${key}`)
        .toContain(`watchlists:outputs.${key}`)
    }
  })

  it.each(["en", ...canonicalLocales])("%s contains every authenticated output artifact error key", (locale) => {
    const resource = readNested(locale)
    for (const key of requiredOutputRuntimeKeys) {
      expect(nestedString(resource, `outputs.${key}`)?.trim(), `${locale}:outputs.${key}`).toBeTruthy()
      expect(readPublic(locale)[`outputs_${key}`]?.message, `${locale}:outputs_${key}`)
        .toBe(nestedString(resource, `outputs.${key}`))
    }
  })

  it.each(Object.entries(publicAliases))("%s mirrors authenticated artifact errors from its canonical locale", (alias, canonical) => {
    for (const key of requiredOutputRuntimeKeys) {
      expect(readPublic(alias)[`outputs_${key}`]?.message, `${alias}:outputs_${key}`)
        .toBe(readPublic(canonical)[`outputs_${key}`]?.message)
    }
  })

  it("uses natural scheduled-empty copy from the real English resource", () => {
    expect(nestedString(readNested("en"), "overview.latest.empty.scheduled"))
      .toBe("Your first briefing is scheduled for {{date}}.")
  })

  it("formats an exact occurrence in the active locale and authoritative timezone", () => {
    expect(formatWatchlistOccurrenceDate(
      "2026-07-12T18:00:00-07:00",
      "America/Los_Angeles",
      "es"
    )).toMatch(/domingo.*12.*julio.*18:00/i)
  })

  it.each(Object.entries(runtimeCorrections))("%s uses reviewed runtime copy instead of fallback-era labels", (locale, expected) => {
    const latest = (readNested(locale).overview as Record<string, unknown>).latest as Record<string, unknown>
    const delivery = latest.delivery as Record<string, string>
    const playback = latest.playback as Record<string, string>
    expect(latest.noUpdates).toBe(expected.noUpdates)
    expect(delivery.reviewRetry).toBe(expected.reviewRetry)
    expect(playback.playAria).toBe(expected.playAria)
    expect(playback.pauseAria).toBe(expected.pauseAria)
    expect(playback.resumeAria).toBe(expected.resumeAria)
  })

  it("resolves corrected accessible copy from the real English resource", async () => {
    const instance = i18next.createInstance()
    await instance.init({
      lng: "en",
      fallbackLng: false,
      resources: { en: { watchlists: readNested("en") } },
      ns: ["watchlists"],
      defaultNS: "watchlists",
      interpolation: { escapeValue: false }
    })
    expect(instance.t("overview.latest.playback.playAria", { name: "Signal Check" }))
      .toBe("Play Signal Check")
    expect(instance.t("overview.latest.delivery.reviewRetry")).toBe("Review and retry")
    expect(instance.t("overview.latest.noUpdates", { noun: "briefing" }))
      .toBe("No qualifying updates were found. A status briefing was saved.")
  })

  it("uses active-locale plural categories for provenance counts", async () => {
    const instance = i18next.createInstance()
    await instance.init({
      lng: "ru",
      fallbackLng: false,
      resources: { ru: { watchlists: readNested("ru") } },
      ns: ["watchlists"],
      defaultNS: "watchlists",
      interpolation: { escapeValue: false }
    })
    expect(instance.t("overview.latest.provenance.sources", { count: 1 })).toContain("1")
    expect(instance.t("overview.latest.provenance.sources", { count: 2 }))
      .not.toBe(instance.t("overview.latest.provenance.sources", { count: 5 }))
  })
})

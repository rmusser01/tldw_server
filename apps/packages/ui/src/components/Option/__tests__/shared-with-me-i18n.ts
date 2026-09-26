import i18n from "i18next"
import { initReactI18next } from "react-i18next"
import common from "@/assets/locale/en/common.json"

void i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  initImmediate: false,
  resources: { en: { common } },
  interpolation: { escapeValue: false }
})

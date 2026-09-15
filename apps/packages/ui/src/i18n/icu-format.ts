import ICU, { type IcuConfig } from "i18next-icu"

declare module "i18next-icu" {
  // The generic name must match the upstream declaration for interface merging.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface IcuInstance<TOptions = IcuConfig> {
    parse(
      res: unknown,
      options: Record<string, unknown>,
      lng: string,
      ns: string,
      key: string,
      info?: { resolved?: { res?: unknown } }
    ): unknown
  }
}

const I18NEXT_VARIABLE_PATTERN = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g

// Convert existing i18next placeholders to ICU arguments before ICU memoizes
// the message template. Interpolating values first would cache the first value.
export default class ICUWithInterpolation extends ICU {
  parse(
    res: unknown,
    options: Record<string, unknown>,
    lng: string,
    ns: string,
    key: string,
    info?: { resolved?: { res?: unknown } }
  ) {
    // Upstream ICU also accepts syntax trees and handles invalid resources via
    // its configured parse-error handler. Only strings need our conversion.
    const icuMessage =
      typeof res === "string"
        ? res.replace(
            I18NEXT_VARIABLE_PATTERN,
            (placeholder, variable: string) =>
              Object.prototype.hasOwnProperty.call(options, variable)
                ? `{${variable}}`
                : placeholder
          )
        : res
    return super.parse(icuMessage, options, lng, ns, key, info)
  }
}

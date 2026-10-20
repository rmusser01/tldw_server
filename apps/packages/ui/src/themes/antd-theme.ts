import { theme, type ThemeConfig } from "antd"
import {
  firstReadableColor,
  mixRgb,
  readableForeground,
  readableForegroundOnAll,
  readableShade,
} from "./contrast"
import type { RGBTriple, ThemeColorTokens, ThemeTypography, ThemeShape, ThemeLayout } from "./types"

type MappingAlgorithm = Extract<
  NonNullable<ThemeConfig["algorithm"]>,
  (...args: never[]) => unknown
>
type MapToken = ReturnType<MappingAlgorithm>
type ComponentsConfig = NonNullable<ThemeConfig["components"]>

/**
 * Components whose keyboard focus outline antd draws in `colorPrimaryBorder`
 * (via genFocusStyle / genFocusOutline). Slider and DatePicker are left out:
 * they use that token for their track and cell borders, not for focus.
 */
const FOCUS_OUTLINE_COMPONENTS = [
  "Breadcrumb",
  "Button",
  "Checkbox",
  "Collapse",
  "Drawer",
  "Dropdown",
  "Menu",
  "Modal",
  "Notification",
  "Pagination",
  "Radio",
  "Segmented",
  "Switch",
  "Tabs",
  "Tour",
  "Tree",
  "Upload",
] as const

/**
 * Convert an RGB space-separated triple (e.g. "47 111 237") to a hex string ("#2f6fed").
 */
export function rgbTripleToHex(triple: RGBTriple): string {
  const parts = triple.trim().split(/\s+/)
  const r = Math.max(0, Math.min(255, parseInt(parts[0] ?? "0", 10)))
  const g = Math.max(0, Math.min(255, parseInt(parts[1] ?? "0", 10)))
  const b = Math.max(0, Math.min(255, parseInt(parts[2] ?? "0", 10)))
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
}

/**
 * A final algorithm that puts back colours the light/dark algorithms derive
 * from their seeds. Without it, antd's dark algorithm turns #5c8dff into
 * #517bdc, so antd and Tailwind primary controls show two different blues
 * (XP-20) and white text on the derived blue fails contrast (AX-07).
 */
function pinMapColors(pins: Partial<MapToken>): MappingAlgorithm {
  return (_seed, mapToken) => ({ ...(mapToken as MapToken), ...pins })
}

const WHITE: RGBTriple = "255 255 255"
const BLACK: RGBTriple = "0 0 0"

/**
 * Hover and pressed shades of a solid fill that carries `foreground` text.
 * antd derives these from its seed, which can move the fill towards the text
 * colour (the light theme's pressed primary put black text at 3.06:1). Moving
 * away from the text colour keeps the label at least as readable as at rest.
 */
function fillStateShades(fill: RGBTriple, foreground: RGBTriple) {
  const away = foreground === BLACK ? WHITE : BLACK
  return {
    hover: rgbTripleToHex(mixRgb(fill, away, 0.12)),
    active: rgbTripleToHex(mixRgb(fill, away, 0.24)),
  }
}

/**
 * Build an Ant Design ThemeConfig from our semantic tokens + dark mode flag.
 * This bridges our CSS custom property system to Ant Design's theme engine
 * so that modals, dropdowns, tables, buttons, etc. all match.
 *
 * Accessibility (UX review 2026-10, #3124):
 * - Primary, danger and link colours are the theme's own values (XP-20).
 * - Text on solid primary/danger fills is black or white, whichever reads
 *   better, matching --color-primary-foreground (AX-07). That covers every
 *   component that draws colorTextLightSolid on those fills, not only Button.
 * - Hover and pressed primary/danger fills move away from their text colour,
 *   so the label stays readable in every state (AX-07).
 * - Placeholders and secondary text use the first theme text colour that
 *   reaches 4.5:1 on the surfaces they sit on. Links use the primary or
 *   strong primary shade when one reaches 4.5:1, otherwise the strong shade
 *   moved towards the text colour until it does; hover and pressed links
 *   move further towards the text colour (AX-07).
 * - Keyboard focus is a 2px outline in the theme focus colour (AX-08).
 */
export function buildAntdThemeConfig(
  tokens: ThemeColorTokens,
  isDark: boolean,
  typography?: ThemeTypography,
  shape?: ThemeShape,
  layout?: ThemeLayout,
): ThemeConfig {
  const primary = rgbTripleToHex(tokens.primary)
  const danger = rgbTripleToHex(tokens.danger)
  const focus = rgbTripleToHex(tokens.focus)
  const onPrimaryTriple = readableForeground(tokens.primary)
  const onDangerTriple = readableForeground(tokens.danger)
  const onPrimary = rgbTripleToHex(onPrimaryTriple)
  const onDanger = rgbTripleToHex(onDangerTriple)
  const primaryStates = fillStateShades(tokens.primary, onPrimaryTriple)
  const dangerStates = fillStateShades(tokens.danger, onDangerTriple)
  const pageSurfaces = [tokens.bg, tokens.surface, tokens.surface2, tokens.elevated]
  const linkTriple = readableShade(
    firstReadableColor([tokens.primary, tokens.primaryStrong], pageSurfaces),
    tokens.text,
    pageSurfaces
  )
  const link = rgbTripleToHex(linkTriple)
  const linkHover = rgbTripleToHex(mixRgb(linkTriple, tokens.text, 0.2))
  const linkActive = rgbTripleToHex(mixRgb(linkTriple, tokens.text, 0.35))
  const secondaryText = rgbTripleToHex(
    firstReadableColor([tokens.textMuted, tokens.text], pageSurfaces)
  )
  // tailwind-shared.css paints inputs and selects --color-surface-2.
  const placeholder = rgbTripleToHex(
    firstReadableColor([tokens.textSubtle, tokens.textMuted, tokens.text], [tokens.surface2])
  )

  const algorithms: MappingAlgorithm[] = [isDark ? theme.darkAlgorithm : theme.defaultAlgorithm]
  if (layout?.density === "compact") {
    algorithms.push(theme.compactAlgorithm)
  }
  algorithms.push(
    pinMapColors({
      colorPrimary: primary,
      colorPrimaryHover: primaryStates.hover,
      colorPrimaryActive: primaryStates.active,
      colorError: danger,
      colorErrorHover: dangerStates.hover,
      colorErrorActive: dangerStates.active,
      colorLink: link,
      colorLinkHover: linkHover,
      colorLinkActive: linkActive,
    })
  )

  const focusOutline = { colorPrimaryBorder: focus }
  const components = Object.fromEntries(
    FOCUS_OUTLINE_COMPONENTS.map((name) => [name, { ...focusOutline }])
  ) as ComponentsConfig
  // The fallback avatar is painted colorTextPlaceholder, which is now a text
  // colour; give it a surface fill and a readable icon instead.
  const avatarFill = tokens.surface2
  const avatarIcon = firstReadableColor([tokens.textMuted, tokens.text], [avatarFill])

  return {
    algorithm: algorithms,
    token: {
      colorPrimary: primary,
      colorSuccess: rgbTripleToHex(tokens.success),
      colorWarning: rgbTripleToHex(tokens.warn),
      colorError: danger,
      colorTextBase: rgbTripleToHex(tokens.text),
      colorBgBase: rgbTripleToHex(tokens.bg),
      colorBorder: rgbTripleToHex(tokens.border),
      colorBgContainer: rgbTripleToHex(tokens.surface),
      colorBgElevated: rgbTripleToHex(tokens.elevated),
      colorLink: link,
      colorTextPlaceholder: placeholder,
      colorTextDescription: secondaryText,
      colorTextTertiary: secondaryText,
      lineWidthFocus: 2,
      fontFamily: typography?.fontFamily ?? "Arimo",
      fontSize: typography?.fontSizeBody ?? 14,
      ...(shape && {
        borderRadius: shape.radiusMd,
        borderRadiusLG: shape.radiusLg,
        borderRadiusSM: shape.radiusSm,
      }),
      boxShadow: tokens.shadowSm,
      boxShadowSecondary: tokens.shadowMd,
    },
    components: {
      ...components,
      Button: { ...focusOutline, primaryColor: onPrimary, dangerColor: onDanger },
      Radio: { ...focusOutline, buttonSolidCheckedColor: onPrimary },
      DatePicker: { colorTextLightSolid: onPrimary },
      // Text on primary and danger fills elsewhere (AX-07).
      Steps: {
        colorTextLightSolid: rgbTripleToHex(readableForegroundOnAll([tokens.primary, tokens.danger])),
      },
      Badge: {
        colorTextLightSolid: rgbTripleToHex(readableForegroundOnAll([tokens.danger, tokens.primary])),
      },
      Tour: { ...focusOutline, colorTextLightSolid: onPrimary },
      Dropdown: { ...focusOutline, colorTextLightSolid: onDanger },
      Tag: { colorTextLightSolid: onPrimary },
      Menu: {
        ...focusOutline,
        darkItemSelectedColor: onPrimary,
        darkDangerItemSelectedColor: onDanger,
      },
      Tree: { ...focusOutline, directoryNodeSelectedColor: onPrimary },
      Avatar: {
        colorTextPlaceholder: rgbTripleToHex(avatarFill),
        colorTextLightSolid: rgbTripleToHex(avatarIcon),
      },
    },
  }
}

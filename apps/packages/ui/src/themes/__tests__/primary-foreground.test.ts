import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Button } from 'antd';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import postcss from 'postcss';
import tailwindcss from 'tailwindcss';
import { afterEach, describe, expect, it } from 'vitest';
import { applyThemeTokens, clearThemeTokens } from '../apply-theme';
import { contrastRatio } from '../contrast';
import { getBuiltinPresets, getDefaultTheme } from '../presets';

const require = createRequire(import.meta.url);
const foregroundVar = '--color-primary-foreground';
const strongForegroundVar = '--color-primary-strong-foreground';
const black = '0 0 0';
const white = '255 255 255';

afterEach(clearThemeTokens);

describe('shared primary button foreground', () => {
  it.each(['tldw-frontend', 'extension'])(
    'generates the existing foreground utility in %s',
    async (client) => {
      const config = require(`../../../../../${client}/tailwind.config.js`);
      const result = await postcss([
        tailwindcss({
          ...config,
          content: [
            {
              raw: '<button class="bg-primary text-primary-foreground">Start</button>',
            },
          ],
        }),
      ]).process('@tailwind utilities;', { from: undefined });
      const foregrounds: string[] = [];
      result.root.walkRules('.text-primary-foreground', (rule) => {
        rule.walkDecls('color', (decl) => {
          foregrounds.push(decl.value);
        });
      });
      expect(foregrounds).toEqual([expect.stringContaining('var(--color-primary-foreground)')]);
    }
  );

  it('defines readable default foregrounds before any custom theme is applied', () => {
    const css = postcss.parse(
      readFileSync(require.resolve('../../assets/tailwind-shared.css'), 'utf8')
    );
    for (const selector of [':root', '.dark']) {
      const colors = new Map<string, string>();
      css.walkRules(selector, (rule) => {
        rule.walkDecls((decl) => {
          colors.set(decl.prop, decl.value);
        });
      });
      expect(colors.get(foregroundVar), selector).toBeTruthy();
      expect(
        contrastRatio(colors.get(foregroundVar)!, colors.get('--color-primary')!),
        selector
      ).toBeGreaterThanOrEqual(4.5);
    }
  });

  it('applies the higher contrast foreground for every built-in palette', () => {
    for (const theme of getBuiltinPresets()) {
      for (const [mode, tokens] of Object.entries(theme.palette)) {
        applyThemeTokens(tokens, theme);
        const foreground = document.documentElement.style.getPropertyValue(foregroundVar);
        expect([black, white], `${theme.id}/${mode}`).toContain(foreground);
        expect(
          contrastRatio(foreground, tokens.primary),
          `${theme.id}/${mode}`
        ).toBeGreaterThanOrEqual(
          Math.max(contrastRatio(black, tokens.primary), contrastRatio(white, tokens.primary))
        );
      }
    }
  });

  it('recomputes the foreground when a custom primary changes across dark, midtone and light colors', () => {
    const theme = getDefaultTheme();
    for (const primary of [
      black,
      '117 117 117',
      '118 118 118',
      '255 0 0',
      '0 255 0',
      '0 0 255',
      white,
    ]) {
      applyThemeTokens({ ...theme.palette.light, primary }, theme);
      const foreground = document.documentElement.style.getPropertyValue(foregroundVar);
      expect([black, white], primary).toContain(foreground);
      expect(contrastRatio(foreground, primary), primary).toBeGreaterThanOrEqual(
        Math.max(contrastRatio(black, primary), contrastRatio(white, primary), 4.5)
      );
    }
  });

  it.each(["tldw-frontend", "extension"])(
    "generates hover and pressed foreground utilities in %s",
    async (client) => {
      const config = require(`../../../../../${client}/tailwind.config.js`)
      const result = await postcss([
        tailwindcss({
          ...config,
          content: [
            {
              raw: '<button class="hover:text-primary-strong-foreground active:text-primary-strong-foreground">Start</button>'
            }
          ]
        })
      ]).process("@tailwind utilities;", { from: undefined })
      const colors: string[] = []
      result.root.walkDecls("color", (decl) => {
        colors.push(decl.value)
      })
      expect(colors).toEqual([
        expect.stringContaining("var(--color-primary-strong-foreground)"),
        expect.stringContaining("var(--color-primary-strong-foreground)")
      ])
    }
  )

  it("defines readable strong foreground defaults in both stylesheet modes", () => {
    const css = postcss.parse(
      readFileSync(require.resolve("../../assets/tailwind-shared.css"), "utf8")
    )
    for (const selector of [":root", ".dark"]) {
      const colors = new Map<string, string>()
      css.walkRules(selector, (rule) => {
        rule.walkDecls((decl) => {
          colors.set(decl.prop, decl.value)
        })
      })
      expect(colors.get(strongForegroundVar), selector).toBeTruthy()
      expect(
        contrastRatio(
          colors.get(strongForegroundVar)!,
          colors.get("--color-primary-strong")!
        ),
        selector
      ).toBeGreaterThanOrEqual(4.5)
    }
  })

  it("derives strong foregrounds for every palette and custom strong-color edit independently of primary", () => {
    for (const theme of getBuiltinPresets()) {
      for (const tokens of Object.values(theme.palette)) {
        for (const primaryStrong of [
          tokens.primaryStrong,
          black,
          "117 117 117",
          "118 118 118",
          white
        ]) {
          applyThemeTokens({ ...tokens, primaryStrong }, theme)
          const foreground =
            document.documentElement.style.getPropertyValue(strongForegroundVar)
          expect([black, white], `${theme.id}/${primaryStrong}`).toContain(
            foreground
          )
          expect(contrastRatio(foreground, primaryStrong)).toBeGreaterThanOrEqual(
            Math.max(
              contrastRatio(black, primaryStrong),
              contrastRatio(white, primaryStrong),
              4.5
            )
          )
        }
      }
    }
  })

  it("clears the strong foreground when returning to stylesheet defaults", () => {
    const theme = getDefaultTheme()
    applyThemeTokens({ ...theme.palette.light, primaryStrong: black }, theme)
    expect(
      document.documentElement.style.getPropertyValue(strongForegroundVar)
    ).toBe(white)
    clearThemeTokens()
    expect(
      document.documentElement.style.getPropertyValue(strongForegroundVar)
    ).toBe("")
  })

  it('clears the derived foreground when returning to stylesheet defaults', () => {
    const theme = getDefaultTheme();
    applyThemeTokens({ ...theme.palette.light, primary: black }, theme);
    expect(document.documentElement.style.getPropertyValue(foregroundVar)).toBe(white);
    clearThemeTokens();
    expect(document.documentElement.style.getPropertyValue(foregroundVar)).toBe('');
  });
});

describe('Ant Design primary bridge boundaries', () => {
  const selectorParser = createRequire(require.resolve('tailwindcss/package.json'))(
    'postcss-selector-parser'
  );

  it.each([
    { name: 'regular primary', props: { type: 'primary' as const }, expected: true },
    {
      name: 'disabled primary',
      props: { type: 'primary' as const, disabled: true },
      expected: false,
    },
    { name: 'danger primary', props: { type: 'primary' as const, danger: true }, expected: false },
    { name: 'ghost primary', props: { type: 'primary' as const, ghost: true }, expected: false },
    {
      name: 'outlined primary',
      props: { type: 'primary' as const, color: 'primary' as const, variant: 'outlined' as const },
      expected: false,
    },
    {
      name: 'link primary',
      props: { color: 'primary' as const, variant: 'link' as const },
      expected: false,
    },
  ])('applies readable normal/hover/pressed pairs only to $name', ({ props, expected }) => {
    const host = document.createElement('div');
    host.innerHTML = renderToStaticMarkup(createElement(Button, props, 'Action'));
    const button = host.querySelector('button')!;
    const css = postcss.parse(
      readFileSync(require.resolve('../../assets/tailwind-shared.css'), 'utf8')
    );
    for (const state of ['normal', 'hover', 'active']) {
      button.classList.remove('test-hover', 'test-active');
      if (state !== 'normal') button.classList.add('test-' + state);
      const styles: Record<string, string> = {};
      css.walkRules((rule) => {
        const ownsFill = rule.nodes.some(
          (node) =>
            node.type === 'decl' &&
            node.prop === 'background-color' &&
            /^rgb\(var\(--color-primary(?:-strong)?\)\)$/.test(node.value)
        );
        if (!ownsFill) return;
        const selector = selectorParser(
          (root: {
            walkPseudos: (
              fn: (pseudo: { value: string; replaceWith: (node: unknown) => void }) => void
            ) => void;
          }) => {
            root.walkPseudos((pseudo) => {
              if (pseudo.value === ':hover' || pseudo.value === ':active') {
                pseudo.replaceWith(
                  selectorParser.className({ value: 'test-' + pseudo.value.slice(1) })
                );
              }
            });
          }
        ).processSync(rule.selector);
        if (!button.matches(selector)) return;
        rule.walkDecls((decl) => {
          if (decl.prop === 'color' || decl.prop === 'background-color') {
            expect(decl.important).toBe(true);
            styles[decl.prop] = decl.value;
          }
        });
      });
      const tone = state === 'normal' ? 'primary' : 'primary-strong';
      expect(styles).toEqual(
        expected
          ? {
              'background-color': 'rgb(var(--color-' + tone + '))',
              color: 'rgb(var(--color-' + tone + '-foreground))',
            }
          : {}
      );
    }
  });
});

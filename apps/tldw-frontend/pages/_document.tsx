import { Html, Head, Main, NextScript } from "next/document";

// NOTE: This is the ONLY executable inline script in the app. The
// Content-Security-Policy in next.config.mjs allowlists it by SHA-256 hash so
// that script-src can drop 'unsafe-inline'. If you edit THEME_BOOTSTRAP_SCRIPT
// below, you MUST regenerate `themeBootstrapScriptHash` in next.config.mjs or
// the browser will block this script (theme flash on load). Regeneration command
// is documented next to that constant.
const THEME_BOOTSTRAP_SCRIPT = `
(() => {
  try {
    const legacyTheme = window.localStorage.getItem("tldw-theme");
    const storedTheme = window.localStorage.getItem("theme") || legacyTheme;
    if (!window.localStorage.getItem("theme") && legacyTheme) {
      window.localStorage.setItem("theme", legacyTheme);
    }
    const prefersDark =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    const shouldUseDark =
      storedTheme === "dark" ||
      !storedTheme ||
      (storedTheme === "system" && prefersDark);
    if (shouldUseDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  } catch (_) {
    // Ignore storage/matchMedia failures and let runtime theme logic recover.
  }
})();
`;

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <script
          id="tldw-theme-bootstrap"
          dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }}
        />
      </Head>
      <body className="antialiased arimo">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

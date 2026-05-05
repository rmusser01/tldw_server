/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "../packages/ui/src/**/*.{ts,tsx,html}",
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--color-bg) / <alpha-value>)",
        surface: "rgb(var(--color-surface) / <alpha-value>)",
        surface2: "rgb(var(--color-surface-2) / <alpha-value>)",
        elevated: "rgb(var(--color-elevated) / <alpha-value>)",
        primary: "rgb(var(--color-primary) / <alpha-value>)",
        primaryStrong: "rgb(var(--color-primary-strong) / <alpha-value>)",
        accent: "rgb(var(--color-accent) / <alpha-value>)",
        success: "rgb(var(--color-success) / <alpha-value>)",
        warn: "rgb(var(--color-warn) / <alpha-value>)",
        danger: "rgb(var(--color-danger) / <alpha-value>)",
        muted: "rgb(var(--color-muted) / <alpha-value>)",
        border: "rgb(var(--color-border) / <alpha-value>)",
        "border-strong": "rgb(var(--color-border-strong) / <alpha-value>)",
        text: "rgb(var(--color-text) / <alpha-value>)",
        "text-muted": "rgb(var(--color-text-muted) / <alpha-value>)",
        "text-subtle": "rgb(var(--color-text-subtle) / <alpha-value>)",
        focus: "rgb(var(--color-focus) / <alpha-value>)",
        state: {
          ready: "rgb(var(--state-ready) / <alpha-value>)",
          unavailable: "rgb(var(--state-unavailable) / <alpha-value>)",
          setupRequired: "rgb(var(--state-setup-required) / <alpha-value>)",
          authRequired: "rgb(var(--state-auth-required) / <alpha-value>)",
          permissionDenied: "rgb(var(--state-permission-denied) / <alpha-value>)",
          degraded: "rgb(var(--state-degraded) / <alpha-value>)",
          retrying: "rgb(var(--state-retrying) / <alpha-value>)",
          blocked: "rgb(var(--state-blocked) / <alpha-value>)",
          empty: "rgb(var(--state-empty) / <alpha-value>)",
          loading: "rgb(var(--state-loading) / <alpha-value>)",
          error: "rgb(var(--state-error) / <alpha-value>)"
        }
      },
      fontFamily: {
        display: ["Space Grotesk", "Inter", "sans-serif"],
        body: ["var(--font-family, Inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-family-mono, 'Courier New')", "monospace"],
        arimo: ["Arimo", "sans-serif"],
      },
      fontSize: {
        body: ["var(--font-size-body, 14px)", { lineHeight: "1.43" }],
        message: ["var(--font-size-message, 15px)", { lineHeight: "1.47" }],
        caption: ["var(--font-size-caption, 12px)", { lineHeight: "1.33" }],
        label: ["var(--font-size-label, 11px)", { lineHeight: "1.27", letterSpacing: "0.04em" }],
      },
      borderRadius: {
        sm: "var(--radius-sm, 2px)",
        md: "var(--radius-md, 6px)",
        lg: "var(--radius-lg, 8px)",
        xl: "var(--radius-xl, 12px)",
        card: "var(--radius-xl, 12px)",
        pill: "9999px",
      },
      boxShadow: {
        sm: "var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.12))",
        md: "var(--shadow-md, 0 6px 18px rgba(0,0,0,0.08))",
        card: "var(--shadow-md, 0 6px 18px rgba(0,0,0,0.16))",
        modal: "var(--shadow-md, 0 10px 30px rgba(0,0,0,0.28))",
      },
      backgroundImage: {
        "bottom-mask":
          "linear-gradient(0deg, transparent 0, rgb(var(--color-bg)) 160px)"
      },
      maskImage: {
        "bottom-fade": "linear-gradient(0deg, transparent 0, #000 160px)"
      },
      keyframes: {
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "10%, 30%, 50%, 70%, 90%": { transform: "translateX(-2px)" },
          "20%, 40%, 60%, 80%": { transform: "translateX(2px)" }
        }
      },
      animation: {
        shake: "shake 0.3s ease-in-out"
      }
    }
  },
  plugins: [require("@tailwindcss/forms"), require("@tailwindcss/typography")]
}

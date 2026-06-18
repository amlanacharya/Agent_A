/** @type {import('tailwindcss').Config} */
//
// Tailwind theme tokens ported verbatim from
// `prototype/data_intelligence_cockpit/DESIGN.md` (Phase 9 source of truth).
//
// The 11 prototype HTML files are visual references only.
// Do not copy their inline color values; Tailwind classes here
// are the canonical name. Prototypes are checked for layout
// patterns only.
//
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#f7f9fb",
          dim: "#d8dadc",
          bright: "#f7f9fb",
        },
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f2f4f6",
        "surface-container": "#eceef0",
        "surface-container-high": "#e6e8ea",
        "surface-container-highest": "#e0e3e5",
        background: "#f7f9fb",
        "surface-variant": "#e0e3e5",
        primary: {
          DEFAULT: "#3525cd",
          container: "#4f46e5",
        },
        "on-primary": "#ffffff",
        "on-primary-container": "#dad7ff",
        "surface-tint": "#4d44e3",
        "primary-fixed": "#e2dfff",
        "primary-fixed-dim": "#c3c0ff",
        "on-primary-fixed": "#0f0069",
        "on-primary-fixed-variant": "#3323cc",
        "inverse-primary": "#c3c0ff",
        secondary: {
          DEFAULT: "#00687a",
          container: "#57dffe",
          fixed: "#acedff",
          "fixed-dim": "#4cd7f6",
        },
        "on-secondary": "#ffffff",
        "on-secondary-container": "#006172",
        "on-secondary-fixed": "#001f26",
        "on-secondary-fixed-variant": "#004e5c",
        tertiary: {
          DEFAULT: "#7e3000",
          container: "#a44100",
          fixed: "#ffdbcc",
          "fixed-dim": "#ffb695",
        },
        "on-tertiary": "#ffffff",
        "on-tertiary-container": "#ffd2be",
        "on-tertiary-fixed": "#351000",
        "on-tertiary-fixed-variant": "#7b2f00",
        "success-teal": "#0d9488",
        "warning-amber": "#d97706",
        "critical-rose": "#e11d48",
        error: {
          DEFAULT: "#ba1a1a",
          container: "#ffdad6",
        },
        "on-error": "#ffffff",
        "on-error-container": "#93000a",
        "on-surface": "#191c1e",
        "on-surface-variant": "#464555",
        "on-background": "#191c1e",
        "text-main": "#0f172a",
        "text-muted": "#64748b",
        "inverse-surface": "#2d3133",
        "inverse-on-surface": "#eff1f3",
        outline: "#777587",
        "outline-variant": "#c7c4d8",
        "border-slate": "#e2e8f0",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        "display-lg": [
          "48px",
          { lineHeight: "56px", fontWeight: "700", letterSpacing: "-0.02em" },
        ],
        "headline-lg": [
          "32px",
          { lineHeight: "40px", fontWeight: "600", letterSpacing: "-0.01em" },
        ],
        "headline-lg-mobile": [
          "24px",
          { lineHeight: "32px", fontWeight: "600" },
        ],
        "headline-md": ["24px", { lineHeight: "32px", fontWeight: "600" }],
        "body-lg": ["18px", { lineHeight: "28px", fontWeight: "400" }],
        "body-md": ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "body-sm": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "data-mono": [
          "14px",
          {
            lineHeight: "20px",
            fontWeight: "500",
            letterSpacing: "0.02em",
          },
        ],
        "label-caps": [
          "12px",
          {
            lineHeight: "16px",
            fontWeight: "600",
            letterSpacing: "0.05em",
          },
        ],
      },
      borderRadius: {
        sm: "0.25rem",
        DEFAULT: "0.5rem",
        md: "0.75rem",
        lg: "1rem",
        xl: "1.5rem",
        full: "9999px",
      },
      spacing: {
        gutter: "24px",
        "stack-sm": "8px",
        "stack-md": "16px",
        "stack-lg": "32px",
        "container-max": "1440px",
        "margin-desktop": "40px",
        "margin-mobile": "16px",
      },
      maxWidth: {
        "container-max": "1440px",
      },
      boxShadow: {
        card: "0px 4px 6px -1px rgba(0,0,0,0.05)",
        popover: "0px 10px 15px -3px rgba(0,0,0,0.1)",
        "metric-active":
          "inset 0 2px 0 0 #4d44e3, 0 4px 6px -1px rgba(0,0,0,0.05)",
      },
    },
  },
  plugins: [],
};

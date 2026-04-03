/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f7f3ea",
          100: "#ece3d0",
          800: "#231f1a",
          900: "#181511",
        },
        ember: "#ff8552",
        aurora: "#6dd3ce",
        ocean: "#276fbf",
      },
      boxShadow: {
        float: "0 24px 80px rgba(20, 19, 16, 0.12)",
        glow: "0 0 0 1px rgba(255,255,255,0.08), 0 16px 48px rgba(21, 27, 38, 0.28)",
      },
      fontFamily: {
        display: ["'Avenir Next'", "'Trebuchet MS'", "sans-serif"],
        body: ["'IBM Plex Sans'", "'Segoe UI'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "'SFMono-Regular'", "monospace"],
      },
    },
  },
  plugins: [],
};

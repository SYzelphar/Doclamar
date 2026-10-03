/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neonPurple: "#8b5cf6",
        neonGreen: "#22c55e",
        neonBlue: "#3b82f6",
        darkBg: "#1f1f2e",
        darkGray: "#2a2a3d",
      },
      boxShadow: {
        neon: "0 0 5px #8b5cf6, 0 0 10px #8b5cf6, 0 0 20px #8b5cf6",
      },
    },
  },
  plugins: [],
};

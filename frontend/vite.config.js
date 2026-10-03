import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Content Security Policy for the packaged app. Only added to production builds:
// Vite's dev server relies on inline scripts for hot reload.
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self' data:",
  "img-src 'self' data:",
  "connect-src http://127.0.0.1:*",
].join('; ')

const injectCsp = {
  name: 'inject-csp',
  apply: 'build',
  transformIndexHtml(html) {
    return html.replace('<head>', `<head>\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`)
  },
}

export default defineConfig({
  base: './', // relative asset paths so Electron can load dist/index.html from disk
  plugins: [react(), injectCsp],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000, // KaTeX + markdown; loaded from disk, not the network
  },
})

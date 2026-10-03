import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react'; // or vue() if using Vue

export default defineConfig({
  base: './', // important for Electron production build
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});

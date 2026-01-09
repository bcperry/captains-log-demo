import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/ready': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/transcribe': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/transcriptions': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})

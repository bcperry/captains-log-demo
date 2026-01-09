/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/__tests__/',
        'src/vite-env.d.ts',
        'src/main.tsx',
        'src/App.tsx',
        // MSAL auth module is difficult to unit test - uses browser APIs and Azure SDK
        'src/auth/AuthProvider.tsx',
        'src/auth/useAuth.ts',
        'src/auth/msalConfig.ts',
        'src/auth/index.ts',
        // Components that depend heavily on MSAL context
        'src/components/Header.tsx',
        'src/components/LoginButton.tsx',
        'src/components/UserInfo.tsx',
        'src/components/ProtectedRoute.tsx',
        'src/components/Layout.tsx',
        'src/components/index.ts',
        // Index files are re-exports only
        'src/hooks/index.ts',
        'src/types/index.ts',
        'src/types/api.ts',
        'src/services/index.ts',
      ],
      thresholds: {
        lines: 70,
        functions: 65,
        branches: 50,
        statements: 70,
      },
    },
  },
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

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build goes to ../web, which server/api.py serves. In dev, /predict is proxied to the API.
export default defineConfig({
  plugins: [react()],
  base: "./", // relative paths so it works under /<project>/ on Pages
  build: { outDir: "../web", emptyOutDir: true },
  server: { proxy: { "/predict": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000" } },
});

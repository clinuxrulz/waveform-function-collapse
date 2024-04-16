import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  publicDir: "../",
  server: {
    fs: {
      strict: false,
    },
  },
  plugins: [
    wasm(),
    topLevelAwait(),
  ],
});

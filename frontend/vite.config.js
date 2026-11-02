import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.VITE_API_PROXY_TARGET;
  const useProxy = !env.VITE_API_BASE_URL && Boolean(proxyTarget);

  return {
    plugins: [react()],
    server: {
      port: 5173,
      ...(useProxy ? {
        proxy: {
          "/api": {
            target: proxyTarget,
            changeOrigin: true,
          },
        },
      } : {}),
    },
  };
});

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",         // self-contained server for the Docker image
  devIndicators: false,         // the dev badge overlaps bottom-left controls on phones
  cacheComponents: true,
  partialPrefetching: true,
  // Baseline hardening for every page. No CSP yet: Next's inline bootstrap
  // scripts need nonces, which would make every page dynamic (docs/SECURITY.md).
  async headers() {
    return [{
      source: "/:path*",
      headers: [
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
      ],
    }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        // When using Colab + Ngrok, paste your Ngrok URL here!
        destination: "https://joye-tetracid-trevor.ngrok-free.dev/api/:path*",
      },
    ];
  },
};

export default nextConfig;

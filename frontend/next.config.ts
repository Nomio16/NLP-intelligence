import type { NextConfig } from "next";

// Set NEXT_PUBLIC_API_URL in .env.local to point to your backend.
// Example for Colab:  NEXT_PUBLIC_API_URL=https://your-url.ngrok-free.dev
// Example for local:  NEXT_PUBLIC_API_URL=http://localhost:8000
// If not set, defaults to localhost:8000
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

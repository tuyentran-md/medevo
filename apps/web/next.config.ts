import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: process.env.NEXT_PUBLIC_MEDEVO_STATIC_REPLAY === "1" ? "export" : undefined,
  trailingSlash: process.env.NEXT_PUBLIC_MEDEVO_STATIC_REPLAY === "1",
};

export default nextConfig;

import type { NextConfig } from "next";

const config: NextConfig = { poweredByHeader: false, outputFileTracingIncludes:{"/api/live/catalog":["./public/inference/catalog.json"]} };
export default config;

import type { NextConfig } from "next";

const config: NextConfig = { poweredByHeader: false, serverExternalPackages:["onnxruntime-web"], outputFileTracingIncludes:{"/api/live/catalog":["./public/inference/catalog.json"],"/api/compute":["./public/inference/**/*","./public/ort/**/*"]} };
export default config;

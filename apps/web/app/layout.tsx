import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GridLens | Forecast foundations",
  description: "Inspect the timing and data boundaries behind GridLens electricity forecasts.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

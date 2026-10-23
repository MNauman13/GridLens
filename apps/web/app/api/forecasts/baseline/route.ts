import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const model = request.nextUrl.searchParams.get("model") ?? "ridge";
  if (!["last_day", "weekly", "ridge"].includes(model)) return NextResponse.json({error:"Unknown baseline"},{status:422});
  const backend = process.env.GRIDLENS_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${backend}/v1/forecasts/baseline?model=${model}`, {cache:"no-store",signal:AbortSignal.timeout(5000)});
    if (!response.ok) return NextResponse.json({error:"Saved forecast unavailable"},{status:response.status});
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({error:"Forecast API unavailable"},{status:503});
  }
}

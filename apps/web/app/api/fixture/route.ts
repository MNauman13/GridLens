import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const issueDate = request.nextUrl.searchParams.get("issue_date") ?? "2024-02-10";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(issueDate)) return NextResponse.json({error:"Invalid issue date"},{status:422});
  const backend = process.env.GRIDLENS_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${backend}/v1/fixture?issue_date=${encodeURIComponent(issueDate)}`, {cache:"no-store",signal:AbortSignal.timeout(5000)});
    if (!response.ok) return NextResponse.json({error:"Fixture validation failed"},{status:response.status});
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({error:"Verification API unavailable"},{status:503});
  }
}

import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const issueDate = request.nextUrl.searchParams.get("issue_date") ?? "2024-02-10";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(issueDate)) return NextResponse.json({error:"Invalid issue date"},{status:422});
  if (!process.env.GRIDLENS_API_URL) {
    const midnight=Date.parse(`${issueDate}T00:00:00Z`);
    if(!Number.isFinite(midnight)||new Date(midnight).toISOString().slice(0,10)!==issueDate)return NextResponse.json({error:"Invalid issue date"},{status:422});
    const at=(offset:number)=>new Date(midnight+offset*86400000).toISOString().replace(".000Z","+00:00");
    return NextResponse.json({fixture_kind:"synthetic_contract_smoke_test",real_demand_data:false,trained_model:false,contract_version:"gb_nd_day_ahead_v1",issued_at:`${issueDate}T09:30:00+00:00`,context_start:at(-7),context_end_exclusive:at(0),target_start:at(1),target_end_exclusive:at(2),input_count:336,gap_intervals:48,output_count:48});
  }
  const backend = process.env.GRIDLENS_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${backend}/v1/fixture?issue_date=${encodeURIComponent(issueDate)}`, {cache:"no-store",signal:AbortSignal.timeout(5000)});
    if (!response.ok) return NextResponse.json({error:"Fixture validation failed"},{status:response.status});
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({error:"Verification API unavailable"},{status:503});
  }
}

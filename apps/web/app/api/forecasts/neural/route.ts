import saved from "../../../../../../src/gridlens/resources/neural_forecasts_v1.json";
import {NextRequest,NextResponse} from "next/server";

export async function GET(request:NextRequest){
  const model=request.nextUrl.searchParams.get("model")??"timexer";
  if(!["transformer","lstm","timexer"].includes(model))return NextResponse.json({error:"Unknown neural model"},{status:422});
  if (!process.env.GRIDLENS_API_URL) return NextResponse.json(saved[model as keyof typeof saved], {headers:{"X-GridLens-Source":"packaged_saved_example"}});
  const backend=process.env.GRIDLENS_API_URL??"http://127.0.0.1:8000";
  try{
    const response=await fetch(`${backend}/v1/forecasts/neural?model=${model}`,{cache:"no-store",signal:AbortSignal.timeout(5000)});
    if(!response.ok)return NextResponse.json({error:"Saved neural forecast unavailable"},{status:response.status});
    return NextResponse.json(await response.json());
  }catch{return NextResponse.json({error:"Forecast API unavailable"},{status:503});}
}

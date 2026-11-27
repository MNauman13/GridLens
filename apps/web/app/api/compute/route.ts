import {NextRequest,NextResponse} from "next/server";
import {serverInference} from "../../../lib/server-inference";
export const runtime="nodejs";
export const maxDuration=30;
export async function POST(request:NextRequest){
  const origin=request.headers.get("origin");
  const protocol=request.headers.get("x-forwarded-proto")?.split(",")[0].trim()??request.nextUrl.protocol.slice(0,-1);
  if(origin&&origin!==`${protocol}://${request.headers.get("host")}`)return NextResponse.json({error:"Cross-origin compute is unavailable"},{status:403});
  if(Number(request.headers.get("content-length")??0)>1024)return NextResponse.json({error:"Request too large"},{status:413});
  try{
    const text=await request.text();if(text.length>1024)return NextResponse.json({error:"Request too large"},{status:413});
    const payload=JSON.parse(text);
    if(!payload||typeof payload!=="object"||Object.keys(payload).sort().join(",")!=="model,target_date"||!["ridge","timexer"].includes(payload.model)||typeof payload.target_date!=="string"||!/^2022-\d{2}-\d{2}$/.test(payload.target_date))return NextResponse.json({error:"Choose an available model and validation date"},{status:422});
    return NextResponse.json(await serverInference(payload.model,payload.target_date),{headers:{"Cache-Control":"no-store"}});
  }catch(error){
    const message=error instanceof Error?error.message:"";
    if(error instanceof SyntaxError)return NextResponse.json({error:"Invalid JSON"},{status:400});
    if(message==="capacity")return NextResponse.json({error:"This worker is busy. Please retry."},{status:429,headers:{"Retry-After":"2"}});
    if(message==="date")return NextResponse.json({error:"Date unavailable"},{status:422});
    console.error("Server inference failed",error);
    return NextResponse.json({error:"Inference unavailable. No saved prediction was substituted."},{status:503});
  }
}

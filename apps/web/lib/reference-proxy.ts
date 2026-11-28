import {NextRequest,NextResponse} from "next/server";

export async function referenceProxy(request:NextRequest,path:string,method:"GET"|"POST"|"DELETE"="GET"){
  const backend=process.env.GRIDLENS_API_URL;
  if(!backend)return NextResponse.json({error:"Python reference service is not connected. Browser inference remains available."},{status:503});
  if(method!=="GET"){
    const origin=request.headers.get("origin");
    const protocol=request.headers.get("x-forwarded-proto")?.split(",")[0].trim()??request.nextUrl.protocol.slice(0,-1);
    const expectedOrigin=`${protocol}://${request.headers.get("host")}`;
    if(origin&&origin!==expectedOrigin)return NextResponse.json({error:"Cross-origin compute requests are not allowed"},{status:403});
  }
  try{
    const response=await fetch(`${backend}${path}`,{method,headers:{"Content-Type":"application/json",cookie:request.headers.get("cookie")??"",...(request.headers.has("idempotency-key")?{"Idempotency-Key":request.headers.get("idempotency-key")!}:{})},body:method==="POST"?await request.text():undefined,cache:"no-store",signal:AbortSignal.timeout(10000)});
    const headers=new Headers({"Content-Type":"application/json","Cache-Control":"no-store"});
    const retry=response.headers.get("retry-after");
    if(retry)headers.set("Retry-After",retry);
    const cookie=response.headers.get("set-cookie");
    if(cookie)headers.set("set-cookie",cookie);
    return new NextResponse(response.body,{status:response.status,headers});
  }catch{return NextResponse.json({error:"Python reference service unavailable. No saved forecast was substituted."},{status:503});}
}

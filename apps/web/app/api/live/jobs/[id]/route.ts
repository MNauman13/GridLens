import {NextRequest,NextResponse} from "next/server";
import {referenceProxy} from "../../../../../lib/reference-proxy";

export async function GET(request:NextRequest,{params}:{params:Promise<{id:string}>}){
  const {id}=await params;
  if(!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id))return NextResponse.json({error:"Invalid job identity"},{status:422});
  return referenceProxy(request,`/v1/jobs/${id}`);
}

import {readFile} from "node:fs/promises";
import path from "node:path";
import {NextResponse} from "next/server";

export const dynamic="force-dynamic";
export async function GET(){
  try{
    const catalog=JSON.parse(await readFile(path.join(process.cwd(),"public/inference/catalog.json"),"utf-8"));
    let available=false;
    let runtimes:Record<string,string[]>={};
    if(process.env.GRIDLENS_API_URL){
      try{
        const reply=await fetch(`${process.env.GRIDLENS_API_URL}/v1/live/catalog`,{cache:"no-store",signal:AbortSignal.timeout(3000)});
        if(reply.ok){const server=await reply.json();available=server.catalog_id===catalog.catalog_id;if(available)runtimes=server.server_runtimes??{};}
      }catch{/* A disconnected reference service never replaces a browser result. */}
    }
    return NextResponse.json({...catalog,server_available:available,server_runtimes:runtimes},{headers:{"Cache-Control":"no-store"}});
  }catch{return NextResponse.json({error:"Verified inference catalog is unavailable"},{status:503});}
}

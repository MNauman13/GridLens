import {NextRequest} from "next/server";
import {referenceProxy} from "../../../../lib/reference-proxy";

export async function POST(request:NextRequest){return referenceProxy(request,"/v1/forecast-jobs","POST");}

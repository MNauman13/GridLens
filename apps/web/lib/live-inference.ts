export type DemoCase = {
  inputs: {issued_at:string;context_start:string;context_end_exclusive:string;target_start:string;target_end_exclusive:string;feature_snapshot_id:string;demand_mw:number[]};
  history:number[][]; target_calendar:number[][]; observed_mw:number[];
};
export type DemoModel = {kind:"ridge"|"timexer";model_id:string;asset:string;sha256:string;browser_runtime:string;reference_runtime:string;scaling?:{means:number[];scales:number[]}};
export type DemoCatalog = {catalog_id:string;dataset_manifest_id:string;evidence_mode:string;evaluation_role:string;browser_parity_max_mw:number;models:Record<"ridge"|"timexer",DemoModel>;cases:{target_date:string;issue_date:string;asset:string;sha256:string;feature_snapshot_id:string}[];server_available:boolean;server_message?:string};
export type DemoResult = {execution_id:string;prediction_mw:number[];observed_mw:number[];inputs:DemoCase["inputs"];model_id:string;runtime:string;execution_location:"browser"|"python_server";pipeline_ms:number;predict_ms:number;model_init_ms:number;model_cache_hit:boolean;result_cache_hit:false;queue_ms?:number};

type RidgeParameters = {x_mean:number[];x_scale:number[];y_mean:number[];weights:number[][]};
type OrtModule = typeof import("onnxruntime-web/wasm");
const loaded = new Map<string, {ridge?:RidgeParameters;session?:import("onnxruntime-web").InferenceSession}>();

async function verifiedFile(asset:string, expected:string):Promise<ArrayBuffer>{
  if(!/^(models|cases)\/[a-zA-Z0-9_.-]+$/.test(asset))throw Error("Invalid artifact path");
  const response=await fetch(`/inference/${asset}?sha256=${expected}`,{cache:"force-cache",signal:AbortSignal.timeout(30000)});
  if(!response.ok)throw Error("The verified model or input file could not be downloaded. Please retry.");
  const buffer=await response.arrayBuffer();
  const digest=await crypto.subtle.digest("SHA-256",buffer);
  const actual=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,"0")).join("");
  if(actual!==expected)throw Error("Artifact checksum failed. No inference was run on the changed file.");
  return buffer;
}

function checkCase(value:DemoCase):void{
  if(value.history.length!==336||value.history.some(row=>row.length!==15||row.some(v=>!Number.isFinite(v)))||value.target_calendar.length!==48||value.target_calendar.some(row=>row.length!==6||row.some(v=>!Number.isFinite(v)))||value.observed_mw.length!==48||value.observed_mw.some(v=>!Number.isFinite(v)))throw Error("Incomplete or nonfinite input window");
}

export async function runBrowser(catalog:DemoCatalog,kind:"ridge"|"timexer",day:string,onStage:(stage:string)=>void):Promise<DemoResult>{
  const start=performance.now();
  const entry=catalog.cases.find(row=>row.target_date===day);
  if(!entry||!day.startsWith("2022-"))throw Error("Select an approved validation date. Locked-test dates are unavailable.");
  onStage("Verifying the historical input window");
  const value=JSON.parse(new TextDecoder().decode(await verifiedFile(entry.asset,entry.sha256))) as DemoCase;
  checkCase(value);
  if(value.inputs.feature_snapshot_id!==entry.feature_snapshot_id)throw Error("Feature snapshot identity mismatch");
  const model=catalog.models[kind];
  const key=model.model_id+":"+model.browser_runtime;
  const cached=loaded.has(key);
  const initStart=performance.now();
  let runtime:OrtModule|undefined;
  if(!cached){
    onStage("Loading verified model weights");
    const bytes=await verifiedFile(model.asset,model.sha256);
    if(kind==="ridge"){
      const ridge=JSON.parse(new TextDecoder().decode(bytes)) as RidgeParameters;
      if(ridge.x_mean.length!==100||ridge.x_scale.length!==100||ridge.y_mean.length!==48||ridge.weights.length!==100||ridge.weights.some(row=>row.length!==48))throw Error("Invalid ridge model shape");
      loaded.set(key,{ridge});
    }else{
      runtime=await import("onnxruntime-web/wasm");
      runtime.env.wasm.numThreads=1;
      runtime.env.wasm.wasmPaths="/ort/";
      const session=await runtime.InferenceSession.create(bytes,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
      loaded.set(key,{session});
    }
  }
  const initMs=cached?0:performance.now()-initStart;
  onStage("Running fresh inference");
  let prediction:number[];
  let predictionMs:number;
  if(kind==="ridge"){
    const ridge=loaded.get(key)?.ridge;
    if(!ridge)throw Error("Ridge model unavailable");
    const demand=value.inputs.demand_mw;
    const features=[...demand.slice(-48),...demand.slice(48,96),...value.target_calendar[0].slice(2)];
    const tick=performance.now();
    prediction=ridge.y_mean.map((mean,slot)=>features.reduce((sum,v,i)=>sum+((v-ridge.x_mean[i])/ridge.x_scale[i])*ridge.weights[i][slot],mean));
    predictionMs=performance.now()-tick;
  }else{
    runtime??=await import("onnxruntime-web/wasm");
    const session=loaded.get(key)?.session;
    const scaling=model.scaling;
    if(!session||!scaling)throw Error("Transformer model unavailable");
    const history=Float32Array.from(value.history.flatMap(row=>row.map((v,i)=>i<3?(v-scaling.means[i])/scaling.scales[i]:v)));
    const calendar=Float32Array.from(value.target_calendar.flat());
    const tick=performance.now();
    const output=await session.run({history:new runtime.Tensor("float32",history,[1,336,15]),calendar:new runtime.Tensor("float32",calendar,[1,48,6])});
    predictionMs=performance.now()-tick;
    prediction=Array.from(output.prediction.data as Float32Array,v=>v*scaling.scales[0]+scaling.means[0]);
  }
  if(prediction.length!==48||prediction.some(v=>!Number.isFinite(v)))throw Error("Model output failed the 48-slot forecast contract");
  return {execution_id:crypto.randomUUID(),prediction_mw:prediction,observed_mw:value.observed_mw,inputs:value.inputs,model_id:model.model_id,runtime:model.browser_runtime,execution_location:"browser",pipeline_ms:performance.now()-start,predict_ms:predictionMs,model_init_ms:initMs,model_cache_hit:cached,result_cache_hit:false};
}

export async function runServer(kind:"ridge"|"timexer",day:string,onStage:(stage:string)=>void):Promise<DemoResult>{
  const started=performance.now();
  const signal=AbortSignal.timeout(30000);
  const response=await fetch("/api/live/jobs",{method:"POST",signal,headers:{"Content-Type":"application/json"},body:JSON.stringify({model:kind,target_date:day})});
  const accepted=await response.json();
  if(!response.ok)throw Error(accepted.detail??accepted.error??"Reference service unavailable");
  for(let attempt=0;attempt<150;attempt++){
    const reply=await fetch(`/api/live/jobs/${accepted.job_id}`,{cache:"no-store",signal});
    const job=await reply.json();
    if(!reply.ok)throw Error(job.detail??job.error??"Could not retrieve inference status");
    if(job.state==="failed")throw Error(job.error??"Reference inference failed");
    if(job.state==="succeeded"){
      const r=job.result;
      return {execution_id:r.execution_id,prediction_mw:r.point.prediction_mw,observed_mw:r.observed_mw,inputs:r.inputs,model_id:r.model_id,runtime:r.runtime,execution_location:"python_server",pipeline_ms:performance.now()-started,predict_ms:r.predict_ms,model_init_ms:0,model_cache_hit:true,result_cache_hit:false,queue_ms:r.queue_ms};
    }
    onStage(job.state==="queued"?"Waiting for the reference worker":"Running reference model inference");
    await new Promise(resolve=>setTimeout(resolve,200));
  }
  throw Error("The reference worker did not finish within 30 seconds");
}

/** Synchronous serverless demo adapter. No durable/distributed queue claims. */
import {createHash,randomUUID} from "node:crypto";
import {readFile} from "node:fs/promises";
import path from "node:path";
import type {DemoCase,DemoCatalog,DemoModel,DemoResult} from "./live-inference";
type Ort=typeof import("onnxruntime-web/wasm");
type Ridge={x_mean:number[];x_scale:number[];y_mean:number[];weights:number[][]};
let active=0;
const models=new Map<string,Promise<{session?:import("onnxruntime-web").InferenceSession;ridge?:Ridge}>>();
async function asset(name:string,digest:string){
  if(!/^(models|cases)\/[a-zA-Z0-9_.-]+$/.test(name))throw Error("Invalid artifact path");
  const bytes=await readFile(path.join(process.cwd(),"public/inference",name));
  if(createHash("sha256").update(bytes).digest("hex")!==digest)throw Error("Artifact integrity failure");
  return bytes;
}
async function load(model:DemoModel){
  const bytes=await asset(model.asset,model.sha256);
  if(model.kind==="ridge")return {ridge:JSON.parse(bytes.toString()) as Ridge};
  const ort:Ort=await import("onnxruntime-web/wasm");
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=path.join(process.cwd(),"public/ort/");
  const session=await ort.InferenceSession.create(bytes,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
  return {session};
}
export async function serverInference(kind:"ridge"|"timexer",day:string):Promise<DemoResult>{
  if(active>=1)throw Error("capacity");
  active++;
  try{
    const started=performance.now();
    const catalog=JSON.parse(await readFile(path.join(process.cwd(),"public/inference/catalog.json"),"utf8")) as DemoCatalog;
    const entry=catalog.cases.find(row=>row.target_date===day);
    if(!entry||!day.startsWith("2022-"))throw Error("date");
    const value=JSON.parse((await asset(entry.asset,entry.sha256)).toString()) as DemoCase;
    if(value.inputs.feature_snapshot_id!==entry.feature_snapshot_id)throw Error("Artifact identity failure");
    const model=catalog.models[kind];const key=model.model_id+":"+model.sha256;
    const hit=models.has(key);const init=performance.now();
    if(!models.has(key))models.set(key,load(model).catch(error=>{models.delete(key);throw error;}));
    const loaded=await models.get(key)!;const initMs=hit?0:performance.now()-init;
    const tick=performance.now();let prediction:number[];
    if(kind==="ridge"){
      const r=loaded.ridge!;const demand=value.inputs.demand_mw;
      const features=[...demand.slice(-48),...demand.slice(48,96),...value.target_calendar[0].slice(2)];
      prediction=r.y_mean.map((mean,slot)=>features.reduce((sum,v,i)=>sum+((v-r.x_mean[i])/r.x_scale[i])*r.weights[i][slot],mean));
    }else{
      const ort:Ort=await import("onnxruntime-web/wasm");const s=model.scaling!;
      const h=Float32Array.from(value.history.flatMap(row=>row.map((v,i)=>i<3?(v-s.means[i])/s.scales[i]:v)));
      const c=Float32Array.from(value.target_calendar.flat());
      const output=await loaded.session!.run({history:new ort.Tensor("float32",h,[1,336,15]),calendar:new ort.Tensor("float32",c,[1,48,6])});
      prediction=Array.from(output.prediction.data as Float32Array,v=>v*s.scales[0]+s.means[0]);
    }
    const predictMs=performance.now()-tick;
    if(prediction.length!==48||prediction.some(v=>!Number.isFinite(v)))throw Error("Output contract failure");
    return {execution_id:randomUUID(),prediction_mw:prediction,observed_mw:value.observed_mw,inputs:value.inputs,model_id:model.model_id,runtime:kind==="ridge"?"node_js_ridge":"node_onnx_wasm_fp32",execution_location:"web_server",pipeline_ms:performance.now()-started,predict_ms:predictMs,model_init_ms:initMs,model_cache_hit:hit,result_cache_hit:false};
  }finally{active--;}
}

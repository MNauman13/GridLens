"use client";
import { useState } from "react";
import evidence from "../data/i2-cache.json";
const labels: Record<string,string> = {reference:"Full-prefix reference",sdpa_only:"SDPA only",cross_kv:"Cross-attention cache",full_kv:"Full KV cache",full_kv_sdpa:"Full KV cache + SDPA"};
export default function CacheBenchmarks(){
  const [index,setIndex]=useState(0);
  const condition=evidence.conditions[index];
  return <section className="panel" id="cache" aria-labelledby="cache-heading">
    <p className="eyebrow">I2 / INCREMENTAL DECODING</p><h2 id="cache-heading">Stop recomputing earlier tokens.</h2>
    <p>The numerical Transformer generates 48 half-hours one at a time. A request-local cache reuses encoder keys and values, then appends each decoder key and value once. Predictions still use the model’s own previous output.</p>
    <label htmlFor="cache-condition">Measurement scope </label><select id="cache-condition" value={index} onChange={event=>setIndex(Number(event.target.value))}>{evidence.conditions.map((row,i)=><option key={i} value={i}>{row.scope==="bundle_predict"?"Complete prediction":"Prepared tensors"} · batch {row.batch_size}</option>)}</select>
    <p>{condition.scope==="bundle_predict"?"Includes preprocessing, inverse scaling and contract validation.":"Prepared tensors exclude preprocessing and output validation."} HTTP and queueing are excluded. Batch latency is per batch, not per forecast.</p>
    <div style={{overflowX:"auto"}}><table style={{width:"100%",textAlign:"left"}}><caption>{condition.calls_per_runtime.toLocaleString()} measured calls per variant, warmups excluded</caption><thead><tr><th scope="col">Execution</th><th scope="col">p50 ms</th><th scope="col">p95 ms</th><th scope="col">p50 ratio</th></tr></thead><tbody>{Object.entries(condition.runtimes).map(([name,row])=><tr key={name}><th scope="row" style={{padding:"12px 8px"}}>{labels[name]}</th><td>{row.p50_ms.toFixed(3)}</td><td>{row.p95_ms.toFixed(3)}</td><td>{(condition.runtimes.reference.p50_ms/row.p50_ms).toFixed(2)}×</td></tr>)}</tbody></table></div>
    <p>All variants use identical weights and matched dates. Maximum absolute drift is {evidence.max_abs_drift_mw.toFixed(3)} MW across {evidence.origins} benchmark dates and {evidence.heldout_origins} disjoint validation dates, checked at batch sizes one and four. The 2024 test set remains excluded from this experiment.</p>
    <p>A batch-one full cache allocates {(evidence.cache_bytes.full_kv/1024).toFixed(0)} KiB of key/value tensors. This excludes weights, activations, allocator overhead and Python memory. Caches are discarded after completion, cancellation or an early close.</p>
    <details><summary>Latency across the 48 generated half-hours</summary><svg viewBox="0 0 600 230" role="img" aria-label="Per-slot median generation latency for the reference and full cache" style={{width:"100%",height:"auto"}}>{["reference","full_kv"].map((name,index)=>{
      const values=evidence.step_p50_ms[name as "reference"|"full_kv"];
      const maximum=Math.max(...evidence.step_p50_ms.reference,...evidence.step_p50_ms.full_kv);
      return <polyline key={name} fill="none" stroke={index?"#a65124":"#28644b"} strokeWidth="3" points={values.map((value,slot)=>`${30+slot/47*540},${190-value/maximum*160}`).join(" ")}/>;
    })}<text x="30" y="220">Slot 1</text><text x="510" y="220">Slot 48</text></svg><p>Green is the reference. Orange is full caching. Timings are instrumented separately from the main benchmark.</p></details>
    <p>{evidence.cpu}, PyTorch {evidence.torch}, two CPU threads. {evidence.raw_samples.toLocaleString()} retained timings. Runtime order was randomised within matched case blocks. These Python measurements do not describe browser WASM performance.</p>
    <p style={{overflowWrap:"anywhere"}}>Report {evidence.report_id}</p><p><a href="https://github.com/MNauman13/GridLens/tree/feat/i2-kv-cache/reports/i2-cache">Inspect raw timings, parity, profiler and memory evidence</a></p>
  </section>;
}

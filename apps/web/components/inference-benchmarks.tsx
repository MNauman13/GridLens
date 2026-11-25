"use client";

import {useState} from "react";
import evidence from "../data/i2-runtime.json";

export default function InferenceBenchmarks(){
  const [scope,setScope]=useState<"bundle_predict"|"prepared_engine">("bundle_predict");
  return <section className="panel" id="inference" aria-labelledby="inference-heading">
    <p className="eyebrow">I2 / MEASURED INFERENCE</p><h2 id="inference-heading">Same weights. Different execution.</h2>
    <p>Compare the manual attention reference with checked PyTorch SDPA. Both retain input, mask and score-overflow checks. The original training implementation and model weights are unchanged.</p>
    <div className="step-controls"><button aria-pressed={scope==="bundle_predict"} onClick={()=>setScope("bundle_predict")}>Bundle predict</button><button aria-pressed={scope==="prepared_engine"} onClick={()=>setScope("prepared_engine")}>Prepared tensors</button></div>
    <p data-testid="i2-scope">{scope==="bundle_predict"?"Includes preprocessing, model forward, inverse scaling and output contract validation.":"Model forward on prepared tensors; excludes preprocessing, inverse scaling and output contract validation."} Neither scope includes HTTP, queueing or browser rendering.</p>
    <div style={{overflowX:"auto",maxWidth:"100%"}}><table style={{width:"100%",textAlign:"left",borderCollapse:"collapse"}}><caption>{evidence.calls_per_condition.toLocaleString()} measured calls per runtime/model/scope; {evidence.warmup_calls_excluded} warmups excluded.</caption><thead><tr><th scope="col">Model / runtime</th><th scope="col">p50 ms</th><th scope="col">p95 ms</th><th scope="col">Sequential forecasts/s</th></tr></thead><tbody>{evidence.models.flatMap(model=>["manual","sdpa"].map(name=>{
      const row=model.comparisons[scope].runtimes[name as "manual"|"sdpa"];
      return <tr key={`${model.kind}-${name}`}><th scope="row" style={{padding:"12px 8px"}}>{model.label} · {name==="manual"?"Manual":"Checked SDPA"}</th><td>{row.p50_ms.toFixed(3)}</td><td>{row.p95_ms.toFixed(3)}</td><td>{row.sequential_capacity_forecasts_per_second.toFixed(1)}</td></tr>;
    }))}</tbody></table></div>
    {evidence.models.map(model=>{
      const comparison=model.comparisons[scope];
      const change=(1-1/comparison.p50_speedup)*100;
      const interval=comparison.mean_speedup_95pct_block_bootstrap;
      return <p key={model.kind} data-testid={`i2-${model.kind}-change`}><strong>{model.label}:</strong> SDPA median latency is {Math.abs(change).toFixed(1)}% {change>=0?"lower":"higher"} ({comparison.p50_speedup.toFixed(2)}× p50 ratio). The descriptive 95% interval for mean speedup is {interval[0].toFixed(2)}–{interval[1].toFixed(2)}×.{interval[0]<=1&&interval[1]>=1&&" The interval includes 1×; a speed improvement is not established."}</p>;
    })}
    <p>Each comparison alternates runtime order using seed 17 over the same 32 validation dates. Intervals resample five paired repeat blocks, with 2,000 bootstrap draws. They describe this timing experiment, not statistical evidence of improved forecast accuracy.</p>
    <p>The earlier full-score preflight candidate increased median bundle latency by {((1/evidence.initial_trial.bundle_p50_speedups.transformer-1)*100).toFixed(1)}% for the numerical Transformer and {((1/evidence.initial_trial.bundle_p50_speedups.timexer-1)*100).toFixed(1)}% for TimeXer. That unsuccessful trial is retained. The revised candidate uses a conservative range certificate for larger score grids, with the exact overflow check as a fallback.</p>
    <h3>Prediction agreement and memory</h3>
    {evidence.models.map(model=><p key={model.kind}><strong>{model.label}:</strong> maximum prediction drift {model.max_abs_drift_mw.toFixed(6)} MW; validation-corpus RMSE {model.rmse_mw.manual.toFixed(3)} MW manual and {model.rmse_mw.sdpa.toFixed(3)} MW SDPA. Median process peak RSS: {(model.median_process_peak_rss_bytes.manual/1048576).toFixed(1)} MiB manual, {(model.median_process_peak_rss_bytes.sdpa/1048576).toFixed(1)} MiB SDPA.</p>)}
    <p>Memory is measured in {evidence.memory_trials_per_condition} fresh Python processes per condition and includes imports, the input corpus and inference. The retained score-overflow guard can still allocate an attention score matrix. These numbers do not isolate tensor memory or establish a memory saving.</p>
    <p><strong>Measured host:</strong> {evidence.cpu}, CPU only, {evidence.torch_threads} Torch threads, PyTorch {evidence.torch}, batch size 1, no prediction cache. {evidence.raw_timing_count.toLocaleString()} raw timing observations. The 2024 test set was not evaluated.</p>
    <p>These are recorded Python CPU benchmarks. The public live demo runs a separate browser WASM runtime; its timings are not directly comparable.</p>
    <details><summary>Benchmark provenance and raw samples</summary><p style={{overflowWrap:"anywhere"}}>Report {evidence.report_id}<br/>Runtime {evidence.runtime_id}</p><p><a href="https://github.com/MNauman13/GridLens/tree/feat/i2-sdpa-inference/reports/i2">Read all predictions, timing samples, profiler operators and memory trials</a></p></details>
  </section>;
}

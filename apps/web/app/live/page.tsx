"use client";

import Link from "next/link";
import {useEffect,useRef,useState} from "react";
import {DemoCatalog,DemoResult,runBrowser,runServer} from "../../lib/live-inference";

export default function LiveForecast(){
  const [catalog,setCatalog]=useState<DemoCatalog|null>(null);
  const [model,setModel]=useState<"ridge"|"timexer">("timexer");
  const [day,setDay]=useState("");
  const [mode,setMode]=useState("browser");
  const [serverRuntime,setServerRuntime]=useState<"manual"|"sdpa">("manual");
  const [stage,setStage]=useState("Loading available models and dates");
  const [running,setRunning]=useState(false);
  const [error,setError]=useState("");
  const [result,setResult]=useState<DemoResult|null>(null);
  const [slot,setSlot]=useState(0);
  const generation=useRef(0);
  useEffect(()=>{const controller=new AbortController();fetch("/api/live/catalog",{signal:controller.signal}).then(async response=>{if(!response.ok)throw Error("The verified inference artifacts are unavailable.");return response.json() as Promise<DemoCatalog>;}).then(value=>{if(!controller.signal.aborted){setCatalog(value);setDay(value.cases[0].target_date);setStage("Ready to run a real forecast");}}).catch(e=>{if(!controller.signal.aborted)setError(e.message);});const ref=generation;return()=>{controller.abort();ref.current++;};},[]);
  function reset(){generation.current++;setResult(null);setError("");setStage("Ready to run a real forecast");}
  async function run(){
    if(!catalog||running)return;
    const request=++generation.current;
    setRunning(true);setError("");setResult(null);
    try{
      const update=(value:string)=>{if(request===generation.current)setStage(value);};
      const value=mode==="browser"?await runBrowser(catalog,model,day,update):await runServer(model,day,update,model==="timexer"?serverRuntime:"manual");
      if(request===generation.current){setResult(value);setStage("Fresh inference complete · 48 half-hour predictions");}
    }catch(e){if(request===generation.current){setError(e instanceof Error?(e.name==="TimeoutError"?"Inference request timed out after 30 seconds. Please retry.":e.message):"Inference failed");setStage("Inference could not complete");}}
    finally{if(request===generation.current)setRunning(false);}
  }
  const bounds=result?[Math.min(...result.prediction_mw,...result.observed_mw)-700,Math.max(...result.prediction_mw,...result.observed_mw)+700]:[0,1];
  const points=(values:number[])=>values.map((v,i)=>`${36+i*648/47},${238-(v-bounds[0])/(bounds[1]-bounds[0])*206}`).join(" ");
  const mae=result?result.prediction_mw.reduce((sum,v,i)=>sum+Math.abs(v-result.observed_mw[i]),0)/48:0;
  return <main className="live-workspace">
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><nav><Link href="/experiments#inference">Inference benchmarks</Link><Link href="/forecasts/neural">Saved examples</Link><Link href="/learn/transformer">Inside the transformer</Link></nav></header>
    <section className="intro"><p className="eyebrow">RUN THE MODEL</p><h1>Forecast the grid.<br/>See the computation.</h1><p className="lede">Choose a historical day and run a trained electricity-demand model. Every click computes a new 48-slot forecast. No account or API key.</p><p className="notice">Real model inference · 2022 validation replay · MW</p></section>
    <section className="panel live-controls" aria-label="Inference controls">
      <div className="live-form">
        <label htmlFor="live-model">Model<select id="live-model" value={model} disabled={running} onChange={e=>{reset();setModel(e.target.value as "ridge"|"timexer");}}><option value="timexer">TimeXer transformer</option><option value="ridge">Ridge baseline</option></select></label>
        <label htmlFor="live-date">Target date<select id="live-date" value={day} disabled={running||!catalog} onChange={e=>{reset();setDay(e.target.value);}}>{catalog?.cases.map(row=><option key={row.target_date} value={row.target_date}>{row.target_date}</option>)}</select></label>
        <label htmlFor="live-mode">Execution<select id="live-mode" value={mode} disabled={running} onChange={e=>{reset();setMode(e.target.value);}}><option value="browser">This browser</option><option value="server" disabled={!catalog?.server_available}>Python reference service{!catalog?.server_available?" (not connected)":""}</option></select></label>
        <button onClick={run} disabled={!catalog||running}>{running?"Computing…":"Run fresh inference"}</button>
      </div>
      {mode==="server"&&model==="timexer"&&<label htmlFor="server-runtime">Python attention runtime<select id="server-runtime" disabled={running} value={serverRuntime} onChange={e=>{reset();setServerRuntime(e.target.value as "manual"|"sdpa");}}><option value="manual">Manual reference</option><option value="sdpa" disabled={!catalog?.server_runtimes?.timexer?.includes("sdpa")}>Checked PyTorch SDPA{!catalog?.server_runtimes?.timexer?.includes("sdpa")?" (not enabled)":""}</option></select></label>}
      <p role="status" aria-live="polite" className="live-status">{stage}</p>
      {error&&<p role="alert" className="error">{error}</p>}
      <p className="hint">Browser mode runs the same trained TimeXer weights through WebAssembly, or the fitted ridge coefficients in JavaScript. It does not replay saved predictions. Python mode uses a bounded server worker when connected.</p>
    </section>
    {result&&<>
      <section className="panel" aria-label="Forecast result">
        <div className="section-heading"><h2>{day} demand forecast</h2><span className="notice">Fresh {result.execution_location==="browser"?"browser":"server"} inference</span></div>
        <div className="live-metrics"><div><span>Model execution</span><strong data-testid="live-predict-ms">{result.predict_ms.toFixed(2)} <small>ms</small></strong></div><div><span>Click-to-result computation</span><strong>{result.pipeline_ms.toFixed(0)} <small>ms</small></strong></div><div><span>This day’s absolute error</span><strong>{mae.toFixed(0)} <small>MW</small></strong></div></div>
        <p className="hint">Green: forecast. Dashed gray: observed demand. The error shown is for this selected validation day, not an unseen-test score.</p>
        <svg viewBox="0 0 720 280" role="img" aria-label="48 freshly computed forecasts compared with observed electricity demand" className="forecast-chart"><line x1="36" x2="684" y1="239" y2="239" stroke="#dce3dc"/><polyline points={points(result.observed_mw)} fill="none" stroke="#82918a" strokeWidth="2" strokeDasharray="6 5"/><polyline points={points(result.prediction_mw)} fill="none" stroke="#23624e" strokeWidth="3"/>{[0,12,24,36,47].map(i=><text key={i} x={36+i*648/47} y="265" textAnchor="middle" fontSize="13" fill="#53656a">{i===47?"23:30":`${String(i/2).padStart(2,"0")}:00`}</text>)}</svg>
        <label htmlFor="live-slot">Inspect half-hour slot {slot+1}<input id="live-slot" type="range" min="0" max="47" value={slot} onChange={e=>setSlot(Number(e.target.value))}/></label><p data-testid="live-slot-value">Forecast {result.prediction_mw[slot].toFixed(2)} MW · Observed {result.observed_mw[slot].toFixed(2)} MW</p>
        <details><summary>All 48 forecast values</summary><div className="live-table"><table><thead><tr><th>UTC slot</th><th>Forecast MW</th><th>Observed MW</th></tr></thead><tbody>{result.prediction_mw.map((v,i)=><tr key={i}><td>{String(Math.floor(i/2)).padStart(2,"0")}:{i%2?"30":"00"}</td><td>{v.toFixed(2)}</td><td>{result.observed_mw[i].toFixed(2)}</td></tr>)}</tbody></table></div></details>
      </section>
      <section className="panel"><h2>What ran</h2><dl className="boundaries"><div><dt>Execution location</dt><dd>{result.execution_location==="browser"?"Your browser CPU":"Python server CPU"}</dd></div><div><dt>Runtime</dt><dd>{result.runtime}</dd></div>{result.runtime_id&&<div><dt>Runtime identity</dt><dd>{result.runtime_id}</dd></div>}<div><dt>Model load</dt><dd>{result.model_cache_hit?"Already loaded; predictions recomputed":`${result.model_init_ms.toFixed(0)} ms download/initialisation`}</dd></div><div><dt>Prediction cache</dt><dd>Not used</dd></div><div><dt>Logical issue time</dt><dd>{result.inputs.issued_at}</dd></div><div><dt>History cutoff</dt><dd>{result.inputs.context_end_exclusive}</dd></div><div><dt>New execution identity</dt><dd data-testid="live-execution-id">{result.execution_id}</dd></div><div><dt>Model identity</dt><dd>{result.model_id}</dd></div></dl><p className="hint">Browser timings describe this device, not the CPU server benchmark. First runs include downloading and initialising the runtime. Revised historical inputs do not establish what was captured at the original issue time. The 2024 test set is not exposed.</p></section>
    </>}
    <footer>Supported by National Energy SO Open Data. Independent project, not endorsed by NESO. <a href="https://www.neso.energy/data-portal/neso-open-licence">Data licence</a> · <a href="https://github.com/MNauman13/GridLens">Source and evaluation evidence</a></footer>
  </main>;
}

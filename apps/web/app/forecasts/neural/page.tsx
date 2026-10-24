"use client";
import Link from "next/link";
import {useEffect,useState} from "react";

type Forecast={prediction_mw:number[];observed_mw:number[];model_id:string;forecast_id:string;dataset_manifest_id:string;validation_report_id:string;inputs:{issued_at:string;context_end_exclusive:string;target_start:string;feature_snapshot_id:string}};
export default function NeuralForecasts(){
  const [model,setModel]=useState("timexer");
  const [data,setData]=useState<Forecast|null>(null);
  const [error,setError]=useState("");
  const [slot,setSlot]=useState(0);
  useEffect(()=>{const controller=new AbortController();fetch(`/api/forecasts/neural?model=${model}`,{signal:controller.signal}).then(async r=>{if(!r.ok)throw Error("Saved neural forecast unavailable. Check the API and published experiment artifacts.");return r.json() as Promise<Forecast>;}).then(value=>{if(!controller.signal.aborted)setData(value);}).catch(e=>{if(!controller.signal.aborted)setError(String(e.message));});return()=>controller.abort();},[model]);
  const bounds=data?[Math.min(...data.prediction_mw,...data.observed_mw)-1000,Math.max(...data.prediction_mw,...data.observed_mw)+1000]:[0,1];
  const points=(values:number[])=>values.map((v,i)=>`${24+i*552/47},${220-(v-bounds[0])/(bounds[1]-bounds[0])*190}`).join(" ");
  return <main><header><Link className="brand" href="/">Grid<span>Lens</span></Link><Link href="/learn/transformer">How these models work</Link></header>
    <section className="intro"><p className="eyebrow">SAVED VALIDATION FORECAST</p><h1>Inspect the neural pilot.</h1><p className="lede">A real trained forecast for 10 February 2022. Retrospective revised data, seed 17, bounded CPU training. This is a validation example, not a live forecast or test-set result.</p></section>
    <section className="panel"><label htmlFor="neural-model">Neural model</label><select id="neural-model" value={model} onChange={e=>{setData(null);setError("");setModel(e.target.value);}}><option value="timexer">TimeXer selected candidate</option><option value="transformer">Autoregressive Transformer</option><option value="lstm">Autoregressive LSTM</option></select>
    {error&&<p role="alert">{error}</p>}{!data&&!error&&<p role="status">Loading saved forecast…</p>}
    {data&&<><p role="status">48 half-hour predictions in MW. Green: forecast. Gray dashed: observed.</p><svg viewBox="0 0 600 250" role="img" aria-label="48-slot neural prediction and observed demand" style={{width:"100%",height:"auto"}}><polyline points={points(data.observed_mw)} fill="none" stroke="#8b98a5" strokeWidth="2" strokeDasharray="6 5"/><polyline points={points(data.prediction_mw)} fill="none" stroke="#4ade80" strokeWidth="3"/></svg>
    <label htmlFor="neural-slot">Half-hour slot: {slot+1}</label><input id="neural-slot" type="range" min="0" max="47" value={slot} onChange={e=>setSlot(Number(e.target.value))}/><p data-testid="neural-slot-value">Forecast {data.prediction_mw[slot].toFixed(2)} MW · Observed {data.observed_mw[slot].toFixed(2)} MW</p>
    <dl className="boundaries"><div><dt>Issued at</dt><dd>{data.inputs.issued_at}</dd></div><div><dt>Historical cutoff</dt><dd>{data.inputs.context_end_exclusive}</dd></div><div><dt>Target begins</dt><dd>{data.inputs.target_start}</dd></div></dl>
    <details><summary>Forecast provenance</summary><dl className="boundaries"><div><dt>Model identity</dt><dd>{data.model_id}</dd></div><div><dt>Forecast identity</dt><dd>{data.forecast_id}</dd></div><div><dt>Dataset manifest</dt><dd>{data.dataset_manifest_id}</dd></div><div><dt>Validation report</dt><dd>{data.validation_report_id}</dd></div><div><dt>Feature snapshot</dt><dd>{data.inputs.feature_snapshot_id}</dd></div></dl></details></>}
    </section><footer><Link href="/forecasts">Compare frozen baselines</Link> · <a href="https://github.com/MNauman13/GridLens">Experiment reports</a></footer></main>;
}

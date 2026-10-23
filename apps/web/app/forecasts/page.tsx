"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Forecast = {
  forecast_id: string; model_kind: string; dataset_id: string; source_sha256: string[];
  inputs: {issued_at:string; context_end_exclusive:string; target_start:string; target_end_exclusive:string; evidence_mode:string; feature_snapshot_id:string};
  point: {prediction_mw:number[]; model_version:string}; observed_mw:number[];
};
const names: Record<string,string> = {last_day:"Last available day",weekly:"Weekly persistence",ridge:"Ridge regression"};

export default function ForecastExplorer() {
  const [model,setModel] = useState("ridge");
  const [forecast,setForecast] = useState<Forecast|null>(null);
  const [error,setError] = useState("");
  const [slot,setSlot] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/forecasts/baseline?model=${model}`, {signal:controller.signal}).then(async response => {
      if (!response.ok) throw new Error("The saved forecast is unavailable. Check the Python API and forecast artifact.");
      const value = await response.json() as Forecast;
      if (value.point.prediction_mw.length !== 48 || value.observed_mw.length !== 48 || [...value.point.prediction_mw,...value.observed_mw].some(x=>!Number.isFinite(x))) throw new Error("The forecast response failed validation.");
      if (!controller.signal.aborted) setForecast(value);
    }).catch(caught => {if(!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Forecast request failed.");});
    return () => controller.abort();
  },[model]);
  const values = forecast ? [...forecast.point.prediction_mw,...forecast.observed_mw] : [0,1];
  const minimum = Math.min(...values)-500, maximum = Math.max(...values)+500;
  const x = (i:number) => 65+i/47*850;
  const y = (v:number) => 260-(v-minimum)/(maximum-minimum)*220;
  const line = (values:number[]) => values.map((v,i)=>`${i===0?"M":"L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
  const stamp = (i:number) => forecast ? new Date(new Date(forecast.inputs.target_start).getTime()+i*1800000).toISOString() : "";
  return <main>
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><span className="phase">Real data / Saved validation forecast</span></header>
    <section className="intro"><p className="eyebrow">THE BASELINE REFERENCE</p><h1>A real forecast.<br/>An inspectable origin.</h1><p className="lede">Explore a saved 48-point National Demand forecast for 10 February 2022. Ridge was fitted on 2018–2021; these models are compared on the same 2022 validation dates.</p><p className="notice">Revised historical data / Validation example / Final 2024 evaluation remains locked</p></section>
    <section className="panel"><label htmlFor="baseline-model">Baseline model</label><select id="baseline-model" value={model} onChange={event=>{setForecast(null);setError("");setModel(event.target.value);}}>{Object.entries(names).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select>
      {error && <p role="alert" className="error">{error}</p>}
      {!forecast && !error && <p role="status">Loading saved forecast...</p>}
      {forecast && <>
        <h2>{names[model]}</h2><p role="status" data-testid="forecast-count">48 verified half-hour predictions / MW / UTC</p>
        <svg className="forecast-chart" viewBox="0 0 960 310" role="img" aria-label="48 half-hour forecast values and observed National Demand in MW">
          <title>Saved forecast and revised observations</title>
          {[0,.5,1].map(t=><g key={t}><line x1="65" x2="915" y1={y(minimum+t*(maximum-minimum))} y2={y(minimum+t*(maximum-minimum))} stroke="#dce3dc"/><text x="5" y={y(minimum+t*(maximum-minimum))+5} fontSize="13">{Math.round(minimum+t*(maximum-minimum))}</text></g>)}
          <path d={line(forecast.observed_mw)} fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="6 4"/>
          <path d={line(forecast.point.prediction_mw)} fill="none" stroke="#23624e" strokeWidth="3"/>
          <circle cx={x(slot)} cy={y(forecast.point.prediction_mw[slot])} r="6" fill="#23624e"/>
          <text x="65" y="292" fontSize="14">00:00 UTC</text><text x="830" y="292" fontSize="14">23:30 UTC</text>
        </svg><p className="hint">Solid green: forecast. Dashed grey: revised observation.</p>
        <label htmlFor="forecast-slot">Inspect forecast slot {slot+1}</label><input id="forecast-slot" className="period-slider" type="range" min="0" max="47" value={slot} onChange={event=>setSlot(Number(event.target.value))}/>
        <p className="point-readout">{stamp(slot)} / Forecast {forecast.point.prediction_mw[slot].toFixed(1)} MW / Observed {forecast.observed_mw[slot].toFixed(1)} MW</p>
        <dl className="boundaries"><div><dt>Issued at</dt><dd data-testid="forecast-issue">{forecast.inputs.issued_at}</dd></div><div><dt>Context cutoff, exclusive</dt><dd data-testid="forecast-cutoff">{forecast.inputs.context_end_exclusive}</dd></div><div><dt>Target starts</dt><dd data-testid="forecast-target">{forecast.inputs.target_start}</dd></div><div><dt>Target ends, exclusive</dt><dd>{forecast.inputs.target_end_exclusive}</dd></div></dl>
        <details><summary>Inspect provenance</summary><dl className="settlement-reading"><div><dt>Forecast ID</dt><dd>{forecast.forecast_id}</dd></div><div><dt>Model version</dt><dd>{forecast.point.model_version}</dd></div><div><dt>Dataset ID</dt><dd>{forecast.dataset_id}</dd></div><div><dt>Feature snapshot</dt><dd>{forecast.inputs.feature_snapshot_id}</dd></div><div><dt>Frozen source SHA-256</dt><dd>{forecast.source_sha256.join(" / ")}</dd></div></dl></details>
      </>}
    </section><footer><Link href="/">Back to the contract</Link><p>Supported by National Energy SO Open Data. Independent project.</p></footer>
  </main>;
}

"use client";
import Link from "next/link";
import {useEffect,useState} from "react";
import {atStep,Method,ReplayEvidence} from "../../lib/replay-engine";

export default function Replay(){
  const [data,setData]=useState<ReplayEvidence|null>(null);const [error,setError]=useState("");
  const [cursor,setCursor]=useState(0);const [method,setMethod]=useState<Method>("rolling");const [level,setLevel]=useState(1);const [playing,setPlaying]=useState(false);
  useEffect(()=>{const abort=new AbortController();fetch('/replay/evidence.json',{signal:abort.signal}).then(async response=>{if(!response.ok)throw Error("The frozen replay evidence is unavailable.");return response.json();}).then((value:ReplayEvidence)=>{setData(value);setMethod(value.selected);try{const saved=JSON.parse(localStorage.getItem('gridlens-replay')??'null');if(saved?.replay_id===value.replay_id&&Number.isInteger(saved.cursor)&&saved.cursor>=0&&saved.cursor<=value.rows.length+1)setCursor(saved.cursor);}catch{/* Start from the first issue when storage is unavailable. */}}).catch(e=>{if(!abort.signal.aborted)setError(e.message);});return()=>abort.abort();},[]);
  useEffect(()=>{if(data)try{localStorage.setItem('gridlens-replay',JSON.stringify({replay_id:data.replay_id,cursor}));}catch{/* Private browsing can disable persistence. */}},[data,cursor]);
  useEffect(()=>{if(!playing||!data)return;const id=setInterval(()=>setCursor(value=>{if(value>=data.rows.length+1){setPlaying(false);return value;}return value+1;}),1200);return()=>clearInterval(id);},[playing,data]);
  const state=data?atStep(data,cursor,method):null;
  function download(){if(!state||!data)return;const lines=['slot_utc,point_mw,lower_mw,upper_mw,observed_mw',...state.row.point.map((v,i)=>`${String(Math.floor(i/2)).padStart(2,'0')}:${i%2?'30':'00'},${v},${v-state.widths[i][level]},${v+state.widths[i][level]},${state.actualVisible?state.row.actual[i]:''}`)];const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download=`gridlens-${state.row.target_date}-${method}.csv`;a.click();URL.revokeObjectURL(url);}
  const max=state?Math.max(...state.row.point.map((v,i)=>v+state.widths[i][level]),...(state.actualVisible?state.row.actual:[])):1;
  const min=state?Math.min(...state.row.point.map((v,i)=>v-state.widths[i][level]),...(state.actualVisible?state.row.actual:[])):0;
  const coord=(v:number,i:number)=>`${30+i/47*640},${230-(v-min)/Math.max(1,max-min)*200}`;
  return <main><header><Link className="brand" href="/">Grid<span>Lens</span></Link><nav><Link href="/live">Fresh inference</Link><Link href="/results">Release results</Link></nav></header>
    <section className="intro"><p className="eyebrow">CAUSAL UNCERTAINTY REPLAY</p><h1>What was known<br/>when the forecast issued?</h1><p className="lede">Advance the clock through 2023. Observe how uncertainty changes as earlier demand measurements become eligible feedback.</p><p>Historical revised-data simulation. This page replays frozen point forecasts. <Link href="/live">Run a model live</Link> for fresh point computation.</p></section>
    {error&&<p role="alert">{error}</p>}{!data&&!error&&<p role="status">Loading verified study evidence</p>}
    {data&&state&&<>
      <section className="panel"><div className="live-form"><label htmlFor="replay-method">Interval method<select id="replay-method" value={method} onChange={e=>setMethod(e.target.value as Method)}><option value="fixed">Fixed residual</option><option value="rolling">Rolling residual</option><option value="ncc">NCC {data.ncc_status==='experimental'?'(experimental)':''}</option></select></label><label htmlFor="replay-level">Marginal coverage<select id="replay-level" value={level} onChange={e=>setLevel(Number(e.target.value))}>{data.levels.map((value,i)=><option key={value} value={i}>{value*100}%</option>)}</select></label></div>
        <div className="step-controls"><button disabled={cursor===0} onClick={()=>{setPlaying(false);setCursor(v=>v-1);}}>Previous issue</button><button disabled={cursor===data.rows.length+1} onClick={()=>setCursor(v=>v+1)}>Next issue</button><button disabled={cursor===data.rows.length+1} onClick={()=>setPlaying(v=>!v)}>{playing?'Pause':'Play'}</button><button onClick={()=>{setPlaying(false);setCursor(0);}}>Reset replay</button></div>
        <p role="status" data-testid="replay-clock">Issue clock {new Date(state.clock).toISOString()} · {state.matured} validation days released</p><label htmlFor="replay-position">Replay position<input id="replay-position" type="range" min="0" max={data.rows.length+1} value={cursor} onChange={e=>{setPlaying(false);setCursor(Number(e.target.value));}}/></label>
        <p>Latest eligible feedback {state.feedbackWatermark??'none'}. Target-day actuals release at 09:00 UTC the following day. A target’s issue is at 09:30 UTC on the previous day.</p>
        {state.degraded&&<p className="notice">Fewer than 60 eligible days. Residual widths use the labelled pooled fallback.</p>}
        {!data.coverage_gate_passed&&<p className="notice">The selected interval candidate missed at least one validation coverage gate. Treat its bands as experimental.</p>}
      </section>
      <section className="panel"><h2>{state.row.target_date} demand and uncertainty</h2><p data-testid="replay-feedback">{state.actualVisible?'Observed demand is now eligible.':'Observed demand is withheld until its simulated release.'}</p>
        <svg viewBox="0 0 700 275" className="forecast-chart" role="img" aria-label="Point forecast and marginal uncertainty band, with actual demand shown only after release"><polygon points={[...state.row.point.map((v,i)=>coord(v+state.widths[i][level],i)),...state.row.point.map((v,i)=>coord(v-state.widths[i][level],i)).reverse()].join(' ')} fill="#28644b" opacity=".17"/><polyline points={state.row.point.map(coord).join(' ')} fill="none" stroke="#23624e" strokeWidth="3"/>{state.actualVisible&&<polyline points={state.row.actual.map(coord).join(' ')} fill="none" stroke="#7d532d" strokeWidth="2" strokeDasharray="5 4"/>}<text x="30" y="260">00:00 UTC</text><text x="590" y="260">23:30 UTC</text></svg>
        <p>Mean band width {(state.widths.reduce((sum,row)=>sum+2*row[level],0)/48).toFixed(0)} MW. Bands describe individual half-hours, with no guarantee that the whole daily curve lies inside.</p><button onClick={download}>Download this forecast CSV</button>
        <p>{method==='ncc'?'NCC thresholds are recorded outputs from the frozen neural controller, including its delayed feedback and bounded TTA.':'Residual thresholds are recomputed in this browser using eligible historical scores.'} Advancing the clock never changes an already issued forecast.</p>
        <details><summary>Forecast provenance</summary><dl className="boundaries"><div><dt>Point model</dt><dd>{data.point_model_id}</dd></div><div><dt>Calibration</dt><dd>{data.calibration_id}</dd></div><div><dt>Replay evidence</dt><dd>{data.replay_id}</dd></div></dl><p>{data.point_source}</p></details>
      </section>
    </>}
    <footer>Supported by National Energy SO Open Data. Independent research implementation. No new coverage theorem is claimed.</footer>
  </main>;
}

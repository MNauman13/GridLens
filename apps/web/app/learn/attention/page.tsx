"use client";
import Link from "next/link";
import {useEffect,useState} from "react";
import data from "../../../data/attention-demo.json";

const stages=["Compare the query and keys","Divide by the square root of head width","Mask, then apply softmax","Mix values, then merge heads"];
const fmt=(v:number)=>v.toFixed(6);

export default function AttentionLesson(){
  const [frame,setFrame]=useState(0),[playing,setPlaying]=useState(false);
  const [causal,setCausal]=useState(false),[position,setPosition]=useState(0),[positionsPlaying,setPositionsPlaying]=useState(false);
  useEffect(()=>{if(!playing)return;const t=setInterval(()=>setFrame(v=>(v+1)%24),850);return()=>clearInterval(t);},[playing]);
  useEffect(()=>{if(!positionsPlaying)return;const t=setInterval(()=>setPosition(v=>(v+1)%16),650);return()=>clearInterval(t);},[positionsPlaying]);
  const head=Math.floor(frame/12),query=Math.floor(frame/4)%3,stage=frame%4;
  const scenario=data.scenarios[causal?"causal":"unmasked"];
  const weights=scenario.weights[head][query],scores=scenario.scores[head][query];
  const choose=(h:number,q:number)=>{setPlaying(false);setFrame(h*12+q*4+stage);};
  return <main>
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><span className="phase">Phase 5 / Attention from scratch</span></header>
    <section className="intro"><p className="eyebrow">WATCH THE MATHEMATICS</p><h1>Compare. Weight.<br/>Mix information.</h1><p className="lede">Three tokens, two heads, exact numbers. CI checks these calculator values against our manual PyTorch operators.</p><p className="notice">Teaching fixture / Identity projections / No trained attention patterns</p></section>
    <section className="panel"><h2>One query, three candidate keys</h2>
      <div className="lesson-controls">
        <div><label htmlFor="attention-head">Attention head</label><select id="attention-head" value={head} onChange={ev=>choose(Number(ev.target.value),query)}><option value="0">Head 1 / channels 1–2</option><option value="1">Head 2 / channels 3–4</option></select></div>
        <div><label htmlFor="attention-query">Query token</label><select id="attention-query" value={query} onChange={ev=>choose(head,Number(ev.target.value))}>{[0,1,2].map(i=><option key={i} value={i}>Token {i+1}</option>)}</select></div>
        <label className="checkbox-label"><input type="checkbox" checked={causal} onChange={ev=>setCausal(ev.target.checked)}/>Block future keys</label>
      </div>
      <p>Compare a query with every key, divide by √2, replace blocked scores with −∞, then normalise with softmax.</p>
      <p className="formula">Attention(Q,K,V) = softmax(QKᵀ / √2 + mask)V</p>
      <p>Query = [{data.heads[head][query].join(", ")}]</p>
      <div className="attention-cards">{weights.map((w,k)=><div className={causal&&k>query?"attention-card blocked":"attention-card"} key={k}>
        <h3>Key {k+1} / [{data.heads[head][k].join(", ")}]</h3><p>Dot / √2 = <strong>{fmt(scores[k])}</strong></p>
        <p>Weight: <strong data-testid={"weight-"+k}>{fmt(w)}</strong>{causal&&k>query?" / blocked":""}</p>
        <div className="weight-track"><div style={{width:(w*100)+"%"}}/></div><p>Value = [{data.heads[head][k].join(", ")}]</p>
      </div>)}</div>
      <p className="point-readout" data-testid="head-output">Head output = [{scenario.head_outputs[head][query].map(fmt).join(", ")}]</p>
      <p className="point-readout">Merged output = [{scenario.merged_output[query].map(fmt).join(", ")}]</p>
      <div className="step-detail" aria-live={playing?"off":"polite"}><p className="eyebrow">STEP {stage+1} / 4</p><h3 data-testid="attention-stage">{stages[stage]}</h3>
      <p>{["Each head compares content in its own projected channels.","Scaling limits score growth as head width increases.","Blocked keys receive zero weight. Allowed weights sum to one.","Weighted values are added. Head results are concatenated and output-projected."][stage]}</p></div>
      <div className="step-controls attention-controls"><button onClick={()=>setPlaying(v=>!v)}>{playing?"Pause attention":"Play attention"}</button><button disabled={frame===0} onClick={()=>{setPlaying(false);setFrame(v=>v-1);}}>Previous step</button><button onClick={()=>{setPlaying(false);setFrame(v=>(v+1)%24);}}>Next step</button></div>
    </section>
    <section className="panel"><h2>Position adds a pattern at several frequencies</h2><p>Sine channels and cosine channels supply order. Higher channel pairs change more slowly.</p>
      <p className="formula">PE(pos, 2i) = sin(pos / 10000^(2i / d_model))</p><p className="formula">PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))</p>
      <label htmlFor="position-index">Position {position}</label><input id="position-index" className="period-slider" type="range" min="0" max="15" value={position} onChange={ev=>{setPositionsPlaying(false);setPosition(Number(ev.target.value));}}/>
      <div className="position-grid">{data.positions[position].map((v,i)=><div key={i}><span>Channel {i}</span><div className="position-track"><div style={{height:(Math.abs(v)*45)+"%",bottom:v>=0?"50%":(50-Math.abs(v)*45)+"%",background:v>=0?"#23624e":"#ad6455"}}/></div><strong data-testid={"position-"+i}>{fmt(v)}</strong></div>)}</div>
      <button onClick={()=>setPositionsPlaying(v=>!v)}>{positionsPlaying?"Pause positions":"Play positions"}</button><p className="hint">Positions are fixed buffers. At position zero, sine channels are 0 and cosine channels are 1. Both animations play only when requested.</p>
    </section><footer><Link href="/forecasts">Explore the real forecasting baselines</Link><p><a href="https://arxiv.org/abs/1706.03762">Vaswani et al., Attention Is All You Need</a></p></footer>
  </main>;
}

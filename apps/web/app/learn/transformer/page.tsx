"use client";
import Link from "next/link";
import {useEffect,useState} from "react";

const stages=[
  {title:"Patch the demand history",body:"Split 336 historical demand readings into 14 non-overlapping patches of 24 half-hours. Each patch becomes a learned 32-channel token."},
  {title:"Mix patches and the global token",body:"Manual multi-head self-attention mixes the 14 patch tokens with one learned global token. Sinusoidal positions distinguish patch order."},
  {title:"Let the global token query external history",body:"Each historical feature becomes a variable token. Only the global token queries these tokens; individual patches do not directly perform external cross-attention."},
  {title:"Predict across the gap",body:"The GridLens adapter produces 96 half-hour values, discards the first 48 issue-day slots, then adds a known-calendar residual to the remaining 48 target-day values."},
];
export default function TransformerLesson(){
  const [mode,setMode]=useState("training");
  const [slot,setSlot]=useState(0);
  const [stage,setStage]=useState(0);
  const [playing,setPlaying]=useState(false);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setStage(s=>(s+1)%4),1800);return()=>clearInterval(timer);},[playing]);
  const answer=[7,4,9];
  return <main>
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><span className="phase">Phases 6–8 / Neural forecasting</span></header>
    <section className="intro"><p className="eyebrow">FROM TOKENS TO ELECTRICITY</p><h1>Train with answers.<br/>Predict without them.</h1><p className="lede">Explore shifted decoder inputs, then follow TimeXer’s patch and global-token route. This schematic explains the operators; its token example is illustrative, not a trained model output.</p></section>
    <section className="panel"><h2>One decoder, two information flows</h2><p>Source tokens: 9, 4, 7, EOS. Desired reversal: 7, 4, 9, EOS. The encoder reads the source; causal self-attention limits each decoder position to its prefix, and cross-attention reads encoder memory.</p>
      <label htmlFor="decoder-mode">Decoder mode</label><select id="decoder-mode" value={mode} onChange={e=>setMode(e.target.value)}><option value="training">Shifted training inputs</option><option value="inference">Generated inference inputs</option></select>
      <label htmlFor="decoder-slot">Output position: {slot+1}</label><input id="decoder-slot" type="range" min="0" max="3" value={slot} onChange={e=>setSlot(Number(e.target.value))}/>
      <div className="timeline"><div className="region active"><span>Available prefix</span><strong data-testid="decoder-prefix">BOS{answer.slice(0,slot).map(v=>`, ${v}`).join("")}</strong></div><div className="region"><span>Prefix comes from</span><strong data-testid="prefix-origin">{mode==="training"?"Previous true tokens":"Previous generated tokens"}</strong></div></div>
      <p role="status">{mode==="training"?"Training computes all shifted positions together under a causal mask. The answer at the current output position is excluded from its input.":"Inference starts with BOS and repeatedly appends a predicted token. EOS stops generation; a maximum length guarantees termination."}</p>
      <p>The original architecture uses residual addition followed by layer normalization, a two-layer ReLU feed-forward network, and embedding scaling by the square root of model width. The toy trainer uses Adam with a warmup and inverse-square-root schedule.</p>
    </section>
    <section className="panel"><h2>TimeXer’s information route</h2><div className="timeline" aria-label="TimeXer operator stages">{["Patches","Self-attention","Global query","Gap-aware head"].map((text,i)=><div key={text} className={i===stage?"region active":"region"}><span>{i+1}</span><strong>{text}</strong></div>)}</div>
      <div aria-live="polite"><h3 data-testid="timexer-stage">{stages[stage].title}</h3><p>{stages[stage].body}</p></div>
      <div className="step-controls"><button disabled={stage===0} onClick={()=>setStage(s=>s-1)}>Previous operator</button><button onClick={()=>setPlaying(p=>!p)}>{playing?"Pause operators":"Play operators"}</button><button disabled={stage===3} onClick={()=>setStage(s=>s+1)}>Next operator</button></div>
      <p>Reference components: patch embedding, global token, variable tokens, self-attention, global-only external attention, post-norm feed-forward blocks and a flattened head. Application changes: 48-slot gap handling, calendar residual, the frozen feature schema and training-only scaling.</p>
    </section>
    <section className="panel"><h2>How we judge a model</h2><p>All candidates use the same 1,460 training origins and 364 validation origins. The numerical Transformer and LSTM generate their own previous outputs for validation. TimeXer predicts directly. Ridge remains eligible to win: complexity alone does not make a better forecast.</p><p>The bounded pilot uses seed 17. Its five TimeXer configurations test external information, patch length, head count and position encoding. Single-seed differences are sensitivity checks, not established superiority. The 2024 test set remains locked.</p><Link href="/forecasts/neural">Inspect saved neural forecasts →</Link></section>
    <footer><Link href="/learn/attention">Attention calculations</Link> · <a href="https://github.com/MNauman13/GridLens">Code, training reports and limitations</a></footer>
  </main>;
}

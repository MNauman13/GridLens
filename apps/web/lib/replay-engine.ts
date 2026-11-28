export type Method="fixed"|"rolling"|"ncc";
export type ReplayEvidence={replay_id:string;point_model_id:string;calibration_id:string;selected:Method;ncc_status:string;coverage_gate_passed:boolean;point_source:string;levels:number[];window:number;minimum:number;rows:{target_date:string;point:number[];actual:number[]}[];initial:Record<Method,{scores:number[][];feedback_watermark:string|null;fixed:number[][]|null}>;thresholds:Record<Method,number[][][]>};
export function issueClock(day:string){return Date.parse(day+"T09:30:00Z")-86400000;}
export function releaseClock(day:string){return Date.parse(day+"T09:00:00Z")+86400000;}
export function quantile(values:number[],level:number){
  if(!values.length||values.some(v=>!Number.isFinite(v))||level<=0||level>=1)throw Error("Invalid finite quantile");
  const sorted=[...values].sort((a,b)=>a-b);
  return sorted[Math.min(sorted.length,Math.ceil((sorted.length+1)*level))-1];
}
export function atStep(evidence:ReplayEvidence,cursor:number,method:Method){
  if(!Number.isInteger(cursor)||cursor<0||cursor>evidence.rows.length+1)throw Error("Replay step out of range");
  const selected=Math.min(cursor,evidence.rows.length-1);const row=evidence.rows[selected];
  const clock=cursor<evidence.rows.length?issueClock(evidence.rows[cursor].target_date):issueClock(evidence.rows.at(-1)!.target_date)+(cursor-evidence.rows.length+1)*86400000;
  const prior=evidence.rows.filter(r=>releaseClock(r.target_date)<=issueClock(row.target_date));
  const scores=[...evidence.initial[method].scores,...prior.map(r=>r.point.map((v,i)=>Math.abs(v-r.actual[i])))].slice(-evidence.window);
  const pooled=scores.length<evidence.minimum;
  let widths:number[][];
  if(method==="ncc")widths=evidence.thresholds.ncc[selected];
  else if(method==="fixed"){
    const initial=evidence.initial.fixed;
    widths=initial.fixed??Array.from({length:48},(_,slot)=>evidence.levels.map(level=>quantile(initial.scores.length<evidence.minimum?initial.scores.flat():initial.scores.map(r=>r[slot]),level)));
  }else widths=Array.from({length:48},(_,slot)=>evidence.levels.map(level=>quantile(pooled?scores.flat():scores.map(r=>r[slot]),level)));
  const matured=evidence.rows.filter(r=>releaseClock(r.target_date)<=clock);
  const visible=releaseClock(row.target_date)<=clock;
  return {row,selected,clock,widths,actualVisible:visible,matured:matured.length,feedbackWatermark:matured.at(-1)?.target_date??evidence.initial[method].feedback_watermark,degraded:pooled};
}

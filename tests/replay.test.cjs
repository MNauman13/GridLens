const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');
const {atStep,quantile}=require('../apps/web/lib/replay-engine.ts');
test('finite quantiles use capped ceil(n+1) ranks without mutating scores',()=>{
  const values=[3,1,2];assert.equal(quantile(values,.8),3);assert.deepEqual(values,[3,1,2]);assert.throws(()=>quantile([], .9));
});
test('causal browser residual replay agrees with Python on every released origin',()=>{
  const evidence=JSON.parse(fs.readFileSync('apps/web/public/replay/evidence.json','utf8'));
  for(let step=0;step<evidence.rows.length;step++)for(const method of ['fixed','rolling']){
    const result=atStep(evidence,step,method);
    assert.equal(result.actualVisible,false);
    result.widths.forEach((row,slot)=>row.forEach((value,level)=>assert.ok(Math.abs(value-evidence.thresholds[method][step][slot][level])<1e-8)));
  }
  const first=atStep(evidence,0,'rolling');const end=atStep(evidence,evidence.rows.length+1,'rolling');
  assert.equal(first.matured,0);assert.equal(end.matured,evidence.rows.length);assert.equal(end.actualVisible,true);
  assert.deepEqual(atStep(evidence,0,'rolling'),first);assert.throws(()=>atStep(evidence,-1,'rolling'));
});

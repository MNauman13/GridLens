import {test,expect} from 'playwright/test';

test('fresh real browser forecasts: both models, changed date, unique executions',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/live');
  const run=page.getByRole('button',{name:'Run fresh inference'});
  await expect(run).toBeEnabled();
  await run.click();
  await expect(page.getByTestId('live-execution-id')).toBeVisible({timeout:25000});
  const first=await page.getByTestId('live-execution-id').textContent();
  const prediction=await page.getByTestId('live-slot-value').textContent();
  await page.getByRole('combobox',{name:'Target date',exact:true}).selectOption('2022-01-12');
  await run.click();
  await expect(page.getByTestId('live-execution-id')).toBeVisible();
  expect(await page.getByTestId('live-execution-id').textContent()).not.toBe(first);
  expect(await page.getByTestId('live-slot-value').textContent()).not.toBe(prediction);
  await page.getByRole('combobox',{name:'Model',exact:true}).selectOption('ridge');await run.click();
  await expect(page.getByTestId('live-execution-id')).toBeVisible();
  await expect(page.getByText('Your browser CPU',{exact:true})).toBeVisible();
  expect(errors).toEqual([]);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:`reports/local/browser-results/live-${test.info().project.name}.png`,fullPage:true});
});

test('corrupt input is rejected without a substitute forecast',async({page})=>{
  await page.route('**/inference/cases/*',r=>r.fulfill({status:200,body:'{}'}));
  await page.goto('/live');await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByRole('alert').filter({hasText:'checksum failed'})).toBeVisible();
  await expect(page.getByTestId('live-execution-id')).toHaveCount(0);
});

test('all 32 browser TimeXer origins agree with the Python reference',async({page})=>{
  test.skip(test.info().project.name!=='chromium-desktop','Full parity matrix runs once; mobile has separate real inference coverage');
  test.setTimeout(120000);
  await page.goto('/live');
  const catalog=await (await page.request.get('/api/live/catalog')).json();
  expect(catalog.server_available).toBe(true);
  let maxDrift=0;const timings:number[]=[];
  for(const entry of catalog.cases){
    await page.getByRole('combobox',{name:'Target date',exact:true}).selectOption(entry.target_date);
    const tick=Date.now();await page.getByRole('button',{name:'Run fresh inference'}).click();
    await expect(page.getByTestId('live-execution-id')).toBeVisible({timeout:25000});
    timings.push(Date.now()-tick);
    const values=await page.locator('tbody tr td:nth-child(2)').allTextContents();
    expect(values).toHaveLength(48);
    const accepted=await page.request.post('/api/live/jobs',{data:{model:'timexer',target_date:entry.target_date}});
    expect(accepted.status()).toBe(202);
    const id=(await accepted.json()).job_id;
    let job;
    for(let i=0;i<100;i++){
      job=await (await page.request.get(`/api/live/jobs/${id}`)).json();
      if(job.state==='succeeded'||job.state==='failed')break;
      await new Promise(r=>setTimeout(r,20));
    }
    expect(job.state).toBe('succeeded');
    values.forEach((v,i)=>{const drift=Math.abs(Number(v)-job.result.point.prediction_mw[i]);maxDrift=Math.max(maxDrift,drift);expect(drift).toBeLessThan(0.5);});
  }
  console.log(JSON.stringify({scope:'32 browser WASM versus real Python forecasts; UI values rounded to 0.01 MW',max_drift_mw:maxDrift,click_to_visible_ms:timings}));
});

test('Python service UI computes both models and never replaces a failed job',async({page})=>{
  await page.goto('/live');
  const execution=page.getByRole('combobox',{name:'Execution',exact:true});
  await execution.selectOption('server');
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByText('Python server CPU',{exact:true})).toBeVisible();
  const first=await page.getByTestId('live-execution-id').textContent();
  await page.getByRole('combobox',{name:'Model',exact:true}).selectOption('ridge');
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByTestId('live-execution-id')).toBeVisible();
  expect(await page.getByTestId('live-execution-id').textContent()).not.toBe(first);
  const denied=await page.request.post('/api/live/jobs',{headers:{origin:'https://untrusted.example'},data:{model:'ridge',target_date:'2022-01-01'}});
  expect(denied.status()).toBe(403);
  await page.route('**/api/live/jobs',route=>route.fulfill({status:503,json:{error:'Reference service unavailable'}}));
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByRole('alert').filter({hasText:'Reference service unavailable'})).toBeVisible();
  await expect(page.getByTestId('live-execution-id')).toHaveCount(0);
});

test('optional server SDPA has its own runtime identity and benchmark link',async({page})=>{
  await page.goto('/live');
  await expect(page.getByRole('link',{name:'Inference benchmarks'})).toHaveAttribute('href','/experiments#inference');
  await page.getByRole('combobox',{name:'Execution',exact:true}).selectOption('server');
  await page.getByRole('combobox',{name:'Python attention runtime',exact:true}).selectOption('sdpa');
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByText('pytorch_sdpa_checked_v2',{exact:true})).toBeVisible();
  await expect(page.getByText('Runtime identity',{exact:true})).toBeVisible();
  await expect(page.getByTestId('live-execution-id')).toBeVisible();
});

test('verified ONNX CPU jobs have separate runtime provenance',async({page})=>{
  await page.goto('/live');
  await page.getByRole('combobox',{name:'Execution',exact:true}).selectOption('server');
  await page.getByRole('combobox',{name:'Python attention runtime',exact:true}).selectOption('onnx');
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByRole('status')).toContainText('Fresh inference complete');
  await expect(page.getByText('onnx_cpu_fp32',{exact:true})).toBeVisible();
  await expect(page.getByText('Runtime identity',{exact:true})).toBeVisible();
});

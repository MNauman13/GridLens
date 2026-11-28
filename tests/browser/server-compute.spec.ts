import {test,expect} from 'playwright/test';
test('deployed server adapter computes fresh forecasts and rejects invalid requests',async({page,request})=>{
  const bad=await request.post('/api/compute',{data:{model:'timexer',target_date:'2024-01-01'}});
  expect(bad.status()).toBe(422);
  const cross=await request.post('/api/compute',{data:{model:'ridge',target_date:'2022-01-01'},headers:{origin:'https://example.org'}});
  expect(cross.status()).toBe(403);
  const first=await request.post('/api/compute',{data:{model:'timexer',target_date:'2022-01-01'}});
  expect(first.status(),await first.text()).toBe(200);
  const a=await first.json();
  expect(a.result_cache_hit).toBe(false);expect(a.execution_location).toBe('web_server');expect(a.prediction_mw).toHaveLength(48);
  const second=await request.post('/api/compute',{data:{model:'timexer',target_date:'2022-01-01'}});
  const b=await second.json();expect(b.execution_id).not.toBe(a.execution_id);expect(b.prediction_mw).toEqual(a.prediction_mw);
  await page.goto('/live');
  await page.getByRole('combobox',{name:'Execution',exact:true}).selectOption('web');
  await page.getByRole('button',{name:'Run fresh inference'}).click();
  await expect(page.getByRole('status')).toContainText('Fresh inference complete');
  await expect(page.getByText('node_onnx_wasm_fp32',{exact:true})).toBeVisible();
  await expect(page.locator('.live-controls [role="alert"]')).toHaveCount(0);
});

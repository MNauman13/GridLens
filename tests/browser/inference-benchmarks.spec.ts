import {test,expect} from 'playwright/test';

test('I2 evidence switches timing scope and discloses accuracy and memory limits',async({page})=>{
  await page.goto('/experiments#inference');
  await expect(page.getByRole('heading',{name:'Same weights. Different execution.'})).toBeVisible();
  await expect(page.getByTestId('i2-scope')).toContainText('Includes preprocessing');
  await page.getByRole('button',{name:'Prepared tensors',exact:true}).click();
  await expect(page.getByTestId('i2-scope')).toContainText('excludes preprocessing');
  await expect(page.getByTestId('i2-transformer-change')).toContainText('SDPA median latency');
  await expect(page.getByTestId('i2-timexer-change')).toContainText('a speed improvement is not established');
  await expect(page.getByText('These are recorded Python CPU benchmarks.',{exact:false})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

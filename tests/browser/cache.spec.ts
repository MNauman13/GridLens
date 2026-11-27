import {test,expect} from 'playwright/test';
test('cache evidence scopes expose measured results',async({page})=>{
  await page.goto('/experiments#cache');
  const section=page.locator('#cache');
  await expect(section.getByRole('heading',{name:'Stop recomputing earlier tokens.'})).toBeVisible();
  await expect(section.getByRole('cell',{name:'22.073',exact:true})).toBeVisible();
  await section.getByLabel('Measurement scope').selectOption('2');
  await expect(section.getByRole('cell',{name:'29.653',exact:true})).toBeVisible();
  await expect(section.getByText('200 measured calls per variant, warmups excluded')).toBeVisible();
  await section.getByText('Latency across the 48 generated half-hours').click();
  await expect(section.getByRole('img')).toBeVisible();
});

import {test,expect} from 'playwright/test';
test('historical uncertainty replay respects feedback and restores position',async({page})=>{
  await page.goto('/replay');
  await expect(page.getByTestId('replay-clock')).toContainText('0 validation days released');
  await expect(page.getByTestId('replay-feedback')).toContainText('withheld');
  const previous=page.getByRole('button',{name:'Previous issue'});
  await expect(previous).toBeDisabled();
  await page.getByRole('button',{name:'Next issue'}).click();
  await expect(page.getByTestId('replay-clock')).toContainText('0 validation days released');
  await page.getByRole('button',{name:'Next issue'}).click();
  await expect(page.getByTestId('replay-clock')).toContainText('1 validation days released');
  await page.reload();
  await expect(page.getByTestId('replay-clock')).toContainText('1 validation days released');
  await page.getByRole('combobox',{name:'Interval method',exact:true}).selectOption('ncc');
  await expect(page.getByText(/NCC thresholds are recorded outputs/)).toBeVisible();
  await page.getByLabel('Replay position',{exact:true}).fill(await page.getByLabel('Replay position',{exact:true}).getAttribute('max')??'0');
  await expect(page.getByTestId('replay-feedback')).toContainText('now eligible');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download this forecast CSV'}).click();
  expect((await download).suggestedFilename()).toMatch(/gridlens-.*-ncc.csv/);
  await page.getByRole('button',{name:'Reset replay'}).click();
  await expect(previous).toBeDisabled();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});
test('results separate validation selection from the frozen test campaign',async({page,request})=>{
  const summary=await (await request.get('/replay/summary.json')).json();
  await page.goto('/results');
  await expect(page.getByRole('heading',{name:'Point forecasting study'})).toBeVisible();
  if(!summary.test){await expect(page.getByRole('button',{name:'2024 frozen test'})).toBeDisabled();return;}
  await page.getByRole('button',{name:'2024 frozen test'}).click();
  await expect(page.getByRole('caption').filter({hasText:'2024 test'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Frozen test campaign'})).toBeVisible();
});

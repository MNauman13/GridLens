import {test,expect} from 'playwright/test';
import AxeBuilder from '@axe-core/playwright';
for(const route of ['/','/live','/replay','/results','/learn/contract'])test(`accessible release page ${route}`,async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.goto(route);
  if(route==='/replay')await expect(page.getByTestId('replay-clock')).toBeVisible();
  if(route==='/results')await expect(page.getByRole('heading',{name:'Point forecasting study'})).toBeVisible();
  const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  expect(result.violations).toEqual([]);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

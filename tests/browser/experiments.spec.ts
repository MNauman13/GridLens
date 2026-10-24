import {test,expect} from "playwright/test";

test("measured comparison identifies the champion and discloses extrapolation failure",async({page})=>{
  await page.goto("/experiments");
  await expect(page.getByTestId("champion-row")).toContainText("patch 48");
  await expect(page.getByTestId("champion-row")).toContainText("1876.62");
  await expect(page.getByRole("heading",{name:"The original Transformer: a useful success and a clear failure"})).toBeVisible();
  await expect(page.getByText("Both bootstrap intervals include zero.",{exact:false})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

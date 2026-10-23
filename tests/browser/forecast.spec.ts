import { test, expect } from "playwright/test";

test("real saved baseline forecast exposes all 48 values and exact provenance boundaries",async({page})=>{
  const errors:string[]=[];
  page.on("pageerror",error=>errors.push(error.message));
  await page.goto("/forecasts");
  await expect(page.getByTestId("forecast-count")).toContainText("48 verified");
  await expect(page.getByTestId("forecast-issue")).toHaveText("2022-02-09T09:30:00Z");
  await expect(page.getByTestId("forecast-cutoff")).toHaveText("2022-02-09T00:00:00Z");
  await expect(page.getByTestId("forecast-target")).toHaveText("2022-02-10T00:00:00Z");
  await expect(page.getByRole("img",{name:"48 half-hour forecast values and observed National Demand in MW"})).toBeVisible();
  await page.getByLabel("Baseline model").selectOption("weekly");
  await expect(page.getByRole("heading",{name:"Weekly persistence",exact:true})).toBeVisible();
  const response=await page.request.get("/api/forecasts/baseline?model=weekly");
  const saved=await response.json();
  expect(saved.point.prediction_mw).toEqual(saved.inputs.demand_mw.slice(48,96));
  await page.getByText("Inspect provenance",{exact:true}).click();
  await expect(page.getByText(saved.forecast_id,{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
  await page.screenshot({path:`reports/local/browser-results/forecast-${test.info().project.name}.png`,fullPage:true});
});

test("forecast API outage displays a useful error",async({page})=>{
  await page.route("**/api/forecasts/baseline?**",route=>route.fulfill({status:503,body:"unavailable"}));
  await page.goto("/forecasts");
  await expect(page.getByRole("main").getByRole("alert")).toContainText("saved forecast is unavailable");
  await expect(page.getByTestId("forecast-count")).toHaveCount(0);
});

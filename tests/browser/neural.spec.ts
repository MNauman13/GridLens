import {test,expect} from "playwright/test";

test("saved neural forecast switches models and shows immutable provenance",async({page})=>{
  const errors:string[]=[];
  page.on("pageerror",error=>errors.push(error.message));
  await page.goto("/forecasts/neural");
  await expect(page.getByTestId("neural-slot-value")).toContainText("MW");
  await page.getByLabel("Neural model",{exact:true}).selectOption("transformer");
  await expect(page.getByTestId("neural-slot-value")).toContainText("MW");
  await page.getByLabel("Half-hour slot:").fill("47");
  await page.getByText("Forecast provenance",{exact:true}).click();
  await expect(page.getByText("Dataset manifest",{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
  await page.screenshot({path:"reports/local/browser-results/neural-"+test.info().project.name+".png",fullPage:true});
});

test("neural forecast outage is visible",async({page})=>{
  await page.route("**/api/forecasts/neural?*",route=>route.fulfill({status:503,json:{error:"unavailable"}}));
  await page.goto("/forecasts/neural");
  await expect(page.getByRole("alert")).toContainText("unavailable");
});

import {test,expect} from "playwright/test";
import fixture from "../../apps/web/data/attention-demo.json";

test("attention controls match checked weights and causal masks",async({page})=>{
  const errors:string[]=[];
  page.on("pageerror",error=>errors.push(error.message));
  await page.goto("/learn/attention");
  await expect(page.getByTestId("weight-0")).toHaveText(fixture.scenarios.unmasked.weights[0][0][0].toFixed(6));
  await page.getByLabel("Block future keys").check();
  await expect(page.getByTestId("weight-0")).toHaveText("1.000000");
  await expect(page.getByTestId("weight-1")).toHaveText("0.000000");
  await expect(page.getByTestId("weight-2")).toHaveText("0.000000");
  await page.getByLabel("Block future keys").uncheck();
  await page.getByLabel("Attention head",{exact:true}).selectOption("1");
  await page.getByLabel("Query token",{exact:true}).selectOption("1");
  for(let k=0;k<3;k++)await expect(page.getByTestId("weight-"+k)).toHaveText(fixture.scenarios.unmasked.weights[1][1][k].toFixed(6));
  await expect(page.getByTestId("head-output")).toHaveText("Head output = ["+fixture.scenarios.unmasked.head_outputs[1][1].map(v=>v.toFixed(6)).join(", ")+"]");
  await page.getByRole("button",{name:"Play attention",exact:true}).click();
  await expect(page.getByTestId("attention-stage")).toHaveText("Divide by the square root of head width");
  await page.getByRole("button",{name:"Pause attention",exact:true}).click();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
test("position animation matches registered-buffer fixture",async({page})=>{
  await page.goto("/learn/attention");
  await expect(page.getByTestId("position-0")).toHaveText("0.000000");
  await expect(page.getByTestId("position-1")).toHaveText("1.000000");
  await page.getByRole("button",{name:"Play positions",exact:true}).click();
  await expect(page.getByTestId("position-0")).toHaveText(fixture.positions[1][0].toFixed(6));
  await page.getByRole("button",{name:"Pause positions",exact:true}).click();
  await page.screenshot({path:"reports/local/browser-results/attention-"+test.info().project.name+".png",fullPage:true});
});

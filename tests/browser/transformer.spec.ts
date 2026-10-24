import {test,expect} from "playwright/test";

test("decoder lesson distinguishes training from generated prefixes",async({page})=>{
  await page.goto("/learn/transformer");
  await expect(page.getByTestId("prefix-origin")).toHaveText("Previous true tokens");
  await page.getByLabel("Decoder mode").selectOption("inference");
  await expect(page.getByTestId("prefix-origin")).toHaveText("Previous generated tokens");
  await page.getByLabel("Output position:").fill("2");
  await expect(page.getByTestId("decoder-prefix")).toHaveText("BOS, 7, 4");
  await page.getByRole("button",{name:"Play operators",exact:true}).click();
  await expect(page.getByTestId("timexer-stage")).toHaveText("Mix patches and the global token");
  await page.getByRole("button",{name:"Pause operators",exact:true}).click();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

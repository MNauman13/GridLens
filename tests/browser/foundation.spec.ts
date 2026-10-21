import { test, expect } from "playwright/test";

test("real API validates the forecast contract and leap-day origin", async ({page}) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading",{level:1})).toContainText("what we know");
  await page.getByRole("button",{name:"Run contract check"}).click();
  await expect(page.getByRole("status")).toContainText("Contract verified");
  await expect(page.getByRole("status")).toContainText("2024-02-11T00:00:00+00:00");
  await expect(page.getByRole("status")).toContainText("336");
  await page.getByLabel("Issue date").fill("2024-02-28");
  await page.getByRole("button",{name:"Run contract check"}).click();
  await expect(page.getByRole("status")).toContainText("2024-02-29T00:00:00+00:00");
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({path:`reports/local/browser-results/foundation-${test.info().project.name}.png`,fullPage:true});
});

test("timeline controls reach delayed feedback and return", async ({page}) => {
  await page.goto("/");
  await expect(page.getByRole("button",{name:"Previous"})).toBeDisabled();
  await page.getByRole("button",{name:"Next",exact:true}).click();
  await expect(page.getByRole("heading",{name:"Respect the issue-day gap"})).toBeVisible();
  await page.getByRole("button",{name:"Next",exact:true}).click();
  await expect(page.getByRole("heading",{name:"Predict the target day"})).toBeVisible();
  await page.getByRole("button",{name:"Next",exact:true}).click();
  await expect(page.getByRole("heading",{name:"Wait for the actuals"})).toBeVisible();
  await expect(page.getByRole("button",{name:"Next",exact:true})).toBeDisabled();
  await page.getByRole("button",{name:"Previous"}).click();
  await expect(page.getByRole("heading",{name:"Predict the target day"})).toBeVisible();
});

test("API outage is visible and does not masquerade as verification", async ({page}) => {
  await page.route("**/api/fixture?**",route => route.fulfill({status:503,contentType:"application/json",body:'{"error":"unavailable"}'}));
  await page.goto("/");
  await page.getByRole("button",{name:"Run contract check"}).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("API is unavailable");
  await expect(page.getByRole("status")).not.toContainText("Contract verified");
});

test("settlement animation distinguishes repeated local hours and UTC", async ({page}) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/learn/settlement");
  await expect(page.getByTestId("period-count")).toContainText("46 settlement periods");
  await page.getByRole("button", {name:"Period 3",exact:true}).click();
  await expect(page.getByTestId("local-start")).toHaveText("2024-03-31T02:00:00+01:00");
  await expect(page.getByTestId("utc-start")).toHaveText("2024-03-31T01:00:00+00:00");
  await page.getByLabel("Day type").selectOption("autumn");
  await expect(page.getByTestId("period-count")).toContainText("50 settlement periods");
  await page.getByRole("button", {name:"Period 3",exact:true}).click();
  await expect(page.getByTestId("local-start")).toHaveText("2024-10-27T01:00:00+01:00");
  await page.getByRole("button", {name:"Period 5",exact:true}).click();
  await expect(page.getByTestId("local-start")).toHaveText("2024-10-27T01:00:00+00:00");
  await expect(page.getByTestId("utc-start")).toHaveText("2024-10-27T01:00:00+00:00");
  await page.getByRole("button", {name:"Play animation"}).click();
  await expect(page.getByRole("button", {name:"Period 6",exact:true})).toHaveAttribute("aria-pressed","true");
  await page.getByRole("button", {name:"Pause animation"}).click();
  await page.getByLabel("Day type").selectOption("ordinary");
  await expect(page.getByTestId("period-count")).toContainText("48 settlement periods");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
  await page.screenshot({path:`reports/local/browser-results/settlement-${test.info().project.name}.png`,fullPage:true});
});

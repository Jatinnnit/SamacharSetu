import {test,expect} from '@playwright/test';

test('reader, bookmarks, comparison, search, language filters and empty states',async({page})=>{
  await page.goto('/?demo=1');
  await expect(page.locator('.story-card')).toHaveCount(9);
  await expect(page.locator('.demo-notice')).toContainText('Fictional');
  await page.getByLabel('Save story',{exact:true}).first().click();
  await page.getByRole('link',{name:/^Saved/}).click();
  await expect(page.locator('.story-card')).toHaveCount(1);
  await page.reload();
  await expect(page.locator('.story-card')).toHaveCount(1);
  await page.getByLabel('Remove saved story',{exact:true}).click();
  await expect(page.getByText('A place for your next good read.')).toBeVisible();
  await page.getByRole('link',{name:'Home',exact:true}).click();
  await expect(page.locator('.story-card')).toHaveCount(9);
  await page.getByLabel('Compare story',{exact:true}).nth(0).click();
  await page.getByLabel('Compare story',{exact:true}).nth(0).click();
  await page.getByRole('link',{name:/^Compare/}).click();
  await expect(page.locator('.compare-grid .paper-panel')).toHaveCount(2);
  await expect(page.getByText('Character TF-IDF cosine similarity', {exact:false})).toBeVisible();
  await page.getByRole('link',{name:'Home',exact:true}).click();
  await page.getByLabel('News language',{exact:true}).selectOption('mr');
  await expect(page.locator('.story-card')).toHaveCount(3);
  await page.getByLabel('Search news',{exact:true}).fill('zzzzunmatched');
  await page.getByRole('button',{name:'Search',exact:true}).click();
  await expect(page.getByText('No stories match these filters.')).toBeVisible();
  await page.getByRole('button',{name:'Clear filters',exact:true}).click();
  await expect(page.locator('.story-card')).toHaveCount(9);
  await page.locator('.story-card h3').first().click();
  await expect(page.getByText('This story is fictional and undated.',{exact:false})).toBeVisible();
  await page.getByRole('link',{name:'Explore NLP',exact:true}).click();
  await expect(page.getByLabel('Text to analyse')).toContainText('fictional scenario');
});

test('lab runs real algorithms, shows evaluation and handles unsupported operations',async({page})=>{
  await page.goto('/lab');
  await page.getByRole('button',{name:'Analyse text',exact:true}).click();
  await expect(page.locator('.statistics')).toBeVisible();
  await expect(page.getByText('Detected',{exact:true})).toBeVisible();
  await expect(page.getByRole('cell',{name:'reading',exact:true}).first()).toBeVisible();
  await page.getByLabel('Analysis language',{exact:true}).selectOption('mr');
  await page.getByLabel('Text to analyse').fill('विद्यार्थी वाचनालयात पुस्तके वाचत आहेत आणि शिक्षक मदत करत आहेत.');
  await page.getByRole('button',{name:'Analyse text',exact:true}).click();
  await expect(page.getByText('POS, NER, stemming and lemmatization are available for English only',{exact:false}).first()).toBeVisible();
  await page.getByRole('tab',{name:'Word embeddings',exact:true}).click();
  await page.getByRole('button',{name:'Inspect vectors',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Neighbours of “india”'})).toBeVisible();
  await page.getByRole('tab',{name:'Advanced NLP',exact:true}).click();
  await page.getByRole('button',{name:'Calculate probabilities',exact:true}).click();
  await expect(page.getByText('perplexity',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Explore latent topics',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Component 1',exact:true})).toBeVisible();
  await page.getByRole('tab',{name:'Model evaluation',exact:true}).click();
  await expect(page.getByText('67,512',{exact:true})).toBeVisible();
  await expect(page.locator('.confusion')).toBeVisible();
  await page.getByRole('tab',{name:'Assignment map',exact:true}).click();
  await expect(page.locator('.mapping-table tbody tr')).toHaveCount(12);
});

test('mobile layout has no horizontal overflow and navigation stays usable',async({page})=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto('/?demo=1');
  await expect(page.locator('.story-card')).toHaveCount(9);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.getByRole('link',{name:'NLP Lab',exact:true}).click();
  await page.getByRole('button',{name:'Analyse text',exact:true}).click();
  await expect(page.locator('.statistics')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.getByRole('tab',{name:'Model evaluation',exact:true}).click();
  await expect(page.locator('.confusion')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
});

test('live provider failure is clear and never masquerades as live fixtures',async({page})=>{
  await page.route('**/api/news?**',route=>route.fulfill({json:{mode:'unavailable',articles:[],notice:'Live news could not be reached.'}}));
  await page.goto('/');
  await expect(page.getByText('The live connection is unavailable.')).toBeVisible();
  await expect(page.locator('.story-card')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Open offline examples',exact:true})).toBeVisible();
});

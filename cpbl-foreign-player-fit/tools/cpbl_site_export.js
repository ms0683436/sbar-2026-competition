// Amendment 7: export a stats.cpbl.com.tw 排行榜 PR表格 (原始數據) the way a person would in a browser
// (the site's agent guidance: use a browser for filters; never call /api/ directly).
// Usage: NODE_PATH=$(npm root -g) node tools/cpbl_site_export.js 投手|打者|球隊投手 <minPA 0|50|100> <out.xlsx> [二軍例行賽]
const { chromium } = require('playwright');
const role = process.argv[2], minpa = process.argv[3] || '0', out = process.argv[4];
async function pick(p, current, label) {
  await p.getByRole('combobox').filter({ hasText: new RegExp('^' + current + '$') }).first().click();
  await p.waitForTimeout(400);
  await p.getByRole('option', { name: label, exact: true }).first().click();
  await p.waitForTimeout(400);
}
(async () => {
  const b = await chromium.launch(); const ctx = await b.newContext({ acceptDownloads: true }); const p = await ctx.newPage();
  await p.goto('https://stats.cpbl.com.tw/rankings?tab=pr-table', { waitUntil: 'networkidle', timeout: 60000 });
  await p.getByRole('button', { name: '原始數據', exact: true }).click();
  if (role !== '打者') await pick(p, '打者', role);
  if (minpa !== "0") await pick(p, "0", minpa);
  if (process.argv[5]) await pick(p, "一軍例行賽", process.argv[5]);
  await p.getByRole('button', { name: '搜尋', exact: true }).click();
  await p.waitForLoadState('networkidle'); await p.waitForTimeout(2000);
  const combos = await p.getByRole('combobox').allInnerTexts();
  console.log('combos', JSON.stringify(combos.slice(0, 6)));
  const [dl] = await Promise.all([p.waitForEvent('download', { timeout: 60000 }),
                                  p.getByRole('button', { name: '匯出表單', exact: true }).click()]);
  await dl.saveAs(out); console.log('saved', out); await b.close();
})().catch(e => { console.error(e.message); process.exit(1); });

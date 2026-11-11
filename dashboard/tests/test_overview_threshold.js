const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

(async () => {
  const nodes = new Map();
  const handlers = {}, requests = [], timers = [];
  let threshold = 0.75;
  const O = {
    $: id => {
      if (!nodes.has(id)) nodes.set(id, {addEventListener() {}, parentElement: {}});
      return nodes.get(id);
    },
    config: {expected_rows: 10, metrics: {recall: 0.5}},
    get threshold() {return threshold;},
    api: url => new Promise(resolve => requests.push({url, resolve})),
    fmt: String, percent: String, decimal: String, esc: String,
    chart() {}, probability: String, label: String, error() {},
  };
  const context = {
    DataCo: O, window: {addEventListener: (name, fn) => handlers[name] = fn},
    setTimeout: fn => {timers.push(fn); return timers.length;}, clearTimeout() {},
    encodeURIComponent,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../frontend/js/overview.js'), 'utf8'), context);
  const initial = handlers.datacoready();
  threshold = 0.9;
  handlers.thresholdchange();
  requests[0].resolve({});
  await initial;
  assert.equal(O.$('total').textContent, undefined, 'Stale response must not overwrite threshold state');
  const latest = timers[0]();
  assert.equal(requests[1].url, '/overview?t=0.9');
  const group = {available: true, rows: [{name: 'A', rate: 0.2}]};
  requests[1].resolve({threshold: 0.9, total: 10, predicted_late_rate: 0.2, high_risk_orders: 1,
    histogram: [], by_category: group, by_month: group, by_region: group, top_orders: []});
  await latest;
  assert.equal(O.$('total').textContent, '10');
  assert.equal(O.$('lateRate').textContent, '0.2');
  assert.equal(O.$('reloadOverview').disabled, false);
  console.log('Overview threshold response and stale-request checks passed.');
})().catch(error => {console.error(error);process.exitCode = 1;});

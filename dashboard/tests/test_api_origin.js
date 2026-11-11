// Run from dashboard/: node tests/test_api_origin.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../frontend/js/api.js'), 'utf8');
for (const [location, expected] of [
  [{hostname: 'dataco.example.com', port: '', protocol: 'https:'}, '/api'],
  [{hostname: 'dataco.example.com', port: '5500', protocol: 'https:'}, '/api'],
  [{hostname: 'localhost', port: '8000', protocol: 'http:'}, '/api'],
  [{hostname: '127.0.0.1', port: '9000', protocol: 'http:'}, '/api'],
  [{hostname: 'localhost', port: '5500', protocol: 'http:'}, 'http://localhost:8000/api'],
]) {
  const context = {location, window: {}, document: {addEventListener() {}}};
  vm.runInNewContext(source, context);
  assert.equal(context.window.DataCo.base, expected);
}
console.log('API origin checks passed (HTTPS hosting and local previews).');

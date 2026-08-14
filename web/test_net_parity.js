'use strict';
// Node has global atob since v16 (deprecated but present); polyfill just in case.
if (typeof atob === 'undefined') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}
const fs = require('fs');
const { HexNetJS } = require('./net.js');

const netJson = JSON.parse(fs.readFileSync(__dirname + '/../runs/az_hex/net.json', 'utf8'));
const cases = JSON.parse(fs.readFileSync(__dirname + '/parity_cases.json', 'utf8'));
const net = new HexNetJS(netJson);

let maxLogitErr = 0, maxValueErr = 0;
for (const [i, tc] of cases.entries()) {
  const planes = new Float32Array(tc.planes);
  const { logits, value } = net.forward(planes, tc.n);
  let le = 0;
  for (let k = 0; k < logits.length; k++) le = Math.max(le, Math.abs(logits[k] - tc.logits[k]));
  const ve = Math.abs(value - tc.value);
  maxLogitErr = Math.max(maxLogitErr, le);
  maxValueErr = Math.max(maxValueErr, ve);
  console.log(`case ${i} (n=${tc.n}): max logit err=${le.toExponential(2)}  value err=${ve.toExponential(2)}  (py=${tc.value.toFixed(4)} js=${value.toFixed(4)})`);
  if (le > 1e-2 || ve > 1e-2) throw new Error(`case ${i} FAILED tolerance`);
}
console.log(`\nall ${cases.length} parity cases passed (max logit err ${maxLogitErr.toExponential(2)}, max value err ${maxValueErr.toExponential(2)})`);

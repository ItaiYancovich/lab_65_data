#!/usr/bin/env python3
"""Assemble the self-contained Hex Lab artifact from its pieces.

The typeof-guarded require()/module.exports blocks in each .js file are
no-ops in a browser (typeof require/module is 'undefined' there), so the
Node-testable sources can be embedded verbatim -- what was tested is what
ships.
"""
import pathlib

here = pathlib.Path(__file__).parent

def read(name):
    return (here / name).read_text()

template = read('index_template.html')
net_json = (here.parent / 'runs' / 'az_hex' / 'net.json').read_text()

out = template
out = out.replace('__CSS__', read('app.css'))
out = out.replace('__HEX_JS__', read('hex.js'))
out = out.replace('__HEURISTICS_JS__', read('heuristics.js'))
out = out.replace('__NET_JS__', read('net.js'))
out = out.replace('__NET_JSON__', net_json)
out = out.replace('__AGENTS_JS__', read('agents.js'))
out = out.replace('__APP_JS__', read('app.js'))

out_path = here / 'hex_lab.html'
out_path.write_text(out)
print(f'wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)')

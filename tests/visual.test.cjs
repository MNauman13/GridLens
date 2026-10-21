// Exercise the actual embedded script without downloading browser dependencies.
// This verifies interaction logic, not browser layout or rendering.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../docs/learning/forecast-timeline.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mount(savedIndex) {
  const elements = new Map();
  for (const name of ['prev', 'next', 'marker', 'detail', 'step']) {
    assert.ok(html.includes('data-' + name), 'Markup must include ' + name);
    elements.set('[data-' + name + ']', {
      disabled: name === 'prev', textContent: '', style: {}, listeners: {},
      addEventListener(event, callback) { this.listeners[event] = callback; },
    });
  }
  const regions = [...html.matchAll(/data-region="([^"]+)"/g)].map(match => ({
    dataset: { region: match[1] }, active: false,
    classList: { toggle(name, on) { this.owner.active = on; } },
  }));
  regions.forEach(region => { region.classList.owner = region; });
  const root = {
    querySelector(selector) { assert.ok(elements.has(selector)); return elements.get(selector); },
    querySelectorAll(selector) { assert.equal(selector, '[data-region]'); return regions; },
  };
  const saved = [];
  const globals = {};
  const window = {
    openai: {
      widgetState: savedIndex === undefined ? null : { privateContent: { index: savedIndex } },
      setWidgetState(state) { saved.push(state); return Promise.resolve(); },
    },
    addEventListener(event, callback) { globals[event] = callback; },
  };
  vm.runInNewContext(source, {
    document: { getElementById(id) { assert.equal(id, 'gridlens-phase0-timeline'); return root; } },
    window,
  });
  return { elements, regions, saved, globals };
}

test('next reveals target then delayed actual release; controls stop at boundaries', () => {
  const { elements, regions, saved } = mount();
  const next = elements.get('[data-next]');
  next.listeners.click();
  assert.match(elements.get('[data-detail]').textContent, /observed demand remains excluded/);
  assert.equal(regions.find(x => x.active).dataset.region, 'gap');
  next.listeners.click();
  assert.match(elements.get('[data-detail]').textContent, /870 minutes/);
  assert.match(elements.get('[data-detail]').textContent, /not known yet/);
  next.listeners.click();
  assert.match(elements.get('[data-detail]').textContent, /Feb 12 at 09:00/);
  assert.equal(next.disabled, true);
  assert.equal(saved.length, 3);
});

test('saved state and previous control restore an earlier explanation', () => {
  const { elements, saved } = mount(3);
  assert.equal(elements.get('[data-step]').textContent, '4 / 4');
  elements.get('[data-prev]').listeners.click();
  assert.equal(elements.get('[data-step]').textContent, '3 / 4');
  assert.equal(elements.get('[data-next]').disabled, false);
  assert.equal(saved[0].privateContent.index, 2);
});

test('invalid saved state is ignored and host restoration is bounded', () => {
  const { elements, globals } = mount(99);
  assert.equal(elements.get('[data-prev]').disabled, true);
  globals['openai:set_globals']({ detail: { globals: { widgetState: { privateContent: { index: 1 } } } } });
  assert.equal(elements.get('[data-step]').textContent, '2 / 4');
});

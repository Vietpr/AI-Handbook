import assert from 'node:assert/strict';
import test from 'node:test';
import { pagePath, withBase } from '../src/lib/url.ts';
import { createSvgCache } from '../src/lib/svg-cache.ts';

test('page URLs have one trailing slash, before query and fragment', () => {
  for (const [input, expected] of [
    ['/blog', '/blog/'], ['/blog/', '/blog/'], ['/', '/'],
    ['/articles/a?lang=vi#tools', '/articles/a/?lang=vi#tools'],
    ['/blog#part?x', '/blog/#part?x'], ['#part', '#part'],
  ]) assert.equal(pagePath(input), expected);
  assert.equal(withBase('blog'), '/blog/');
  assert.equal(withBase(), '/');
});

test('assets and endpoints must not become directories', () => {
  for (const path of ['/search.json', '/favicon.svg', '/code/example.py?download=1#code'])
    assert.equal(pagePath(path), path);
});

test('SVG cache is least-recently-used and bounded by entry count', () => {
  const cache = createSvgCache(2);
  cache.set('a', '<svg>a</svg>');
  cache.set('b', '<svg>b</svg>');
  assert.equal(cache.get('a'), '<svg>a</svg>');
  cache.set('c', '<svg>c</svg>');
  assert.equal(cache.get('b'), undefined);
  assert.equal(cache.get('a'), '<svg>a</svg>');
  assert.equal(cache.get('c'), '<svg>c</svg>');
});

test('SVG cache bounds memory, including keys, and handles replacements', () => {
  const cache = createSvgCache(5, 10);
  cache.set('aa', '1234');
  cache.set('bb', '123');
  assert.equal(cache.get('aa'), undefined);
  cache.set('bb', '1');
  cache.set('cc', '1234');
  assert.equal(cache.get('bb'), '1');
  assert.equal(cache.get('cc'), '1234');
  cache.set('too-large', '1234567890');
  assert.equal(cache.get('too-large'), undefined);
  assert.equal(cache.get('bb'), '1');
});

test('different source/theme keys cannot reuse the same cached diagram', () => {
  const cache = createSvgCache();
  const light = JSON.stringify(['page', 0, 'A --> B', 'neutral', 'sans-serif']);
  const dark = JSON.stringify(['page', 0, 'A --> B', 'dark', 'sans-serif']);
  cache.set(light, '<svg>light</svg>');
  assert.equal(cache.get(dark), undefined);
});

test('a disabled cache does not retain SVGs', () => {
  const cache = createSvgCache(0);
  cache.set('a', 'svg');
  assert.equal(cache.get('a'), undefined);
});

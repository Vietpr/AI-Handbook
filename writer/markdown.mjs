/*
 * Frontmatter reading and writing.
 *
 * Parsing uses js-yaml — the same parser Astro itself uses for frontmatter —
 * so anything the writer accepts is read identically by the build. Writing goes
 * through a small canonical serializer instead, so generated files look exactly
 * like the hand-authored ones already in src/data/posts.
 */
import yaml from 'js-yaml';

export const FIELD_ORDER = [
  'title', 'description', 'domain', 'topic', 'type', 'section',
  'language', 'translationKey', 'order', 'pubDate', 'featured', 'draft',
];

const FENCE = /^---[ \t]*\r?\n/;

export function splitFrontmatter(input) {
  const text = input.replace(/^﻿/, '').replace(/\r\n/g, '\n');
  if (!FENCE.test(text)) return { raw: '', body: text.replace(/^\n+/, ''), hasFrontmatter: false };
  const rest = text.slice(text.indexOf('\n') + 1);
  const end = rest.search(/^---[ \t]*$/m);
  if (end === -1) return { raw: '', body: text, hasFrontmatter: false, error: 'Frontmatter is opened with --- but never closed.' };
  const raw = rest.slice(0, end);
  const after = rest.slice(end);
  const body = after.slice(after.indexOf('\n') + 1).replace(/^\n+/, '');
  return { raw, body, hasFrontmatter: true };
}

export function parseFrontmatter(input) {
  const split = splitFrontmatter(input);
  if (split.error) return { data: {}, body: split.body, hasFrontmatter: false, error: split.error };
  if (!split.hasFrontmatter) return { data: {}, body: split.body, hasFrontmatter: false };
  try {
    const data = yaml.load(split.raw, { schema: yaml.JSON_SCHEMA }) ?? {};
    if (typeof data !== 'object' || Array.isArray(data)) {
      return { data: {}, body: split.body, hasFrontmatter: true, error: 'Frontmatter is not a set of key / value pairs.' };
    }
    return { data, body: split.body, hasFrontmatter: true };
  } catch (error) {
    return { data: {}, body: split.body, hasFrontmatter: true, error: `Frontmatter is not valid YAML: ${error.reason || error.message}` };
  }
}

/* Writing ----------------------------------------------------------------- */

const quote = value => `"${String(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;

/* Plain (unquoted) is only safe for text YAML cannot misread as something else. */
const plainSafe = value =>
  /^[A-Za-z0-9][A-Za-z0-9 _.,/()+&'’–—-]*$/.test(value) &&
  value === value.trim() &&
  !/^(true|false|null|yes|no|on|off)$/i.test(value) &&
  !/^[-+]?[0-9]+(\.[0-9]+)?$/.test(value);

export function toDateString(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  /* Frontmatter dates are calendar days, so format from UTC parts and never
     let a local timezone shift the date by one. */
  const iso = date.toISOString();
  return iso.slice(0, 10);
}

function line(key, value) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value === 'boolean' || typeof value === 'number') return `${key}: ${value}`;
  if (key === 'pubDate') {
    const day = toDateString(value);
    return day ? `${key}: ${day}` : null;
  }
  if (Array.isArray(value)) {
    if (!value.length) return `${key}: []`;
    return [`${key}:`, ...value.map(item => `  - ${plainSafe(String(item)) ? item : quote(item)}`)].join('\n');
  }
  return `${key}: ${quote(value)}`;
}

export function serializePost(data, body) {
  const lines = [];
  for (const key of FIELD_ORDER) {
    const rendered = line(key, data[key]);
    if (rendered !== null) lines.push(rendered);
  }
  const text = String(body ?? '').replace(/\r\n/g, '\n').replace(/^\n+/, '').replace(/\s*$/, '');
  return `---\n${lines.join('\n')}\n---\n${text}\n`;
}

/* Helpers ----------------------------------------------------------------- */

export function slugify(input) {
  return String(input)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[đĐ]/g, 'd')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
    .replace(/-+$/, '');
}

export function firstHeading(body) {
  const match = String(body).match(/^#[ \t]+(.+?)[ \t]*#*$/m);
  return match ? match[1].trim() : '';
}

/* Drops the leading H1 when it was only there to carry the title. */
export function stripLeadingHeading(body) {
  return String(body).replace(/^\s*#[ \t]+.+?[ \t]*#*\n+/, '');
}

/* Every image reference in the body, so publishing never breaks one silently. */
export function scanImages(body) {
  const found = [];
  const markdown = /!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+["'][^"']*["'])?\s*\)/g;
  const html = /<img\b[^>]*\bsrc\s*=\s*["']([^"']+)["']/gi;
  for (const pattern of [markdown, html]) {
    let match;
    while ((match = pattern.exec(body)) !== null) found.push(match[1]);
  }
  return [...new Set(found)];
}

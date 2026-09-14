/*
 * Writer configuration.
 *
 * Everything secret lives in .env.writer (gitignored) or the real environment.
 * Nothing here is ever sent to the browser.
 */
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/* The writer imports src/*.ts directly (schema, taxonomy, site name), which
   needs Node's built-in type stripping. Fail loudly instead of throwing a
   confusing syntax error deep inside an import. */
const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 22 || (major === 22 && minor < 18)) {
  console.error(`The writer needs Node 22.18 or newer (running ${process.versions.node}).`);
  console.error('With nvm:  nvm install 22 && nvm use 22');
  process.exit(1);
}

export const ROOT = path.resolve(fileURLToPath(new URL('..', import.meta.url)));

export const PATHS = {
  root: ROOT,
  posts: path.join(ROOT, 'src', 'data', 'posts'),
  publicDir: path.join(ROOT, 'public'),
  images: path.join(ROOT, 'public', 'images'),
  site: path.join(ROOT, 'dist'),
  work: path.join(ROOT, '.writer'),
  buildOut: path.join(ROOT, '.writer', 'build'),
  previewPosts: path.join(ROOT, '.writer', 'preview', 'posts'),
  previewSite: path.join(ROOT, '.writer', 'preview', 'site'),
  ui: path.join(ROOT, 'writer', 'ui'),
  globalCss: path.join(ROOT, 'src', 'styles', 'global.css'),
  envFile: path.join(ROOT, '.env.writer'),
};

/* A deliberately small .env reader: KEY=value, # comments, optional quotes.
   Values already present in the real environment always win, so a host's
   secret manager can override the file. */
function loadEnvFile(file) {
  if (!existsSync(file)) return;
  for (const rawLine of readFileSync(file, 'utf8').split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = value;
  }
}
loadEnvFile(PATHS.envFile);

const flag = (value, fallback = false) =>
  value === undefined ? fallback : ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase());

export const CONFIG = {
  port: Number(process.env.WRITER_PORT || 4322),
  host: process.env.WRITER_HOST || '127.0.0.1',
  username: process.env.WRITER_USERNAME || '',
  passwordHash: process.env.WRITER_PASSWORD_HASH || '',
  sessionSecret: process.env.WRITER_SESSION_SECRET || '',
  sessionHours: Number(process.env.WRITER_SESSION_HOURS || 12),
  trustProxy: flag(process.env.WRITER_TRUST_PROXY),
  secureCookies: flag(process.env.WRITER_SECURE_COOKIES),
  /* Passed through to every build so publishes match how the site is served. */
  siteUrl: process.env.SITE_URL || '',
  basePath: process.env.BASE_PATH || '',
};

export function configProblems() {
  const problems = [];
  if (!CONFIG.username) problems.push('WRITER_USERNAME is not set.');
  if (!CONFIG.passwordHash) problems.push('WRITER_PASSWORD_HASH is not set.');
  if (!CONFIG.sessionSecret) problems.push('WRITER_SESSION_SECRET is not set.');
  else if (CONFIG.sessionSecret.length < 32) problems.push('WRITER_SESSION_SECRET is shorter than 32 characters.');
  return problems;
}

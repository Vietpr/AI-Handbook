/*
 * The local store: articles are files under src/data/posts, and a change is
 * only allowed to go live once `astro build` has accepted it.
 *
 * Building before swapping is what makes a publish safe. The new site is
 * assembled in a scratch directory; dist/ is only replaced once that succeeded,
 * so a change that would break the site is reported instead of shipped and the
 * previous site keeps serving throughout.
 *
 * Implements the contract documented in store.mjs.
 */
import { spawn } from 'node:child_process';
import { readdir, readFile, writeFile, rename, unlink, mkdir, copyFile, rm, stat, access } from 'node:fs/promises';
import { constants, existsSync } from 'node:fs';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import { CONFIG, PATHS } from './config.mjs';
import { slugProblem } from './rules.mjs';

const ASTRO_BIN = path.join(PATHS.root, 'node_modules', 'astro', 'bin', 'astro.mjs');
const PREVIEW_BASE = '/write/preview';
const ANSI = new RegExp(String.fromCharCode(27) + '\\[[0-9;]*m', 'g');

/* Paths ------------------------------------------------------------------- */

/* Never build a path from author input without checking where it landed. */
function postPath(slug) {
  if (slugProblem(slug)) throw new Error('Invalid slug.');
  const target = path.resolve(PATHS.posts, `${slug}.md`);
  if (path.dirname(target) !== path.resolve(PATHS.posts)) throw new Error('Invalid slug.');
  return target;
}

const isFile = file => access(file, constants.F_OK).then(() => true, () => false);

async function writeAtomic(target, contents) {
  const temporary = path.join(path.dirname(target), `.${path.basename(target)}.${randomBytes(6).toString('hex')}.tmp`);
  await writeFile(temporary, contents, 'utf8');
  await rename(temporary, target);
}

const removeIfPresent = async slug => {
  try {
    await unlink(postPath(slug));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
};

/* Builds ------------------------------------------------------------------- */

/* One build at a time: they share .astro/ and the output directories. */
let queue = Promise.resolve();
let running = null;

function serialize(label, task) {
  const result = queue.then(async () => {
    running = label;
    try {
      return await task();
    } finally {
      running = null;
    }
  });
  queue = result.then(() => undefined, () => undefined);
  return result;
}

function childEnv(overrides) {
  const env = { ...process.env, FORCE_COLOR: '0', NODE_ENV: 'production' };
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined || value === '') delete env[key];
    else env[key] = value;
  }
  return env;
}

function runAstro(args, overrides) {
  return new Promise(resolve => {
    const child = spawn(process.execPath, [ASTRO_BIN, ...args], {
      cwd: PATHS.root,
      env: childEnv(overrides),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';
    const collect = chunk => {
      output += chunk;
      if (output.length > 200000) output = output.slice(-200000);
    };
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);
    child.on('error', error => resolve({ ok: false, log: `Could not start Astro: ${error.message}` }));
    child.on('close', code => {
      const log = output.replace(ANSI, '').trim();
      resolve({ ok: code === 0, code, log: log.length > 12000 ? `...\n${log.slice(-12000)}` : log });
    });
  });
}

async function swapInto(target, source) {
  const retired = `${target}.retired`;
  await rm(retired, { recursive: true, force: true });
  const hadTarget = existsSync(target);
  if (hadTarget) await rename(target, retired);
  try {
    await rename(source, target);
  } catch (error) {
    if (hadTarget) await rename(retired, target);
    throw error;
  }
  await rm(retired, { recursive: true, force: true });
}

function buildSite() {
  return serialize('publish', async () => {
    await mkdir(PATHS.work, { recursive: true });
    await rm(PATHS.buildOut, { recursive: true, force: true });
    /* --force clears the content-layer cache, so a preview build can never
       leave a stale entry behind in the published output. */
    const result = await runAstro(
      ['build', '--force', '--outDir', path.relative(PATHS.root, PATHS.buildOut)],
      { WRITER_POSTS_DIR: undefined, SITE_URL: CONFIG.siteUrl, BASE_PATH: CONFIG.basePath },
    );
    if (!result.ok) return result;
    await swapInto(PATHS.site, PATHS.buildOut);
    return result;
  });
}

/* Reading ------------------------------------------------------------------ */

async function listSlugs() {
  const entries = await readdir(PATHS.posts, { withFileTypes: true });
  return entries
    .filter(entry => entry.isFile() && entry.name.endsWith('.md'))
    .map(entry => entry.name.slice(0, -3))
    .sort();
}

const exists = slug => isFile(postPath(slug));

async function read(slug) {
  try {
    return await readFile(postPath(slug), 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

/* Writing ------------------------------------------------------------------ */

async function commit({ writes = [], deletes = [], publish = false }) {
  /* Remember exactly what was there so a failed build can be undone. */
  const previous = new Map();
  for (const { slug } of writes) if (!previous.has(slug)) previous.set(slug, await read(slug));
  for (const slug of deletes) if (!previous.has(slug)) previous.set(slug, await read(slug));

  const apply = async () => {
    for (const { slug, contents } of writes) await writeAtomic(postPath(slug), contents);
    for (const slug of deletes) await removeIfPresent(slug);
  };

  const undo = async () => {
    for (const [slug, contents] of previous) {
      if (contents === null) await removeIfPresent(slug);
      else await writeAtomic(postPath(slug), contents);
    }
  };

  await apply();
  if (!publish) return { ok: true, log: '', published: false };

  const build = await buildSite();
  if (build.ok) return { ok: true, log: build.log, published: true };

  /* The build never reached the swap, so dist/ still holds the previous site.
     Restoring the files is enough to put content and site back in step. */
  await undo();
  return { ok: false, log: build.log, published: false };
}

const republish = () => buildSite();

async function archive(slug, contents) {
  const trash = path.join(PATHS.work, 'trash');
  await mkdir(trash, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const target = path.join(trash, `${slug}.${stamp}.md`);
  await writeFile(target, contents, 'utf8');
  return path.relative(PATHS.root, target);
}

/* Preview ------------------------------------------------------------------ */

async function stageDraft(slug, contents) {
  await rm(PATHS.previewPosts, { recursive: true, force: true });
  await mkdir(PATHS.previewPosts, { recursive: true });

  const entries = await readdir(PATHS.posts, { withFileTypes: true });
  await Promise.all(entries
    .filter(entry => entry.isFile() && entry.name.endsWith('.md') && entry.name !== `${slug}.md`)
    .map(entry => copyFile(path.join(PATHS.posts, entry.name), path.join(PATHS.previewPosts, entry.name))));

  await writeFile(path.join(PATHS.previewPosts, `${slug}.md`), contents, 'utf8');
}

function preview(slug, contents) {
  return serialize('preview', async () => {
    await stageDraft(slug, contents);
    await rm(PATHS.previewSite, { recursive: true, force: true });
    const result = await runAstro(
      ['build', '--outDir', path.relative(PATHS.root, PATHS.previewSite)],
      { WRITER_POSTS_DIR: PATHS.previewPosts, BASE_PATH: PREVIEW_BASE, SITE_URL: CONFIG.siteUrl },
    );
    return { ...result, url: `${PREVIEW_BASE}/articles/${slug}/` };
  });
}

/* Assets ------------------------------------------------------------------- */

async function listAssets() {
  await mkdir(PATHS.images, { recursive: true });
  const entries = await readdir(PATHS.images, { withFileTypes: true });
  const assets = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const info = await stat(path.join(PATHS.images, entry.name));
    assets.push({ name: entry.name, url: `/images/${entry.name}`, size: info.size });
  }
  return assets.sort((a, b) => a.name.localeCompare(b.name));
}

/* "/images/x.png" is relative to the site root; anything else is relative to
   the article, which is how Astro resolves it during the build. */
async function assetExists(reference) {
  const clean = String(reference).split(/[?#]/)[0];
  if (!clean) return false;
  const target = clean.startsWith('/')
    ? path.resolve(PATHS.publicDir, `.${clean}`)
    : path.resolve(PATHS.posts, clean);
  const roots = [path.resolve(PATHS.publicDir), path.resolve(PATHS.posts)];
  if (!roots.some(root => target.startsWith(root + path.sep))) return false;
  return isFile(target);
}

async function writeAsset(name, bytes) {
  await mkdir(PATHS.images, { recursive: true });
  const target = path.resolve(PATHS.images, name);
  if (path.dirname(target) !== path.resolve(PATHS.images)) throw new Error('Invalid file name.');
  try {
    await writeFile(target, bytes, { flag: 'wx' });
    return true;
  } catch (error) {
    if (error.code === 'EEXIST') return false;
    throw error;
  }
}

/* -------------------------------------------------------------------------- */

export default {
  name: 'filesystem',
  listSlugs,
  exists,
  read,
  commit,
  republish,
  archive,
  preview,
  busy: () => running,
  listAssets,
  assetExists,
  writeAsset,
  deployment: () => ({ basePath: CONFIG.basePath || '/', siteUrl: CONFIG.siteUrl || '' }),
};

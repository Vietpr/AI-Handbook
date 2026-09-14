/*
 * One article, up to two languages.
 *
 * The editor works on a "pair": metadata shared by both versions, plus a
 * title, description, slug and body per language. On disk that is one or two
 * ordinary Markdown files joined by `translationKey`; nothing else on the site
 * knows the pair existed. Saving a pair is one store commit, so either both
 * versions go live or neither does.
 */
import * as content from './content.mjs';
import { store } from './store.mjs';
import { serializePost, toDateString } from './markdown.mjs';

const LANGS = ['en', 'vi'];
const SHARED = ['section', 'domain', 'topic', 'type', 'order', 'pubDate', 'featured'];
const PER_VERSION = ['title', 'description'];

class PairError extends Error {
  constructor(status, message, extra = {}) {
    super(message);
    this.status = status;
    this.extra = extra;
  }
}
const fail = (status, message, extra) => {
  throw new PairError(status, message, extra);
};

const text = value => (typeof value === 'string' ? value.trim() : '');

/* Reading ------------------------------------------------------------------ */

function versionOf(post) {
  return {
    slug: post.slug,
    title: String(post.data.title ?? ''),
    description: String(post.data.description ?? ''),
    body: post.body,
    draft: post.data.draft === true,
    parseError: post.parseError,
  };
}

function sharedOf(post) {
  const data = post.data;
  return {
    section: data.section ?? '',
    domain: data.domain ?? '',
    topic: data.topic ?? '',
    type: data.type ?? '',
    order: data.order ?? '',
    pubDate: toDateString(data.pubDate) || '',
    featured: data.featured === true,
  };
}

/* The other language of `post`, if a file with the same key exists. */
async function siblingOf(post) {
  const key = post.data.translationKey;
  if (!key) return null;
  const summaries = await content.listPosts();
  const match = summaries.find(item =>
    item.slug !== post.slug && item.translationKey === key && item.language !== post.data.language);
  return match ? content.readPost(match.slug) : null;
}

export async function getPair(slug) {
  if (content.slugProblem(slug)) fail(400, 'Invalid slug.');
  const post = await content.readPost(slug);
  if (!post) fail(404, `No article named "${slug}".`);
  const sibling = await siblingOf(post);

  const versions = {};
  versions[post.data.language ?? 'en'] = versionOf(post);
  if (sibling) versions[sibling.data.language] = versionOf(sibling);

  return {
    ok: true,
    key: post.data.translationKey ?? null,
    shared: sharedOf(post),
    versions,
  };
}

/* Validation --------------------------------------------------------------- */

/*
 * Turns the editor's pair into one frontmatter object per present language.
 * A version is present when the author put anything into it; an untouched tab
 * is simply not an article yet.
 */
function versionsIn(payload) {
  const out = {};
  for (const lang of LANGS) {
    const raw = payload.versions?.[lang];
    if (!raw || typeof raw !== 'object') continue;
    const version = {
      slug: text(raw.slug),
      currentSlug: text(raw.currentSlug) || null,
      title: text(raw.title),
      description: text(raw.description),
      body: typeof raw.body === 'string' ? raw.body : '',
      hidden: raw.hidden === true,
    };
    const present = version.slug || version.title || version.description || version.body.trim();
    if (present || version.currentSlug) out[lang] = { ...version, present: Boolean(present) };
  }
  return out;
}

async function check(payload, { forSave }) {
  const shared = payload.shared && typeof payload.shared === 'object' ? payload.shared : {};
  const versions = versionsIn(payload);
  const langs = Object.keys(versions);
  const fields = [];
  const publish = payload.publish === true;

  if (!langs.length) fields.push({ field: 'en.title', message: 'Write the article in at least one language.' });

  /* An existing version whose tab was wiped is a removal, and removals are
     explicit — otherwise a stray select-all + delete would unpublish a page. */
  for (const lang of langs) {
    const version = versions[lang];
    if (version.currentSlug && !version.present) {
      fields.push({ field: `${lang}.title`, message: 'This version is empty. Use "Remove this version" if you want it gone.' });
    }
  }

  const currentSlugs = langs.map(lang => versions[lang].currentSlug).filter(Boolean);
  const existingKeys = new Set();
  for (const slug of currentSlugs) {
    const post = await content.readPost(slug);
    if (post?.data.translationKey) existingKeys.add(post.data.translationKey);
  }
  /* Keep the key the files already carry; mint one only when a second
     language first appears. */
  let key = text(payload.key) || [...existingKeys][0] || null;
  if (!key && langs.length > 1) key = versions.en?.slug || versions.vi?.slug || null;

  const summaries = await content.listPosts();
  const frontmatters = {};
  const warnings = [];

  for (const lang of langs) {
    const version = versions[lang];
    if (!version.present) continue;

    const slugIssue = content.slugProblem(version.slug);
    if (slugIssue) fields.push({ field: `${lang}.slug`, message: slugIssue });
    else {
      const taken = summaries.find(post => post.slug === version.slug && !currentSlugs.includes(post.slug));
      if (taken) fields.push({ field: `${lang}.slug`, message: `An article with the slug "${version.slug}" already exists.` });
    }

    const candidate = {
      ...shared,
      title: version.title,
      description: version.description,
      language: lang,
      ...(key ? { translationKey: key } : {}),
      draft: publish ? version.hidden : true,
    };
    const validation = content.validateFrontmatter(candidate);
    for (const issue of validation.errors) {
      fields.push({ field: PER_VERSION.includes(issue.field) ? `${lang}.${issue.field}` : issue.field, message: issue.message });
    }
    if (validation.valid) {
      frontmatters[lang] = validation.data;
      warnings.push(...(await content.collectWarnings(validation.data, version.body, {
        slug: version.slug, currentSlug: version.currentSlug, key,
      })).map(note => (langs.length > 1 ? `${lang.toUpperCase()}: ${note}` : note)));
    }
  }

  if (versions.en?.present && versions.vi?.present && versions.en.slug && versions.en.slug === versions.vi.slug) {
    fields.push({ field: 'vi.slug', message: 'The two languages need different slugs.' });
  }

  /* One file per language per key, or the site could not tell them apart. */
  if (key) {
    for (const lang of langs) {
      const other = summaries.find(post =>
        post.translationKey === key && post.language === lang && !currentSlugs.includes(post.slug));
      if (other) fields.push({ field: `${lang}.slug`, message: `"${other.title}" is already the ${lang.toUpperCase()} version of this article.` });
    }
  }

  /* Shared-field errors come back once, not once per language. */
  const seen = new Set();
  const unique = fields.filter(issue => !seen.has(`${issue.field}|${issue.message}`) && seen.add(`${issue.field}|${issue.message}`));

  if (forSave && unique.length) fail(422, 'Some metadata is not valid yet.', { fields: unique });
  return { key, versions, frontmatters, fields: unique, warnings: [...new Set(warnings)] };
}

export async function validatePair(payload) {
  const result = await check(payload, { forSave: false });
  return { ok: true, valid: result.fields.length === 0, fields: result.fields, warnings: result.warnings };
}

/* Saving ------------------------------------------------------------------- */

export async function savePair(payload) {
  const { key, versions, frontmatters, warnings } = await check(payload, { forSave: true });
  const langs = Object.keys(frontmatters);

  const writes = [];
  const deletes = [];
  let wasPublic = false;
  let willBePublic = false;

  for (const lang of langs) {
    const version = versions[lang];
    const data = frontmatters[lang];
    writes.push({ slug: version.slug, contents: serializePost(data, version.body) });
    if (version.currentSlug && version.currentSlug !== version.slug) deletes.push(version.currentSlug);
    if (data.draft !== true) willBePublic = true;
    if (version.currentSlug) {
      const previous = await content.readPost(version.currentSlug);
      if (previous && previous.data.draft !== true) wasPublic = true;
    }
  }

  const result = await store.commit({
    writes,
    deletes,
    publish: wasPublic || willBePublic,
    message: `${langs.some(lang => versions[lang].currentSlug) ? 'Update' : 'Add'} ${langs.map(lang => versions[lang].slug).join(' + ')}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so nothing was changed.', { buildLog: result.log });

  const saved = {};
  for (const lang of langs) {
    saved[lang] = { slug: versions[lang].slug, draft: frontmatters[lang].draft === true };
    if (versions[lang].currentSlug && versions[lang].currentSlug !== versions[lang].slug && wasPublic) {
      warnings.push(`${lang.toUpperCase()}: the old address /articles/${versions[lang].currentSlug} no longer exists.`);
    }
  }

  return { ok: true, key, versions: saved, built: result.published, warnings };
}

/* Whole-pair actions from the dashboard -------------------------------------- */

async function loadSlugs(slugs) {
  if (!Array.isArray(slugs) || !slugs.length || slugs.length > 2) fail(400, 'Expected one or two slugs.');
  const posts = [];
  for (const slug of slugs) {
    if (content.slugProblem(slug)) fail(400, 'Invalid slug.');
    const post = await content.readPost(slug);
    if (!post) fail(404, `No article named "${slug}".`);
    posts.push(post);
  }
  return posts;
}

export async function setPairState(payload) {
  const posts = await loadSlugs(payload.slugs);
  const draft = payload.draft === true;
  const writes = [];
  for (const post of posts) {
    const validation = content.validateFrontmatter({ ...post.data, draft });
    if (!validation.valid) fail(422, `"${post.slug}" has invalid metadata, so its state cannot be changed.`, { fields: validation.errors });
    writes.push({ slug: post.slug, contents: serializePost(validation.data, post.body) });
  }
  const wasPublic = posts.some(post => post.data.draft !== true);
  const result = await store.commit({
    writes,
    publish: wasPublic || !draft,
    message: `${draft ? 'Unpublish' : 'Publish'} ${posts.map(post => post.slug).join(' + ')}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so the change was rolled back.', { buildLog: result.log });
  return { ok: true, draft, built: result.published };
}

export async function deletePair(payload) {
  const posts = await loadSlugs(payload.slugs);
  const expected = posts.map(post => post.slug).sort().join(' ');
  const given = String(payload.confirm || '').trim().split(/\s+/).filter(Boolean).sort().join(' ');
  if (given !== expected) fail(400, 'Type the slug of every version to confirm the deletion.');

  const backups = [];
  for (const post of posts) backups.push(await store.archive(post.slug, post.raw));

  const result = await store.commit({
    deletes: posts.map(post => post.slug),
    publish: posts.some(post => post.data.draft !== true),
    message: `Delete ${posts.map(post => post.slug).join(' + ')}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so nothing was deleted.', { buildLog: result.log });
  return { ok: true, built: result.published, backups };
}

/*
 * The writer's JSON API.
 *
 * Every handler here runs behind requireSession() in server.mjs except login
 * and session, so hiding the UI is never what protects an action. Mutations
 * also require an X-Writer-Request header, which a cross-site form cannot set.
 *
 * Persistence goes through the store (see store.mjs) — this file never touches
 * a filesystem, a build or a repository directly.
 */
import { SITE_NAME } from './site-info.mjs';
import * as content from './content.mjs';
import { store } from './store.mjs';
import { LIMITS, IMAGE_EXTENSIONS } from './rules.mjs';
import {
  parseFrontmatter, serializePost, slugify, firstHeading, stripLeadingHeading, toDateString,
} from './markdown.mjs';

const today = () => new Date().toISOString().slice(0, 10);

/* Filenames arrive from a browser, so treat them as text rather than paths. */
const fileName = value => String(value).split(/[/\\]/).pop() || '';
const extensionOf = value => {
  const dot = fileName(value).lastIndexOf('.');
  return dot > 0 ? fileName(value).slice(dot).toLowerCase() : '';
};
const stemOf = value => {
  const name = fileName(value);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
};

class ApiError extends Error {
  constructor(status, message, extra = {}) {
    super(message);
    this.status = status;
    this.extra = extra;
  }
}

const fail = (status, message, extra) => {
  throw new ApiError(status, message, extra);
};

export { ApiError };

/* Shared save path -------------------------------------------------------- */

async function prepare(payload, { currentSlug = null } = {}) {
  const slug = String(payload.slug || '').trim();
  const slugIssue = content.slugProblem(slug);
  if (slugIssue) fail(400, slugIssue, { field: 'slug' });

  const body = typeof payload.body === 'string' ? payload.body : '';
  const frontmatter = { ...(payload.frontmatter || {}) };
  if (typeof payload.publish === 'boolean') frontmatter.draft = !payload.publish;

  const validation = content.validateFrontmatter(frontmatter);
  if (!validation.valid) fail(422, 'Some metadata is not valid yet.', { fields: validation.errors });

  const warnings = await content.collectWarnings(validation.data, body, { slug, currentSlug });
  return { slug, body, data: validation.data, warnings };
}

/* Handlers ---------------------------------------------------------------- */

export async function meta() {
  const deployment = store.deployment();
  return {
    ok: true,
    site: { name: SITE_NAME, basePath: deployment.basePath, siteUrl: deployment.siteUrl },
    taxonomy: await content.taxonomy(),
    today: today(),
  };
}

export async function listPosts() {
  return { ok: true, posts: await content.listPosts(), building: store.busy() };
}

export async function getPost(slug) {
  if (content.slugProblem(slug)) fail(400, 'Invalid slug.');
  const post = await content.readPost(slug);
  if (!post) fail(404, `No article named "${slug}".`);
  return {
    ok: true,
    slug,
    frontmatter: {
      ...post.data,
      pubDate: toDateString(post.data.pubDate) || '',
    },
    body: post.body,
    parseError: post.parseError,
  };
}

export async function createPost(payload) {
  const { slug, body, data, warnings } = await prepare(payload);
  if (await store.exists(slug)) fail(409, `An article with the slug "${slug}" already exists.`, { field: 'slug' });

  const result = await store.commit({
    writes: [{ slug, contents: serializePost(data, body) }],
    publish: data.draft !== true,
    message: `Add ${slug}`,
  });
  if (!result.ok) {
    fail(422, 'The site build failed, so nothing was published. The article was not saved.', { buildLog: result.log });
  }

  return { ok: true, slug, warnings, built: result.published, draft: data.draft === true };
}

export async function updatePost(currentSlug, payload) {
  if (content.slugProblem(currentSlug)) fail(400, 'Invalid slug.');
  const previous = await content.readPost(currentSlug);
  if (!previous) fail(404, `No article named "${currentSlug}".`);
  const wasPublic = previous.data.draft !== true;

  const { slug, body, data, warnings } = await prepare(payload, { currentSlug });
  const renamed = slug !== currentSlug;
  if (renamed && (await store.exists(slug))) {
    fail(409, `An article with the slug "${slug}" already exists.`, { field: 'slug' });
  }

  const result = await store.commit({
    writes: [{ slug, contents: serializePost(data, body) }],
    deletes: renamed ? [currentSlug] : [],
    publish: wasPublic || data.draft !== true,
    message: renamed ? `Rename ${currentSlug} to ${slug}` : `Update ${slug}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so the change was rolled back.', { buildLog: result.log });

  if (renamed && wasPublic) {
    warnings.push(`The old address /articles/${currentSlug} no longer exists — update any links pointing at it.`);
  }

  return { ok: true, slug, renamed, warnings, built: result.published, draft: data.draft === true };
}

export async function setPostState(slug, payload) {
  if (content.slugProblem(slug)) fail(400, 'Invalid slug.');
  const post = await content.readPost(slug);
  if (!post) fail(404, `No article named "${slug}".`);

  const draft = payload.draft === true;
  const validation = content.validateFrontmatter({ ...post.data, draft });
  if (!validation.valid) {
    fail(422, 'This article has invalid metadata, so its state cannot be changed.', { fields: validation.errors });
  }

  const result = await store.commit({
    writes: [{ slug, contents: serializePost(validation.data, post.body) }],
    publish: post.data.draft !== true || !draft,
    message: draft ? `Unpublish ${slug}` : `Publish ${slug}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so the change was rolled back.', { buildLog: result.log });

  return { ok: true, slug, draft, built: result.published };
}

export async function deletePost(slug, payload) {
  if (content.slugProblem(slug)) fail(400, 'Invalid slug.');
  if (payload.confirm !== slug) fail(400, 'Type the slug to confirm the deletion.');
  const post = await content.readPost(slug);
  if (!post) fail(404, `No article named "${slug}".`);

  /* Deleting is meant to be final on the site, but a copy costs nothing and
     turns a misclick into an inconvenience instead of a loss. */
  const backup = await store.archive(slug, post.raw);

  const result = await store.commit({
    deletes: [slug],
    publish: post.data.draft !== true,
    message: `Delete ${slug}`,
  });
  if (!result.ok) fail(422, 'The site build failed, so the article was restored.', { buildLog: result.log });

  return { ok: true, slug, built: result.published, backup };
}

export async function validate(payload) {
  const slug = String(payload.slug || '').trim();
  const body = typeof payload.body === 'string' ? payload.body : '';
  const currentSlug = payload.currentSlug ? String(payload.currentSlug) : null;

  const fields = [];
  const slugIssue = content.slugProblem(slug);
  if (slugIssue) fields.push({ field: 'slug', message: slugIssue });
  else if (slug !== currentSlug && (await store.exists(slug))) {
    fields.push({ field: 'slug', message: `An article with the slug "${slug}" already exists.` });
  }

  const validation = content.validateFrontmatter(payload.frontmatter || {});
  fields.push(...validation.errors);

  const warnings = validation.valid ? await content.collectWarnings(validation.data, body, { slug, currentSlug }) : [];
  return { ok: true, valid: fields.length === 0, fields, warnings };
}

export async function preview(payload) {
  const slug = content.slugProblem(payload.slug) ? 'draft-preview' : String(payload.slug);
  const body = typeof payload.body === 'string' ? payload.body : '';
  const validation = content.validateFrontmatter(payload.frontmatter || {});
  if (!validation.valid) fail(422, 'Fill in the required metadata before previewing.', { fields: validation.errors });

  /* A preview always renders, even while the article is still a draft. */
  const contents = serializePost({ ...validation.data, draft: false }, body);
  const result = await store.preview(slug, contents);
  if (!result.ok) fail(422, 'The preview could not be rendered.', { buildLog: result.log });
  return { ok: true, url: result.url };
}

export async function rebuild() {
  const result = await store.republish();
  if (!result.ok) fail(422, 'The site build failed.', { buildLog: result.log });
  return { ok: true, built: true, log: result.log };
}

/* Import ------------------------------------------------------------------ */

const IMPORTABLE = new Set(['.md', '.markdown', '.txt']);

export async function importMarkdown(payload) {
  const filename = String(payload.filename || 'article.md');
  const text = typeof payload.text === 'string' ? payload.text : '';
  if (!text.trim()) fail(400, 'The file is empty.');
  if (new TextEncoder().encode(text).length > LIMITS.markdownBytes) fail(413, 'The file is larger than 1 MB.');
  if (!IMPORTABLE.has(extensionOf(filename))) {
    fail(415, 'Only .md, .markdown and .txt files can be imported.');
  }

  const parsed = parseFrontmatter(text);
  const notes = [];
  if (parsed.error) notes.push(parsed.error);

  const source = parsed.data || {};
  const frontmatter = {};
  const filled = [];
  const missing = [];

  /* Only copy fields the schema knows and the value actually fits. */
  const taxonomy = await content.taxonomy();
  const pick = (key, allowed) => {
    const value = source[key];
    if (value === undefined || value === null || value === '') return false;
    const trimmed = String(value).trim();
    if (allowed && !allowed.includes(trimmed)) {
      notes.push(`Ignored ${key}: "${trimmed}" is not one of ${allowed.join(', ')}.`);
      return false;
    }
    frontmatter[key] = trimmed;
    filled.push(key);
    return true;
  };

  pick('title');
  pick('description');
  pick('topic');
  pick('translationKey');
  pick('domain', taxonomy.domains);
  pick('type', taxonomy.types);
  pick('section', taxonomy.sections);
  pick('language', taxonomy.languages);

  for (const key of ['pubDate']) {
    const day = source[key] === undefined ? null : toDateString(source[key]);
    if (day) {
      frontmatter[key] = day;
      filled.push(key);
    } else if (source[key] !== undefined) {
      notes.push(`Ignored ${key}: "${source[key]}" is not a date.`);
    }
  }

  for (const key of ['order']) {
    const value = Number(source[key]);
    if (Number.isInteger(value) && value > 0) {
      frontmatter[key] = value;
      filled.push(key);
    } else if (source[key] !== undefined) {
      notes.push(`Ignored ${key}: "${source[key]}" is not a positive whole number.`);
    }
  }

  for (const key of ['featured', 'draft']) {
    if (typeof source[key] === 'boolean') {
      frontmatter[key] = source[key];
      filled.push(key);
    }
  }

  /* Safe to infer: an opening H1 that is clearly the article title. */
  let body = parsed.body;
  if (!frontmatter.title) {
    const heading = firstHeading(body);
    if (heading) {
      frontmatter.title = heading;
      body = stripLeadingHeading(body);
      notes.push(`Title taken from the first heading: "${heading}".`);
      filled.push('title');
    }
  }

  /* Everything else is a decision, not a derivation, so it stays empty. */
  for (const key of ['description', 'domain', 'topic', 'type', 'section']) {
    if (frontmatter[key] === undefined) missing.push(key);
  }
  if (!frontmatter.title) missing.push('title');

  const declared = String(source.slug || '');
  const slug = content.slugProblem(declared) ? slugify(frontmatter.title || stemOf(filename)) : declared;

  return {
    ok: true,
    slug,
    slugTaken: slug ? await store.exists(slug) : false,
    frontmatter,
    body,
    filled: [...new Set(filled)],
    missing,
    notes,
    hadFrontmatter: parsed.hasFrontmatter,
    imageWarnings: await content.imageWarnings(body),
  };
}

/* Images ------------------------------------------------------------------ */

const IMAGE_TYPES = {
  '.png': bytes => bytes.length > 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47,
  '.jpg': bytes => bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff,
  '.jpeg': bytes => bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff,
  '.gif': bytes => ascii(bytes, 0, 4) === 'GIF8',
  '.webp': bytes => ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 12) === 'WEBP',
  '.avif': bytes => ascii(bytes, 4, 8) === 'ftyp',
};

const ascii = (bytes, from, to) => String.fromCharCode(...bytes.slice(from, to));

function decodeBase64(value) {
  const binary = atob(String(value));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function listImages() {
  return { ok: true, images: await store.listAssets() };
}

export async function uploadImage(payload) {
  const extension = extensionOf(payload.filename || '');
  if (!IMAGE_EXTENSIONS.includes(extension)) {
    fail(415, 'Images must be .png, .jpg, .gif, .webp or .avif. SVG is not accepted because it can carry scripts.');
  }

  let bytes;
  try {
    bytes = decodeBase64(payload.data || '');
  } catch {
    fail(400, 'The image data could not be read.');
  }
  if (!bytes.length) fail(400, 'The image is empty.');
  if (bytes.length > LIMITS.imageBytes) fail(413, 'The image is larger than 4 MB.');
  if (!IMAGE_TYPES[extension](bytes)) fail(415, `That file does not look like a ${extension.slice(1).toUpperCase()} image.`);

  const stem = slugify(stemOf(payload.filename || '')) || 'image';
  let name = `${stem}${extension}`;
  let counter = 2;
  while (!(await store.writeAsset(name, bytes))) {
    name = `${stem}-${counter}${extension}`;
    counter += 1;
    if (counter > 200) fail(409, 'Too many files with that name.');
  }

  return { ok: true, name, url: `/images/${name}`, markdown: `![${stem}](/images/${name})` };
}

/* Helpers used by the editor form ----------------------------------------- */

export async function suggest(payload) {
  return {
    ok: true,
    slug: slugify(payload.title || ''),
    order: await content.nextOrder(String(payload.section || ''), String(payload.domain || '')),
    today: today(),
  };
}

/*
 * What an article is, and what makes one publishable.
 *
 * Validation runs the same Zod schema the build uses (src/content.schema.ts),
 * so "valid here" and "valid in the build" cannot drift apart. Nothing in this
 * file knows where articles are kept — it goes through the store, so the same
 * rules apply whatever the backend is.
 */
import { postSchema, DOMAINS, TYPES, SECTIONS, LANGUAGES } from '../src/content.schema.ts';
import { KNOWLEDGE } from '../src/data/knowledge.ts';
import { ALGORITHM_GUIDES } from '../src/data/algorithms.ts';
import { store } from './store.mjs';
import { slugProblem, SLUG_PATTERN } from './rules.mjs';
import {
  parseFrontmatter, serializePost, slugify, scanImages, toDateString, FIELD_ORDER,
} from './markdown.mjs';


/* Reading ----------------------------------------------------------------- */

export async function readPost(slug) {
  const raw = await store.read(slug);
  if (raw === null) return null;
  const parsed = parseFrontmatter(raw);
  return { slug, raw, data: parsed.data, body: parsed.body, parseError: parsed.error || null };
}

export async function listPosts() {
  const slugs = await store.listSlugs();
  const posts = await Promise.all(slugs.map(async slug => {
    const post = await readPost(slug);
    if (!post) return null;
    const result = postSchema.safeParse(post.data);
    const data = result.success ? result.data : post.data;
    return {
      slug,
      title: String(data.title ?? slug),
      description: String(data.description ?? ''),
      section: data.section ?? null,
      domain: data.domain ?? null,
      topic: data.topic ?? null,
      type: data.type ?? null,
      language: data.language ?? null,
      translationKey: data.translationKey ?? null,
      order: typeof data.order === 'number' ? data.order : null,
      draft: data.draft === true,
      featured: data.featured === true,
      pubDate: toDateString(data.pubDate) || null,
      broken: Boolean(post.parseError) || !result.success,
      brokenReason: post.parseError
        || (result.success ? null : formatIssues(result.error).map(issue => `${issue.field}: ${issue.message}`).join('; ')),
    };
  }));
  return posts.filter(Boolean);
}

/* Validation -------------------------------------------------------------- */

/* Zod's messages are written for developers; the form needs one short line. */
function plainMessage(message) {
  if (/^Invalid option/.test(message)) return 'Choose one.';
  if (/expected number/.test(message)) return 'Enter a whole number.';
  if (/expected date/.test(message)) return 'Enter a date.';
  if (/expected string, received undefined/.test(message)) return 'Required.';
  return message;
}

function formatIssues(error) {
  return error.issues.map(issue => ({
    field: issue.path.length ? issue.path.join('.') : '_',
    message: plainMessage(issue.message),
  }));
}

/* Turns the browser's JSON into the object shape the schema expects. */
export function normalizeFrontmatter(input = {}) {
  const data = {};
  const text = key => (typeof input[key] === 'string' ? input[key].trim() : undefined);

  data.title = text('title');
  data.description = text('description');
  data.section = text('section');
  /* domain, topic and type only mean something in one section each, so an
     empty value is simply absent rather than an error. */
  const domain = text('domain');
  if (domain) data.domain = domain;
  const topic = text('topic');
  if (topic) data.topic = topic;
  const type = text('type');
  if (type) data.type = type;
  data.language = text('language') || 'en';
  const translationKey = text('translationKey');
  if (translationKey) data.translationKey = translationKey;
  data.pubDate = text('pubDate');
  data.featured = input.featured === true;
  data.draft = input.draft === true;

  if (input.order !== '' && input.order !== null && input.order !== undefined) data.order = Number(input.order);

  for (const key of Object.keys(data)) if (data[key] === undefined) delete data[key];
  return data;
}

export function validateFrontmatter(input) {
  const normalized = normalizeFrontmatter(input);
  const result = postSchema.safeParse(normalized);
  if (!result.success) return { valid: false, errors: formatIssues(result.error), data: null };
  /* Store the plain YYYY-MM-DD strings back so the serializer stays stable. */
  const data = { ...result.data };
  data.pubDate = toDateString(result.data.pubDate);
  const ordered = {};
  for (const key of FIELD_ORDER) if (data[key] !== undefined) ordered[key] = data[key];
  return { valid: true, errors: [], data: ordered };
}

/* Warnings ----------------------------------------------------------------- */

/* Image references are checked on their own too, because an import is worth
   warning about before any metadata has been filled in. */
export async function imageWarnings(body) {
  const warnings = [];
  for (const reference of scanImages(body || '')) {
    if (/^(https?:)?\/\//i.test(reference) || reference.startsWith('data:')) {
      warnings.push(`Image "${reference}" is loaded from another site - it will break if that site goes away.`);
      continue;
    }
    if (!(await store.assetExists(reference))) {
      warnings.push(`Image "${reference}" does not resolve to a file - the build will fail or the image will 404.`);
    }
  }
  return warnings;
}

/* Warnings are never blocking: they describe how the site will treat the
   article, which is easy to get wrong and impossible to see from the form. */
export async function collectWarnings(data, body, { slug, currentSlug, key } = {}) {
  const warnings = [];
  if (!data) return warnings;

  if (data.section === 'Learn' && data.order === undefined) {
    warnings.push('No order set — the article sorts last inside its chapter.');
  }

  if (data.section === 'Algorithms' && data.type === 'Algorithm') {
    const known = ALGORITHM_GUIDES.some(guide => guide.topic.toLowerCase() === String(data.topic).toLowerCase());
    if (!known) {
      warnings.push(`Topic "${data.topic}" is not in the Algorithms study table, so the article is only reachable by direct link.`);
    }
  }

  if (data.section === 'Blog' && data.order !== undefined) {
    warnings.push('Blog posts are sorted by date; order has no effect here.');
  }

  if (typeof data.order === 'number' && data.section === 'Learn') {
    /* The other language of the same article shares its order on purpose. */
    const pairKey = key ?? data.translationKey ?? null;
    const clash = (await listPosts()).find(post =>
      post.slug !== currentSlug && post.slug !== slug &&
      !(pairKey && post.translationKey === pairKey) &&
      !post.draft && post.section === data.section && post.domain === (data.domain ?? null) && post.order === data.order);
    if (clash) warnings.push(`Order ${data.order} is already used by "${clash.title}" in the same chapter.`);
  }

  warnings.push(...(await imageWarnings(body)));

  /* The page already renders the title as its heading. */
  if (/^\s*#\s+\S/.test(String(body))) {
    warnings.push('The body starts with a "#" heading, which will appear as a second title under the real one. Put the title in the Title field and use "##" for sections.');
  }

  if (!String(body).trim()) warnings.push('The article body is empty.');

  return warnings;
}

/* Options the editor form needs ------------------------------------------- */

export async function taxonomy() {
  const posts = await listPosts();
  const topics = [...new Set(posts.map(post => post.topic).filter(Boolean))].sort();
  return {
    domains: [...DOMAINS],
    types: [...TYPES],
    sections: [...SECTIONS],
    languages: [...LANGUAGES],
    chapters: KNOWLEDGE.map(chapter => ({ id: chapter.id, index: chapter.index, title: chapter.title })),
    algorithmTopics: ALGORITHM_GUIDES.map(guide => guide.topic),
    topics,
    slugs: posts.map(post => post.slug),
  };
}

export async function nextOrder(section, domain) {
  if (section !== 'Learn') return null;
  const posts = await listPosts();
  const used = posts
    .filter(post => post.section === section && post.domain === domain && typeof post.order === 'number')
    .map(post => post.order);
  return used.length ? Math.max(...used) + 1 : 1;
}

export {
  slugProblem, SLUG_PATTERN,
  slugify, parseFrontmatter, serializePost, toDateString,
  KNOWLEDGE, ALGORITHM_GUIDES,
};

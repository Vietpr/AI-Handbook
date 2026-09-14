/*
 * An article can exist in two languages: two files with the same
 * `translationKey`. Everything that lists articles goes through here so a
 * translated pair is counted once, numbered once and shown in the language
 * the reader picked.
 */
import type { CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;
export type Lang = Post['data']['language'];

const keyOf = (post: Post) => post.data.translationKey ?? post.id;

/* Groups posts by translation key, keeping the order they were given in. */
export function groupsOf(posts: Post[]): Post[][] {
  const groups = new Map<string, Post[]>();
  for (const post of posts) {
    const key = keyOf(post);
    const group = groups.get(key);
    if (group) group.push(post);
    else groups.set(key, [post]);
  }
  return [...groups.values()];
}

/* A translated pair counts as one article. */
export const countArticles = (posts: Post[]) => groupsOf(posts).length;

/* The same article in the other language, if it exists. */
export function translationOf(post: Post, posts: Post[]) {
  return posts.find(
    other => other.id !== post.id && keyOf(other) === keyOf(post) && other.data.language !== post.data.language,
  );
}

/*
 * Class that shows a post only when the interface is in its language. Posts
 * without a translation get no class, so they show in every language.
 */
export function langClass(post: Post, posts: Post[]) {
  return translationOf(post, posts) ? `lang-${post.data.language}` : undefined;
}

/* One post per article, preferring `lang` when the article has a translation. */
export function pickFor(lang: Lang, posts: Post[]) {
  return groupsOf(posts).map(group => group.find(post => post.data.language === lang) ?? group[0]);
}

export const byOrder = (a: Post, b: Post) => (a.data.order ?? 999) - (b.data.order ?? 999);

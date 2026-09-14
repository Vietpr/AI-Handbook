import { getCollection } from 'astro:content';
import { withBase } from '../lib/url';

export async function GET() {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return new Response(JSON.stringify(posts.map(post => ({
    title: post.data.title,
    description: post.data.description,
    domain: post.data.domain ?? '',
    topic: post.data.topic ?? '',
    type: post.data.type,
    section: post.data.section,
    language: post.data.language,
    href: withBase(`/articles/${post.id}`),
  }))), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}

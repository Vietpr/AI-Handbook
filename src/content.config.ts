import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { postSchema } from './content.schema';

/* The writer points this at a scratch copy when it builds a preview; every
   normal build uses the real content directory. */
const postsBase = process.env.WRITER_POSTS_DIR || './src/data/posts';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: postsBase }),
  schema: postSchema,
});

export const collections = { posts };

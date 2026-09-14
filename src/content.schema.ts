/*
 * Single source of truth for article metadata.
 *
 * Both the Astro build (via src/content.config.ts) and the writer server
 * (writer/content.mjs, plain Node) import this file, so an article accepted by
 * the writer is an article the build accepts. Keep it free of `astro:content`
 * imports — that virtual module only exists inside Astro's Vite pipeline.
 */
import { z } from 'astro/zod';

/* The six Learn chapters. Only Learn articles carry a domain. */
export const DOMAINS = [
  'Foundations', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'Generative AI', 'AI Systems',
] as const;

/* Only the three the site actually branches on: Algorithm and LeetCode drive
   the two lists on /algorithms, Concept is everything else. */
export const TYPES = ['Concept', 'Algorithm', 'LeetCode'] as const;

export const SECTIONS = ['Learn', 'Algorithms', 'Blog'] as const;
export const LANGUAGES = ['en', 'vi'] as const;

export const postSchema = z.object({
  /* min(1) matters: a bare z.string() accepts "", which would happily build an
     article with no title at all. */
  title: z.string().trim().min(1, 'Required.'),
  description: z.string().trim().min(1, 'Required.'),
  section: z.enum(SECTIONS),
  /* Which Learn chapter the article belongs to. Meaningless elsewhere. */
  domain: z.enum(DOMAINS).optional(),
  /* Which row of the /algorithms study table links here. Meaningless elsewhere. */
  topic: z.string().trim().min(1, 'Required.').optional(),
  type: z.enum(TYPES).default('Concept'),
  language: z.enum(LANGUAGES).default('en'),
  /* Two files with the same key are the same article in two languages. The
     site shows one of them at a time and links them to each other. */
  translationKey: z.string().trim().min(1).optional(),
  order: z.number().int().positive().optional(),
  pubDate: z.coerce.date(),
  featured: z.boolean().default(false),
  draft: z.boolean().default(false),
}).superRefine((data, ctx) => {
  if (data.section === 'Learn' && !data.domain) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['domain'], message: 'Choose a chapter.' });
  }
  if (data.section === 'Algorithms' && !data.topic) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['topic'], message: 'Choose a topic.' });
  }
  /* /algorithms only lists study guides and LeetCode solutions, so a Concept
     in this section would exist without being reachable from anywhere. */
  if (data.section === 'Algorithms' && data.type === 'Concept') {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['type'], message: 'Choose Study guide or LeetCode.' });
  }
});

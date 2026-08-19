import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/data/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    domain: z.enum(['Foundations', 'Machine Learning', 'Computer Vision', 'Generative AI', 'AI Systems', 'Algorithms']),
    topic: z.string(),
    type: z.enum([
      'Concept', 'Repo Breakdown', 'Paper Breakdown', 'Experiment',
      'From Scratch', 'Production Note', 'Algorithm', 'LeetCode'
    ]),
    section: z.enum(['Handbook', 'Algorithm', 'Blog']),
    language: z.enum(['en', 'vi']).default('en'),
    translationKey: z.string().optional(),
    level: z.enum(['Beginner', 'Intermediate', 'Advanced']),
    priority: z.enum(['Essential', 'High', 'Medium', 'Optional']).optional(),
    order: z.number().int().positive().optional(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    readingTime: z.number().int().positive(),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    prerequisites: z.array(z.string()).default([]),
  }),
});

export const collections = { posts };

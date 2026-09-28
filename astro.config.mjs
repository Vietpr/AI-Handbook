import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const site = process.env.SITE_URL || 'https://vietpr.github.io';
const base = process.env.BASE_PATH || '/';

export default defineConfig({
  output: 'static',
  trailingSlash: 'always',
  site,
  base,
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'viewport',
  },
  markdown: {
    /* $…$ and $$…$$ render to KaTeX HTML at build time; the CSS is imported
       by ArticleLayout. Needs @astrojs/markdown-remark for the unified pipeline. */
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
      smartypants: false,
    }),
    shikiConfig: {
      theme: 'github-dark-default',
      wrap: true,
    },
  },
});

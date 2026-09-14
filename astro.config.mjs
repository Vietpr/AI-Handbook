import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const site = process.env.SITE_URL || 'https://vietpr.github.io';
const base = process.env.BASE_PATH || '/';

export default defineConfig({
  output: 'static',
  site,
  base,
  markdown: {
    /* $…$ and $$…$$ render to KaTeX HTML at build time; the CSS is imported
       by ArticleLayout. Needs @astrojs/markdown-remark for the unified pipeline. */
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: false }]],
    shikiConfig: {
      theme: 'github-dark-default',
      wrap: true,
    },
  },
});

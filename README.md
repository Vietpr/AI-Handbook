# AI Field Notes — v3

A minimal personal knowledge site built with Astro + TypeScript.

The navigation is intentionally small:

- **Home** — short introduction: who Pham Van Viet is, what he works on, and why the site exists.
- **Learn** — the AI Handbook, organized from foundations to advanced AI systems.
- **Algorithms** — data structures, algorithms, problem-solving patterns and LeetCode notes.
- **Blog** — personal technical writing, interesting repositories, experiments and engineering lessons.

Deep Dives and Build are no longer separate sections. Detailed analysis, implementation, repo anatomy and trade-offs belong directly inside the relevant Handbook article.

## Run locally

```bash
npm install
npm run dev
```

Production build:

```bash
npm run build
npm run preview
```

Astro outputs the static site to `dist/`.

## Content

The starter currently contains 25 AI articles. Existing starter articles are English. The interface itself supports EN / VI, and new articles can declare either language with frontmatter.

### Add an AI Handbook article

```yaml
---
title: "Your title"
description: "Short summary"
domain: "Generative AI"
topic: "RAG"
type: "Concept"
section: "Handbook"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 5
pubDate: 2026-08-19
readingTime: 12
featured: false
draft: false
prerequisites:
  - Embeddings
---
```

### Add an Algorithm / LeetCode article

```yaml
---
title: "Two Pointers"
description: "How the two-pointer pattern reduces a search space."
domain: "Algorithms"
topic: "patterns"
type: "Algorithm"
section: "Algorithm"
language: "vi"
level: "Beginner"
order: 1
pubDate: 2026-08-19
readingTime: 8
featured: false
draft: false
prerequisites: []
---
```

For a LeetCode solution use `type: "LeetCode"` and `topic: "leetcode"`.

### Add a Blog post

Use `section: "Blog"`. The Blog is intentionally empty in this starter so it begins with your own writing.

## Bilingual behavior

The `EN / VI` control switches navigation and core interface copy. Article language is content metadata (`language: "en" | "vi"`). Existing Handbook content has not been machine-translated, to avoid filling the site with low-quality technical translations.

## Main files

- `src/config.ts` — site identity and navigation.
- `src/content.config.ts` — content schema.
- `src/data/knowledge.ts` — AI roadmap / taxonomy.
- `src/data/algorithms.ts` — Algorithms taxonomy.
- `src/data/posts/` — Markdown content.
- `src/pages/index.astro` — Home.
- `src/pages/learn.astro` — AI Handbook.
- `src/pages/algorithms.astro` — Algorithms.
- `src/pages/blog.astro` — Blog.
- `src/styles/global.css` — visual system.
- `CONTENT_GUIDE.md` — writing conventions.

## GitHub Pages / custom domain

Root or custom domain:

```bash
SITE_URL=https://notes.example.com npm run build
```

Repository Pages such as `https://username.github.io/ai-field-notes/`:

```bash
SITE_URL=https://username.github.io \
BASE_PATH=/ai-field-notes \
npm run build
```

All internal URLs use `src/lib/url.ts`, so a GitHub Pages base path is preserved.

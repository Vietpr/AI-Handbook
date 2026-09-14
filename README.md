# Technical Handbook — v3

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

### Add a Learn article

```yaml
---
title: "Your title"
description: "Short summary"
domain: "Generative AI"
section: "Learn"
language: "en"
order: 5
pubDate: 2026-08-19
featured: false
draft: false
---
```

`domain` is the chapter — one of the six Learn chapters in
`src/data/knowledge.ts` — and `order` sorts the article inside it. Both only
exist for `section: "Learn"`. Add `translationKey` when the same article exists
in the other language (see **Translations** below).

### Add an Algorithm / LeetCode article

```yaml
---
title: "Two Pointers"
description: "How the two-pointer pattern reduces a search space."
topic: "Two Pointers"
type: "Algorithm"
section: "Algorithms"
language: "vi"
pubDate: 2026-08-19
featured: false
draft: false
---
```

`topic` names the row of the study table on `/algorithms` that links to the
article (see `src/data/algorithms.ts`), and `type` decides which list it joins
there: `Algorithm` for a study guide, `LeetCode` for a solution. Both are
required for this section and meaningless outside it.

### Add a Blog post

Use `section: "Blog"`. The Blog is intentionally empty in this starter so it begins with your own writing.

## Bilingual behavior

The site opens in English. The `EN / VI` control switches the interface —
navigation, labels, empty states — and the choice is remembered.

Articles are written in one language each (`language: "en" | "vi"`). An
article page always shows the interface in the article's own language, so the
chrome and the text never disagree. On that page the `EN / VI` control does one
of two things:

- if the article also exists in the other language, it is a link to that
  version (and a "Read this in English" / "Đọc bản tiếng Việt" link sits in the
  meta row);
- if it does not, the other language is greyed out.

The first article someone reads sets their interface language when they have
not picked one yet.

### Translations

Two files with the same `translationKey` are one article in two languages:

```yaml
# how-attention-works.md          # co-che-attention.md
language: "en"                    language: "vi"
translationKey: "attention"       translationKey: "attention"
```

Every list on the site — chapter pages, sidebar, home, blog, the Algorithms
table — counts such a pair once and shows the version matching the interface
language (`src/lib/translations.ts`). Previous / next links stay within the
reader's language where a translation exists. Pages carry `hreflang` links
for search engines. Nothing is machine-translated.

## Main files

- `src/config.ts` — site identity and navigation.
- `src/content.config.ts` — content collection definition.
- `src/content.schema.ts` — the article schema, shared by the build and the writer.
- `src/data/knowledge.ts` — AI roadmap / taxonomy.
- `src/data/algorithms.ts` — Algorithms taxonomy.
- `src/data/posts/` — Markdown content.
- `src/lib/url.ts` — base-path aware URLs.
- `src/lib/labels.ts` — Vietnamese labels for priority / type.
- `src/lib/translations.ts` — groups the two languages of an article.
- `src/pages/index.astro` — Home.
- `src/pages/learn.astro` — AI Handbook.
- `src/pages/algorithms.astro` — Algorithms.
- `src/pages/blog.astro` — Blog.
- `src/layouts/ArticleLayout.astro` — article shell, reading progress, code copy.
- `src/components/HandbookSidebar.astro` — collapsible chapter navigation.
- `src/components/ArticleToc.astro` — table of contents (sticky rail and collapsible).
- `src/components/TextSize.astro` — reader-controlled article text size.
- `src/styles/global.css` — visual system.
- `writer/` — the private publishing console (see below).
- `CONTENT_GUIDE.md` — writing conventions.

## Requirements

Astro 7 requires **Node.js 22.12 or newer**, and the writer needs **22.18 or
newer** (it imports the TypeScript schema directly). With `nvm`:

```bash
nvm install 22 && nvm use 22
```

## Reading experience

Articles get a sticky table of contents with scroll tracking on wide screens, a
collapsible one below 1280px, a reading progress bar, copy buttons on code
blocks and previous / next links. The handbook sidebar collapses to the current
chapter; on phones it is replaced by an in-page chapter list.

Readers can resize the article text with the `A− / A+` control
(`src/components/TextSize.astro`). It renders twice — in the article meta row and
in the sticky table-of-contents rail — but CSS shows only one at a time: the rail
copy above 1280px, the meta-row copy below. It has four steps, is remembered in
`localStorage` and is restored before first paint alongside theme and language.
Everything inside `.prose` is sized in `em` against the `--prose-size` custom
property, so headings, tables and code scale together.

## Writer — publishing from a browser

`writer/` is a private console for creating, importing, previewing, editing and
deleting articles without a checkout of this repository. It is a separate Node
process; the public site and the build are untouched by it.

### Setting it up

```bash
npm run writer:init     # asks for a username and password once
npm run writer          # serves the site and the console
```

`writer:init` writes `.env.writer` (chmod 600, gitignored) with a scrypt password
hash and a random session secret. The plain password is never stored. Without
that file the console refuses every request, including login.

Then open `http://127.0.0.1:4322/write/`. The public site is served from the same
port, so `http://127.0.0.1:4322/` is the site itself.

Optional settings, in `.env.writer` or the environment:

| Variable | Meaning |
| --- | --- |
| `WRITER_PORT` | Port to listen on (default `4322`). |
| `WRITER_HOST` | Interface to bind (default `127.0.0.1`). |
| `WRITER_TRUST_PROXY` | Honour `X-Forwarded-For` / `X-Forwarded-Proto`. |
| `WRITER_SECURE_COOKIES` | Force the `Secure` cookie flag. |
| `WRITER_SESSION_HOURS` | Session lifetime (default 12). |
| `SITE_URL`, `BASE_PATH` | Passed to every build, same as `npm run build`. |

To reach it from another machine, put it behind HTTPS — a reverse proxy or a
tunnel — and set `WRITER_TRUST_PROXY=1`. Do not bind `0.0.0.0` on a public
interface without TLS in front: the session cookie would travel in the clear.

### Writing an article

**New article** opens the details screen first: the section, then only the
fields that section needs — chapter and order for Learn, topic and kind for
Algorithms, nothing extra for Blog — and, side by side, an **English** and a
**Tiếng Việt** panel with a title, slug and description each. The slug follows
the title until you edit it. Fill one panel or both; an empty panel is simply
not a version yet. **Continue** validates and moves to the writing screen.

The writing screen is the Markdown body on the left and the preview on the
right. The preview rebuilds by itself a moment after you stop typing (untick
*Auto* to refresh by hand or with Ctrl+Enter). Language tabs switch the body;
**Details** goes back to the first screen. Ctrl+S saves a draft.

**Publish** writes every language you filled in as one commit, so both
versions go live together or neither does. Tick *Keep this version hidden* on
a panel to publish the other language while that one stays a draft. *Remove
this version* deletes one language and leaves the other alone.

On disk this is two ordinary Markdown files joined by `translationKey`; the
writer picks the key and you never have to type it.

### How publishing works

Publishing writes a real `.md` file into `src/data/posts/` and then runs a real
`astro build`. The new site only replaces `dist/` once that build succeeds, so a
mistake that would break the build is reported instead of shipped, and the
previous site stays up. An article created here is byte-for-byte the kind of file
you would have written by hand — same schema, same field order, same quoting.

Preview works the same way: it builds the whole site into a scratch directory
with `BASE_PATH=/write/preview`, using a copy of the content plus the draft. The
preview you see is the actual article page — same layout, typography, table of
contents, Shiki highlighting and reader controls — because it *is* the built
page, not a second renderer. A build takes about two seconds.

Drafts (`draft: true`) are only written to disk; they never trigger a rebuild and
never appear on the site, which is the behaviour the schema already had.

Deleting removes the file and rebuilds, keeping a copy under `.writer/trash/`.

### Where content lives

`writer/store.mjs` is the only place that knows what "stored" and "published"
mean. Everything above it — validation, warnings, frontmatter handling, the API
and the whole interface — is written against a small contract:

```
listSlugs / exists / read
commit({ writes, deletes, publish, message })
republish / archive / preview / busy
listAssets / assetExists / writeAsset
deployment
```

`commit` is the transaction: writes and deletes are applied together, and when
`publish` is true the change has to be live before it resolves — otherwise the
store puts the previous content back. Nothing above the store can leave a
half-applied change behind.

`writer/store-fs.mjs` is the local implementation: real files under
`src/data/posts`, with `astro build` deciding whether a change may go live.
Moving the writer onto hosted infrastructure — commit through a Git provider's
API and let the host rebuild — means writing one more module against that
contract and changing the import on the last line of `store.mjs`. The schema,
the frontmatter serializer, the validation, the import parser and the entire
interface carry over untouched; only `writer/auth.mjs` (Node crypto) and the
HTTP shell (`server.mjs`, `http.mjs`, `config.mjs`) are tied to running on Node.

### What it validates

The console runs `src/content.schema.ts` — the same Zod schema the build uses —
so nothing can be published that the build would reject. On top of that it warns
about things that are valid but probably not what you meant: a Learn article
with a missing or duplicated `order`, a study guide whose topic is not in the
study table, image references that do not resolve, and a slug change that
would break an existing URL.

### Scope

Single author, no signup, no roles. Authentication is checked on the server for
every mutation, not just in the interface. `/write` is never part of `npm run
build` output, so deploying `dist/` anywhere never exposes it.

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

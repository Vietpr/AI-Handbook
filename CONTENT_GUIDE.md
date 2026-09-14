# Content Guide

The site has three content areas: **Learn**, **Algorithms**, and **Blog**.

## Learn (`section: "Learn"`)

Use this for structured AI knowledge. A Learn article should be complete enough that a beginner can build the right mental model without jumping across several posts.

Suggested structure:

1. What problem are we trying to solve?
2. Intuition / mental model.
3. How the mechanism works step by step.
4. Math, data flow or architecture when useful.
5. A small implementation or code path.
6. Common misunderstandings and failure cases.
7. Engineering trade-offs / production implications.
8. Related concepts and what to learn next.

Repository or paper analysis belongs **inside the relevant Learn article** when it helps explain the idea. There is no separate Deep Dives section.

Every Learn article belongs to a chapter (`domain`, one of the five in `src/data/knowledge.ts`). Use `order` so it sits in the right place inside that chapter.

## Algorithms (`section: "Algorithms"`)

Use this for data structures, algorithms, problem-solving patterns and LeetCode solutions. Every article here names a `topic` from the study table in `src/data/algorithms.ts` and a `type` — `Algorithm` for a study guide, `LeetCode` for a solution — which is how `/algorithms` finds it.

For a concept / pattern (`type: "Algorithm"`):

1. Problem shape.
2. Brute-force idea.
3. Key observation.
4. Algorithm / invariant.
5. Complexity.
6. Implementation.
7. Common mistakes.
8. Related problems.

For LeetCode (`type: "LeetCode"`): explain the reasoning before the accepted code. Avoid turning the article into a code dump.

## Blog (`section: "Blog"`)

Blog is intentionally lighter and more personal. Good examples:

- an interesting repository you found;
- a lesson from building an AI system;
- an experiment result;
- a technical opinion;
- a short note about a paper or tool;
- something worth revisiting later.

A Blog post can later become a Learn article if the idea becomes stable, reusable knowledge.

## Languages

Use `language: "en"` or `language: "vi"` in frontmatter. The site UI supports EN / VI independently of article language.

To publish the same article in both languages, give both files the same `translationKey` (the writer does this for you). The site then treats them as one article: listed once, shown in the reader's language, linked to each other.

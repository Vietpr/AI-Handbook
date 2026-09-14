/*
 * Rules that hold no matter where articles are stored.
 *
 * Deliberately free of imports so both the pure logic and any store
 * implementation can depend on it without pulling in a runtime.
 */

export const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export function slugProblem(slug) {
  if (!slug) return 'Slug is required.';
  if (slug.length > 80) return 'Slug is longer than 80 characters.';
  if (!SLUG_PATTERN.test(slug)) return 'Slug may only contain lowercase letters, digits and single hyphens.';
  return null;
}

export const LIMITS = {
  markdownBytes: 1024 * 1024,      // an imported .md file
  imageBytes: 4 * 1024 * 1024,     // one uploaded image
  requestBytes: 8 * 1024 * 1024,   // hard cap on any request body
  loginAttempts: 8,
  loginWindowMs: 15 * 60 * 1000,
};

export const SESSION_COOKIE = 'writer_session';

export const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif'];

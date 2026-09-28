const rawBase = import.meta.env?.BASE_URL || '/';
const base = rawBase === '/' ? '' : rawBase.replace(/\/$/, '');

/* Static pages are directories on GitHub Pages; assets keep their filenames. */
export function pagePath(path: string) {
  const boundary = path.search(/[?#]/);
  const pathname = boundary < 0 ? path : path.slice(0, boundary);
  const suffix = boundary < 0 ? '' : path.slice(boundary);
  if (!pathname || pathname.endsWith('/') || /\.[^/]+$/.test(pathname)) return path;
  return `${pathname}/${suffix}`;
}

export function withBase(path = '/') {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${base}${pagePath(normalized)}` || '/';
}

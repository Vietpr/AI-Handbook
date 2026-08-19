const rawBase = import.meta.env.BASE_URL || '/';
const base = rawBase === '/' ? '' : rawBase.replace(/\/$/, '');

export function withBase(path = '/') {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalized}` || '/';
}

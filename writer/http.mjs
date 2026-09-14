/*
 * Small HTTP helpers: bounded body reading, JSON replies and a static file
 * server that refuses to serve anything outside the directory it was given.
 */
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { LIMITS } from './rules.mjs';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.map': 'application/json; charset=utf-8',
};

export const mimeFor = file => MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';

export function send(response, status, body, headers = {}) {
  const payload = typeof body === 'string' || Buffer.isBuffer(body) ? body : '';
  response.writeHead(status, {
    'Content-Length': Buffer.byteLength(payload),
    'X-Content-Type-Options': 'nosniff',
    ...headers,
  });
  response.end(payload);
}

export function sendJson(response, status, data, headers = {}) {
  send(response, status, JSON.stringify(data), {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    ...headers,
  });
}

export function readBody(request, limit = LIMITS.requestBytes) {
  return new Promise((resolve, reject) => {
    const declared = Number(request.headers['content-length'] || 0);
    if (declared > limit) {
      reject(Object.assign(new Error('Request body is too large.'), { status: 413 }));
      request.destroy();
      return;
    }
    const chunks = [];
    let size = 0;
    request.on('data', chunk => {
      size += chunk.length;
      if (size > limit) {
        reject(Object.assign(new Error('Request body is too large.'), { status: 413 }));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => resolve(Buffer.concat(chunks)));
    request.on('error', reject);
  });
}

export async function readJson(request, limit = LIMITS.requestBytes) {
  const raw = await readBody(request, limit);
  if (!raw.length) return {};
  try {
    const value = JSON.parse(raw.toString('utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw Object.assign(new Error('Expected a JSON object.'), { status: 400 });
    }
    return value;
  } catch (error) {
    throw Object.assign(new Error(error.status ? error.message : 'Request body is not valid JSON.'), { status: 400 });
  }
}

/* Static files ------------------------------------------------------------ */

/* Resolves a URL path inside `root` or returns null. Anything that escapes the
   root — encoded traversal, absolute paths, null bytes — resolves to null. */
export function resolveInside(root, urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath);
  } catch {
    return null;
  }
  if (decoded.includes('\0')) return null;
  const base = path.resolve(root);
  const target = path.resolve(base, `.${decoded.startsWith('/') ? decoded : `/${decoded}`}`);
  if (target !== base && !target.startsWith(base + path.sep)) return null;
  return target;
}

async function fileInfo(target) {
  try {
    const info = await stat(target);
    if (info.isDirectory()) {
      const index = path.join(target, 'index.html');
      const indexInfo = await stat(index);
      return indexInfo.isFile() ? { file: index, info: indexInfo } : null;
    }
    return info.isFile() ? { file: target, info } : null;
  } catch {
    return null;
  }
}

/*
 * Serves one file from `root`. Every lookup hits the filesystem, so a publish
 * that swaps the site directory is visible on the very next request.
 */
export async function serveStatic(request, response, root, urlPath, { headers = {}, fallback } = {}) {
  const target = resolveInside(root, urlPath);
  if (!target) {
    send(response, 400, 'Bad request');
    return true;
  }

  let found = await fileInfo(target);
  /* Pretty URLs: /articles/foo -> /articles/foo/index.html */
  if (!found && !path.extname(target)) found = await fileInfo(path.join(target, 'index.html'));

  if (!found) {
    if (!fallback) return false;
    found = await fileInfo(path.resolve(root, fallback));
    if (!found) return false;
    response.statusCode = 404;
  }

  const tag = `W/"${found.info.size}-${Number(found.info.mtimeMs).toString(36)}"`;
  const common = {
    'Content-Type': mimeFor(found.file),
    'X-Content-Type-Options': 'nosniff',
    ETag: tag,
    ...headers,
  };

  if (request.headers['if-none-match'] === tag) {
    response.writeHead(304, common);
    response.end();
    return true;
  }

  response.writeHead(response.statusCode === 404 ? 404 : 200, {
    ...common,
    'Content-Length': found.info.size,
  });
  if (request.method === 'HEAD') {
    response.end();
    return true;
  }
  createReadStream(found.file).pipe(response);
  return true;
}

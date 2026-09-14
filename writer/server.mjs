#!/usr/bin/env node
/*
 * Technical Handbook writer.
 *
 * One process serves two things:
 *   - the public site, straight from dist/ (unchanged static output);
 *   - /write, a private console for creating, importing, previewing,
 *     editing and deleting articles.
 *
 * Publishing writes a real Markdown file and runs a real `astro build`, so a
 * published article is indistinguishable from one written by hand.
 */
import http from 'node:http';
import path from 'node:path';
import { existsSync } from 'node:fs';
import { CONFIG, PATHS, configProblems } from './config.mjs';
import { SESSION_COOKIE } from './rules.mjs';
import {
  verifyPassword, createSession, readSession, sessionCookie, parseCookies,
  loginBlockedFor, noteFailedLogin, clearLoginAttempts,
} from './auth.mjs';
import { send, sendJson, serveStatic, readJson } from './http.mjs';
import * as api from './api.mjs';
import * as pairs from './pairs.mjs';
import { store } from './store.mjs';

const PROBLEMS = configProblems();

const PRIVATE_HEADERS = {
  'Cache-Control': 'no-store, max-age=0',
  'X-Robots-Tag': 'noindex, nofollow',
  'Referrer-Policy': 'same-origin',
};

const SHELL_HEADERS = {
  ...PRIVATE_HEADERS,
  'Content-Security-Policy': [
    "default-src 'self'",
    "img-src 'self' data: blob:",
    "style-src 'self' 'unsafe-inline'",
    "script-src 'self'",
    "frame-src 'self'",
    "form-action 'none'",
    "base-uri 'none'",
    "object-src 'none'",
  ].join('; '),
  'X-Frame-Options': 'SAMEORIGIN',
};

const clientKey = request => {
  if (CONFIG.trustProxy) {
    const forwarded = String(request.headers['x-forwarded-for'] || '').split(',')[0].trim();
    if (forwarded) return forwarded;
  }
  return request.socket.remoteAddress || 'unknown';
};

const isSecure = request =>
  CONFIG.secureCookies ||
  (CONFIG.trustProxy && String(request.headers['x-forwarded-proto'] || '').split(',')[0].trim() === 'https');

const sessionOf = request => readSession(parseCookies(request.headers.cookie || '')[SESSION_COOKIE]);

/* API ---------------------------------------------------------------------- */

const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

async function handleApi(request, response, route) {
  const { method } = request;

  if (PROBLEMS.length) {
    sendJson(response, 503, {
      ok: false,
      error: 'The writer is not configured yet. Run `npm run writer:init` on the server.',
      problems: PROBLEMS,
    }, PRIVATE_HEADERS);
    return;
  }

  /* A cross-site page can send a form POST but cannot add this header. */
  if (MUTATING.has(method) && request.headers['x-writer-request'] !== '1') {
    sendJson(response, 403, { ok: false, error: 'Missing writer request header.' }, PRIVATE_HEADERS);
    return;
  }

  const session = sessionOf(request);

  if (route === 'session' && method === 'GET') {
    sendJson(response, 200, {
      ok: true,
      authenticated: Boolean(session),
      username: session ? session.username : null,
      expires: session ? session.expires : null,
    }, PRIVATE_HEADERS);
    return;
  }

  if (route === 'login' && method === 'POST') {
    const key = clientKey(request);
    const blocked = loginBlockedFor(key);
    if (blocked) {
      sendJson(response, 429, { ok: false, error: `Too many attempts. Try again in ${blocked} seconds.` }, PRIVATE_HEADERS);
      return;
    }
    const payload = await readJson(request, 4096);
    const username = String(payload.username || '');
    const password = String(payload.password || '');
    /* Check the password either way so a wrong username is not faster. */
    const passwordOk = verifyPassword(password, CONFIG.passwordHash);
    if (!passwordOk || username !== CONFIG.username) {
      noteFailedLogin(key);
      sendJson(response, 401, { ok: false, error: 'Wrong username or password.' }, PRIVATE_HEADERS);
      return;
    }
    clearLoginAttempts(key);
    const { token } = createSession(username);
    sendJson(response, 200, { ok: true, username }, {
      ...PRIVATE_HEADERS,
      'Set-Cookie': sessionCookie(token, CONFIG.sessionHours * 3600, isSecure(request)),
    });
    return;
  }

  if (route === 'logout' && method === 'POST') {
    sendJson(response, 200, { ok: true }, {
      ...PRIVATE_HEADERS,
      'Set-Cookie': sessionCookie('', 0, isSecure(request)),
    });
    return;
  }

  if (!session) {
    sendJson(response, 401, { ok: false, error: 'Sign in to continue.' }, PRIVATE_HEADERS);
    return;
  }

  const body = MUTATING.has(method) ? await readJson(request) : {};
  const post = route.startsWith('posts/') ? route.slice('posts/'.length) : null;

  /* The editor works on pairs (one article, up to two languages). */
  if (route === 'pair' && method === 'POST') return reply(response, await pairs.savePair(body));
  if (route === 'pair/validate' && method === 'POST') return reply(response, await pairs.validatePair(body));
  if (route === 'pair/state' && method === 'POST') return reply(response, await pairs.setPairState(body));
  if (route === 'pair/delete' && method === 'POST') return reply(response, await pairs.deletePair(body));
  if (route.startsWith('pair/') && method === 'GET') return reply(response, await pairs.getPair(route.slice('pair/'.length)));

  if (route === 'meta' && method === 'GET') return reply(response, await api.meta());
  if (route === 'posts' && method === 'GET') return reply(response, await api.listPosts());
  if (route === 'posts' && method === 'POST') return reply(response, await api.createPost(body));
  if (route === 'preview' && method === 'POST') return reply(response, await api.preview(body));
  if (route === 'validate' && method === 'POST') return reply(response, await api.validate(body));
  if (route === 'import' && method === 'POST') return reply(response, await api.importMarkdown(body));
  if (route === 'suggest' && method === 'POST') return reply(response, await api.suggest(body));
  if (route === 'rebuild' && method === 'POST') return reply(response, await api.rebuild());
  if (route === 'images' && method === 'GET') return reply(response, await api.listImages());
  if (route === 'images' && method === 'POST') return reply(response, await api.uploadImage(body));

  if (post) {
    const [slug, action] = post.split('/');
    if (!action && method === 'GET') return reply(response, await api.getPost(slug));
    if (!action && method === 'PUT') return reply(response, await api.updatePost(slug, body));
    if (!action && method === 'DELETE') return reply(response, await api.deletePost(slug, body));
    if (action === 'state' && method === 'POST') return reply(response, await api.setPostState(slug, body));
  }

  sendJson(response, 404, { ok: false, error: 'Unknown endpoint.' }, PRIVATE_HEADERS);
}

const reply = (response, data) => sendJson(response, 200, data, PRIVATE_HEADERS);

/* Writer UI ---------------------------------------------------------------- */

async function handleWriter(request, response, url) {
  const rest = url.pathname.slice('/write'.length) || '/';

  if (rest === '/' || rest === '') {
    if (url.pathname === '/write') {
      send(response, 302, '', { Location: '/write/', ...PRIVATE_HEADERS });
      return true;
    }
    return serveStatic(request, response, PATHS.ui, '/index.html', { headers: SHELL_HEADERS });
  }

  if (rest === '/global.css') {
    return serveStatic(request, response, path.dirname(PATHS.globalCss), '/global.css', { headers: PRIVATE_HEADERS });
  }

  if (rest.startsWith('/preview/')) {
    if (!sessionOf(request)) {
      send(response, 401, 'Sign in to view previews.', { 'Content-Type': 'text/plain; charset=utf-8', ...PRIVATE_HEADERS });
      return true;
    }
    if (!existsSync(PATHS.previewSite)) {
      send(response, 404, 'No preview has been rendered yet.', { 'Content-Type': 'text/plain; charset=utf-8', ...PRIVATE_HEADERS });
      return true;
    }
    const served = await serveStatic(request, response, PATHS.previewSite, rest.slice('/preview'.length), { headers: PRIVATE_HEADERS });
    if (!served) send(response, 404, 'Not in this preview.', { 'Content-Type': 'text/plain; charset=utf-8', ...PRIVATE_HEADERS });
    return true;
  }

  return serveStatic(request, response, PATHS.ui, rest, { headers: PRIVATE_HEADERS });
}

/* Server ------------------------------------------------------------------- */

const server = http.createServer(async (request, response) => {
  let url;
  try {
    url = new URL(request.url, 'http://localhost');
  } catch {
    send(response, 400, 'Bad request');
    return;
  }

  try {
    if (url.pathname === '/api/writer' || url.pathname.startsWith('/api/writer/')) {
      const route = url.pathname.slice('/api/writer/'.length).replace(/\/+$/, '');
      await handleApi(request, response, route);
      return;
    }

    if (url.pathname === '/write' || url.pathname.startsWith('/write/')) {
      const handled = await handleWriter(request, response, url);
      if (!handled) send(response, 404, 'Not found', { 'Content-Type': 'text/plain; charset=utf-8', ...PRIVATE_HEADERS });
      return;
    }

    if (request.method !== 'GET' && request.method !== 'HEAD') {
      send(response, 405, 'Method not allowed', { Allow: 'GET, HEAD' });
      return;
    }

    /* Everything else is the public site, read from disk on every request so a
       publish is live the moment its build finishes. */
    const served = await serveStatic(request, response, PATHS.site, url.pathname, { fallback: '404.html' });
    if (!served) send(response, 404, 'Not found', { 'Content-Type': 'text/plain; charset=utf-8' });
  } catch (error) {
    const status = error.status || (error instanceof api.ApiError ? error.status : 500);
    if (url.pathname.startsWith('/api/writer')) {
      sendJson(response, status, { ok: false, error: error.message || 'Something went wrong.', ...(error.extra || {}) }, PRIVATE_HEADERS);
    } else {
      send(response, status, 'Something went wrong.', { 'Content-Type': 'text/plain; charset=utf-8' });
    }
    if (status >= 500) console.error(error);
  }
});

server.on('clientError', (error, socket) => {
  if (socket.writable) socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
});

async function start() {
  if (!existsSync(PATHS.site)) {
    console.log('No dist/ yet — building the site once before starting.');
    const result = await store.republish();
    if (!result.ok) {
      console.error(result.log);
      console.error('\nThe initial build failed. Fix the content, then start the writer again.');
      process.exit(1);
    }
  }

  server.listen(CONFIG.port, CONFIG.host, () => {
    const where = `http://${CONFIG.host}:${CONFIG.port}`;
    console.log(`Site   ${where}/`);
    console.log(`Writer ${where}/write/`);
    if (PROBLEMS.length) {
      console.log('\nThe writer is disabled until it is configured:');
      for (const problem of PROBLEMS) console.log(`  - ${problem}`);
      console.log('  Run: npm run writer:init');
    }
  });
}

const stop = () => server.close(() => process.exit(0));
process.on('SIGINT', stop);
process.on('SIGTERM', stop);

start();

export { server };

/*
 * Password hashing, signed sessions and login throttling.
 *
 * The password is never stored, only a scrypt hash. The session cookie carries
 * no privileges of its own: it is a signed statement of "who", verified with a
 * server-side secret on every request.
 */
import { randomBytes, scryptSync, timingSafeEqual, createHmac } from 'node:crypto';
import { CONFIG } from './config.mjs';
import { LIMITS, SESSION_COOKIE } from './rules.mjs';

const SCRYPT = { N: 16384, r: 8, p: 1, keylen: 64 };

export function hashPassword(password) {
  const salt = randomBytes(16);
  const key = scryptSync(password, salt, SCRYPT.keylen, { N: SCRYPT.N, r: SCRYPT.r, p: SCRYPT.p });
  return ['scrypt', SCRYPT.N, SCRYPT.r, SCRYPT.p, salt.toString('base64'), key.toString('base64')].join('$');
}

export function verifyPassword(password, stored) {
  try {
    const [scheme, N, r, p, salt, hash] = String(stored).split('$');
    if (scheme !== 'scrypt') return false;
    const expected = Buffer.from(hash, 'base64');
    const actual = scryptSync(password, Buffer.from(salt, 'base64'), expected.length, {
      N: Number(N), r: Number(r), p: Number(p),
      /* scrypt's default memory cap is below what N=16384 needs. */
      maxmem: 256 * 1024 * 1024,
    });
    return timingSafeEqual(expected, actual);
  } catch {
    return false;
  }
}

/* Sessions --------------------------------------------------------------- */

const b64url = buffer => Buffer.from(buffer).toString('base64url');
const sign = payload => createHmac('sha256', CONFIG.sessionSecret).update(payload).digest();

export function createSession(username) {
  const expires = Date.now() + CONFIG.sessionHours * 60 * 60 * 1000;
  const payload = b64url(JSON.stringify({ u: username, exp: expires }));
  return { token: `${payload}.${b64url(sign(payload))}`, expires };
}

export function readSession(token) {
  if (!token || !CONFIG.sessionSecret) return null;
  const dot = token.indexOf('.');
  if (dot < 1) return null;
  const payload = token.slice(0, dot);
  const provided = Buffer.from(token.slice(dot + 1), 'base64url');
  const expected = sign(payload);
  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) return null;
  try {
    const data = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
    if (!data || typeof data.exp !== 'number' || Date.now() > data.exp) return null;
    /* A renamed user must not keep an old session alive. */
    if (data.u !== CONFIG.username) return null;
    return { username: data.u, expires: data.exp };
  } catch {
    return null;
  }
}

export function sessionCookie(token, maxAgeSeconds, secure) {
  const parts = [
    `${SESSION_COOKIE}=${token}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Strict',
    `Max-Age=${maxAgeSeconds}`,
  ];
  if (secure) parts.push('Secure');
  return parts.join('; ');
}

export function parseCookies(header = '') {
  const jar = {};
  for (const part of header.split(';')) {
    const eq = part.indexOf('=');
    if (eq < 1) continue;
    jar[part.slice(0, eq).trim()] = decodeURIComponent(part.slice(eq + 1).trim());
  }
  return jar;
}

/* Login throttling -------------------------------------------------------- */

const attempts = new Map();

export function loginBlockedFor(key) {
  const record = attempts.get(key);
  if (!record) return 0;
  if (Date.now() > record.until) {
    attempts.delete(key);
    return 0;
  }
  return record.count >= LIMITS.loginAttempts ? Math.ceil((record.until - Date.now()) / 1000) : 0;
}

export function noteFailedLogin(key) {
  const now = Date.now();
  const record = attempts.get(key);
  if (!record || now > record.until) {
    attempts.set(key, { count: 1, until: now + LIMITS.loginWindowMs });
    return;
  }
  record.count += 1;
  record.until = now + LIMITS.loginWindowMs;
}

export function clearLoginAttempts(key) {
  attempts.delete(key);
}

#!/usr/bin/env node
/*
 * One-time writer setup: asks for a username and password, then writes
 * .env.writer with a scrypt hash and a fresh session secret.
 *
 * The plain password is never stored and never leaves this process.
 */
import { createInterface } from 'node:readline';
import { Writable } from 'node:stream';
import { writeFile, readFile, chmod } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { PATHS } from './config.mjs';
import { hashPassword } from './auth.mjs';

const interactive = Boolean(process.stdin.isTTY);

/* Interactive input goes through readline, whose echo we can mute so the
   password is never printed. Piped input is read in one go instead: readline
   emits every buffered line at once and would drop the answers we have not
   asked for yet. */
let muted = false;
const rl = interactive
  ? createInterface({
    input: process.stdin,
    output: new Writable({
      write(chunk, encoding, callback) {
        if (!muted) process.stdout.write(chunk, encoding);
        callback();
      },
    }),
    terminal: true,
  })
  : null;

let piped = null;

async function ask(question) {
  if (interactive) return new Promise(resolve => rl.question(question, resolve));
  if (piped === null) {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    piped = Buffer.concat(chunks).toString('utf8').split('\n');
  }
  process.stdout.write(question);
  const value = piped.shift() ?? '';
  process.stdout.write('\n');
  return value;
}

async function askSecret(question) {
  if (!interactive) return ask(question);
  process.stdout.write(question);
  muted = true;
  const value = await ask('');
  muted = false;
  process.stdout.write('\n');
  return value;
}

const done = () => rl?.close();
const quoteIfNeeded = value => (/[^A-Za-z0-9_./:-]/.test(value) ? JSON.stringify(value) : value);
const line = (key, value) => `${key}=${quoteIfNeeded(value)}`;

const stop = message => {
  console.error(message);
  done();
  process.exit(1);
};

async function main() {
  if (existsSync(PATHS.envFile)) {
    const answer = (await ask('.env.writer already exists. Overwrite it? [y/N] ')).trim().toLowerCase();
    if (answer !== 'y' && answer !== 'yes') {
      console.log('Left the existing file alone.');
      done();
      return;
    }
  }

  const username = (await ask('Username: ')).trim();
  if (!username) stop('A username is required.');

  const password = await askSecret('Password: ');
  if (password.length < 10) stop('Use at least 10 characters - this is the only thing protecting the writer.');

  const again = await askSecret('Repeat password: ');
  if (password !== again) stop('The two passwords do not match.');

  /* Keep any non-secret settings the file already carried. */
  const previous = existsSync(PATHS.envFile) ? await readFile(PATHS.envFile, 'utf8') : '';
  const keep = previous
    .split('\n')
    .filter(row => /^(WRITER_PORT|WRITER_HOST|WRITER_TRUST_PROXY|WRITER_SECURE_COOKIES|WRITER_SESSION_HOURS|SITE_URL|BASE_PATH)=/.test(row.trim()));

  const contents = [
    '# Writer credentials. Keep this file out of version control.',
    line('WRITER_USERNAME', username),
    line('WRITER_PASSWORD_HASH', hashPassword(password)),
    line('WRITER_SESSION_SECRET', randomBytes(32).toString('hex')),
    '',
    '# Optional:',
    '#   WRITER_PORT=4322',
    '#   WRITER_HOST=127.0.0.1        set to 0.0.0.0 only behind a proxy or tunnel',
    '#   WRITER_TRUST_PROXY=1         honour X-Forwarded-For / X-Forwarded-Proto',
    '#   WRITER_SECURE_COOKIES=1      force the Secure cookie flag',
    '#   WRITER_SESSION_HOURS=12',
    '#   SITE_URL=https://your-domain',
    '#   BASE_PATH=/subpath',
    ...keep,
    '',
  ].join('\n');

  await writeFile(PATHS.envFile, contents, 'utf8');
  await chmod(PATHS.envFile, 0o600);

  console.log(`\nWrote ${PATHS.envFile} (permissions 600).`);
  console.log('Start the writer with:  npm run writer');
  done();
}

main().catch(error => {
  console.error(error.message);
  done();
  process.exit(1);
});

/*
 * Writer console.
 *
 * Four views on one page: login, the article list, the details form and the
 * writing screen. All content changes go through /api/writer, which does its
 * own authentication and validation — nothing here is a security boundary,
 * only an interface.
 *
 * An article is edited in two steps. Details first: where it lives (section,
 * chapter or topic) and, per language, its title, slug and description. Then
 * writing: the Markdown body beside a preview that is a real build of the
 * site, rebuilt a moment after you stop typing.
 */

const API = '/api/writer';
const LANGS = [
  { code: 'en', label: 'English' },
  { code: 'vi', label: 'Tiếng Việt' },
];
const langLabel = code => LANGS.find(lang => lang.code === code)?.label ?? code;

/* Helpers ------------------------------------------------------------------ */

const $ = selector => document.querySelector(selector);

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function announce(message) {
  $('#live-region').textContent = message;
}

/* Same rules as writer/markdown.mjs, so the slug the form proposes is the slug
   the server would accept. */
function slugify(input) {
  return String(input)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[đĐ]/g, 'd')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
    .replace(/-+$/, '');
}

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data || {};
  }
}

async function api(route, { method = 'GET', body } = {}) {
  const response = await fetch(`${API}/${route}`, {
    method,
    credentials: 'same-origin',
    headers: {
      'X-Writer-Request': '1',
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  let data = {};
  try {
    data = await response.json();
  } catch {
    /* Non-JSON reply: fall through to the status-based message below. */
  }

  if (response.status === 401 && route !== 'login' && route !== 'session') {
    state.session = null;
    showLogin('Your session expired. Sign in again.');
    throw new ApiError('Signed out.', 401, data);
  }
  if (!response.ok || data.ok === false) {
    throw new ApiError(data.error || `Request failed (${response.status}).`, response.status, data);
  }
  return data;
}

/* State -------------------------------------------------------------------- */

const state = {
  session: null,
  meta: null,
  posts: [],
  filter: 'all',
  search: '',
  editor: null,
  view: 'login',
  busy: false,
};

const VIEWS = ['login', 'dashboard', 'details', 'write'];
function show(view) {
  state.view = view;
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== view;
  window.scrollTo(0, 0);
}

/* Login -------------------------------------------------------------------- */

function showLogin(message) {
  show('login');
  const error = $('#login-error');
  error.textContent = message || '';
  error.hidden = !message;
  $('#login-pass').value = '';
  $('#login-user').focus();
}

$('#login-form').addEventListener('submit', async event => {
  event.preventDefault();
  const submit = $('#login-submit');
  const error = $('#login-error');
  submit.disabled = true;
  submit.textContent = 'Signing in…';
  error.hidden = true;
  try {
    const result = await api('login', {
      method: 'POST',
      body: { username: $('#login-user').value, password: $('#login-pass').value },
    });
    state.session = { username: result.username };
    $('#login-pass').value = '';
    await enter();
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
    if (Array.isArray(failure.data.problems)) {
      error.textContent = `${failure.message} (${failure.data.problems.join(' ')})`;
    }
  } finally {
    submit.disabled = false;
    submit.textContent = 'Sign in';
  }
});

/* Dashboard ---------------------------------------------------------------- */

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'Learn', label: 'Learn' },
  { id: 'Algorithms', label: 'Algorithms' },
  { id: 'Blog', label: 'Blog' },
  { id: 'draft', label: 'Drafts' },
];

function renderFilters() {
  const host = $('#dash-filters');
  host.replaceChildren(...FILTERS.map(filter =>
    el('button', {
      class: 'w-filter',
      type: 'button',
      'aria-pressed': String(state.filter === filter.id),
      onClick: () => {
        state.filter = filter.id;
        renderFilters();
        renderList();
      },
      text: filter.label,
    })));
}

/* One row per article: the files that share a translation key form a group. */
function groupPosts(posts) {
  const groups = new Map();
  for (const post of posts) {
    const key = post.translationKey || post.slug;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(post);
  }
  const rank = post => (post.language === 'en' ? 0 : 1);
  return [...groups.values()].map(versions => {
    versions.sort((a, b) => rank(a) - rank(b));
    return { versions, primary: versions[0] };
  });
}

function visibleGroups() {
  const needle = state.search.trim().toLowerCase();
  return groupPosts(state.posts).filter(({ versions, primary }) => {
    if (state.filter === 'draft' && !versions.some(post => post.draft)) return false;
    if (state.filter !== 'all' && state.filter !== 'draft' && primary.section !== state.filter) return false;
    if (!needle) return true;
    return versions.some(post => [post.title, post.slug, post.topic, post.domain].filter(Boolean)
      .some(field => String(field).toLowerCase().includes(needle)));
  });
}

function renderList() {
  const host = $('#dash-list');
  const groups = visibleGroups();

  if (!groups.length) {
    host.replaceChildren(el('p', {
      class: 'w-empty',
      text: state.posts.length ? 'No article matches this filter.' : 'No articles yet. Create one to get started.',
    }));
    return;
  }

  host.replaceChildren(...groups.map(({ versions, primary }) => {
    const live = versions.find(post => !post.draft);
    const allDraft = versions.every(post => post.draft);
    const slugs = versions.map(post => post.slug);
    return el('div', { class: 'w-row' }, [
      el('div', { class: 'w-row-main' }, [
        el('strong', { text: primary.title }),
        el('span', { text: versions.map(post => `${String(post.language || 'en').toUpperCase()} /articles/${post.slug}`).join('  ·  ') }),
      ]),
      el('div', { class: 'w-row-where', text: [primary.section, primary.domain ?? primary.topic].filter(Boolean).join(' · ') || '—' }),
      el('div', { class: 'w-row-status' }, versions.map(post =>
        post.broken
          ? el('span', { class: 'w-chip broken', title: post.brokenReason || '', text: `${post.language.toUpperCase()} broken` })
          : el('span', { class: post.draft ? 'w-chip draft' : 'w-chip live', text: `${String(post.language || 'en').toUpperCase()} ${post.draft ? 'draft' : 'live'}` }),
      )),
      el('div', { class: 'w-row-actions' }, [
        el('button', { class: 'w-btn small', type: 'button', text: 'Edit', onClick: () => openEditor(primary.slug) }),
        live
          ? el('a', { class: 'w-btn small', href: `/articles/${live.slug}`, target: '_blank', rel: 'noreferrer', text: 'View' })
          : null,
        el('button', {
          class: 'w-btn small',
          type: 'button',
          text: allDraft ? 'Publish' : 'Unpublish',
          onClick: () => setPairState(slugs, primary.title, !allDraft),
        }),
        el('button', { class: 'w-btn small danger', type: 'button', text: 'Delete', onClick: () => confirmDelete(versions, primary.title) }),
      ]),
    ]);
  }));
}

function dashStatus(message, kind = '') {
  const node = $('#dash-status');
  node.textContent = message || '';
  node.className = `w-status ${kind}`;
  node.hidden = !message;
}

async function loadPosts() {
  const result = await api('posts');
  state.posts = result.posts;
  const groups = groupPosts(state.posts);
  const live = groups.filter(group => group.versions.some(post => !post.draft)).length;
  const drafts = groups.length - live;
  const translated = groups.filter(group => group.versions.length > 1).length;
  $('#dash-summary').textContent =
    `${live} published${drafts ? `, ${drafts} draft${drafts === 1 ? '' : 's'}` : ''}` +
    `${translated ? `, ${translated} in both languages` : ''} — publishing rebuilds the site.`;
  renderList();
}

async function setPairState(slugs, title, draft) {
  dashStatus(draft ? `Unpublishing “${title}”…` : `Publishing “${title}”…`, 'busy');
  try {
    await api('pair/state', { method: 'POST', body: { slugs, draft } });
    await loadPosts();
    dashStatus(draft ? 'Unpublished. It is no longer on the site.' : 'Published and live.', 'good');
  } catch (failure) {
    dashStatus(failure.message, 'bad');
    if (failure.data.buildLog) openBuildLog(failure.message, failure.data.buildLog);
  }
}

/* Modals ------------------------------------------------------------------- */

function openModal(build) {
  const root = $('#modal-root');
  const close = () => {
    root.replaceChildren();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = event => {
    if (event.key === 'Escape') close();
  };
  document.addEventListener('keydown', onKey);

  const card = el('div', { class: 'w-modal-card', role: 'dialog', 'aria-modal': 'true' });
  const backdrop = el('div', {
    class: 'w-modal',
    onClick: event => {
      if (event.target === backdrop) close();
    },
  }, [card]);

  build(card, close);
  root.replaceChildren(backdrop);
  const focusable = card.querySelector('input, textarea, button');
  if (focusable) focusable.focus();
  return close;
}

function openBuildLog(title, log) {
  openModal((card, close) => {
    card.append(
      el('h2', { text: title }),
      el('p', { text: 'Astro reported this while building the site. Nothing was published.' }),
      el('pre', { text: log, style: 'max-height:44vh;overflow:auto;padding:12px 14px;border-radius:6px;background:var(--code);color:var(--code-text);font:12px/1.6 var(--mono);white-space:pre-wrap' }),
      el('div', { class: 'w-modal-actions' }, [
        el('button', { class: 'w-btn primary', type: 'button', text: 'Close', onClick: close }),
      ]),
    );
  });
}

function confirmDelete(versions, title) {
  const slugs = versions.map(post => post.slug);
  const expected = slugs.join(' ');
  openModal((card, close) => {
    const input = el('input', { type: 'text', id: 'delete-confirm', autocomplete: 'off', spellcheck: 'false' });
    const error = el('p', { class: 'w-status bad', hidden: true });
    const run = async () => {
      const typed = input.value.trim().split(/\s+/).filter(Boolean).sort().join(' ');
      if (typed !== [...slugs].sort().join(' ')) {
        error.textContent = slugs.length > 1 ? 'Type both slugs, separated by a space.' : 'The slug does not match.';
        error.hidden = false;
        return;
      }
      error.hidden = true;
      try {
        const result = await api('pair/delete', { method: 'POST', body: { slugs, confirm: input.value } });
        close();
        await loadPosts();
        dashStatus(`Deleted. ${result.backups.length === 1 ? 'A copy was' : 'Copies were'} kept in .writer/trash.`, 'good');
      } catch (failure) {
        error.textContent = failure.message;
        error.hidden = false;
        if (failure.data.buildLog) openBuildLog(failure.message, failure.data.buildLog);
      }
    };

    const addresses = versions.map(post => `/articles/${post.slug}`).join(' and ');
    card.append(
      el('h2', { text: versions.length > 1 ? 'Delete both versions?' : 'Delete this article?' }),
      el('p', {}, [
        `“${title}” will be removed from the site` + (versions.every(post => post.draft) ? '' : ` and ${addresses} will stop working`) + '. ',
        'A copy is kept in .writer/trash so this is recoverable from the server.',
      ]),
      el('div', { class: 'w-field' }, [
        el('label', { for: 'delete-confirm', text: `Type ${expected} to confirm` }),
        input,
      ]),
      error,
      el('div', { class: 'w-modal-actions' }, [
        el('button', { class: 'w-btn', type: 'button', text: 'Cancel', onClick: close }),
        el('button', { class: 'w-btn danger', type: 'button', text: 'Delete', onClick: run }),
      ]),
    );
    input.addEventListener('keydown', event => {
      if (event.key === 'Enter') run();
    });
  });
}

/* Import ------------------------------------------------------------------- */

function openImport() {
  openModal((card, close) => {
    const status = el('p', { class: 'w-status', hidden: true });
    const input = el('input', { type: 'file', accept: '.md,.markdown,.txt', class: 'w-sr', id: 'import-file' });

    const handle = async file => {
      if (!file) return;
      if (file.size > 1024 * 1024) {
        status.textContent = 'That file is larger than 1 MB.';
        status.className = 'w-status bad';
        status.hidden = false;
        return;
      }
      status.textContent = `Reading ${file.name}…`;
      status.className = 'w-status busy';
      status.hidden = false;
      try {
        const text = await file.text();
        const result = await api('import', { method: 'POST', body: { filename: file.name, text } });
        close();
        startImported(result);
      } catch (failure) {
        status.textContent = failure.message;
        status.className = 'w-status bad';
      }
    };

    const drop = el('div', {
      class: 'w-drop',
      tabindex: '0',
      role: 'button',
      onClick: () => input.click(),
      onKeydown: event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          input.click();
        }
      },
      onDragover: event => {
        event.preventDefault();
        drop.classList.add('over');
      },
      onDragleave: () => drop.classList.remove('over'),
      onDrop: event => {
        event.preventDefault();
        drop.classList.remove('over');
        handle(event.dataTransfer.files[0]);
      },
    }, [
      el('strong', { text: 'Drop a Markdown file here' }),
      el('span', { text: 'or click to choose one — .md, .markdown or .txt' }),
    ]);

    input.addEventListener('change', () => handle(input.files[0]));

    card.append(
      el('h2', { text: 'Import Markdown' }),
      el('p', { text: 'The file is parsed and loaded into the editor. Nothing is published until you say so.' }),
      drop,
      input,
      status,
      el('div', { class: 'w-modal-actions' }, [
        el('button', { class: 'w-btn', type: 'button', text: 'Cancel', onClick: close }),
      ]),
    );
  });
}

function startImported(result) {
  const notes = [...result.notes, ...result.imageWarnings];
  if (!result.hadFrontmatter) notes.unshift('The file had no frontmatter, so the details below are mostly empty.');
  if (result.missing.length) notes.push(`Still needed: ${result.missing.join(', ')}.`);
  if (result.slugTaken) notes.push(`An article with the slug "${result.slug}" already exists — choose another one.`);

  const lang = result.frontmatter.language === 'vi' ? 'vi' : 'en';
  const { title = '', description = '', language, translationKey, ...shared } = result.frontmatter;
  openEditorWith({
    mode: 'new',
    key: translationKey || null,
    shared: { pubDate: state.meta.today, ...shared },
    versions: {
      [lang]: { slug: result.slugTaken ? '' : result.slug, title, description, body: result.body, slugEdited: true },
    },
    active: lang,
    notes,
    view: 'details',
  });
}

/* Images ------------------------------------------------------------------- */

function openImages() {
  openModal(async (card, close) => {
    const status = el('p', { class: 'w-status', hidden: true });
    const gallery = el('div', { class: 'w-images' });
    const input = el('input', { type: 'file', accept: 'image/*', class: 'w-sr', id: 'image-file' });

    const insert = url => {
      insertAtCursor(`![](${url})`);
      close();
    };

    const refresh = async () => {
      try {
        const result = await api('images');
        gallery.replaceChildren(...result.images.map(image =>
          el('figure', { onClick: () => insert(image.url), title: `Insert ${image.url}` }, [
            el('img', { src: image.url, alt: image.name, loading: 'lazy' }),
            el('figcaption', { text: image.name }),
          ])));
        if (!result.images.length) gallery.replaceChildren(el('p', { class: 'w-status', text: 'No images uploaded yet.' }));
      } catch (failure) {
        status.textContent = failure.message;
        status.className = 'w-status bad';
        status.hidden = false;
      }
    };

    input.addEventListener('change', async () => {
      const file = input.files[0];
      if (!file) return;
      status.textContent = `Uploading ${file.name}…`;
      status.className = 'w-status busy';
      status.hidden = false;
      try {
        const buffer = await file.arrayBuffer();
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
        const result = await api('images', { method: 'POST', body: { filename: file.name, data: btoa(binary) } });
        status.textContent = `Uploaded as ${result.url}. It goes live with the next publish.`;
        status.className = 'w-status good';
        await refresh();
      } catch (failure) {
        status.textContent = failure.message;
        status.className = 'w-status bad';
      }
    });

    card.append(
      el('h2', { text: 'Images' }),
      el('p', { text: 'Files live in public/images and are copied into the site by the next build. Click one to insert it at the cursor.' }),
      el('button', { class: 'w-btn', type: 'button', text: 'Upload an image', onClick: () => input.click() }),
      input,
      status,
      gallery,
      el('div', { class: 'w-modal-actions' }, [
        el('button', { class: 'w-btn', type: 'button', text: 'Close', onClick: close }),
      ]),
    );
    refresh();
  });
}

/* Details: fields ---------------------------------------------------------- */

/*
 * Shared by every language of the article. `when` limits a field to one
 * section: only Learn articles sit in a chapter, only Algorithms articles are
 * tied to a topic row on /algorithms. Blog needs neither.
 */
const SHARED_FIELDS = [
  { key: 'section', label: 'Section', type: 'select', required: true,
    options: () => state.meta.taxonomy.sections.map(value => ({ value, label: value })) },
  { key: 'domain', label: 'Chapter', type: 'select', required: true, when: 'Learn',
    options: () => state.meta.taxonomy.chapters.map(chapter => ({ value: chapter.title, label: `${chapter.index} · ${chapter.title}` })) },
  { key: 'order', label: 'Order in chapter', type: 'order', when: 'Learn' },
  { key: 'topic', label: 'Topic', type: 'select', required: true, when: 'Algorithms',
    options: () => state.meta.taxonomy.algorithmTopics.map(value => ({ value, label: value })) },
  { key: 'type', label: 'Kind', type: 'select', required: true, when: 'Algorithms',
    options: () => [{ value: 'Algorithm', label: 'Study guide' }, { value: 'LeetCode', label: 'LeetCode solution' }] },
  { key: 'pubDate', label: 'Published', type: 'date', required: true },
];

/* Written once per language. */
const VERSION_FIELDS = [
  { key: 'title', label: 'Title', type: 'text', required: true },
  { key: 'slug', label: 'Slug', type: 'slug', required: true },
  { key: 'description', label: 'Description', type: 'textarea', required: true },
];

const sharedInputs = {};
const verInputs = { en: {}, vi: {} };

const sharedValue = key => {
  const node = sharedInputs[key];
  if (!node) return undefined;
  return node.type === 'checkbox' ? node.checked : node.value;
};

function markDirty(key) {
  const editor = state.editor;
  if (!editor) return;
  editor.dirty = true;
  if (key) editor.touched.add(key);
  if (editor.previewUrl && !autoOn()) $('#preview-stale').hidden = false;
  updateTitles();
  scheduleValidate();
  if (state.view === 'write') schedulePreview();
}

function buildInput(field, id) {
  if (field.type === 'textarea') return el('textarea', { id, rows: '2' });
  if (field.type === 'select') {
    return el('select', { id }, [
      el('option', { value: '', text: 'Choose…', disabled: true }),
      ...field.options().map(option => el('option', { value: option.value, text: option.label })),
    ]);
  }
  if (field.type === 'date') return el('input', { type: 'date', id });
  if (field.type === 'order') return el('input', { type: 'number', id, min: '1', step: '1' });
  return el('input', { type: 'text', id, autocomplete: 'off', spellcheck: field.type === 'slug' ? 'false' : 'true' });
}

function buildSharedField(field) {
  const wrap = el('div', { class: 'w-field', dataset: { field: field.key, scope: 'shared' } });
  const id = `f-${field.key}`;
  wrap.append(el('label', { for: id, text: field.label + (field.required ? '' : ' (optional)') }));
  const node = buildInput(field, id);
  sharedInputs[field.key] = node;
  wrap.append(node);

  const touch = () => markDirty(field.key);
  node.addEventListener('input', touch);
  node.addEventListener('change', touch);

  if (field.key === 'section') {
    node.addEventListener('change', updateConditionalFields);
  }
  if (field.key === 'domain') {
    /* A new article usually goes at the end of its chapter. */
    node.addEventListener('change', () => {
      if (!sharedInputs.order.value) suggestOrder();
    });
  }
  if (field.type === 'order') {
    wrap.append(el('p', { class: 'hint' }, [
      'Sorts the article inside its chapter · ',
      el('button', { type: 'button', text: 'use the next free number', onClick: suggestOrder }),
    ]));
  }

  wrap.append(el('p', { class: 'error', hidden: true }));
  return wrap;
}

async function suggestOrder() {
  try {
    const result = await api('suggest', {
      method: 'POST',
      body: { section: sharedValue('section'), domain: sharedValue('domain') },
    });
    if (result.order === null || result.order === undefined) return;
    sharedInputs.order.value = result.order;
    markDirty('order');
  } catch (failure) {
    editorStatus(failure.message, 'bad');
  }
}

function buildLangPanel(lang) {
  const code = lang.code;
  const inputs = verInputs[code];

  const removeButton = el('button', {
    class: 'w-btn small danger',
    type: 'button',
    text: 'Remove this version',
    dataset: { removeLang: code },
    hidden: true,
    onClick: () => removeVersion(code),
  });
  const head = el('div', { class: 'w-lang-head' }, [
    el('strong', { text: lang.label }),
    el('span', { class: 'w-lang-state', dataset: { langState: code } }),
    removeButton,
  ]);

  const fields = VERSION_FIELDS.map(field => {
    const wrap = el('div', { class: 'w-field', dataset: { field: field.key, scope: 'version', lang: code } });
    const id = `f-${code}-${field.key}`;
    wrap.append(el('label', { for: id, text: field.label }));
    const node = buildInput(field, id);
    inputs[field.key] = node;
    wrap.append(node);

    const touch = () => markDirty(`${code}.${field.key}`);
    node.addEventListener('input', touch);
    node.addEventListener('change', touch);

    if (field.key === 'title') {
      /* The slug follows the title until the author edits the slug by hand
         or the article has an address on the site already. */
      node.addEventListener('input', () => {
        const version = state.editor?.versions[code];
        if (!version || version.slugEdited || version.currentSlug) return;
        inputs.slug.value = slugify(node.value);
        markDirty(`${code}.slug`);
      });
    }
    if (field.key === 'slug') {
      node.addEventListener('input', () => {
        const version = state.editor?.versions[code];
        if (version) version.slugEdited = node.value.trim() !== '';
      });
      wrap.append(el('p', { class: 'hint', text: 'The address: /articles/… — follows the title until you change it.' }));
    }
    wrap.append(el('p', { class: 'error', hidden: true }));
    return wrap;
  });

  const hidden = el('input', { type: 'checkbox', id: `f-${code}-hidden`, onChange: () => markDirty(`${code}.hidden`) });
  inputs.hidden = hidden;
  const hiddenField = el('div', { class: 'w-field', dataset: { field: 'hidden', scope: 'version', lang: code } }, [
    el('span', { class: 'w-field-legend', text: 'On publish' }),
    el('label', { class: 'w-check', title: 'Publish the other language but keep this one as a draft for now.' }, [hidden, 'Keep this version hidden']),
  ]);

  return el('section', { class: 'w-lang-panel', dataset: { lang: code } }, [head, ...fields, hiddenField]);
}

function buildForms() {
  const featured = el('input', { type: 'checkbox', id: 'f-featured', onChange: () => markDirty('featured') });
  sharedInputs.featured = featured;
  $('#shared-grid').replaceChildren(
    ...SHARED_FIELDS.map(buildSharedField),
    el('div', { class: 'w-field', dataset: { field: 'featured', scope: 'shared' } }, [
      el('span', { class: 'w-field-legend', text: 'Home page' }),
      el('label', { class: 'w-check' }, [featured, 'Feature this article']),
    ]),
  );
  $('#lang-grid').replaceChildren(...LANGS.map(buildLangPanel));
}

/* Shows only the fields that mean something for the chosen section. */
function updateConditionalFields() {
  const section = sharedValue('section');
  for (const field of SHARED_FIELDS) {
    if (!field.when) continue;
    const wrap = $(`.w-field[data-scope="shared"][data-field="${field.key}"]`);
    wrap.hidden = field.when !== section;
  }
}

/* Details: reading the form ------------------------------------------------ */

const blankVersion = () => ({
  currentSlug: null, slug: '', title: '', description: '', body: '', hidden: false, draft: null, slugEdited: false,
});

function collectShared() {
  const section = sharedValue('section');
  const shared = { section, pubDate: sharedValue('pubDate'), featured: sharedInputs.featured.checked };
  if (section === 'Learn') {
    shared.domain = sharedValue('domain');
    shared.order = sharedValue('order');
  }
  if (section === 'Algorithms') {
    shared.topic = sharedValue('topic');
    shared.type = sharedValue('type');
  }
  return shared;
}

/* Copies a language's inputs back into its record and returns the record. */
function collectVersion(code) {
  const version = state.editor.versions[code];
  const inputs = verInputs[code];
  version.title = inputs.title.value;
  version.slug = inputs.slug.value;
  version.description = inputs.description.value;
  version.hidden = inputs.hidden.checked;
  if (state.editor.active === code && state.view === 'write') version.body = $('#body-input').value;
  return version;
}

const isPresent = version =>
  Boolean(version.slug.trim() || version.title.trim() || version.description.trim() || version.body.trim());

function collectPair(publish) {
  const editor = state.editor;
  const versions = {};
  for (const lang of LANGS) {
    const version = collectVersion(lang.code);
    if (!isPresent(version) && !version.currentSlug) continue;
    versions[lang.code] = {
      currentSlug: version.currentSlug,
      slug: version.slug,
      title: version.title,
      description: version.description,
      body: version.body,
      hidden: version.hidden,
    };
  }
  return { key: editor.key, shared: collectShared(), versions, publish };
}

/* Loads a language record into its panel. */
function fillVersion(code) {
  const version = state.editor.versions[code];
  const inputs = verInputs[code];
  inputs.title.value = version.title;
  inputs.slug.value = version.slug;
  inputs.description.value = version.description;
  inputs.hidden.checked = version.hidden;
}

function versionStateText(version) {
  if (version.currentSlug) return version.draft ? 'draft' : 'live';
  return isPresent(version) ? 'not saved' : 'not written';
}

function updateLangStates() {
  const editor = state.editor;
  if (!editor) return;
  for (const lang of LANGS) {
    const version = collectVersion(lang.code);
    const chip = $(`[data-lang-state="${lang.code}"]`);
    chip.textContent = versionStateText(version);
    chip.className = `w-lang-state ${version.currentSlug ? (version.draft ? 'draft' : 'live') : ''}`;
    $(`[data-remove-lang="${lang.code}"]`).hidden = !version.currentSlug;
  }
}

/* Validation --------------------------------------------------------------- */

function visibleErrors(errors) {
  if (!state.editor) return errors;
  if (state.editor.showErrors) return errors;
  return errors.filter(issue => state.editor.touched.has(issue.field));
}

function setFieldErrors(errors) {
  for (const wrap of document.querySelectorAll('.w-field[data-scope]')) {
    wrap.classList.remove('bad');
    const message = wrap.querySelector('.error');
    if (message) {
      message.hidden = true;
      message.textContent = '';
    }
  }
  for (const issue of errors) {
    const [first, second] = issue.field.split('.');
    const wrap = second === undefined
      ? $(`.w-field[data-scope="shared"][data-field="${first}"]`)
      : $(`.w-field[data-lang="${first}"][data-field="${second}"]`);
    if (!wrap) continue;
    wrap.classList.add('bad');
    const message = wrap.querySelector('.error');
    if (message) {
      message.textContent = issue.message;
      message.hidden = false;
    }
  }
}

/* Both screens show the same notes, so switching between them loses nothing. */
function renderNotes({ problems = [], warnings = [], log = '' }) {
  for (const host of [$('#details-notes'), $('#editor-notes')]) {
    const items = [
      ...problems.map(text => el('li', { class: 'problem', text })),
      ...warnings.map(text => el('li', { class: 'warn', text })),
    ];
    const children = [];
    if (items.length) children.push(el('ul', {}, items));
    if (log) children.push(el('pre', { text: log }));
    host.replaceChildren(...children);
    host.hidden = !children.length;
  }
}

function editorStatus(message, kind = '') {
  const node = state.view === 'details' ? $('#details-status') : $('#editor-status');
  node.textContent = message || '';
  node.className = `w-status ${kind}`;
  if (message) announce(message);
}

let validateTimer = null;
function scheduleValidate() {
  clearTimeout(validateTimer);
  validateTimer = setTimeout(runValidate, 700);
}

async function runValidate() {
  if (!state.editor) return null;
  try {
    const result = await api('pair/validate', { method: 'POST', body: collectPair(true) });
    if (!state.editor) return null;
    const shown = visibleErrors(result.fields);
    setFieldErrors(shown);
    renderNotes({ warnings: [...state.editor.notes, ...result.warnings] });

    const problems = { en: 0, vi: 0, shared: 0 };
    for (const issue of shown) {
      const [first, second] = issue.field.split('.');
      if (second !== undefined) problems[first] += 1;
      else problems.shared += 1;
    }
    state.editor.problems = problems;
    updateLangStates();
    renderTabs();
    return result;
  } catch (failure) {
    editorStatus(failure.message, 'bad');
    return null;
  }
}

/* Details: moving on -------------------------------------------------------- */

function updateTitles() {
  const editor = state.editor;
  if (!editor) return;
  const active = collectVersion(editor.active);
  const other = LANGS.map(lang => collectVersion(lang.code)).find(version => version.title.trim());
  const name = active.title.trim() || other?.title.trim() || 'Untitled';
  const states = LANGS
    .filter(lang => editor.versions[lang.code].currentSlug)
    .map(lang => `${lang.code.toUpperCase()} ${editor.versions[lang.code].draft ? 'draft' : 'live'}`);
  const title = `${name} — ${states.length ? states.join(' · ') : 'new'}`;
  $('#editor-title').textContent = title;
  $('#details-title').textContent = editor.mode === 'new' && !states.length ? 'New article' : title;
}

function showDetails({ focus } = {}) {
  /* Opened to add a language: Continue should land in that language. */
  if (state.editor) state.editor.wantedLang = focus || null;
  show('details');
  updateConditionalFields();
  updateLangStates();
  updateTitles();
  editorStatus('');
  runValidate();
  const target = focus ? verInputs[focus]?.title : null;
  if (target) target.focus();
}

async function continueToWrite() {
  const editor = state.editor;
  editor.showErrors = true;
  const validation = await runValidate();
  if (!validation) return;
  if (validation.fields.length) {
    editorStatus('Fix the fields marked in red first.', 'bad');
    const bad = document.querySelector('.w-field.bad input, .w-field.bad select, .w-field.bad textarea');
    if (bad) bad.focus();
    return;
  }
  const present = LANGS.map(lang => lang.code).filter(code => isPresent(editor.versions[code]));
  if (editor.wantedLang && present.includes(editor.wantedLang)) editor.active = editor.wantedLang;
  if (!present.includes(editor.active)) editor.active = present[0] || 'en';
  editor.wantedLang = null;
  enterWrite();
}

/* Writing: language tabs ---------------------------------------------------- */

function loadBody() {
  const editor = state.editor;
  $('#body-input').value = editor.versions[editor.active].body;
  $('#markdown-label').textContent = `Markdown · ${langLabel(editor.active)}`;
}

function enterWrite() {
  const editor = state.editor;
  show('write');
  loadBody();
  if (editor.previewLang && editor.previewLang !== editor.active) resetPreview();
  renderTabs();
  updateTitles();
  editorStatus('');
  runValidate();
  schedulePreview(400);
  $('#body-input').focus();
}

function switchLanguage(code) {
  const editor = state.editor;
  if (!editor || editor.active === code) return;
  collectVersion(editor.active);
  const target = editor.versions[code];
  if (!isPresent(target) && !target.currentSlug) {
    /* Nothing written in this language yet: it starts with a title. */
    showDetails({ focus: code });
    return;
  }
  editor.active = code;
  loadBody();
  if (editor.previewLang !== code) resetPreview();
  editorStatus('');
  renderTabs();
  updateTitles();
  runValidate();
  schedulePreview(300);
}

function renderTabs() {
  const editor = state.editor;
  if (!editor) return;
  const tabs = LANGS.map(lang => {
    const version = editor.versions[lang.code];
    const problems = editor.problems?.[lang.code] ?? 0;
    const present = isPresent(version) || version.currentSlug;
    return el('button', {
      class: 'w-tab',
      type: 'button',
      role: 'tab',
      'aria-selected': String(editor.active === lang.code),
      dataset: { lang: lang.code },
      title: present ? undefined : `Add a ${lang.label} version`,
      onClick: () => switchLanguage(lang.code),
    }, [
      present ? lang.label : `+ ${lang.label}`,
      present ? el('span', { class: `w-tab-state ${version.currentSlug ? (version.draft ? 'draft' : 'live') : ''}`, text: versionStateText(version) }) : null,
      problems ? el('span', { class: 'w-tab-count', text: `${problems} to fix` }) : null,
    ]);
  });
  const shared = editor.problems?.shared ?? 0;
  if (shared) {
    tabs.push(el('button', {
      class: 'w-tab-note',
      type: 'button',
      text: `${shared} detail${shared === 1 ? '' : 's'} to fix — open Details`,
      onClick: () => showDetails(),
    }));
  }
  $('#lang-tabs').replaceChildren(...tabs);
}

/* Markdown toolbar --------------------------------------------------------- */

const TOOLS = [
  { label: 'H2', title: 'Heading 2', run: () => prefixLines('## ') },
  { label: 'H3', title: 'Heading 3', run: () => prefixLines('### ') },
  { label: 'B', title: 'Bold', run: () => wrapSelection('**', '**') },
  { label: 'I', title: 'Italic', run: () => wrapSelection('_', '_') },
  { label: '</>', title: 'Inline code', mono: true, run: () => wrapSelection('`', '`') },
  { label: '{ }', title: 'Code block', mono: true, run: () => wrapSelection('```text\n', '\n```') },
  { label: 'Link', title: 'Link', run: () => wrapSelection('[', '](https://)') },
  { label: 'List', title: 'Bulleted list', run: () => prefixLines('- ') },
  { label: 'Quote', title: 'Blockquote', run: () => prefixLines('> ') },
  { label: 'Table', title: 'Table', run: () => insertAtCursor('\n| Column | Column |\n| --- | --- |\n| Value | Value |\n') },
];

function textarea() {
  return $('#body-input');
}

function replaceRange(start, end, text, selectStart, selectEnd) {
  const node = textarea();
  node.setRangeText(text, start, end, 'preserve');
  node.focus();
  if (selectStart !== undefined) node.setSelectionRange(selectStart, selectEnd ?? selectStart);
  bodyChanged();
}

function wrapSelection(before, after) {
  const node = textarea();
  const { selectionStart: start, selectionEnd: end } = node;
  const selected = node.value.slice(start, end);
  replaceRange(start, end, `${before}${selected}${after}`, start + before.length, start + before.length + selected.length);
}

function prefixLines(prefix) {
  const node = textarea();
  const { selectionStart: start, selectionEnd: end } = node;
  const lineStart = node.value.lastIndexOf('\n', start - 1) + 1;
  const lineEnd = node.value.indexOf('\n', end) === -1 ? node.value.length : node.value.indexOf('\n', end);
  const block = node.value.slice(lineStart, lineEnd);
  const updated = block.split('\n').map(line => (line.startsWith(prefix) ? line.slice(prefix.length) : prefix + line)).join('\n');
  replaceRange(lineStart, lineEnd, updated, lineStart, lineStart + updated.length);
}

function insertAtCursor(text) {
  const node = textarea();
  replaceRange(node.selectionStart, node.selectionEnd, text, node.selectionStart + text.length);
}

function buildTools() {
  $('#md-tools').replaceChildren(...TOOLS.map(tool =>
    el('button', {
      class: `w-tool${tool.mono ? ' mono' : ''}`,
      type: 'button',
      title: tool.title,
      'aria-label': tool.title,
      text: tool.label,
      onClick: tool.run,
    })));
}

function bodyChanged() {
  const editor = state.editor;
  if (!editor) return;
  editor.versions[editor.active].body = $('#body-input').value;
  markDirty(`${editor.active}.body`);
}

/* Preview ------------------------------------------------------------------ */

const autoOn = () => $('#auto-preview').checked;

function resetPreview(message) {
  const frame = $('#preview-frame');
  frame.hidden = true;
  frame.removeAttribute('src');
  $('#preview-idle').replaceChildren(
    el('strong', { text: 'Nothing rendered yet.' }),
    el('span', { text: message || 'The preview appears a moment after you stop typing. It is a real build of the site, so it shows exactly what will be published.' }),
  );
  $('#preview-idle').hidden = false;
  $('#preview-stale').hidden = true;
  if (state.editor) {
    state.editor.previewUrl = null;
    state.editor.previewLang = null;
    state.editor.previewSignature = null;
  }
}

let previewTimer = null;
let previewBusy = false;
let previewQueued = false;

/* Debounced: one build a moment after the last change, never one per key. */
function schedulePreview(delay = 1500) {
  clearTimeout(previewTimer);
  if (state.view !== 'write' || !autoOn()) return;
  previewTimer = setTimeout(() => refreshPreview({ auto: true }), delay);
}

async function refreshPreview({ auto = false } = {}) {
  const editor = state.editor;
  if (!editor || state.view !== 'write') return;
  clearTimeout(previewTimer);
  if (previewBusy) {
    previewQueued = true;
    return;
  }

  const active = editor.active;
  const version = collectVersion(active);
  const signature = JSON.stringify([active, collectShared(), {
    slug: version.slug, title: version.title, description: version.description, body: version.body,
  }]);
  if (auto && signature === editor.previewSignature) return;

  if (!auto) editor.showErrors = true;
  const validation = await runValidate();
  if (!validation || state.view !== 'write') return;

  const blocking = validation.fields.filter(issue => !issue.field.includes('.') || issue.field.startsWith(`${active}.`));
  if (blocking.length) {
    const labels = [...SHARED_FIELDS, ...VERSION_FIELDS];
    const names = [...new Set(blocking.map(issue => {
      const key = issue.field.split('.').pop();
      return labels.find(field => field.key === key)?.label ?? key;
    }))];
    $('#preview-idle').replaceChildren(
      el('strong', { text: 'Cannot preview yet.' }),
      el('span', { text: 'Fill in these first:' }),
      el('ul', {}, names.map(name => el('li', { text: name }))),
      el('button', { class: 'w-btn small', type: 'button', text: 'Open details', onClick: () => showDetails() }),
    );
    $('#preview-idle').hidden = false;
    $('#preview-frame').hidden = true;
    $('#preview-stale').hidden = true;
    editor.previewUrl = null;
    editorStatus(auto ? 'Preview paused — finish the details first.' : 'Fix the details before previewing.', auto ? '' : 'bad');
    return;
  }
  if (!isPresent(version)) {
    if (!auto) editorStatus('Write something in this language first.', 'bad');
    return;
  }

  previewBusy = true;
  editorStatus('Building the preview…', 'busy');
  try {
    const shared = collectShared();
    const result = await api('preview', {
      method: 'POST',
      body: {
        slug: version.slug,
        frontmatter: { ...shared, title: version.title, description: version.description, language: active },
        body: version.body,
      },
    });
    const frame = $('#preview-frame');
    frame.src = `${result.url}?t=${Date.now()}`;
    frame.hidden = false;
    $('#preview-idle').hidden = true;
    $('#preview-stale').hidden = true;
    editor.previewUrl = result.url;
    editor.previewLang = active;
    editor.previewSignature = signature;
    editorStatus('Preview is up to date.', 'good');
  } catch (failure) {
    editorStatus(failure.message, 'bad');
    if (failure.data.buildLog) renderNotes({ problems: [failure.message], log: failure.data.buildLog });
  } finally {
    previewBusy = false;
    if (previewQueued) {
      previewQueued = false;
      schedulePreview(300);
    }
  }
}

/* Saving ------------------------------------------------------------------- */

async function save(publish) {
  const editor = state.editor;
  editor.showErrors = true;
  const validation = await runValidate();
  if (!validation) return;
  if (!validation.valid) {
    showDetails();
    editorStatus('Some details are missing or invalid — fix the fields marked in red.', 'bad');
    return;
  }

  const pair = collectPair(publish);
  const count = Object.keys(pair.versions).length;

  setBusy(true);
  editorStatus(publish ? 'Publishing — building the site…' : 'Saving the draft…', 'busy');
  try {
    const result = await api('pair', { method: 'POST', body: pair });
    editor.mode = 'edit';
    editor.key = result.key;
    for (const [code, saved] of Object.entries(result.versions)) {
      const version = editor.versions[code];
      version.currentSlug = saved.slug;
      version.slug = saved.slug;
      version.slugEdited = true;
      version.draft = saved.draft;
      verInputs[code].slug.value = saved.slug;
    }
    editor.dirty = false;
    editor.notes = [];
    editor.touched = new Set();
    editor.showErrors = false;
    updateLangStates();
    renderTabs();
    updateTitles();
    renderNotes({ warnings: result.warnings || [] });
    editorStatus(
      publish
        ? `Published${count > 1 ? ' in both languages' : ''}. The site has been rebuilt.`
        : `Draft saved${count > 1 ? ' in both languages' : ''}. It is not on the public site.`,
      'good',
    );
  } catch (failure) {
    editorStatus(failure.message, 'bad');
    if (Array.isArray(failure.data.fields)) {
      showDetails();
      setFieldErrors(failure.data.fields);
      editorStatus(failure.message, 'bad');
    }
    if (failure.data.buildLog) renderNotes({ problems: [failure.message], log: failure.data.buildLog });
  } finally {
    setBusy(false);
  }
}

function removeVersion(code) {
  const editor = state.editor;
  const version = editor.versions[code];
  if (!version.currentSlug) return;
  const label = langLabel(code);
  openModal((card, close) => {
    const error = el('p', { class: 'w-status bad', hidden: true });
    card.append(
      el('h2', { text: `Remove the ${label} version?` }),
      el('p', { text: `/articles/${version.currentSlug} will be deleted${version.draft ? '' : ' and will stop working on the site'}. The other language stays as it is. A copy is kept in .writer/trash.` }),
      error,
      el('div', { class: 'w-modal-actions' }, [
        el('button', { class: 'w-btn', type: 'button', text: 'Cancel', onClick: close }),
        el('button', {
          class: 'w-btn danger',
          type: 'button',
          text: 'Remove',
          onClick: async () => {
            try {
              await api(`posts/${version.currentSlug}`, { method: 'DELETE', body: { confirm: version.currentSlug } });
              close();
              editor.versions[code] = blankVersion();
              fillVersion(code);
              const other = LANGS.find(lang => lang.code !== code && editor.versions[lang.code].currentSlug);
              if (editor.active === code && other) editor.active = other.code;
              editor.touched = new Set();
              if (editor.previewLang === code) resetPreview();
              updateLangStates();
              updateTitles();
              editorStatus(`${label} version removed.`, 'good');
              runValidate();
            } catch (failure) {
              error.textContent = failure.message;
              error.hidden = false;
            }
          },
        }),
      ]),
    );
  });
}

function setBusy(busy) {
  state.busy = busy;
  for (const button of document.querySelectorAll('#view-details [data-action], #view-write [data-action]')) button.disabled = busy;
}

/* Editor entry points ------------------------------------------------------ */

function openEditorWith({ mode, key = null, shared = {}, versions = {}, active = 'en', notes = [], view = 'details' }) {
  state.editor = {
    mode,
    key,
    active,
    versions: {
      en: { ...blankVersion(), ...(versions.en || {}) },
      vi: { ...blankVersion(), ...(versions.vi || {}) },
    },
    dirty: false,
    notes,
    touched: new Set(),
    showErrors: false,
    problems: { en: 0, vi: 0, shared: 0 },
    previewUrl: null,
    previewLang: null,
    previewSignature: null,
    wantedLang: null,
  };

  for (const field of SHARED_FIELDS) {
    const given = shared[field.key];
    sharedInputs[field.key].value = given === undefined || given === null ? '' : String(given);
  }
  sharedInputs.featured.checked = shared.featured === true;
  for (const lang of LANGS) fillVersion(lang.code);

  resetPreview();
  $('#details-heading').textContent = mode === 'new' ? 'New article' : 'Article details';
  renderNotes({ warnings: notes });
  setBusy(false);

  if (view === 'write') enterWrite();
  else showDetails({ focus: mode === 'new' ? active : null });
}

function newArticle() {
  openEditorWith({
    mode: 'new',
    shared: { section: 'Learn', pubDate: state.meta.today, featured: false },
    active: 'en',
    view: 'details',
  });
}

async function openEditor(slug) {
  try {
    const pair = await api(`pair/${slug}`);
    const versions = {};
    const notes = [];
    const anyLive = Object.values(pair.versions).some(version => !version.draft);
    for (const [code, version] of Object.entries(pair.versions)) {
      versions[code] = {
        currentSlug: version.slug,
        slug: version.slug,
        title: version.title,
        description: version.description,
        body: version.body,
        /* Ticked only when this language was deliberately held back while
           the other went live; a plain draft should publish on Publish. */
        hidden: version.draft && anyLive,
        draft: version.draft,
        slugEdited: true,
      };
      if (version.parseError) notes.push(`${code.toUpperCase()}: ${version.parseError}`);
    }
    const opened = Object.keys(versions).find(code => versions[code].slug === slug) || Object.keys(versions)[0] || 'en';
    openEditorWith({ mode: 'edit', key: pair.key, shared: pair.shared, versions, active: opened, notes, view: 'write' });
  } catch (failure) {
    dashStatus(failure.message, 'bad');
  }
}

async function leaveEditor() {
  if (state.editor && state.editor.dirty) {
    const stay = !window.confirm('This article has unsaved changes. Leave anyway?');
    if (stay) return;
  }
  clearTimeout(previewTimer);
  clearTimeout(validateTimer);
  state.editor = null;
  show('dashboard');
  dashStatus('');
  await loadPosts();
}

/* Wiring ------------------------------------------------------------------- */

const ACTIONS = {
  new: newArticle,
  import: openImport,
  images: openImages,
  back: leaveEditor,
  details: () => {
    if (state.editor) collectVersion(state.editor.active);
    showDetails();
  },
  continue: continueToWrite,
  preview: () => refreshPreview({ auto: false }),
  'preview-open': () => {
    if (state.editor && state.editor.previewUrl) window.open(state.editor.previewUrl, '_blank', 'noreferrer');
    else editorStatus('Render a preview first.', 'bad');
  },
  'save-draft': () => save(false),
  publish: () => save(true),
  theme: () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('theme', next);
    } catch {
      /* Nothing to persist to; the toggle still works for this session. */
    }
    /* The preview is a real page that reads the same stored theme, but it was
       loaded before the switch — move it over so the two panes match. */
    const preview = $('#preview-frame').contentDocument;
    if (preview) preview.documentElement.dataset.theme = next;
  },
  logout: async () => {
    await api('logout', { method: 'POST' });
    state.session = null;
    state.editor = null;
    showLogin('Signed out.');
  },
  rebuild: async () => {
    dashStatus('Rebuilding the site…', 'busy');
    try {
      await api('rebuild', { method: 'POST' });
      dashStatus('Site rebuilt from the current files.', 'good');
    } catch (failure) {
      dashStatus(failure.message, 'bad');
      if (failure.data.buildLog) openBuildLog(failure.message, failure.data.buildLog);
    }
  },
};

document.addEventListener('click', event => {
  const trigger = event.target.closest('[data-action]');
  if (!trigger || !ACTIONS[trigger.dataset.action]) return;
  event.preventDefault();
  ACTIONS[trigger.dataset.action]();
});

document.addEventListener('click', event => {
  const trigger = event.target.closest('[data-view-set]');
  if (!trigger) return;
  const view = trigger.dataset.viewSet;
  $('#panes').dataset.view = view;
  for (const button of document.querySelectorAll('[data-view-set]')) {
    button.setAttribute('aria-pressed', String(button.dataset.viewSet === view));
  }
  if (view !== 'write' && state.editor && !state.editor.previewUrl) refreshPreview({ auto: true });
});

$('#auto-preview').addEventListener('change', () => {
  if (autoOn()) {
    $('#preview-stale').hidden = true;
    schedulePreview(200);
  }
});

$('#dash-search').addEventListener('input', event => {
  state.search = event.target.value;
  renderList();
});

$('#body-input').addEventListener('input', bodyChanged);

$('#body-input').addEventListener('keydown', event => {
  if (event.key !== 'Tab' || event.shiftKey) return;
  event.preventDefault();
  insertAtCursor('  ');
});

/* Enter in a one-line details field moves on, like a form. */
$('#view-details').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.target.tagName !== 'INPUT' || event.target.type === 'checkbox') return;
  event.preventDefault();
  continueToWrite();
});

document.addEventListener('keydown', event => {
  if (!(event.metaKey || event.ctrlKey) || !state.editor) return;
  if (event.key === 's' && state.view === 'write') {
    event.preventDefault();
    save(false);
  }
  if (event.key === 'Enter') {
    event.preventDefault();
    if (state.view === 'write') refreshPreview({ auto: false });
    else if (state.view === 'details') continueToWrite();
  }
});

window.addEventListener('beforeunload', event => {
  if (state.editor && state.editor.dirty) event.preventDefault();
});

/* Start -------------------------------------------------------------------- */

async function enter() {
  state.meta = await api('meta');
  $('#brand-name').textContent = state.meta.site.name;
  document.title = `Writer — ${state.meta.site.name}`;
  buildForms();
  buildTools();
  renderFilters();
  show('dashboard');
  await loadPosts();
}

async function boot() {
  try {
    const session = await api('session');
    if (session.authenticated) {
      state.session = { username: session.username };
      await enter();
      return;
    }
  } catch (failure) {
    showLogin(failure.status === 503 ? failure.message : '');
    return;
  }
  showLogin('');
}

boot();

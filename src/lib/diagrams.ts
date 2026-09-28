import { createSvgCache } from './svg-cache';

type Theme = 'dark' | 'neutral';
type Mermaid = {
  initialize: (config: { startOnLoad: boolean; theme: Theme; fontFamily: string }) => void;
  render: (id: string, source: string) => Promise<{ svg: string }>;
};

const svgCache = createSvgCache();
let library: Promise<Mermaid> | undefined;
let renderQueue: Promise<unknown> = Promise.resolve();
let nextId = 0;
const sources = new WeakMap<HTMLPreElement, string>();
const cacheKey = (page: string, index: number, source: string, theme: Theme, font: string) =>
  JSON.stringify([page.replace(/\/$/, ''), index, source, theme, font]);

function showSvg(pre: HTMLPreElement, svg: string, theme: Theme) {
  pre.innerHTML = svg;
  pre.classList.remove('diagram-pending');
  pre.classList.add('mermaid', 'rendered');
  pre.removeAttribute('style');
  pre.removeAttribute('aria-busy');
  pre.dataset.diagramTheme = theme;
  const drawn = pre.querySelector('svg');
  const natural = parseFloat(drawn?.style.maxWidth || '');
  if (drawn && natural) drawn.style.minWidth = `${Math.min(natural, 520)}px`;
}

function placeholder(pre: HTMLPreElement, vi: boolean) {
  pre.classList.add('mermaid', 'diagram-pending');
  pre.removeAttribute('style');
  pre.setAttribute('aria-busy', 'true');
  pre.textContent = vi ? 'Sơ đồ sẽ hiển thị khi bạn cuộn tới đây.' : 'Diagram loads as you scroll closer.';
}

/* Astro restores scroll before page-load. Prepare the incoming document while
   it is still detached, so Back sees the cached diagrams' actual dimensions. */
export function prepareDiagrams(doc: Document, url: URL) {
  const theme: Theme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'neutral';
  const font = getComputedStyle(document.body).fontFamily;
  doc.querySelectorAll<HTMLPreElement>('.prose pre[data-language="mermaid"]').forEach((pre, index) => {
    const source = pre.textContent ?? '';
    sources.set(pre, source);
    const cached = svgCache.get(cacheKey(url.pathname, index, source, theme, font));
    if (cached) showSvg(pre, cached, theme);
    else placeholder(pre, doc.documentElement.lang === 'vi');
  });
}

/* One shared load, including when the reader changes articles mid-download.
   Removing an already-started script during a swap does not undo its load. */
function loadMermaid(): Promise<Mermaid> {
  const browser = window as Window & { mermaid?: Mermaid };
  if (browser.mermaid) return Promise.resolve(browser.mermaid);
  if (!library) {
    library = new Promise<Mermaid>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js';
      script.async = true;
      const timer = window.setTimeout(() => fail(new Error('Mermaid download timed out')), 15000);
      const fail = (error: Error) => {
        clearTimeout(timer);
        script.remove();
        reject(error);
      };
      script.onload = () => {
        clearTimeout(timer);
        if (browser.mermaid) resolve(browser.mermaid);
        else fail(new Error('Mermaid did not initialize'));
      };
      script.onerror = () => fail(new Error('Mermaid download failed'));
      document.head.appendChild(script);
    }).catch(error => {
      library = undefined;
      throw error;
    });
  }
  return library;
}

export function mountDiagrams(
  diagrams: HTMLPreElement[],
  signal: AbortSignal,
  onRender: (pre: HTMLPreElement) => void,
) {
  if (signal.aborted || !diagrams.length) return;
  const root = document.documentElement;
  const readTheme = (): Theme => root.dataset.theme === 'dark' ? 'dark' : 'neutral';
  let theme = readTheme();
  const fontFamily = getComputedStyle(document.body).fontFamily;
  const page = location.pathname.replace(/\/$/, '');
  const vi = root.lang === 'vi';
  const timers = new Set<ReturnType<typeof setTimeout>>();

  const entries = diagrams.map((pre, index) => ({
    pre, index, source: sources.get(pre) ?? pre.textContent ?? '',
    near: false,
    drawnTheme: pre.dataset.diagramTheme as Theme | undefined,
    pendingTheme: undefined as Theme | undefined,
    failedTheme: undefined as Theme | undefined,
  }));
  type Entry = typeof entries[number];
  const keyFor = (entry: Entry, mode: Theme) =>
    cacheKey(page, entry.index, entry.source, mode, fontFamily);
  const alive = (entry: Entry, mode: Theme) =>
    !signal.aborted && entry.pre.isConnected && theme === mode;

  const show = (entry: Entry, svg: string, mode: Theme) => {
    if (!alive(entry, mode)) return;
    const { pre } = entry;
    showSvg(pre, svg, mode);
    entry.drawnTheme = mode;
    onRender(pre);
  };

  const request = (entry: Entry) => {
    const mode = theme;
    if (!alive(entry, mode) || entry.drawnTheme === mode || entry.pendingTheme === mode) return;
    const key = keyFor(entry, mode);
    const cached = svgCache.get(key);
    if (cached) {
      show(entry, cached, mode);
      return;
    }
    if (!entry.near || entry.failedTheme === mode) return;
    entry.pendingTheme = mode;

    /* Let the short page transition finish first. One diagram per task keeps
       scrolling/clicks responsive instead of blocking on a whole article. */
    const timer = setTimeout(() => {
      timers.delete(timer);
      const task = renderQueue.then(async () => {
        if (!alive(entry, mode)) return;
        const mermaid = await loadMermaid();
        if (!alive(entry, mode)) return;
        await new Promise<void>(resolve => setTimeout(resolve, 0));
        if (!alive(entry, mode)) return;
        mermaid.initialize({ startOnLoad: false, theme: mode, fontFamily });
        const { svg } = await mermaid.render(`handbook-diagram-${++nextId}`, entry.source);
        // Cache completed work, but never insert it into a departed/stale page.
        svgCache.set(key, svg);
        show(entry, svg, mode);
      }).catch(error => {
        if (!alive(entry, mode)) return;
        entry.failedTheme = mode;
        entry.pre.classList.remove('diagram-pending');
        entry.pre.removeAttribute('aria-busy');
        // A CDN/parse failure still leaves the original diagram readable.
        if (!entry.drawnTheme) entry.pre.textContent = entry.source;
        console.warn('Diagram could not be drawn:', error);
      }).finally(() => {
        if (entry.pendingTheme === mode) entry.pendingTheme = undefined;
      });
      renderQueue = task;
    }, 120);
    timers.add(timer);
  };

  entries.forEach(entry => {
    const { pre } = entry;
    sources.set(pre, entry.source);
    if (entry.drawnTheme) onRender(pre);
    else placeholder(pre, vi);
    /* Restore ALL cached SVGs immediately, including those above the viewport,
       so browser Back can recover the previous reading position. */
    request(entry);
  });

  const byElement = new Map(entries.map(entry => [entry.pre, entry]));
  const observer = typeof IntersectionObserver === 'undefined' ? undefined : new IntersectionObserver(changes => {
    changes.forEach(change => {
      const entry = byElement.get(change.target as HTMLPreElement);
      if (!entry) return;
      entry.near = change.isIntersecting;
      if (entry.near) request(entry);
    });
  }, { rootMargin: '400px 0px' });
  if (observer) entries.forEach(entry => observer.observe(entry.pre));
  else entries.forEach(entry => { entry.near = true; request(entry); });

  const themeObserver = new MutationObserver(() => {
    const next = readTheme();
    if (next === theme || signal.aborted) return;
    theme = next;
    entries.forEach(request);
  });
  themeObserver.observe(root, { attributes: true, attributeFilter: ['data-theme'] });

  signal.addEventListener('abort', () => {
    observer?.disconnect();
    themeObserver.disconnect();
    timers.forEach(clearTimeout);
    timers.clear();
  }, { once: true });
}

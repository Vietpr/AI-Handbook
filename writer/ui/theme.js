/* Restores the site's theme before first paint so the console never flashes.
   Loaded synchronously from <head>; kept in its own file because the writer
   page runs under a script-src 'self' content security policy. */
(() => {
  try {
    const saved = localStorage.getItem('theme');
    const dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  } catch {
    document.documentElement.dataset.theme = 'light';
  }
})();

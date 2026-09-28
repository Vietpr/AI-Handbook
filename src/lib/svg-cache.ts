/* Bounded, session-only LRU cache. Keys include source, theme and font, so
   editing content or switching themes cannot reuse an unrelated SVG. */
export function createSvgCache(maxEntries = 96, maxCharacters = 2_000_000) {
  const entries = new Map<string, string>();
  let characters = 0;
  const remove = (key: string) => {
    const value = entries.get(key);
    if (value === undefined) return;
    characters -= key.length + value.length;
    entries.delete(key);
  };
  return {
    get(key: string) {
      const value = entries.get(key);
      if (value !== undefined) {
        entries.delete(key);
        entries.set(key, value);
      }
      return value;
    },
    set(key: string, value: string) {
      remove(key);
      const size = key.length + value.length;
      if (maxEntries < 1 || size > maxCharacters) return;
      entries.set(key, value);
      characters += size;
      while (entries.size > maxEntries || characters > maxCharacters) {
        remove(entries.keys().next().value!);
      }
    },
  };
}

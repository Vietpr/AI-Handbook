/*
 * Where articles live, and what "published" means.
 *
 * Everything above this line — validation, warnings, frontmatter handling, the
 * whole API and the whole UI — is written against the contract below and knows
 * nothing about files, builds, Git or hosting. Moving the writer onto a
 * different backend is a matter of writing one more module and changing the
 * import on the last line of this file.
 *
 * ---------------------------------------------------------------------------
 * The contract
 * ---------------------------------------------------------------------------
 *
 * listSlugs()                 -> string[]            every article id, sorted
 * exists(slug)                -> boolean
 * read(slug)                  -> string | null       the raw .md text
 *
 * commit(change)              -> { ok, log, published }
 *
 *     change = {
 *       writes:  [{ slug, contents }],   files to create or replace
 *       deletes: [slug],                 files to remove
 *       publish: boolean,                must the public site reflect this now?
 *       message: string,                 what happened, in one line
 *     }
 *
 *     Writes and deletes are applied together. When `publish` is true the
 *     change has to be live on the public site before this resolves. If that
 *     fails, the store puts the previous content back and returns ok:false with
 *     a log the author can read. A store may never leave a half-applied change
 *     behind.
 *
 * republish()                 -> { ok, log }         make the site match content
 * archive(slug, contents)     -> string | null       keep a copy of a deletion
 *
 * preview(slug, contents)     -> { ok, log, url }    render one draft privately
 * busy()                      -> string | null       what is running right now
 *
 * listAssets()                -> [{ name, url, size }]
 * assetExists(reference)      -> boolean             "/images/x.png" or "./x.png"
 * writeAsset(name, bytes)     -> boolean             false when the name is taken
 *
 * deployment()                -> { basePath, siteUrl }
 *
 * ---------------------------------------------------------------------------
 *
 * store-fs.mjs — the local implementation: real files under src/data/posts,
 * and `astro build` deciding whether a change is allowed to go live.
 *
 * A hosted implementation (commit through the Git provider's API and let the
 * host rebuild) fits the same contract: `commit` becomes one commit, `publish`
 * stops meaning "build now" and starts meaning "push to the deployed branch",
 * and `preview` points at a preview deployment instead of a local build.
 */
import filesystem from './store-fs.mjs';

export const store = filesystem;

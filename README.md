# How To Scale Your Model with AMD

A Distill-style Jekyll site, published at https://clarkechong.github.io/scale-your-amd.

The theme is a trimmed-down version of the one behind
[How To Scale Your Model](https://jax-ml.github.io/scaling-book), which itself uses
the [al-folio](https://github.com/alshedivat/al-folio) `distill` layout. Only the
pieces this book needs were kept: the distill layout and stylesheet, MathJax and
KaTeX math, Tufte-style margin sidenotes, dark mode, and the `figure` and
`details` helpers. See `LICENSE` (MIT) for attribution.

## Running locally

You need Ruby 3.1 or newer.

```bash
bundle install
bundle exec jekyll serve --livereload
```

The site is then at http://127.0.0.1:4000/scale-your-amd/. Note the trailing
path: `/` on its own returns 404, because `baseurl` in `_config.yml` is part of
every URL. If `baseurl` stops matching the repository name, every stylesheet and
script 404s and the site renders unstyled.

`--livereload` rebuilds and refreshes the browser on save, which takes about a
tenth of a second. It serves the reload socket on port 35729, so over SSH you
need that port forwarded as well as 4000. Edits to `_config.yml` are the one
thing not picked up automatically; restart the server after those.

## Layout

`index.md` at the repository root is the landing page. Every chapter and appendix is
one Markdown file in `pages/`, and all fifteen already exist as skeletons: correct
front matter and navigation, the section headings, and a `> **To write.**`
blockquote per section carrying the brief for it.

```bash
grep -rn '^> \*\*To write' pages/ | wc -l   # remaining work, by section
```

`docs/structure.md` is the roadmap those skeletons were generated from. It is the
document to argue with about what a chapter should contain; the skeletons are just
its shape in Jekyll form.

## Adding or editing a chapter

Copy any file in `pages/` as a starting point and set the front matter:

- `section_number` is the chapter number shown under the title. Appendices set
  `section_label` instead (`"Appendix A"`), which the layout uses in its place.
- `previous_section_url` / `next_section_url` are **site-root-relative** paths such
  as `/pages/moe`, and `previous_section_name` / `next_section_name` are their
  labels. The layout pipes the URLs through `relative_url`, so they pick up
  `baseurl` automatically and work from any directory depth. These drive both the
  arrows in the navbar and the line under the title. Nothing is inferred, so
  remember to update the neighbouring chapter too.
- `toc` is the sidebar table of contents. Each `name` must match a heading in the
  body exactly, because the anchor is the slugified name. Subsections nest under
  their parent entry:

  ```yaml
  toc:
    - name: Peak FLOPs by Dtype
      subsections:
        - name: Numeric Formats, Once, for the Whole Book
  ```

  **Avoid apostrophes and slashes in headings.** Kramdown deletes them when it
  generates heading IDs while Liquid's `slugify` converts them to hyphens, so the
  two disagree and the sidebar link 404s without the build complaining. Colons,
  commas and question marks are handled the same way by both.

Inside the body you can use:

- `[text]({{ '/pages/moe' | relative_url }})` for an internal link. A bare
  `/pages/moe` drops `baseurl` and 404s on the deployed site. Keep Liquid out of
  Markdown table cells: it renders before kramdown so it does work, but the pipe in
  a filter reads like a cell delimiter and the next person to edit the table will
  assume it is broken.
- `<d-footnote>...</d-footnote>` for a margin sidenote.
- `$x$` for inline math and `$$...$$` for display math. Inline math is rendered by
  the KaTeX bundled with the distill template; MathJax handles numbered AMS
  equations.
- `{% include figure.liquid path="assets/img/name.png" class="img-fluid" caption="..." %}`
  for a captioned figure. Wrap unused ones in `{% comment %}` rather than an HTML
  comment, because Liquid expands includes inside `<!-- -->` and leaves a broken
  image request in the built page.
- `{% include notation.liquid %}` for the book's notation table, so it cannot drift
  between chapters.
- `{% details Click here for the answer. %} ... {% enddetails %}` for a collapsible
  block, which is how worked-problem answers are hidden.

## Deployment

`.github/workflows/deploy.yml` builds the site and publishes it with
`actions/deploy-pages` on every push to `main`. This requires
**Settings → Pages → Source** to be set to **GitHub Actions**. There is no
`gh-pages` branch.

The workflow passes `--baseurl` from `actions/configure-pages`, so the deployed
path is correct even if the repository is renamed.

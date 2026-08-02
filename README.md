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
bundle exec jekyll serve
```

The site is then at http://127.0.0.1:4000/scale-your-amd/. The `baseurl` in
`_config.yml` is part of that path; if it stops matching the repository name,
every stylesheet and script 404s and the site renders unstyled.

## Adding a chapter

Each chapter is one Markdown file at the repository root with `layout: distill`.
Copy `profiling.md` as a starting point and set the front matter:

- `section_number` is the part number shown under the title.
- `previous_section_url` / `next_section_url` are relative links, and
  `previous_section_name` / `next_section_name` are their labels. These drive both
  the arrows in the navbar and the line under the title. Nothing is inferred, so
  remember to update the neighbouring chapter too.
- `toc` is the sidebar table of contents. Each `name` must match a heading in the
  body exactly, because the anchor is the slugified name.

Inside the body you can use:

- `<d-footnote>...</d-footnote>` for a margin sidenote.
- `$x$` for inline math and `$$...$$` for display math. Inline math is rendered by
  the KaTeX bundled with the distill template; MathJax handles numbered AMS
  equations.
- `{% include figure.liquid path="assets/img/name.png" class="img-fluid" caption="..." %}`
  for a captioned figure.
- `{% details Click here for the answer. %} ... {% enddetails %}` for a collapsible
  block, which is how worked-problem answers are hidden.

## Deployment

`.github/workflows/deploy.yml` builds the site and publishes it with
`actions/deploy-pages` on every push to `main`. This requires
**Settings → Pages → Source** to be set to **GitHub Actions**. There is no
`gh-pages` branch.

The workflow passes `--baseurl` from `actions/configure-pages`, so the deployed
path is correct even if the repository is renamed.

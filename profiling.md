---
layout: distill
title: "How to Profile AMD GPU Programs"
description: "A placeholder chapter that exercises every feature the layout supports: sidenotes, math, figures, code blocks and collapsible worked answers. Replace the prose, keep the machinery."
date: 2026-08-02

section_number: 1

previous_section_url: "../"
previous_section_name: "Part 0: Intro"

next_section_url: ""
next_section_name: "Part 2: TBD"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

# The table of contents is declared here, not inferred. Every `name` must match
# a heading in the body or its anchor link will 404.
toc:
  - name: "A Thousand-Foot View of the Stack"
  - name: "Reading a Profile"
  - subsections:
    - name: "Trace Viewer"
  - name: "Worked Problems"
---

## A Thousand-Foot View of the Stack

Body text. A sidenote lives in the margin on wide screens and collapses into the
appendix on narrow ones.<d-footnote>This is a sidenote. It is written inline with
a `d-footnote` tag and gets numbered automatically.</d-footnote>

**Bold topic sentences** are the house style: a reader skimming only the bold
should still follow the argument.

## Reading a Profile

Inline math renders with MathJax, so an arithmetic-per-byte ratio like
$2 \cdot B \cdot D \cdot F / (2BD + 2DF)$ works inline, and display math works
too:

$$\text{T}_{\text{math}} = \frac{2 \cdot B \cdot D \cdot F}{C}$$

A fenced code block gets syntax highlighting and a copy button:

```python
import torch

x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
y = x @ x
```

### Trace Viewer

Figures go through the `figure` include so they pick up captions and lazy
loading. Drop the image in `assets/img/` first, then uncomment the line below.
It is wrapped in a Liquid comment rather than an HTML one because Liquid still
expands includes inside `<!-- -->`, which would leave a broken image request in
the built page.

{% comment %}
{% include figure.liquid path="assets/img/example.png" class="img-fluid" caption="<b>Figure:</b> caption text." %}
{% endcomment %}

## Worked Problems

**Question 1:** State a problem the reader can actually attempt, with enough
numbers to check their answer.

{% details Click here for the answer. %}

The answer goes here, hidden until clicked. This is the `details` Liquid tag from
`_plugins/details.rb`, and its body is rendered as Markdown, so $x^2$ and code
blocks both work inside it.

{% enddetails %}
